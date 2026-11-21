"""A nudge is read by the person it was written for.

The old coach routes sit under /api/, where nobody is asked to sign in. One
took the user from the query string and one looked a nudge up by its id
alone. While everyone shared three personas that showed nothing new. With an
account per person it hands anyone anybody's nudges, with the budgets,
merchants and purchases in their tool trace."""

import json
import os
import uuid

import pytest
from sqlalchemy import text

from banko_ai.coach.agent import CoachAgent
from banko_ai.coach.tools import explain_nudge
from banko_ai.db.engine import get_engine
from tests._app import app_client, stop_apps
from tests._money import drop_user, make_user

DB = os.getenv("DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB, reason="DATABASE_URL not set")


def _nudge(user_id: str, message: str) -> str:
    with get_engine().begin() as conn:
        signal = conn.execute(text("""
            INSERT INTO spending_signals (user_id, signal_type, severity, payload, idempotency_key)
            VALUES (:u, 'budget_threshold', 'warn', CAST(:p AS JSONB), :k) RETURNING signal_id
        """), {"u": user_id, "p": '{"category": "dining", "pct_used": 0.82}', "k": f"scope:{uuid.uuid4()}"}).scalar()
        return str(conn.execute(text("""
            INSERT INTO coach_nudges (signal_id, user_id, message, tool_trace, provider_used)
            VALUES (:s, :u, :m, CAST(:t AS JSONB), 'stub') RETURNING nudge_id
        """), {"s": signal, "u": user_id, "m": message,
               "t": '[{"tool": "get_recent_transactions", "result": [{"merchant": "Blue Bottle"}]}]'}).scalar())


@pytest.fixture
def two_people():
    ana, ben = make_user(), make_user()
    nudges = {ana: _nudge(ana, "Ana is at 82% of her dining budget"),
              ben: _nudge(ben, "Ben is at 82% of his dining budget")}
    yield ana, ben, nudges
    with get_engine().begin() as conn:
        for user in (ana, ben):
            conn.execute(text("DELETE FROM coach_nudges WHERE user_id = :u"), {"u": user})
            conn.execute(text("DELETE FROM spending_signals WHERE user_id = :u"), {"u": user})
    drop_user(ana)
    drop_user(ben)
    stop_apps()


def _signed_in(monkeypatch, user_id: str):
    client = app_client(monkeypatch)
    with client.session_transaction() as session:
        session["user_id"] = user_id
    return client


def test_the_list_is_the_signed_in_users_whatever_the_query_names(monkeypatch, two_people):
    ana, ben, _ = two_people
    client = _signed_in(monkeypatch, ben)
    listed = client.get(f"/api/coach/nudges?user_id={ana}").get_json()["nudges"]
    assert [n["message"] for n in listed] == ["Ben is at 82% of his dining budget"]


def test_one_nudge_is_shown_to_its_owner_only(monkeypatch, two_people):
    ana, ben, nudges = two_people
    client = _signed_in(monkeypatch, ben)
    mine = client.get(f"/api/coach/nudges/{nudges[ben]}")
    assert mine.status_code == 200 and mine.get_json()["message"].startswith("Ben")
    hers = client.get(f"/api/coach/nudges/{nudges[ana]}")
    assert hers.status_code == 404 and "Ana" not in hers.get_data(as_text=True)


def test_the_coachs_own_lookup_is_for_its_user(two_people):
    ana, ben, nudges = two_people
    assert explain_nudge(user_id=ben, nudge_id=nudges[ben], database_url=DB)["message"].startswith("Ben")
    assert explain_nudge(user_id=ben, nudge_id=nudges[ana], database_url=DB) == {}
    assert explain_nudge(user_id=ben, nudge_id="not-a-nudge-id", database_url=DB) == {}


def test_a_follow_up_about_a_nudge_reads_only_the_askers(two_people):
    """The planner names the tool and its arguments; the user it runs for is
    the one who asked. (It used to fail outright: the agent hands every tool
    the user, and this one did not take it.)"""
    ana, ben, nudges = two_people

    def planner_that_asks_for(nudge_id: str):
        replies = iter([json.dumps({"steps": [{"tool": "explain_nudge", "args": {"nudge_id": nudge_id}}]}),
                        "Here is what that nudge was about."])
        return lambda messages: next(replies)

    own = CoachAgent(database_url=DB, llm_invoker=planner_that_asks_for(nudges[ben])).converse(
        ben, "why did I get this?", context={"nudge_id": nudges[ben]})
    assert own["tool_trace"][0]["result"]["message"].startswith("Ben")
    hers = CoachAgent(database_url=DB, llm_invoker=planner_that_asks_for(nudges[ana])).converse(
        ben, "why did she get this?", context={"nudge_id": nudges[ana]})
    assert hers["tool_trace"][0].get("result") == {} and "Ana" not in json.dumps(hers["tool_trace"])
