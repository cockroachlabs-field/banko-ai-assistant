"""The Agents feed: real decisions, receipts and nudges for this user,
stable ids, and confidence only when an agent recorded one."""

import json
import os
import uuid
from datetime import date

import pytest
from sqlalchemy import text

from banko_ai.db.engine import get_engine
from banko_ai.services import agent_feed
from banko_ai.services.seeding import seed_user
from tests._app import app_client, stop_apps
from tests._money import drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    stop_apps()


@pytest.fixture(scope="module")
def member():
    uid = make_user("diner")
    seed_user(uid, months=1, today=date.today())
    yield uid
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM receipts WHERE user_id = CAST(:u AS UUID)"), {"u": uid})
        c.execute(text("DELETE FROM coach_nudges WHERE user_id = CAST(:u AS UUID)"), {"u": uid})
    drop_user(uid)


def _expense(uid):
    with get_engine().connect() as c:
        return str(c.execute(text("SELECT expense_id FROM expenses WHERE user_id = CAST(:u AS UUID) LIMIT 1"), {"u": uid}).scalar())


def test_activity_merges_real_sources_with_stable_ids(member):
    eid = _expense(member)
    with get_engine().begin() as c:
        c.execute(text("INSERT INTO receipts (user_id, status, expense_id, extracted) VALUES (CAST(:u AS UUID), 'posted', "
                       "CAST(:x AS UUID), CAST(:e AS JSONB))"), {"u": member, "x": eid, "e": json.dumps({"merchant": "Corner Market", "amount": 9.5})})
        c.execute(text("INSERT INTO coach_nudges (user_id, message) VALUES (CAST(:u AS UUID), 'Coffee is up')"), {"u": member})
        c.execute(text("INSERT INTO agent_decisions (decision_type, context, reasoning, confidence) VALUES "
                       "('fraud_analysis', CAST(:ctx AS JSONB), 'Looks normal', 0.12)"), {"ctx": json.dumps({"expense_id": eid})})
    items = agent_feed.activity(member)
    agents = {i["agent"] for i in items}
    assert {"receipt", "coach", "fraud"} <= agents
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids))
    fraud = next(i for i in items if i["agent"] == "fraud")
    assert fraud["confidence"] == pytest.approx(0.12)
    assert all(i["confidence"] is None for i in items if i["agent"] in ("receipt", "coach"))


def test_activity_never_shows_another_users_items(member):
    other = make_user()
    try:
        assert agent_feed.activity(other) == []
    finally:
        drop_user(other)


def test_api_and_stats(monkeypatch, member):
    c = app_client(monkeypatch, REPORTING_FOLLOWER_READS="false")
    with c.session_transaction() as s:
        s["user_id"] = member
    assert "items" in c.get("/api/v1/agents/activity").get_json()
    stats = c.get("/api/v1/agents/stats").get_json()
    assert {"agents", "today"} <= set(stats)


def test_old_page_has_no_invented_confidence():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "banko_ai"
    for path in [root / "templates" / "pages" / "agents.html", root / "static" / "js" / "pages" / "agents.js"]:
        assert "0.9" not in path.read_text()
