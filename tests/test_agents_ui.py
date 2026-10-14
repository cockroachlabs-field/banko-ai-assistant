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


def _seeded(style):
    uid = make_user(style)
    seed_user(uid, months=1, today=date.today())
    return uid


def _drop_with_feed(uid):
    with get_engine().begin() as c:
        for table in ("receipts", "coach_nudges", "assistant_actions"):
            c.execute(text(f"DELETE FROM {table} WHERE user_id = CAST(:u AS UUID)"), {"u": uid})
        # agent_decisions has no user column: match what these tests write.
        c.execute(text("DELETE FROM agent_decisions WHERE context->>'user_id' = :u OR context->>'expense_id' IN "
                       "(SELECT expense_id::STRING FROM expenses WHERE user_id = CAST(:u AS UUID))"), {"u": uid})
    drop_user(uid)


@pytest.fixture(scope="module")
def member():
    uid = _seeded("diner")
    yield uid
    _drop_with_feed(uid)


@pytest.fixture(scope="module")
def neighbor():
    uid = _seeded("saver")
    yield uid
    _drop_with_feed(uid)


def _expense(uid):
    with get_engine().connect() as c:
        return str(c.execute(text("SELECT expense_id FROM expenses WHERE user_id = CAST(:u AS UUID) LIMIT 1"), {"u": uid}).scalar())


def _one_of_everything(uid, name):
    """A receipt, a nudge, a fraud check on one of uid's purchases, a budget
    check (which names the user, not an expense) and a proposal."""
    eid = _expense(uid)
    with get_engine().begin() as c:
        def one(sql, **params):
            return str(c.execute(text(sql), params).scalar())
        return {
            "receipt": "receipt:" + one("INSERT INTO receipts (user_id, status, expense_id, extracted) VALUES (CAST(:u AS UUID), "
                                        "'posted', CAST(:x AS UUID), CAST(:e AS JSONB)) RETURNING document_id",
                                        u=uid, x=eid, e=json.dumps({"merchant": f"{name} Market", "amount": 9.5})),
            "nudge": "nudge:" + one("INSERT INTO coach_nudges (user_id, message) VALUES (CAST(:u AS UUID), :m) RETURNING nudge_id",
                                    u=uid, m=f"{name}: coffee is up"),
            "fraud": "decision:" + one("INSERT INTO agent_decisions (decision_type, context, reasoning, confidence) VALUES "
                                       "('fraud_analysis', CAST(:ctx AS JSONB), :r, 0.05) RETURNING decision_id",
                                       ctx=json.dumps({"expense_id": eid}), r=f"{name}: looks normal"),
            "budget": "decision:" + one("INSERT INTO agent_decisions (decision_type, context, reasoning, confidence) VALUES "
                                        "('budget_check', CAST(:ctx AS JSONB), :r, 0.85) RETURNING decision_id",
                                        ctx=json.dumps({"user_id": uid, "status": "on_track"}), r=f"{name}: on track"),
            "action": "action:" + one("INSERT INTO assistant_actions (user_id, kind, args, summary, expires_at) VALUES "
                                      "(CAST(:u AS UUID), 'budget', CAST('{}' AS JSONB), :s, now() + INTERVAL '1 hour') "
                                      "RETURNING action_id", u=uid, s=f"Set {name}'s Groceries budget to $450.00 a month"),
        }


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


def test_activity_never_shows_another_users_items(member, neighbor):
    # Both users have data in every source, the shared agent_decisions included.
    mine, theirs = _one_of_everything(member, "Member"), _one_of_everything(neighbor, "Neighbor")
    seen_by_member = agent_feed.activity(member, limit=200)
    seen_by_neighbor = agent_feed.activity(neighbor, limit=200)
    assert set(mine.values()) <= {i["id"] for i in seen_by_member}
    assert set(theirs.values()) <= {i["id"] for i in seen_by_neighbor}
    assert not set(theirs.values()) & {i["id"] for i in seen_by_member}
    assert not set(mine.values()) & {i["id"] for i in seen_by_neighbor}
    assert "Neighbor" not in json.dumps(seen_by_member) and "Member" not in json.dumps(seen_by_neighbor)
    other = make_user()
    try:
        assert agent_feed.activity(other) == []
    finally:
        drop_user(other)


def test_budget_checks_and_proposals_show_and_count_today(member):
    ids = _one_of_everything(member, "Today")
    with get_engine().begin() as c:
        unknown = "decision:" + str(c.execute(text(
            "INSERT INTO agent_decisions (decision_type, context, reasoning, confidence) VALUES "
            "('workflow_execution', CAST(:ctx AS JSONB), 'Ran two steps', 0.9) RETURNING decision_id"),
            {"ctx": json.dumps({"user_id": member})}).scalar())
    items = {i["id"]: i for i in agent_feed.activity(member, limit=200)}
    assert items[ids["budget"]]["agent"] == "budget" and items[ids["budget"]]["confidence"] == pytest.approx(0.85)
    proposal = items[ids["action"]]
    assert proposal["agent"] == "assistant" and proposal["confidence"] is None
    assert proposal["title"] == "Set Today's Groceries budget to $450.00 a month"
    assert proposal["detail"] == "Waiting for your confirmation"
    # A decision type no agent mapping knows is not passed off as the assistant's.
    assert items[unknown]["agent"] == "workflow"
    today = agent_feed.stats(member)["today"]
    assert today.get("budget", 0) >= 1 and today.get("assistant", 0) >= 1


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
