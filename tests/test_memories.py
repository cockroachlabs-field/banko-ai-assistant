"""Long-term memories: per user, deduplicated, recalled by meaning."""

import os

import pytest
from sqlalchemy import text

from banko_ai.db.engine import get_engine
from banko_ai.services import memories as M
from tests._money import drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture
def user():
    uid = make_user()
    yield uid
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM memories WHERE user_id = CAST(:u AS UUID)"), {"u": uid})
    drop_user(uid)


def test_remember_and_recall_by_meaning(user):
    M.remember(user, "I'm saving for a trip to Japan next fall")
    M.remember(user, "I prefer paying with my rewards card")
    hits = M.recall(user, "what am I saving money for?")
    assert hits and "Japan" in hits[0]["content"]
    assert hits[0]["kind"] == "goal"


def test_remember_deduplicates_and_counts_use(user):
    first = M.remember(user, "My rent is due on the first")
    again = M.remember(user, "My rent is due on the first.")
    assert first["created"] and not again["created"]
    assert again["memory_id"] == first["memory_id"]
    assert len(M.list_memories(user)) == 1


def test_recall_bumps_use_count(user):
    m = M.remember(user, "I usually buy coffee at Blue Bottle")
    M.recall(user, "where do I buy coffee")
    with get_engine().connect() as c:
        n = c.execute(text("SELECT use_count FROM memories WHERE memory_id = CAST(:m AS UUID)"),
                      {"m": m["memory_id"]}).scalar()
    assert n == 1


def test_memories_never_cross_users(user):
    other = make_user()
    try:
        M.remember(other, "I'm saving for a boat")
        assert M.recall(user, "saving for a boat") == []
        assert M.forget(user, M.list_memories(other)[0]["memory_id"]) is False
    finally:
        with get_engine().begin() as c:
            c.execute(text("DELETE FROM memories WHERE user_id = CAST(:u AS UUID)"), {"u": other})
        drop_user(other)


def test_forget_matching_removes_the_closest(user):
    M.remember(user, "I'm saving for a trip to Japan")
    M.remember(user, "I prefer cash at farmers markets")
    removed = M.forget_matching(user, "the Japan trip")
    assert removed == ["I'm saving for a trip to Japan"]
    assert [m["content"] for m in M.list_memories(user)] == ["I prefer cash at farmers markets"]


def test_kind_for():
    assert M.kind_for("my goal is an emergency fund") == "goal"
    assert M.kind_for("I prefer debit") == "preference"
    assert M.kind_for("my landlord is Greenway") == "fact"


def test_blank_memory_is_refused(user):
    with pytest.raises(ValueError):
        M.remember(user, "   ")


def test_agent_memory_access_count_increments():
    import inspect

    from banko_ai.agents import base_agent
    src = inspect.getsource(base_agent)
    assert "access_count = access_count + 1" in src
