"""The assistant graph with a scripted model: routing, receipt follow-ups,
memory, language, proposals, caching, and checkpoints that survive a new
process."""

import json
import os
import uuid
from datetime import date

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import text

from banko_ai.assistant import graph as G
from banko_ai.db.engine import get_engine
from banko_ai.services import memories, threads
from banko_ai.services.seeding import seed_user
from tests._chat import ScriptedChat
from tests._money import drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture(scope="module")
def member():
    uid = make_user("diner")
    seed_user(uid, months=2, today=date.today())
    yield uid
    with get_engine().begin() as c:
        for t in ("search_cache", "answer_cache", "memories", "assistant_actions", "receipts"):
            c.execute(text(f"DELETE FROM {t} WHERE user_id = CAST(:u AS UUID)"), {"u": uid})
    drop_user(uid)


@pytest.fixture
def thread(member):
    t = threads.create_thread(member)
    yield t["thread_id"]
    threads.delete_thread(member, t["thread_id"])


@pytest.fixture
def g():
    return G.build_graph(InMemorySaver())


def _turn(g, member, thread, message, chat, **kw):
    events = []
    out = G.run_turn(member, thread, message, chat=chat, graph=g, on_event=events.append, **kw)
    return out, events


def test_aggregation_answers_from_sql_and_streams(g, member, thread):
    chat = ScriptedChat("- Groceries are steady.")
    out, events = _turn(g, member, thread, "how much did I spend on groceries in the last 30 days", chat)
    assert out["route"]["kind"] == "aggregate"
    assert "You spent **$" in out["answer"] and "- Groceries are steady." in out["answer"]
    assert "".join(e["text"] for e in events if e["type"] == "token") == out["answer"]
    assert any(e["type"] == "step" and e["label"] == "Aggregation SQL" for e in events)


def test_receipt_follow_up_uses_the_active_receipt(g, member, thread):
    doc = str(uuid.uuid4())
    with get_engine().begin() as c:
        c.execute(text("INSERT INTO receipts (document_id, user_id, status, extracted) VALUES "
                       "(CAST(:d AS UUID), CAST(:u AS UUID), 'posted', CAST(:x AS JSONB))"),
                  {"d": doc, "u": member, "x": json.dumps({"merchant": "Trader Joe's", "amount": 23.47,
                                                           "items": ["Pinot Noir wine"], "date": "2026-10-10"})})
    G.set_active_receipt(thread, {"document_id": doc, "merchant": "Trader Joe's", "items": ["Pinot Noir wine"],
                                  "amount": 23.47}, graph=g)
    chat = ScriptedChat(lambda msgs: "It came to $23.47 at Trader Joe's.")
    out, _ = _turn(g, member, thread, "how much was this receipt?", chat)
    assert out["route"]["kind"] == "receipt"
    assert "23.47" in chat.prompts()[-1] and "Trader Joe" in chat.prompts()[-1]
    out2, _ = _turn(g, member, thread, "was the wine on it?", ScriptedChat("Yes, Pinot Noir."))
    assert out2["route"]["kind"] == "receipt"


def test_remember_and_forget(g, member, thread):
    out, _ = _turn(g, member, thread, "remember that I'm saving for a kayak", ScriptedChat("unused"))
    assert out["route"]["kind"] == "remember" and "kayak" in out["answer"]
    assert any("kayak" in m["content"] for m in memories.list_memories(member))
    out, _ = _turn(g, member, thread, "forget about the kayak", ScriptedChat("unused"))
    assert not any("kayak" in m["content"] for m in memories.list_memories(member))


def test_language_applies_to_deterministic_answers(g, member, thread):
    chat = ScriptedChat("Je ne peux aider qu'avec vos finances.")
    out, _ = _turn(g, member, thread, "what's the capital of France?", chat, language_code="fr-FR")
    assert out["route"]["kind"] == "offtopic" and out["answer"].startswith("Je ne peux")
    assert "French" in chat.prompts()[-1]


def _cite_first_fact(msgs):
    import re
    eid = re.search(r'"expense_id": "([0-9a-f-]{36})"', msgs[-1]["content"]).group(1)
    return f"You had pizza here [txn:{eid}]."


def test_plan_path_cites_and_caches(g, member, thread):
    chat = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"query": "pizza"}}]}',
                        '{"steps": []}', _cite_first_fact)
    out, events = _turn(g, member, thread, "where do I usually get pizza late at night?", chat)
    assert out["route"]["kind"] == "plan" and not out["cached"]
    cites = [e for e in events if e["type"] == "citation"]
    assert len(cites) == 1 and out["citations"][0]["expense_id"] == cites[0]["expense_id"]
    again, _ = _turn(g, member, thread, "where do I usually get pizza late at night?", ScriptedChat("unused"))
    assert again["cached"] and again["answer"] == out["answer"]


def test_a_broken_answer_cache_does_not_break_the_answer(g, member, thread, monkeypatch):
    from banko_ai.services import caches

    def boom(*a, **k):
        raise RuntimeError("cache down")
    monkeypatch.setattr(caches, "get_answer", boom)
    monkeypatch.setattr(caches, "put_answer", boom)
    chat = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"query": "coffee"}}]}',
                        '{"steps": []}', "You buy coffee often.")
    out, _ = _turn(g, member, thread, "where do I buy coffee on weekends?", chat)
    assert out["route"]["kind"] == "plan" and out["answer"] == "You buy coffee often." and not out["cached"]


def test_proposals_are_events(g, member, thread):
    chat = ScriptedChat('{"steps": [{"tool": "propose_budget", "args": {"category": "Groceries", "amount": "400"}}]}',
                        '{"steps": []}', "I can set that for you.")
    out, events = _turn(g, member, thread, "set my grocery budget to 400", chat)
    props = [e for e in events if e["type"] == "proposal"]
    assert props and props[0]["status"] == "pending" and out["proposals"] == props


def test_history_is_bounded_and_titles_the_thread(g, member, thread, monkeypatch):
    monkeypatch.setenv("ASSISTANT_HISTORY_WINDOW", "4")
    from banko_ai.config.settings import reset_config
    reset_config()
    for i in range(4):
        _turn(g, member, thread, f"what's the capital of country {i}?", ScriptedChat("Only money questions."))
    assert len(G.thread_messages(thread, graph=g)) == 4
    assert threads.get_thread(member, thread)["title"].startswith("What's the capital")


def test_context_persists_for_the_thread(g, member, thread):
    G.set_context(thread, {"kind": "transaction", "id": str(uuid.uuid4())}, graph=g)
    chat = ScriptedChat('{"steps": []}', '{"steps": []}', "Here is what I see.")
    out, _ = _turn(g, member, thread, "how much did I spend on groceries last month", chat)
    assert out["route"]["kind"] == "plan"
    assert G.thread_state(thread, graph=g)["context"]["kind"] == "transaction"


def test_checkpoints_survive_a_new_process(member, thread):
    from banko_ai.assistant import checkpoint
    G.set_active_receipt(thread, {"document_id": "d-restart", "merchant": "REI", "items": [], "amount": 99.0},
                         graph=G.build_graph(checkpoint.get_saver()))
    fresh = G.build_graph(checkpoint.get_saver())
    state = fresh.get_state({"configurable": {"thread_id": thread}}).values
    assert state["active_receipt"]["merchant"] == "REI"
    checkpoint.delete_checkpoints(thread)
    assert fresh.get_state({"configurable": {"thread_id": thread}}).values == {}
