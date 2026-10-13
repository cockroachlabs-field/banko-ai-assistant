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
    again, again_events = _turn(g, member, thread, "where do I usually get pizza late at night?",
                                ScriptedChat("unused"))
    assert again["cached"] and again["answer"] == out["answer"]
    again_cites = [e for e in again_events if e["type"] == "citation"]
    assert len(again_cites) == 1 and again_cites[0]["expense_id"] == cites[0]["expense_id"]
    assert [c["expense_id"] for c in again["citations"]] == [cites[0]["expense_id"]]


def _granite_cites_first_fact(msgs):
    """granite3.3:8b wraps the real id in angle brackets and adds a bare placeholder."""
    import re
    eid = re.search(r'"expense_id": "([0-9a-f-]{36})"', msgs[-1]["content"]).group(1)
    return f"You had pizza here [txn:<{eid}>] and once more [txn:<expense_id>]."


def _cold_answer_cache(member):
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM answer_cache WHERE user_id = CAST(:u AS UUID)"), {"u": member})


def test_granite_markers_resolve_or_disappear_and_the_cache_keeps_the_clean_text(g, member, thread):
    q = "which pizza places did I order from late at night?"
    _cold_answer_cache(member)
    chat = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"query": "pizza"}}]}',
                        '{"steps": []}', _granite_cites_first_fact)
    out, events = _turn(g, member, thread, q, chat)
    cites = [e for e in events if e["type"] == "citation"]
    assert len(cites) == 1 and cites[0]["amount"] > 0
    eid = cites[0]["expense_id"]
    assert out["answer"] == f"You had pizza here [txn:{eid}] and once more."
    with get_engine().connect() as c:
        cached = c.execute(text("SELECT answer FROM answer_cache WHERE user_id = CAST(:u AS UUID) "
                                "AND query_text = :q"), {"u": member, "q": q}).scalar()
    assert cached == out["answer"]
    fresh = threads.create_thread(member)["thread_id"]
    try:
        again, again_events = _turn(g, member, fresh, q, ScriptedChat("unused"))
    finally:
        threads.delete_thread(member, fresh)
    assert again["cached"] and again["answer"] == out["answer"]
    assert [e["expense_id"] for e in again_events if e["type"] == "citation"] == [eid]


def test_a_placeholder_marker_never_reaches_the_answer(g, member, thread):
    from tests.test_assistant_answers import GRANITE_PLACEHOLDER
    chat = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"query": "dining out"}}]}',
                        '{"steps": []}', GRANITE_PLACEHOLDER)
    out, events = _turn(g, member, thread, "where could I trim my dining out spending?", chat)
    assert "txn" not in out["answer"] and "could save money. Consider" in out["answer"]
    assert out["citations"] == [] and not [e for e in events if e["type"] == "citation"]


def test_a_memory_id_in_a_marker_is_dropped(g, member, thread):
    mid = memories.remember(member, "I'm saving for a trip to Japan")["memory_id"]
    chat = ScriptedChat('{"steps": [{"tool": "recall", "args": {"query": "what am I saving for"}}]}',
                        '{"steps": []}', f"You are saving for a trip to Japan. * Goal: Saving for Japan [txn:<{mid}>]")
    out, events = _turn(g, member, thread, "what savings goal am I working toward?", chat)
    assert "txn" not in out["answer"] and mid not in out["answer"]
    assert out["answer"].endswith("Saving for Japan")
    assert not [e for e in events if e["type"] == "citation"]


def _expense_of(member):
    with get_engine().connect() as c:
        return str(c.execute(text("SELECT expense_id FROM expenses WHERE user_id = CAST(:u AS UUID) LIMIT 1"),
                             {"u": member}).scalar())


def test_a_receipt_answer_with_an_angle_bracket_marker_cites_the_receipt(g, member, thread):
    doc, eid = str(uuid.uuid4()), _expense_of(member)
    with get_engine().begin() as c:
        c.execute(text("INSERT INTO receipts (document_id, user_id, status, extracted, expense_id) VALUES "
                       "(CAST(:d AS UUID), CAST(:u AS UUID), 'posted', CAST(:x AS JSONB), CAST(:e AS UUID))"),
                  {"d": doc, "u": member, "e": eid,
                   "x": json.dumps({"merchant": "Trader Joe's", "amount": 25.94, "items": ["Pinot Noir wine"],
                                    "date": "2026-10-13"})})
    G.set_active_receipt(thread, {"document_id": doc, "merchant": "Trader Joe's", "items": ["Pinot Noir wine"],
                                  "amount": 25.94, "expense_id": eid}, graph=g)
    chat = ScriptedChat(f"The amount of this receipt from Trader Joe's was $25.94 [txn:<{eid}>].")
    out, events = _turn(g, member, thread, "how much was this receipt?", chat)
    assert out["route"]["kind"] == "receipt"
    assert out["answer"] == f"The amount of this receipt from Trader Joe's was $25.94 [txn:{eid}]."
    cites = [e for e in events if e["type"] == "citation"]
    assert [c["expense_id"] for c in cites] == [eid] and cites[0]["amount"] == 25.94
    facts = chat.prompts()[-1]
    assert f'"cite": "[txn:{eid}]"' in facts


def _latest_action(member):
    with get_engine().connect() as c:
        return str(c.execute(text("SELECT action_id FROM assistant_actions WHERE user_id = CAST(:u AS UUID) "
                                  "ORDER BY created_at DESC LIMIT 1"), {"u": member}).scalar())


def test_a_proposal_answer_says_it_waits_for_confirmation(g, member, thread):
    chat = ScriptedChat('{"steps": [{"tool": "propose_budget", "args": {"category": "Groceries", "amount": "450"}}]}',
                        '{"steps": []}',
                        lambda msgs: f"Your groceries budget has been set to $450.00 a month. "
                                     f"[txn:<{_latest_action(member)}>]")
    out, events = _turn(g, member, thread, "set my groceries budget to 450", chat)
    action_id = out["proposals"][0]["action_id"]
    assert out["answer"].endswith("\n\nWaiting for your confirmation: Set your Groceries budget to $450.00 a month.")
    assert "txn" not in out["answer"] and action_id not in out["answer"]
    assert not [e for e in events if e["type"] == "citation"]
    facts = chat.prompts()[-1]
    assert action_id not in facts and "waiting for you to confirm" in facts
    assert "".join(e["text"] for e in events if e["type"] == "token").endswith(
        "Waiting for your confirmation: Set your Groceries budget to $450.00 a month.")


def test_the_confirmation_line_is_translated(g, member, thread):
    chat = ScriptedChat('{"steps": [{"tool": "propose_budget", "args": {"category": "Groceries", "amount": "460"}}]}',
                        '{"steps": []}', "Listo para confirmar.",
                        "Esperando tu confirmación: fija tu presupuesto de Groceries en $460.00 al mes.")
    out, _ = _turn(g, member, thread, "set my groceries budget to 460", chat, language_code="es-ES")
    assert out["answer"] == ("Listo para confirmar.\n\n"
                             "Esperando tu confirmación: fija tu presupuesto de Groceries en $460.00 al mes.")
    assert "Waiting for your confirmation: Set your Groceries budget to $460.00 a month." in chat.prompts()[-1]
    assert "Spanish" in chat.prompts()[-1]


def _answer_rows(member, query):
    with get_engine().connect() as c:
        return c.execute(text("SELECT count(*) FROM answer_cache WHERE user_id = CAST(:u AS UUID) "
                              "AND query_text = :q"), {"u": member, "q": query}).scalar()


def test_a_proposal_turn_is_not_cached(g, member, thread):
    q = "please set my dining budget to 250 a month"
    chat = ScriptedChat('{"steps": [{"tool": "propose_budget", "args": {"category": "Restaurant", "amount": "250"}}]}',
                        '{"steps": []}', "I can set that for you.")
    out, _ = _turn(g, member, thread, q, chat)
    assert out["route"]["kind"] == "plan" and out["proposals"]
    assert _answer_rows(member, q) == 0
    chat2 = ScriptedChat('{"steps": [{"tool": "propose_budget", "args": {"category": "Restaurant", "amount": "250"}}]}',
                         '{"steps": []}', "I can set that for you.")
    again, events = _turn(g, member, thread, q, chat2)
    assert not again["cached"] and any(e["type"] == "proposal" for e in events)


def test_an_empty_answer_is_not_cached(g, member, thread):
    q = "which of my subscriptions renew around the holidays?"
    out, _ = _turn(g, member, thread, q, ScriptedChat('{"steps": []}', '{"steps": []}', ""))
    assert out["route"]["kind"] == "plan" and out["answer"] == ""
    assert _answer_rows(member, q) == 0


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
