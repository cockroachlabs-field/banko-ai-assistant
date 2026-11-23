"""The assistant graph with a scripted model: routing, receipt follow-ups,
memory, language, proposals, caching, and checkpoints that survive a new
process."""

import json
import os
import threading
import uuid
from datetime import date, timedelta

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import text

from banko_ai.assistant import background
from banko_ai.assistant import graph as G
from banko_ai.db.engine import get_engine
from banko_ai.services import memories, threads
from banko_ai.services.seeding import seed_user
from tests._chat import ScriptedChat, reworded
from tests._money import drop_user, make_user, seeded_diner

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture(scope="module")
def member():
    uid = seeded_diner(2)
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
    background.drain()  # the cache writes the turn left behind, landed before the next one
    return out, events


def test_aggregation_answers_from_sql_and_streams(g, member, thread):
    chat = ScriptedChat("- Groceries are steady.")
    out, events = _turn(g, member, thread, "how much did I spend on groceries in the last 30 days", chat)
    assert out["route"]["kind"] == "aggregate"
    assert "You spent **$" in out["answer"] and "- Groceries are steady." in out["answer"]
    assert "".join(e["text"] for e in events if e["type"] == "token") == out["answer"]
    assert any(e["type"] == "step" and e["label"] == "Aggregation SQL" for e in events)


def _token_log(monkeypatch):
    """What the nodes emit, in order, as they emit it (not as the caller
    happens to drain the stream)."""
    log = []
    real = G._emit

    def spy(event):
        if event.get("type") == "token":
            log.append(("token", event["text"]))
        real(event)
    monkeypatch.setattr(G, "_emit", spy)
    return log


def test_an_english_aggregation_streams_the_card_before_the_insights(g, member, thread, monkeypatch):
    log = _token_log(monkeypatch)

    def insights(msgs):
        log.append(("insights call", ""))
        return "- Restaurants are steady.\n- Fridays cost the most.\n"
    chat = ScriptedChat(insights)
    out, events = _turn(g, member, thread, "how much did I spend on restaurants in the last 45 days", chat)
    kinds = [kind for kind, _ in log]
    assert kinds.index("token") < kinds.index("insights call")
    assert log[0][1].startswith("You spent **$")  # the SQL card goes out before any model work
    assert [kind for kind, _ in chat.calls] == ["stream"]
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert tokens == out["answer"] and out["answer"].endswith("\n\n- Restaurants are steady.\n- Fridays cost the most.")
    again, again_events = _turn(g, member, thread, "how much did I spend on restaurants in the last 45 days",
                                ScriptedChat("unused"))
    assert again["cached"] and again["answer"] == out["answer"]
    assert "".join(e["text"] for e in again_events if e["type"] == "token") == out["answer"]


def test_a_spanish_aggregation_streams_the_translation(g, member, thread, monkeypatch):
    log = _token_log(monkeypatch)
    translated = "Gastaste **$1** en Restaurant.\n\n- Estable."

    def translation(msgs):
        log.append(("translation call", ""))
        return translated
    chat = ScriptedChat("- Steady.", translation)
    out, events = _turn(g, member, thread, "how much did I spend on restaurants in the last 50 days", chat,
                        language_code="es-ES")
    assert [kind for kind, _ in chat.calls] == ["complete", "stream"]
    assert [kind for kind, _ in log][0] == "translation call"  # nothing goes out in English first
    translate_prompt = chat.calls[1][1][-1]["content"]
    assert "Spanish" in translate_prompt and "You spent **$" in translate_prompt and "- Steady." in translate_prompt
    tokens = "".join(e["text"] for e in events if e["type"] == "token")
    assert tokens == out["answer"] == translated


def _unreachable(*args, **kwargs):
    raise ConnectionError("the gateway node went away")


def test_a_query_plan_the_database_cannot_give_keeps_the_answer(g, member, thread, monkeypatch):
    from banko_ai.utils import aggregations

    monkeypatch.setattr(aggregations, "explain_aggregation", _unreachable)
    out, events = _turn(g, member, thread, "how much did I spend on groceries in the last 33 days",
                        ScriptedChat("- Groceries are steady."))
    assert out["route"]["kind"] == "aggregate" and "You spent **$" in out["answer"]
    assert [e for e in events if e["type"] == "error"] == []
    assert {e["label"]: e for e in events if e["type"] == "step"}["EXPLAIN ANALYZE"]["status"] == "warning"
    assert out["meta"]["explain"] is None and "db_ms" in out["meta"]


class _BreaksMidStream(ScriptedChat):
    """A model whose stream drops after its first words."""

    def stream(self, messages, *, temperature=0.3):
        self.calls.append(("stream", messages))
        yield "- Groceries are"
        raise ConnectionError("the model went away")


def test_an_answer_cut_short_keeps_its_words_but_is_never_cached(g, member, thread):
    question = "how much did I spend on groceries in the last 37 days"
    out, events = _turn(g, member, thread, question, _BreaksMidStream())
    assert out["answer"].endswith("- Groceries are")
    assert "".join(e["text"] for e in events if e["type"] == "token") == out["answer"]
    again, _ = _turn(g, member, thread, question, ScriptedChat("- Groceries are steady."))
    assert not again["cached"] and again["answer"].endswith("- Groceries are steady.")


def test_a_translation_cut_short_is_never_cached(g, member, thread):
    class TranslationBreaks(ScriptedChat):
        def stream(self, messages, *, temperature=0.3):
            self.calls.append(("stream", messages))
            yield "Gastaste **$"
            raise ConnectionError("the model went away")

    question = "how much did I spend on groceries in the last 39 days"
    out, _ = _turn(g, member, thread, question, TranslationBreaks("- Steady."), language_code="es-ES")
    assert out["answer"] == "Gastaste **$"
    again, _ = _turn(g, member, thread, question, ScriptedChat("- Estable.", "Gastaste **$1** en Groceries."),
                     language_code="es-ES")
    assert not again["cached"] and again["answer"] == "Gastaste **$1** en Groceries."


def test_a_regional_aggregation_traces_pruning_and_the_serving_region(g, member, thread, monkeypatch):
    from banko_ai.utils import aggregations
    from banko_ai.utils.aggregations import AggregationResult
    used = {}

    def run(intent, user_id, url, region=None, merchant=None):
        used["region"] = region
        return AggregationResult(operation="sum", category="Restaurant", window_start=intent.window_start,
                                 window_end=intent.window_end, total=42.0, count=2, average=21.0,
                                 first_date=None, last_date=None,
                                 rows=[{"date": "2026-10-01", "merchant": "Bistro", "amount": 21.0}] * 2)
    monkeypatch.setattr(aggregations, "run_aggregation", run)
    monkeypatch.setattr(aggregations, "explain_aggregation", lambda *a, **k: "• group\n  regions: us-west-2\n")
    monkeypatch.setattr(G, "money_region", lambda conn, user_id, url: "us-east-1")
    monkeypatch.setattr(G, "read_region", lambda user_id, url: "us-east-1")
    out, events = _turn(g, member, thread, "how much did I spend on restaurants in the last 20 days",
                        ScriptedChat("- Two dinners."))
    steps = {e["label"]: e for e in events if e["type"] == "step"}
    assert steps["Region pruning"]["detail"] == "Scoped to us-east-1 (REGIONAL BY ROW)"
    assert steps["EXPLAIN ANALYZE"]["detail"] == "plan captured, served by us-west-2"
    assert steps["Narrative cache"]["status"] == "miss"
    labels = [e["label"] for e in events if e["type"] == "step"]
    assert labels.index("Region pruning") < labels.index("Aggregation SQL") < labels.index("Narrative cache")
    assert used["region"] == "us-east-1" and out["meta"]["region"] == "us-west-2"


def test_an_aggregation_counts_rows_outside_the_home_region(g, member, thread, monkeypatch):
    """A user can have purchases written outside their home region (before
    the tables were regional, say); a sum pinned to the home region would
    miss them and answer "no matching transactions", so the aggregation
    names the region only when read_region says all their rows live there."""
    from banko_ai.utils import aggregations
    from banko_ai.utils.aggregations import AggregationResult
    used = {}

    def run(intent, user_id, url, region=None, merchant=None):
        used["region"] = region
        return AggregationResult(operation="sum", category="Coffee", window_start=intent.window_start,
                                 window_end=intent.window_end, total=9.5, count=2, average=4.75,
                                 first_date=None, last_date=None,
                                 rows=[{"date": "2026-10-01", "merchant": "Blue Bottle", "amount": 4.75}] * 2)
    monkeypatch.setattr(aggregations, "run_aggregation", run)
    monkeypatch.setattr(aggregations, "explain_aggregation", lambda *a, **k: "• group\n  regions: us-east-1, eu-west-1\n")
    monkeypatch.setattr(G, "money_region", lambda conn, user_id, url: "us-east-1")
    monkeypatch.setattr(G, "read_region", lambda user_id, url: None)
    out, events = _turn(g, member, thread, "what did I spend on coffee", ScriptedChat("- Two coffees."))
    labels = [e["label"] for e in events if e["type"] == "step"]
    assert used["region"] is None and "Region pruning" not in labels
    assert "$9.50" in out["answer"]


def test_an_aggregation_narrative_is_kept_as_long_as_a_cached_answer(g, member, thread, monkeypatch):
    """Asked again twenty minutes later, the same question with the same
    figures must still find its narrative: it lives as long as a cached
    answer, not the search cache's ten minutes."""
    from banko_ai.config.settings import get_config
    from banko_ai.services import caches
    kept = []
    real = caches.put_search
    monkeypatch.setattr(caches, "put_search", lambda *a, **kw: kept.append(kw) or real(*a, **kw))
    _turn(g, member, thread, "how much did I spend on groceries in the last 60 days", ScriptedChat("- Steady."))
    assert background.drain()
    narratives = [kw for kw in kept if kw.get("stat") == "narrative"]
    assert narratives and narratives[0].get("minutes") == get_config().cache_ttl_hours * 60


def test_the_readme_question_sums_one_category(g, member, thread):
    out, _ = _turn(g, member, thread, "what did I spend on dining last month?", ScriptedChat("- steady"))
    assert out["route"]["kind"] == "aggregate"
    assert "on Restaurant from" in out["answer"] and "all categories" not in out["answer"]


def test_an_unknown_subject_is_said_not_summed(g, member, thread):
    quiet = ScriptedChat("unused")
    out, events = _turn(g, member, thread, "how much did I spend on zorbing last month?", quiet)
    assert out["route"]["kind"] == "aggregate"
    assert out["answer"] == "I couldn't find a category or merchant called 'zorbing'."
    assert quiet.calls == [] and "Aggregation SQL" not in [e.get("label") for e in events]
    es = ScriptedChat("No encontré ninguna categoría ni comercio llamado 'zorbing'.")
    out, _ = _turn(g, member, thread, "how much did I spend on zorbing last month?", es, language_code="es-ES")
    assert out["answer"].startswith("No encontré")
    assert "I couldn't find a category or merchant called 'zorbing'." in es.prompts()[-1]


def test_the_smoke_plan_now_finds_dining_out(g, member, thread):
    """granite's real plan for "where could I cut back on dining out?"."""
    plan = ('{"steps": [{"tool": "search_transactions", "args": {"category": "dining out"}}, '
            '{"tool": "spend_summary", "args": {"category": "dining out"}}]}')
    chat = ScriptedChat(plan, '{"steps": []}', "You spent the most at the bistro.")
    out, events = _turn(g, member, thread, "where could I cut back on dining out lately?", chat)
    tools = [(e["label"], e["status"]) for e in events if e["type"] == "step" and e["label"].startswith("Tool ")]
    assert tools == [("Tool search_transactions", "warning"), ("Tool spend_summary", "success")]
    assert '"category": "Restaurant"' in chat.prompts()[-1]


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


def test_the_last_route_decides_what_a_bare_pronoun_means(g, member, thread):
    doc = str(uuid.uuid4())
    with get_engine().begin() as c:
        c.execute(text("INSERT INTO receipts (document_id, user_id, status, extracted) VALUES "
                       "(CAST(:d AS UUID), CAST(:u AS UUID), 'posted', CAST(:x AS JSONB))"),
                  {"d": doc, "u": member, "x": json.dumps({"merchant": "Harbor Deli", "amount": 18.20,
                                                           "items": ["Turkey club"], "date": "2026-10-11"})})
    G.set_active_receipt(thread, {"document_id": doc, "merchant": "Harbor Deli", "items": ["Turkey club"],
                                  "amount": 18.20}, graph=g)
    assert G.thread_state(thread, graph=g)["last_route"] == "receipt"  # the upload is a turn about it
    out, _ = _turn(g, member, thread, "is it a duplicate?", ScriptedChat("No, it isn't."))
    assert out["route"]["kind"] == "receipt"
    agg, _ = _turn(g, member, thread, "what's the total I spent on restaurants this month?",
                   ScriptedChat("- steady"))
    assert agg["route"]["kind"] == "aggregate"
    assert G.thread_state(thread, graph=g)["last_route"] == "aggregate"
    after, _ = _turn(g, member, thread, "is it a good idea to save more?",
                     ScriptedChat('{"steps": []}', '{"steps": []}', "Saving more helps."))
    assert after["route"]["kind"] != "receipt"


def test_without_an_active_receipt_this_receipt_means_the_latest_one(g, member, thread):
    doc = str(uuid.uuid4())
    with get_engine().begin() as c:
        c.execute(text("INSERT INTO receipts (document_id, user_id, status, extracted) VALUES "
                       "(CAST(:d AS UUID), CAST(:u AS UUID), 'posted', CAST(:x AS JSONB))"),
                  {"d": doc, "u": member, "x": json.dumps({"merchant": "Latest Lane Books", "amount": 31.40,
                                                           "items": ["Atlas"], "date": "2026-10-12"})})
    chat = ScriptedChat("That receipt from Latest Lane Books came to $31.40.")
    out, events = _turn(g, member, thread, "how much was this receipt?", chat)
    assert out["route"]["kind"] == "receipt" and "Latest Lane Books" in chat.prompts()[-1]
    assert [e["detail"] for e in events if e["type"] == "step" and e["label"] == "Receipt details"] == [
        "your latest receipt"]
    assert G.thread_state(thread, graph=g)["active_receipt"]["document_id"] == doc


def test_no_receipt_at_all_says_so_in_the_users_language(g):
    uid = make_user()
    tid = threads.create_thread(uid)["thread_id"]
    try:
        quiet = ScriptedChat("unused")
        out, _ = _turn(g, uid, tid, "how much was this receipt?", quiet)
        assert out["route"]["kind"] == "receipt"
        assert out["answer"] == "I don't see a receipt in this conversation yet. Upload one and ask me about it."
        assert quiet.calls == []
        es = ScriptedChat("Todavía no veo un recibo en esta conversación. Sube uno y pregúntame.")
        out, _ = _turn(g, uid, tid, "how much was this receipt?", es, language_code="es-ES")
        assert out["answer"].startswith("Todavía no veo un recibo")
        assert "I don't see a receipt in this conversation yet." in es.prompts()[-1]
    finally:
        threads.delete_thread(uid, tid)
        drop_user(uid)


def test_a_short_message_mid_conversation_is_not_off_topic(g, member, thread):
    _turn(g, member, thread, "how much did I spend on groceries in the last 30 days", ScriptedChat("- steady"))
    out, _ = _turn(g, member, thread, "thanks!", ScriptedChat('{"steps": []}', '{"steps": []}', "Any time."))
    assert out["route"]["kind"] == "plan" and out["answer"] == "Any time."
    more, _ = _turn(g, member, thread, "tell me more", ScriptedChat('{"steps": []}', '{"steps": []}', "Sure."))
    assert more["route"]["kind"] == "plan"
    # Only this conversation gives them a meaning, so their text never keys a cached answer.
    assert _answer_rows(member, "thanks!") == 0 and _answer_rows(member, "tell me more") == 0


def test_a_plan_whose_steps_all_fail_still_searches(g, member, thread):
    """granite planned search_transactions without its required query."""
    chat = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"category": "dining out"}}]}',
                        '{"steps": []}', "You eat out a few times a week.")
    out, events = _turn(g, member, thread, "where could I cut back on dining out?", chat)
    tools = [(e["label"], e["status"]) for e in events if e["type"] == "step" and e["label"].startswith("Tool ")]
    assert tools == [("Tool search_transactions", "warning"), ("Tool search_transactions", "success")]
    assert '"expense_id"' in chat.prompts()[-1]  # the answer is grounded in the fallback search


def test_an_answer_without_one_successful_tool_is_not_cached(g, member, thread, monkeypatch):
    from banko_ai.assistant import tools as T

    def down(*args, **kwargs):
        raise RuntimeError("search down")
    monkeypatch.setattr(T.txns, "semantic_search", down)
    q = "which weekend outings could I cut from my budget?"
    chat = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"category": "fun"}}]}',
                        '{"steps": []}', "unused: no model writes this reply")
    out, _ = _turn(g, member, thread, q, chat)
    assert out["route"]["kind"] == "plan" and out["answer"] == G.LOOKUP_FAILED
    assert _answer_rows(member, q) == 0


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


def _no_card_kept(member):
    """The cards code writes from SQL are kept for a repeat: a test about how
    one was read starts with none, whatever an earlier test asked."""
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM search_cache WHERE user_id = CAST(:u AS UUID)"), {"u": member})


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


def test_what_the_user_told_banko_reaches_the_model_that_writes_from_it(g, member, thread, monkeypatch):
    """Memories are recalled where an answer is written from them, no longer
    at the start of every turn: by the planner, and beside a receipt."""
    told = [{"memory_id": str(uuid.uuid4()), "kind": "preference", "content": "I split every bill with Kai",
             "distance": 0.2}]
    monkeypatch.setattr(memories, "recall", lambda *a, **k: told)
    _cold_answer_cache(member)
    chat = ScriptedChat('{"steps": []}', '{"steps": []}', "Half of it is Kai's.")
    out, _ = _turn(g, member, thread, "how should I think about what I owe this month?", chat)
    assert out["route"]["kind"] == "plan"
    planner, writer = chat.calls[0][1], chat.calls[-1][1]
    assert any("I split every bill with Kai" in m["content"] for m in planner)        # the planner knew
    assert any("I split every bill with Kai" in m["content"] for m in writer)         # and so did the answer

    doc = str(uuid.uuid4())
    with get_engine().begin() as c:
        c.execute(text("INSERT INTO receipts (document_id, user_id, status, extracted) VALUES "
                       "(CAST(:d AS UUID), CAST(:u AS UUID), 'posted', CAST(:x AS JSONB))"),
                  {"d": doc, "u": member, "x": json.dumps({"merchant": "Kai's Diner", "amount": 31.0,
                                                           "items": ["Two lunches"], "date": "2026-10-10"})})
    G.set_active_receipt(thread, {"document_id": doc, "merchant": "Kai's Diner", "items": ["Two lunches"],
                                  "amount": 31.0}, graph=g)
    chat = ScriptedChat(lambda msgs: "It came to $31.00, so $15.50 each.")
    out, events = _turn(g, member, thread, "how much was this receipt?", chat)
    assert out["route"]["kind"] == "receipt"
    assert "I split every bill with Kai" in "\n".join(chat.prompts())
    assert "Memory recall" in [e["label"] for e in events if e["type"] == "step"]


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


def _fresh_thread_turn(g, member, message, chat):
    fresh = threads.create_thread(member)["thread_id"]
    try:
        return _turn(g, member, fresh, message, chat)
    finally:
        threads.delete_thread(member, fresh)


def test_forgetting_a_memory_retires_cached_answers(g, member, thread):
    mid = memories.remember(member, "I'm saving for a sailboat")["memory_id"]
    q = "what big purchase am I saving up for?"
    first, _ = _turn(g, member, thread, q, ScriptedChat(
        '{"steps": [{"tool": "recall", "args": {"query": "saving for"}}]}', '{"steps": []}',
        "You're saving for a sailboat."))
    assert not first["cached"] and _answer_rows(member, q) == 1
    assert memories.forget(member, mid)
    again, _ = _fresh_thread_turn(g, member, q, ScriptedChat('{"steps": []}', '{"steps": []}', "No goal saved."))
    assert not again["cached"] and again["answer"] == "No goal saved."


def test_a_follow_up_skips_the_answer_cache(g, member, thread):
    _cold_answer_cache(member)
    first, _ = _turn(g, member, thread, "where do I buy groceries most often?", ScriptedChat(
        '{"steps": [{"tool": "search_transactions", "args": {"query": "groceries"}}]}', '{"steps": []}',
        "Mostly at the corner market."))
    assert first["route"]["kind"] == "plan"
    q = "what about grocery purchases at night?"
    follow, events = _turn(g, member, thread, q, ScriptedChat(
        '{"steps": [{"tool": "search_transactions", "args": {"query": "groceries at night"}}]}', '{"steps": []}',
        "A few late runs."))
    assert follow["route"]["kind"] == "plan" and not follow["cached"]
    assert "Answer cache" not in [e["label"] for e in events if e["type"] == "step"]
    assert _answer_rows(member, q) == 0
    # The same words opening a new conversation are not a follow-up.
    opener, opener_events = _fresh_thread_turn(g, member, q, ScriptedChat(
        '{"steps": [{"tool": "search_transactions", "args": {"query": "groceries at night"}}]}', '{"steps": []}',
        "A few late runs."))
    assert [e["status"] for e in opener_events if e["type"] == "step" and e["label"] == "Answer cache"] == ["miss"]
    assert _answer_rows(member, q) == 1


def test_the_memory_reply_speaks_to_the_user(g, member, thread):
    out, _ = _turn(g, member, thread, "remember that I'm saving for a new canoe", ScriptedChat("unused"))
    assert out["answer"] == "Got it. I'll remember that you're saving for a new canoe."
    gone, _ = _turn(g, member, thread, "forget about the new canoe", ScriptedChat("unused"))
    assert gone["answer"] == "Done, I forgot that you're saving for a new canoe."


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


def _phases(events) -> list[str]:
    """The running labels a turn sent, in order, after checking that each
    one finished later under its own key (the chip the bubble swaps)."""
    steps = [e for e in events if e["type"] == "step"]
    running = [e for e in steps if e["status"] == "running"]
    assert len({r["key"] for r in running}) == len(running), "every running phase has its own key"
    for r in running:
        later = steps[steps.index(r) + 1:]
        assert [e["status"] != "running" for e in later if e.get("key") == r["key"]] == [True], r
    return [r["label"] for r in running]


def test_long_phases_say_they_started_and_finish_under_the_same_key(g, member, thread):
    _cold_answer_cache(member)
    chat = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"query": "coffee"}}]}',
                        '{"steps": []}', "You buy coffee most mornings.")
    _, events = _turn(g, member, thread, "where do I buy coffee before work most days?", chat)
    # Memories are recalled once the cache has said no: a hit is replayed without them.
    assert _phases(events) == ["Checking the answer cache", "Recalling what you told me", "Planning the steps",
                               "Searching your transactions", "Planning the steps", "Reading the query plan",
                               "Writing the answer"]
    done = {e["key"]: e["label"] for e in events if e["type"] == "step" and e.get("key") and e["status"] != "running"}
    assert list(done.values()) == ["Answer cache", "Memory recall", "Planner", "Tool search_transactions", "Planner",
                                   "EXPLAIN ANALYZE", "LLM answer"]
    _, agg = _turn(g, member, thread, "how much did I spend on groceries in the last 37 days",
                   ScriptedChat("- Groceries are steady."))
    # A total recalls nothing: no memory has a part in it.
    assert _phases(agg) == ["Running the numbers in CockroachDB", "Writing the answer", "Reading the query plan"]


def test_cache_writes_leave_the_answer_path_and_drain_lands_them(g, member, thread, monkeypatch):
    """respond hands the answer cache write to the background, so the turn
    returns (and the answer goes out) while the write is still held."""
    from banko_ai.services import caches
    gate = threading.Event()
    real_put = caches.put_answer

    def held_put(*args, **kwargs):
        gate.wait(5)
        return real_put(*args, **kwargs)

    monkeypatch.setattr(caches, "put_answer", held_put)
    _cold_answer_cache(member)
    q = "which coffee shops did I stop at on weekday mornings?"
    chat = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"query": "coffee"}}]}',
                        '{"steps": []}', "Mostly the corner cafe.")
    out = G.run_turn(member, thread, q, chat=chat, graph=g)
    assert not out["cached"] and _answer_rows(member, q) == 0
    gate.set()
    assert background.drain()
    assert _answer_rows(member, q) == 1
    again, events = _fresh_thread_turn(g, member, q, ScriptedChat("unused"))
    assert again["cached"] and again["answer"] == out["answer"]
    assert [e["status"] for e in events if e["type"] == "step" and e["label"] == "Answer cache"] == ["hit"]


def test_implicit_memory_is_extracted_off_the_answer_path(g, member, thread):
    released, seen = threading.Event(), {}

    def extraction(msgs):
        seen["released"] = released.wait(5)
        return '{"content": "I usually buy coffee at Blue Bottle", "kind": "preference"}'

    chat = ScriptedChat('{"steps": []}', '{"steps": []}', "Noted, that's a few visits a week.", extraction)
    out = G.run_turn(member, thread, "i usually buy coffee at blue bottle, is that a lot?", chat=chat, graph=g)
    assert out["route"]["kind"] == "plan" and out["answer"].startswith("Noted")
    released.set()
    assert background.drain()
    assert seen["released"] is True
    saved = [m for m in memories.list_memories(member) if "Blue Bottle" in m["content"]]
    assert len(saved) == 1
    memories.forget(member, saved[0]["memory_id"])


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


def test_the_model_never_sees_the_page_cursor():
    facts = G._facts([{"tool": "search_transactions", "ok": True,
                       "result": {"items": [{"expense_id": "e1", "merchant": "Blue Bottle"}], "next_cursor": "bzEw"}}])
    assert "Blue Bottle" in facts and "next_cursor" not in facts and "bzEw" not in facts


def _searched(g, member, thread, question, query):
    chat = ScriptedChat(json.dumps({"steps": [{"tool": "search_transactions", "args": {"query": query}}]}),
                        '{"steps": []}', "Here are your purchases.")
    return _turn(g, member, thread, question, chat)[0]


def test_a_search_answer_carries_its_next_page(g, member, thread):
    out = _searched(g, member, thread, "find my dinners out", "dinner")
    search = out["meta"]["search"]
    assert search["q"] == "dinner" and search["next"]
    assert search["category"] is None and search["from"] is None and search["to"] is None


def test_show_more_pages_the_last_search_without_the_model(g, member, thread):
    from banko_ai.services import transactions as txns
    _searched(g, member, thread, "show my grocery runs", "groceries")
    quiet = ScriptedChat("the model should not be asked")
    out, events = _turn(g, member, thread, "show more", quiet)
    assert out["route"]["kind"] == "more" and quiet.calls == []
    first = [i["expense_id"] for i in txns.semantic_search(member, "groceries", limit=10)["items"]]
    shown = out["meta"]["search"]["items"]
    assert len(shown) == 10 and not {i["expense_id"] for i in shown} & set(first)
    assert all(i["amount"] > 0 and i["date"] and i["merchant"] for i in shown)
    assert out["answer"] == "Here are 10 more matches for “groceries”."
    assert any(e["type"] == "step" and e["label"] == "Next page" for e in events)
    again, _ = _turn(g, member, thread, "next 10", quiet)
    assert again["route"]["kind"] == "more"
    assert not {i["expense_id"] for i in again["meta"]["search"]["items"]} & {i["expense_id"] for i in shown}


def test_show_more_after_the_last_page_says_so(g, member, thread):
    g.update_state(G._config(thread), {"last_search": {"q": "pizza", "category": None, "from": None, "to": None,
                                                       "next": None}}, as_node="learn")
    out, _ = _turn(g, member, thread, "show more", ScriptedChat("unused"))
    assert out["route"]["kind"] == "more"
    assert out["answer"] == "That's every match I found for “pizza”."


def test_a_turn_without_a_search_ends_the_paging(g, member, thread):
    _searched(g, member, thread, "list my coffee stops", "coffee")
    _turn(g, member, thread, "how much did I spend on groceries in the last 30 days", ScriptedChat("- Steady."))
    out, _ = _turn(g, member, thread, "show more", ScriptedChat('{"steps": []}', '{"steps": []}', "Sure."))
    assert out["route"]["kind"] != "more"


def _top_merchant(member):
    with get_engine().connect() as c:
        return c.execute(text("""
            SELECT merchant, count(*), sum(expense_amount) FROM expenses WHERE user_id = CAST(:u AS UUID)
            GROUP BY merchant ORDER BY count(*) DESC, merchant LIMIT 1"""), {"u": member}).fetchone()


def test_a_visit_question_is_answered_from_sql_without_the_model(g, member, thread):
    merchant, visits, total = _top_merchant(member)
    quiet = ScriptedChat("the model should not be asked")
    out, events = _turn(g, member, thread, f"did I go to {merchant.lower()}?", quiet)
    assert out["route"]["kind"] == "visits" and quiet.calls == []
    assert f"**{merchant}** {visits} times" in out["answer"] and f"${total:,.2f}" in out["answer"]
    assert out["meta"]["explain"] and "db_ms" in out["meta"]
    assert any(e["type"] == "step" and e["label"] == "Visits SQL" for e in events)


def test_a_visit_answer_survives_a_query_plan_the_database_cannot_give(g, member, thread, monkeypatch):
    _no_card_kept(member)
    monkeypatch.setattr(G.txns, "explain_merchant_visits", _unreachable)
    merchant, visits, _ = _top_merchant(member)
    out, events = _turn(g, member, thread, f"did I go to {merchant.lower()}?", ScriptedChat("unused"))
    assert out["route"]["kind"] == "visits" and f"**{merchant}** {visits} times" in out["answer"]
    assert [e for e in events if e["type"] == "error"] == []
    assert {e["label"]: e for e in events if e["type"] == "step"}["EXPLAIN ANALYZE"]["status"] == "warning"
    assert out["meta"]["explain"] is None


def _kite_shop(member):
    from banko_ai.services.accounts import list_accounts
    from banko_ai.services.ledger import post_purchase
    acct = next(a for a in list_accounts(member)["accounts"] if a["kind"] == "credit")
    last_month = (date.today().replace(day=1) - timedelta(days=1)).replace(day=3)
    for day, amount in ((last_month, "42.00"), (date.today() - timedelta(days=400), "18.50")):
        post_purchase(member, acct["account_id"], amount=amount, merchant="Zephyr Kites", category="Shopping",
                      expense_date=day)
    return last_month


def test_a_visit_window_counts_that_window_only(g, member, thread):
    last_month = _kite_shop(member)
    out, _ = _turn(g, member, thread, "did I go to Zephyr Kites last month?", ScriptedChat("unused"))
    assert out["route"]["kind"] == "visits"
    assert "**Zephyr Kites** 1 time " in out["answer"] and "$42.00" in out["answer"]
    assert str(last_month) in out["answer"] and "$18.50" not in out["answer"]
    none, _ = _turn(g, member, thread, "have I been to Zephyr Kites this month?", ScriptedChat("unused"))
    assert none["answer"].startswith("I don't see a visit to **Zephyr Kites**")
    assert f"The last one was on {last_month}." in none["answer"]


def test_a_visit_card_is_translated_in_another_language(g, member, thread):
    merchant, visits, _ = _top_merchant(member)
    chat = ScriptedChat(reworded)
    out, _ = _turn(g, member, thread, f"have I been to {merchant}?", chat, language_code="es-ES")
    assert out["answer"].startswith(f"Fuiste a **{merchant}** {visits} times since ") and len(chat.calls) == 1


def test_a_place_that_is_not_a_merchant_goes_to_the_planner(g, member, thread):
    chat = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"query": "moon base"}}]}',
                        '{"steps": []}', "I found no visits to a moon base.")
    out, events = _turn(g, member, thread, "did I go to the moon base?", chat)
    assert out["route"]["kind"] == "plan" and chat.calls
    assert any(e["type"] == "step" and e["label"] == "Merchant lookup" and e["status"] == "warning" for e in events)


def test_an_answer_with_any_failed_tool_call_is_not_cached(g, member, thread):
    """One search ran and one was rejected: the answer rests on half the
    evidence, so it is never replayed from the cache."""
    q = "which dinners out could I skip?"
    chat = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"query": "dinner", "date_from": "lately"}},'
                        ' {"tool": "search_transactions", "args": {"query": "dinner"}}]}',
                        '{"steps": []}', "Here are a few dinners you could skip.")
    out, _ = _turn(g, member, thread, q, chat)
    assert out["route"]["kind"] == "plan" and out["answer"].startswith("Here are a few dinners")
    assert _answer_rows(member, q) == 0


def _search_raises(monkeypatch, only: str | None = None):
    """The transaction search fails as a lost connection would, for every
    query or for one. The search cache goes with the connection, so an
    earlier test's cached result never stands in for the search."""
    from banko_ai.services import caches
    from banko_ai.services import transactions as txns
    real = txns.semantic_search
    monkeypatch.setattr(caches, "get_search", lambda *args, **kwargs: None)

    def search(user_id, query, **kw):
        if only is None or query == only:
            raise RuntimeError("connection reset by peer")
        return real(user_id, query, **kw)

    monkeypatch.setattr(txns, "semantic_search", search)


def test_when_every_lookup_raised_the_reply_says_so(g, member, thread, monkeypatch):
    """A lookup that raised leaves the model no facts, and an answer written
    from none reads as "you have none". The reply is a fixed line instead, and
    no model writes one."""
    _search_raises(monkeypatch)
    chat = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"query": "coffee"}}]}',
                        '{"steps": []}', "I don't have any information about your coffee purchases.")
    out, events = _turn(g, member, thread, "where do I get my coffee before work?", chat)
    assert out["route"]["kind"] == "plan" and out["answer"] == G.LOOKUP_FAILED
    assert "".join(e["text"] for e in events if e["type"] == "token") == out["answer"]
    assert not [kind for kind, _ in chat.calls if kind == "stream"]
    assert "connection reset" not in json.dumps(events, default=str)


def test_when_one_lookup_raised_the_answer_carries_a_note(g, member, thread, monkeypatch):
    _search_raises(monkeypatch, only="restaurants")
    chat = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"query": "coffee"}},'
                        ' {"tool": "search_transactions", "args": {"query": "restaurants"}}]}',
                        '{"steps": []}', "Here are your coffee purchases.")
    out, events = _turn(g, member, thread, "compare my coffee and restaurant purchases", chat)
    assert out["answer"] == f"Here are your coffee purchases.\n\n{G.LOOKUP_PARTLY_FAILED}"
    assert "".join(e["text"] for e in events if e["type"] == "token") == out["answer"]


def test_a_lookup_that_declined_adds_no_note(g, member, thread):
    """Arguments the planner got wrong are its own mistake, and a merchant the
    user never visited is an answer. Neither is a lookup that broke."""
    chat = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"query": "dinner", "date_from": "lately"}},'
                        ' {"tool": "spend_summary", "args": {"merchant": "NoSuchMerchant"}},'
                        ' {"tool": "search_transactions", "args": {"query": "dinner"}}]}',
                        '{"steps": []}', "Here are a few dinners.")
    out, _ = _turn(g, member, thread, "which dinners out could I skip?", chat)
    assert out["answer"] == "Here are a few dinners."


def test_the_incomplete_note_is_in_the_users_language(g, member, thread, monkeypatch):
    _search_raises(monkeypatch, only="restaurantes")
    note = "Una de mis consultas no funcionó ahora, así que esto puede estar incompleto."
    chat = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"query": "café"}},'
                        ' {"tool": "search_transactions", "args": {"query": "restaurantes"}}]}',
                        '{"steps": []}', "Aquí están tus compras de café.", note)
    out, events = _turn(g, member, thread, "compara mis compras de café y restaurantes", chat, language_code="es-ES")
    assert out["answer"] == f"Aquí están tus compras de café.\n\n{note}"
    assert "".join(e["text"] for e in events if e["type"] == "token") == out["answer"]


def test_the_failed_lookup_line_is_in_the_users_language(g, member, thread, monkeypatch):
    _search_raises(monkeypatch)
    chat = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"query": "café"}}]}',
                        '{"steps": []}', "No pude consultarlo ahora mismo. Vuelve a preguntar en un momento.")
    out, _ = _turn(g, member, thread, "muéstrame mis compras de café", chat, language_code="es-ES")
    assert out["answer"] == "No pude consultarlo ahora mismo. Vuelve a preguntar en un momento."
    assert G.LOOKUP_FAILED in chat.prompts()[-1]  # the line itself was what the model translated


def _recall_raises(monkeypatch, error):
    def recall(*args, **kwargs):
        raise error
    monkeypatch.setattr(memories, "recall", recall)


def test_a_model_that_is_still_loading_is_one_line_in_the_log(g, member, thread, monkeypatch, caplog):
    """In the first seconds after a boot the embedding model is still
    loading. The turn goes on without what needs it and the log says so in a
    line: a traceback there reads as a failure."""
    import logging

    from banko_ai.utils.embeddings import ModelWarmingUp
    _recall_raises(monkeypatch, ModelWarmingUp("embedding model all-MiniLM-L6-v2 is still loading"))
    caplog.set_level(logging.INFO, logger="banko.assistant")
    _cold_answer_cache(member)
    out, _ = _turn(g, member, thread, "where could I cut back on eating out these days?",
                   ScriptedChat('{"steps": []}', '{"steps": []}', "Nowhere."))
    assert out["route"]["kind"] == "plan" and out["answer"] == "Nowhere."
    about = [r for r in caplog.records if "memory recall" in r.getMessage()]
    assert [r.getMessage() for r in about] == ["memory recall skipped: the embedding model is still loading"]
    assert about[0].exc_info is None and about[0].levelno == logging.INFO


def test_any_other_reason_to_skip_a_step_keeps_its_traceback(g, member, thread, monkeypatch, caplog):
    import logging
    _recall_raises(monkeypatch, RuntimeError("connection reset by peer"))
    caplog.set_level(logging.INFO, logger="banko.assistant")
    _cold_answer_cache(member)
    out, _ = _turn(g, member, thread, "where could I cut back on eating out this season?",
                   ScriptedChat('{"steps": []}', '{"steps": []}', "Nowhere."))
    assert out["route"]["kind"] == "plan" and out["answer"] == "Nowhere."
    about = [r for r in caplog.records if "memory recall" in r.getMessage()]
    assert [r.getMessage() for r in about] == ["memory recall skipped"]
    assert about[0].exc_info is not None and about[0].levelno == logging.WARNING


def test_the_model_hears_which_counts_were_capped():
    from banko_ai.assistant import graph as G
    facts = G._facts([{"tool": "cashflow", "ok": True, "result": {"months": []}, "clamped": {"months": 12}}])
    assert '"capped_at": {"months": 12}' in facts
    assert "capped_at" not in G._facts([{"tool": "cashflow", "ok": True, "result": {"months": []}}])


def test_a_streamed_translation_in_html_paragraphs_is_stored_as_markdown(g, member, thread):
    question = "how much did I spend on groceries in the last 41 days"
    out, _ = _turn(g, member, thread, question, ScriptedChat("- Estable.", "<p>Gastaste **$1** en Groceries.</p>"),
                   language_code="es-ES")
    assert out["answer"] == "Gastaste **$1** en Groceries."


def _purchases(member: str, start, end) -> list[tuple]:
    with get_engine().connect() as c:
        return c.execute(text("SELECT expense_amount, merchant FROM expenses WHERE user_id = CAST(:u AS UUID) "
                              "AND expense_date >= :s AND expense_date < :e "
                              "ORDER BY expense_amount DESC, expense_date DESC, expense_id"),
                         {"u": member, "s": start, "e": end}).fetchall()


def test_the_biggest_purchase_comes_from_sql_and_no_model_picks_it(g, member, thread):
    """It went to the planner, whose tools hand the model the rows most like
    the wording or the most recent ones, never the largest: a month's rent
    was left out, and a June purchase was once named as last month's."""
    from datetime import date
    first_this = date.today().replace(day=1)
    first_last = (first_this.replace(year=first_this.year - 1, month=12) if first_this.month == 1
                  else first_this.replace(month=first_this.month - 1))
    top = _purchases(member, first_last, first_this)
    chat = ScriptedChat("the model was asked, and it should not have been")
    out, events = _turn(g, member, thread, "What can you tell me about my biggest purchase last month?", chat)
    assert out["route"]["kind"] == "aggregate" and chat.calls == []
    assert out["answer"].startswith(f"Your biggest purchase from {first_last} to ")
    assert f"**${float(top[0][0]):,.2f}** at **{top[0][1]}**" in out["answer"]
    assert f"${float(top[1][0]):,.2f}" in out["answer"] and f"out of {len(top)} purchases" in out["answer"]
    assert any(e["type"] == "step" and e["label"] == "Aggregation SQL" for e in events)


def test_the_top_purchases_are_listed_in_order(g, member, thread):
    from datetime import date
    top = _purchases(member, date(date.today().year, 1, 1), date(date.today().year + 1, 1, 1))
    chat = ScriptedChat("unused")
    out, _ = _turn(g, member, thread, "show me my top 5 purchases this year", chat)
    listed = [line for line in out["answer"].splitlines() if line.startswith("| 20")]
    assert chat.calls == [] and len(listed) == 5
    assert [line.split("|")[-2].strip() for line in listed] == [f"${float(a):,.2f}" for a, _ in top[:5]]


def test_the_biggest_purchase_in_another_language_is_translated_not_rewritten(g, member, thread):
    chat = ScriptedChat(reworded)
    out, _ = _turn(g, member, thread, "what was my biggest purchase last month?", chat, language_code="es-ES")
    assert out["answer"].startswith("Tu compra más grande") and len(chat.calls) == 1
    assert "Your biggest purchase" in chat.prompts()[-1]            # the card it was asked to translate


# ── a total from the planner's tool brings its plan ─────────────────────
# Which tool a short question gets is the model's choice. "coffee" was the
# search on one model and spend_summary on the next, and on gpt-4.1 seven
# characters of the tool list moved it from one to the other. The search
# brought its plan; a total from the tool brought none.

TOTAL_OF_DINING = '{"steps": [{"tool": "spend_summary", "args": {"category": "dining out"}}]}'


def _planned(g, member, thread, message, chat):
    """A turn the planner answered. A question the router sends elsewhere
    would pass these tests for nothing."""
    out, events = _turn(g, member, thread, message, chat)
    assert out["route"]["kind"] == "plan", out["route"]
    return out, events


def _plans_asked(monkeypatch, plan="• group (scalar)\n") -> list:
    """Every totals statement whose plan was asked for, and a fixed plan back."""
    from banko_ai.utils import aggregations
    asked = []

    def plan_of(intent, user_id, database_url, region=None, merchant=None):
        asked.append((intent.operation, intent.subject, intent.window_start, intent.window_end, merchant, user_id))
        return plan
    monkeypatch.setattr(aggregations, "explain_aggregation", plan_of)
    return asked


def test_an_answer_from_the_totals_tool_carries_its_plan(g, member, thread):
    _cold_answer_cache(member)
    chat = ScriptedChat(TOTAL_OF_DINING, '{"steps": []}', "Most of it went to the bistro.")
    out, events = _planned(g, member, thread, "where does my dining money tend to go?", chat)
    assert out["route"]["kind"] == "plan" and out["answer"] == "Most of it went to the bistro."
    plan = out["meta"]["explain"]
    assert plan and "group (scalar)" in plan and "expenses@" in plan
    done = {e["key"]: e for e in events if e["type"] == "step" and e.get("key") and e["status"] != "running"}
    assert [e["label"] for e in done.values()] == ["Answer cache", "Memory recall", "Planner", "Tool spend_summary",
                                                   "Planner", "EXPLAIN ANALYZE", "LLM answer"]
    assert done["explain"]["detail"] == "plan captured"
    assert _phases(events)[-2:] == ["Reading the query plan", "Writing the answer"]


def test_the_plan_is_of_the_last_total_the_planner_asked_for(g, member, thread, monkeypatch):
    """gpt-4.1 asks for everything first, and for the category once it has
    seen that. The answer is written about the second."""
    asked = _plans_asked(monkeypatch)
    _cold_answer_cache(member)
    start, end = date.today() - timedelta(days=40), date.today() - timedelta(days=10)
    chat = ScriptedChat('{"steps": [{"tool": "spend_summary", "args": {}}]}',
                        '{"steps": [{"tool": "spend_summary", "args": {"category": "dining out", '
                        f'"date_from": "{start}", "date_to": "{end}"}}}}]}}', "Dining came to a fair sum.")
    out, _ = _planned(g, member, thread, "dining in broad strokes, please", chat)
    assert asked == [("sum", "Restaurant", start, end + timedelta(days=1), None, member)]
    assert out["meta"]["explain"] == "• group (scalar)\n"


def test_a_total_at_a_merchant_shows_the_plan_of_that_statement(g, member, thread, monkeypatch):
    asked = _plans_asked(monkeypatch)
    _cold_answer_cache(member)
    merchant = _a_merchant(member)
    chat = ScriptedChat(json.dumps({"steps": [{"tool": "spend_summary", "args": {"merchant": merchant.upper()}}]}),
                        '{"steps": []}', "You are a regular there.")
    _planned(g, member, thread, "am I a regular at that place by now?", chat)
    today = date.today()
    assert asked == [("sum", None, today - timedelta(days=30), today + timedelta(days=1), merchant, member)]


def _a_merchant(member) -> str:
    with get_engine().connect() as c:
        return c.execute(text("SELECT merchant FROM expenses WHERE user_id = CAST(:u AS UUID) "
                              "ORDER BY expense_date DESC LIMIT 1"), {"u": member}).scalar()


def _visit_plans_asked(monkeypatch, plan="• group (scalar) of visits\n") -> list:
    """Every merchant whose visits statement was asked for its plan."""
    asked = []
    monkeypatch.setattr(G.txns, "explain_merchant_visits",
                        lambda user_id, name, **kw: asked.append((user_id, name, kw.get("date_from"), kw.get("date_to")))
                        or plan)
    return asked


def test_a_merchants_visits_from_the_planner_carry_their_plan(g, member, thread):
    """ "starbucks" on gpt-4.1 ends at merchant_history. The visits route
    shows the plan of that very statement; from the planner it showed none."""
    _cold_answer_cache(member)
    merchant = _a_merchant(member)
    chat = ScriptedChat(json.dumps({"steps": [{"tool": "merchant_history", "args": {"merchant": merchant.lower()}}]}),
                        '{"steps": []}', "You go there often.")
    out, events = _planned(g, member, thread, "tell me about my habit at that place", chat)
    plan = out["meta"]["explain"]
    assert plan and "group (scalar)" in plan and "expenses@" in plan
    assert {e["label"]: e for e in events if e["type"] == "step"}["EXPLAIN ANALYZE"]["detail"] == "plan captured"


def test_the_plan_is_of_the_last_lookup_that_has_one(g, member, thread, monkeypatch):
    """A total of everything, then the merchant's visits: the answer is
    written about the merchant, so that is the statement to show. And the
    other way round."""
    totals, visits = _plans_asked(monkeypatch), _visit_plans_asked(monkeypatch)
    merchant = _a_merchant(member)
    history = {"tool": "merchant_history", "args": {"merchant": merchant}}
    _cold_answer_cache(member)
    chat = ScriptedChat('{"steps": [{"tool": "spend_summary", "args": {}}]}', json.dumps({"steps": [history]}),
                        "You go there often.")
    out, _ = _planned(g, member, thread, "what does that place cost me, in numbers?", chat)
    assert out["meta"]["explain"] == "• group (scalar) of visits\n"
    assert (totals, visits) == ([], [(member, merchant, None, None)])

    del visits[:]
    chat = ScriptedChat(json.dumps({"steps": [history]}), '{"steps": [{"tool": "spend_summary", "args": {}}]}',
                        "A small part of the whole.")
    out, _ = _planned(g, member, thread, "how does my spending at that place weigh against the rest?", chat)
    assert out["meta"]["explain"] == "• group (scalar)\n" and visits == [] and len(totals) == 1


def test_a_merchant_nobody_went_to_has_no_visits_to_explain(g, member, thread, monkeypatch):
    visits = _visit_plans_asked(monkeypatch)
    _cold_answer_cache(member)
    chat = ScriptedChat('{"steps": [{"tool": "merchant_history", "args": {"merchant": "No Such Place"}}]}',
                        '{"steps": []}', "You have never been there.")
    out, events = _planned(g, member, thread, "does that place show up in my purchases at all?", chat)
    assert visits == [] and "explain" not in out["meta"]
    assert "EXPLAIN ANALYZE" not in [e.get("label") for e in events]


def test_a_search_keeps_its_plan_beside_a_total(g, member, thread, monkeypatch):
    """The search is what "show more" continues, and its plan is the one shown."""
    asked = _plans_asked(monkeypatch)
    monkeypatch.setattr(G.txns, "explain_semantic", lambda *a, **k: "• vector search\n")
    _cold_answer_cache(member)
    chat = ScriptedChat('{"steps": [{"tool": "spend_summary", "args": {"category": "dining out"}}, '
                        '{"tool": "search_transactions", "args": {"query": "dinner"}}]}',
                        '{"steps": []}', "A few dinners stand out.")
    out, _ = _planned(g, member, thread, "which dinners stand out against the usual?", chat)
    assert out["meta"]["explain"] == "• vector search\n" and asked == []


def test_a_totals_plan_the_database_cannot_give_keeps_the_answer(g, member, thread, monkeypatch):
    from banko_ai.utils import aggregations
    monkeypatch.setattr(aggregations, "explain_aggregation", _unreachable)
    _cold_answer_cache(member)
    chat = ScriptedChat(TOTAL_OF_DINING, '{"steps": []}', "Most of it went to the bistro.")
    out, events = _planned(g, member, thread, "where does my dining money usually end up?", chat)
    assert out["answer"] == "Most of it went to the bistro." and [e for e in events if e["type"] == "error"] == []
    assert {e["label"]: e for e in events if e["type"] == "step"}["EXPLAIN ANALYZE"]["status"] == "warning"
    assert "explain" not in out["meta"] and "db_ms" in out["meta"]


def test_a_total_that_was_declined_has_no_plan_of_its_own(g, member, thread, monkeypatch):
    """Nothing ran for it. The search that grounds the answer instead brings its plan."""
    asked = _plans_asked(monkeypatch)
    monkeypatch.setattr(G.txns, "explain_semantic", lambda *a, **k: "• vector search\n")
    _cold_answer_cache(member)
    chat = ScriptedChat('{"steps": [{"tool": "spend_summary", "args": {"merchant": "NoSuchMerchant"}}]}',
                        '{"steps": []}', "I found no such place.")
    out, _ = _planned(g, member, thread, "what do I owe that odd little shop?", chat)
    assert asked == [] and out["meta"]["explain"] == "• vector search\n"


def test_a_declined_total_beside_another_lookup_is_not_asked_for_a_plan(g, member, thread, monkeypatch):
    """Nothing grounds this answer in a search: another lookup worked. The
    total that was declined ran no statement, so no plan is read for it."""
    asked = _plans_asked(monkeypatch)
    _cold_answer_cache(member)
    chat = ScriptedChat('{"steps": [{"tool": "spend_summary", "args": {"merchant": "NoSuchMerchant"}}, '
                        '{"tool": "budget_status", "args": {}}]}', '{"steps": []}', "Your budgets hold.")
    out, events = _planned(g, member, thread, "is that odd little shop hurting my budgets?", chat)
    assert asked == [] and "explain" not in out["meta"]
    assert "EXPLAIN ANALYZE" not in [e.get("label") for e in events]


def test_a_lookup_with_no_statement_to_show_brings_no_plan(g, member, thread, monkeypatch):
    asked = _plans_asked(monkeypatch)
    _cold_answer_cache(member)
    chat = ScriptedChat('{"steps": [{"tool": "budget_status", "args": {}}]}', '{"steps": []}', "Your budgets hold.")
    out, events = _planned(g, member, thread, "are my budgets holding up so far?", chat)
    assert asked == [] and "explain" not in out["meta"]
    assert "EXPLAIN ANALYZE" not in [e.get("label") for e in events]


def test_the_plan_under_a_size_answer_is_of_the_statement_that_ordered_the_purchases(g, member, thread):
    """It was the plan of the totals statement, which the rows in the answer
    did not come from."""
    _no_card_kept(member)
    out, events = _turn(g, member, thread, "What was my biggest purchase last month?", ScriptedChat("unused"))
    plan = out["meta"]["explain"]
    assert plan and "expense_amount" in plan and ("top-k" in plan or "sort" in plan)
    assert "group (scalar)" not in plan
    assert any(e["type"] == "step" and e["label"] == "EXPLAIN ANALYZE" for e in events)


def test_a_size_question_about_nothing_the_user_has_goes_to_the_planner(g, member, thread):
    """"My biggest moon base purchase": no category and none of this user's
    merchants. The totals say so and stop. Here the words between "biggest"
    and "purchase" are as likely not a name at all, so the question goes
    where it went before this route existed."""
    chat = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"query": "moon base"}}]}',
                        '{"steps": []}', "I found nothing bought at a moon base.")
    out, events = _turn(g, member, thread, "what was my biggest moon base purchase last month?", chat)
    assert out["route"]["kind"] == "plan" and out["answer"].startswith("I found nothing bought at a moon base")
    assert "couldn't find a category" not in out["answer"]
    assert any(e["type"] == "step" and e["label"] == "Category lookup" and e["status"] == "warning" for e in events)


def test_a_size_question_with_no_purchases_under_its_subject_goes_to_the_planner(g, member, thread):
    """Income is a category, and no purchase carries it: "No matching
    transactions on Income" read as no income at all."""
    chat = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"query": "salary deposit"}}]}',
                        '{"steps": []}', "Your largest deposit was your salary.")
    out, events = _turn(g, member, thread, "my largest income transaction", chat)
    assert out["route"]["kind"] == "plan" and out["answer"].startswith("Your largest deposit")
    assert "No matching transactions" not in out["answer"]
    assert any(e["type"] == "step" and e["label"] == "Aggregation SQL" and e["status"] == "warning" for e in events)


def test_no_purchases_in_the_window_is_said_without_the_model(g, member, thread):
    quiet = ScriptedChat("the model should not be asked")
    out, _ = _turn(g, member, thread, "what was my biggest purchase in March 2019?", quiet)
    assert out["route"]["kind"] == "aggregate" and quiet.calls == []
    assert out["answer"] == "You have no purchases from 2019-03-01 to 2019-03-31."


def test_ties_come_in_one_order_and_fewer_than_asked_are_all_listed(g, member, thread):
    """Three purchases of one amount, two of them on one day: latest first,
    then by id, every time. Five were asked for and four exist."""
    from banko_ai.services.accounts import list_accounts
    from banko_ai.services.ledger import post_purchase
    acct = next(a for a in list_accounts(member)["accounts"] if a["kind"] == "credit")["account_id"]
    same_day = {}
    for merchant, amount, day in (("Tie Harbor", "700.00", date(2024, 3, 12)), ("Tie Summit", "700.00", date(2024, 3, 12)),
                                  ("Tie Meadow", "700.00", date(2024, 3, 10)), ("Small Dune", "20.00", date(2024, 3, 5))):
        expense_id = post_purchase(member, acct, amount=amount, merchant=merchant, category="Travel", expense_date=day)
        if day == date(2024, 3, 12):
            same_day[expense_id] = merchant
    out, _ = _turn(g, member, thread, "show me my top 5 purchases in March 2024", ScriptedChat("unused"))
    assert out["route"]["kind"] == "aggregate"
    assert out["answer"].splitlines()[0] == "Your 4 biggest purchases from 2024-03-01 to 2024-03-31, out of 4:"
    listed = [line.split("|")[2].strip() for line in out["answer"].splitlines() if line.startswith("| 2024")]
    assert listed == [same_day[i] for i in sorted(same_day)] + ["Tie Meadow", "Small Dune"]
    again, _ = _turn(g, member, thread, "show me my top 5 purchases in March 2024", ScriptedChat("unused"))
    assert again["answer"] == out["answer"]


def test_a_size_question_is_not_about_the_receipt_on_the_table(g, member, thread):
    doc = str(uuid.uuid4())
    with get_engine().begin() as c:
        c.execute(text("INSERT INTO receipts (document_id, user_id, status, extracted) VALUES "
                       "(CAST(:d AS UUID), CAST(:u AS UUID), 'posted', CAST(:x AS JSONB))"),
                  {"d": doc, "u": member, "x": json.dumps({"merchant": "Trader Joe's", "amount": 23.47,
                                                           "items": ["Pinot Noir wine"], "date": "2026-10-10"})})
    G.set_active_receipt(thread, {"document_id": doc, "merchant": "Trader Joe's", "items": ["Pinot Noir wine"],
                                  "amount": 23.47}, graph=g)
    quiet = ScriptedChat("the model should not be asked")
    out, _ = _turn(g, member, thread, "What was my biggest purchase last month?", quiet)
    assert out["route"]["kind"] == "aggregate" and quiet.calls == []
    assert out["answer"].startswith("Your biggest purchase from ")
    about_it, _ = _turn(g, member, thread, "what was the biggest charge on that receipt?",
                        ScriptedChat("The wine, part of the $23.47 total."))
    assert about_it["route"]["kind"] == "receipt" and about_it["answer"].startswith("The wine")


def test_the_biggest_purchase_after_a_list_stays_with_the_conversation(g, member, thread):
    """Asked after a groceries total, "what was the biggest purchase?" means
    the biggest of those. It used to get the last 90 days of everything."""
    _turn(g, member, thread, "how much did I spend on groceries in the last 40 days", ScriptedChat("- Steady."))
    chat = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"query": "groceries"}}]}',
                        '{"steps": []}', "The biggest grocery purchase was at Local Market.")
    out, _ = _turn(g, member, thread, "what was the biggest purchase?", chat)
    assert out["route"]["kind"] == "plan" and out["answer"].startswith("The biggest grocery purchase")
    # The biggest of the groceries, here. In another conversation the same
    # words mean another list, so the answer is never kept under them.
    assert _answer_rows(member, "what was the biggest purchase?") == 0
    fresh, _ = _turn(g, member, thread, "what was my biggest purchase last month?", ScriptedChat("unused"))
    assert fresh["route"]["kind"] == "aggregate"          # its own window: a new question, mid-conversation or not


def test_the_biggest_purchase_at_a_merchant_or_in_a_category(g, member, thread):
    merchant, _, _ = _top_merchant(member)
    start, end = date.today() - timedelta(days=89), date.today() + timedelta(days=1)
    with get_engine().connect() as c:
        at_merchant = c.execute(text(
            "SELECT max(expense_amount), count(*) FROM expenses WHERE user_id = CAST(:u AS UUID) AND merchant = :m "
            "AND expense_date >= :s AND expense_date < :e"), {"u": member, "m": merchant, "s": start, "e": end}).one()
        dining = c.execute(text(
            "SELECT expense_amount, merchant FROM expenses WHERE user_id = CAST(:u AS UUID) "
            "AND shopping_type = 'Restaurant' AND expense_date >= :s AND expense_date < :e "
            "ORDER BY expense_amount DESC, expense_date DESC, expense_id LIMIT 1"),
            {"u": member, "s": date(date.today().year, 1, 1), "e": end}).one()
    quiet = ScriptedChat("the model should not be asked")
    out, _ = _turn(g, member, thread, f"what was my biggest purchase at {merchant.lower()}?", quiet)
    assert out["route"]["kind"] == "aggregate" and quiet.calls == []
    assert out["answer"].startswith(f"Your biggest purchase at {merchant} from {start} to ")
    assert f"**${float(at_merchant[0]):,.2f}**" in out["answer"].splitlines()[0]
    out, _ = _turn(g, member, thread, "What was my priciest dining purchase this year?", quiet)
    assert out["route"]["kind"] == "aggregate" and quiet.calls == []
    assert out["answer"].startswith("Your biggest Restaurant purchase from ")
    assert f"**${float(dining[0]):,.2f}** at **{dining[1]}**" in out["answer"].splitlines()[0]


# A conversation with a receipt on the table, and the answer cache.

def _receipt_on_the_table(g, member, thread, merchant="Blue Bottle Coffee", amount=9.75) -> str:
    doc = str(uuid.uuid4())
    with get_engine().begin() as c:
        c.execute(text("INSERT INTO receipts (document_id, user_id, status, extracted) VALUES "
                       "(CAST(:d AS UUID), CAST(:u AS UUID), 'posted', CAST(:x AS JSONB))"),
                  {"d": doc, "u": member, "x": json.dumps({"merchant": merchant, "amount": amount, "category": "Coffee",
                                                           "items": ["Latte", "Croissant"], "date": "2026-11-21"})})
    G.set_active_receipt(thread, {"document_id": doc, "merchant": merchant, "items": ["Latte", "Croissant"],
                                  "amount": amount}, graph=g)
    return doc


def _cache_steps(events) -> list[str]:
    return [e["status"] for e in events if e["type"] == "step" and e["label"] == "Answer cache"]


def test_the_same_question_about_a_receipt_comes_from_the_cache_the_second_time(g, member, thread):
    """Asked twice in a row with a receipt on the table, a question cost two
    model calls and the trace never mentioned a cache. A conversation with a
    receipt stayed away from the answer cache altogether, because the same
    words about another receipt mean something else. The receipt is part of
    the key now."""
    _receipt_on_the_table(g, member, thread)
    q = "how much was this receipt?"
    first, events = _turn(g, member, thread, q, ScriptedChat("It came to $9.75 at Blue Bottle Coffee."))
    assert first["route"]["kind"] == "receipt" and not first["cached"] and _cache_steps(events) == ["miss"]
    again, again_events = _turn(g, member, thread, q, ScriptedChat("the model was asked again"))
    assert again["cached"] and again["answer"] == first["answer"] and again["saved_tokens"] > 0
    assert _cache_steps(again_events) == ["hit"]
    assert "LLM answer" not in [e["label"] for e in again_events if e["type"] == "step"]
    in_spanish, spanish_events = _turn(g, member, thread, q, ScriptedChat("Fueron $9.75 en Blue Bottle Coffee."),
                                       language_code="es-ES")
    assert not in_spanish["cached"] and _cache_steps(spanish_events) == ["miss"]     # another language, another answer


def test_another_receipt_never_gets_the_first_ones_answer(g, member, thread):
    _receipt_on_the_table(g, member, thread)
    q = "how much was this receipt?"
    _turn(g, member, thread, q, ScriptedChat("It came to $9.75 at Blue Bottle Coffee."))
    other = threads.create_thread(member)["thread_id"]
    try:
        _receipt_on_the_table(g, member, other, merchant="Harbor Deli", amount=18.20)
        out, events = _turn(g, member, other, q, ScriptedChat("It came to $18.20 at Harbor Deli."))
        assert not out["cached"] and "18.20" in out["answer"] and _cache_steps(events) == ["miss"]
    finally:
        threads.delete_thread(member, other)


def test_a_question_that_leans_on_the_turn_before_is_still_never_cached(g, member, thread):
    _receipt_on_the_table(g, member, thread)
    _turn(g, member, thread, "how much was this receipt?", ScriptedChat("It came to $9.75."))
    q = "was the latte on it?"                              # "it": after another turn it means something else
    first, events = _turn(g, member, thread, q, ScriptedChat("Yes, one latte."))
    again, again_events = _turn(g, member, thread, q, ScriptedChat("Yes, the latte is on it."))
    assert not first["cached"] and not again["cached"] and again["answer"] == "Yes, the latte is on it."
    assert _cache_steps(events) == [] and _cache_steps(again_events) == []


def test_a_new_purchase_retires_what_was_cached_about_a_receipt(g, member, thread):
    from banko_ai.services import ledger
    from banko_ai.services.accounts import list_accounts
    _receipt_on_the_table(g, member, thread)
    q = "how much was this receipt?"
    _turn(g, member, thread, q, ScriptedChat("It came to $9.75."))
    card = next(a for a in list_accounts(member)["accounts"] if a["kind"] == "credit")
    ledger.post_purchase(member, card["account_id"], amount="3.10", merchant="Corner Kiosk", category="Other")
    out, events = _turn(g, member, thread, q, ScriptedChat("It came to $9.75, as before."))
    assert not out["cached"] and out["answer"] == "It came to $9.75, as before." and _cache_steps(events) == ["miss"]


def test_an_open_question_beside_a_receipt_is_cached_with_it(g, member, thread):
    """"where do I get my coffee before work" twice, with a coffee receipt
    on the table: an open question the planner answers, and the second ask
    comes from the cache."""
    _receipt_on_the_table(g, member, thread)
    q = "where do I get my coffee before work"
    plan = ('{"steps": [{"tool": "search_transactions", "args": {"query": "coffee"}}]}', '{"steps": []}')
    first, events = _turn(g, member, thread, q, ScriptedChat(*plan, "You bought coffee a few times."))
    assert first["route"]["kind"] == "plan" and not first["cached"] and _cache_steps(events) == ["miss"]
    again, again_events = _turn(g, member, thread, q, ScriptedChat("the model was asked again"))
    assert again["cached"] and again["answer"] == first["answer"] and _cache_steps(again_events) == ["hit"]
    # The same words in a conversation without the receipt are another question: its answer was never theirs.
    elsewhere, elsewhere_events = _fresh_thread_turn(g, member, q, ScriptedChat(*plan, "A few coffees, no receipt here."))
    assert not elsewhere["cached"] and _cache_steps(elsewhere_events) == ["miss"]
