"""The per-answer DB Ops trace: every row is something the server really
did for that request, it names the provider the way the persona shows it,
and failures leave a neutral error step instead of the exception."""

import json
import os
import uuid
from datetime import date, timedelta

import pytest
from sqlalchemy import text

from banko_ai.utils import request_trace as rt
from tests._chat import ScriptedChat

MAYA = "00000000-0000-0000-0000-0000000000a1"
needs_db = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


def test_trace_step_is_a_noop_without_an_active_trace():
    rt.trace_step("cache", "Embedding cache", "miss", ms=1.0, status="miss")
    assert rt.current_trace() == []


def test_trace_step_records_in_order_with_an_active_trace():
    token = rt.start_trace()
    try:
        rt.trace_step("route", "Intent routing", "retrieval")
        rt.trace_step("cache", "Embedding cache", "hit", ms=2.345, status="hit")
        rt.trace_step("odd", "Unknown status", status="bogus")
        steps = rt.current_trace()
    finally:
        rt.reset_trace(token)
    assert [s["label"] for s in steps] == ["Intent routing", "Embedding cache", "Unknown status"]
    assert steps[1]["status"] == "hit" and steps[1]["ms"] == 2.3
    assert steps[2]["status"] == "info"
    assert all("t" in s for s in steps)
    assert rt.current_trace() == []


def test_serving_regions_and_index_parse_the_plan():
    plan = ("• scan\n  regions: us-east-1, us-central-1\n"
            "  table: expenses@idx_expenses_user_embedding\n  regions: us-east-1")
    assert rt.serving_regions(plan) == "us-central-1, us-east-1"
    assert rt.plan_indexes(plan) == ["idx_expenses_user_embedding"]
    assert rt.serving_regions("") is None


@pytest.fixture
def chat(monkeypatch):
    """The chat page's model, scripted, on a graph with an in-memory checkpointer."""
    from langgraph.checkpoint.memory import InMemorySaver

    from banko_ai.assistant import graph as G
    from banko_ai.web import assistant_api

    holder = {"chat": ScriptedChat("- Dining is steady.")}
    monkeypatch.setattr(assistant_api, "_chat", lambda: holder["chat"])
    monkeypatch.setattr(G, "_graph", G.build_graph(InMemorySaver()))
    return holder


@pytest.fixture
def trace_client(monkeypatch, chat):
    monkeypatch.setenv("AI_SERVICE", "ollama")
    monkeypatch.setenv("AI_DISPLAY_PROVIDER", "watsonx")
    monkeypatch.setenv("AI_DISPLAY_MODEL", "ibm/granite-3-3-8b-instruct")
    from banko_ai.db.engine import get_engine
    from banko_ai.utils.embeddings import load_embedding_model
    from banko_ai.web.app import create_app

    app = create_app()
    app.extensions["banko_readiness"].wait_for("schema", 60)
    app.extensions["banko_readiness"].wait_for("personas", 60)
    model = load_embedding_model(wait_seconds=600)

    engine = get_engine()
    ids = [str(uuid.uuid4()) for _ in range(3)]
    vec = "[" + ",".join(str(float(x)) for x in model.encode("restaurant dinner")) + "]"
    with engine.begin() as c:
        for i, eid in enumerate(ids):
            c.execute(text("""
                INSERT INTO expenses (expense_id, user_id, expense_date, expense_amount,
                  shopping_type, description, merchant, payment_method, embedding)
                VALUES (:id, :u, :d, :a, 'Restaurant', 'trace test dinner', :m, 'Credit Card',
                        CAST(:e AS VECTOR(384)))
            """), {"id": eid, "u": MAYA, "d": date.today() - timedelta(days=3 + i),
                   "a": 20 + i, "m": f"TraceBistro{i}", "e": vec})
        # Every test starts cold: an answer an earlier run cached must not come back as a hit.
        for table in ("search_cache", "answer_cache"):
            c.execute(text(f"DELETE FROM {table} WHERE user_id = CAST(:u AS UUID)"), {"u": MAYA})

    client = app.test_client()
    client.post("/login", data={"username": "maya"})
    yield client
    with engine.begin() as c:
        c.execute(text("DELETE FROM expenses WHERE expense_id = ANY(CAST(:ids AS UUID[]))"), {"ids": ids})
    app.extensions["banko_warmup_stop"].set()


def _ask(client, question):
    """One turn on a fresh thread; the raw SSE body and its parsed events."""
    tid = client.post("/api/v1/assistant/threads", json={}).get_json()["thread_id"]
    body = client.post(f"/api/v1/assistant/threads/{tid}/messages", json={"message": question}).get_data(as_text=True)
    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        events.append((lines["event"], json.loads(lines["data"])))
    return body, events


def _answer(client, question):
    _, events = _ask(client, question)
    answer = dict(events)["answer"]
    return answer, answer["meta"]["trace"]


def _assert_no_leaks(trace):
    blob = str(trace).lower()
    for word in ("ollama", "granite3.3:8b", "localhost", "postgresql://", "11434"):
        assert word not in blob, word


AGG_QUESTION = "How much did I spend on restaurants in the last 30 days?"


@needs_db
def test_an_aggregation_answer_carries_its_trace(trace_client):
    answer, trace = _answer(trace_client, AGG_QUESTION)
    labels, kinds = [s["label"] for s in trace], [s["kind"] for s in trace]
    assert "SQL" in trace[labels.index("Intent routing")]["detail"]
    assert labels.index("Intent routing") < labels.index("Aggregation SQL") < kinds.index("llm")
    assert kinds[-1] == "total"
    # The steps the old page showed: the narrative cache on a miss too, and the plan.
    narrative = next(s for s in trace if s["label"] == "Narrative cache")
    assert narrative["status"] == "miss"
    assert labels.index("Aggregation SQL") < labels.index("Narrative cache") < labels.index("EXPLAIN ANALYZE")
    llm = trace[kinds.index("llm")]
    assert "IBM watsonx" in llm["detail"] and "ibm/granite-3-3-8b-instruct" in llm["detail"]
    assert llm["ms"] is not None
    assert answer["meta"]["db_ms"] is not None and answer["meta"]["explain"]
    _assert_no_leaks(trace)


@needs_db
def test_a_repeat_aggregation_names_no_model(trace_client):
    """SQL reruns and the prose comes back from the narrative cache, so the
    trace shows the hit and no model step: no model ran."""
    _answer(trace_client, AGG_QUESTION)
    _, trace = _answer(trace_client, AGG_QUESTION)
    labels = [s["label"] for s in trace]
    assert trace[labels.index("Narrative cache")]["status"] == "hit"
    assert "Aggregation SQL" in labels
    assert "llm" not in [s["kind"] for s in trace]
    _assert_no_leaks(trace)


@needs_db
def test_a_retrieval_answer_carries_its_trace(trace_client, chat):
    chat["chat"] = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"query": "restaurant dinners"}}]}',
                                '{"steps": []}', "You had a few dinners.")
    answer, trace = _answer(trace_client, "Show me my restaurant dinners")
    labels, kinds = [s["label"] for s in trace], [s["kind"] for s in trace]
    assert trace[labels.index("Intent routing")]["detail"].startswith("plan")
    assert trace[labels.index("Answer cache")]["status"] == "miss"
    assert "Search cache" in labels
    assert (labels.index("Answer cache") < labels.index("Tool search_transactions")
            < labels.index("EXPLAIN ANALYZE") < labels.index("LLM answer") < kinds.index("llm"))
    llm = trace[kinds.index("llm")]
    assert "IBM watsonx" in llm["detail"] and "3 calls" in llm["detail"]
    assert kinds[-1] == "total"
    assert answer["meta"]["explain"]
    _assert_no_leaks(trace)


@needs_db
def test_a_failed_answer_keeps_a_neutral_trace(trace_client, chat):
    class Down:
        def complete(self, messages, *, temperature=0.3):
            raise RuntimeError("Cannot reach Ollama at http://localhost:11434 secret-token")

        def stream(self, messages, *, temperature=0.3):
            raise RuntimeError("Cannot reach Ollama at http://localhost:11434 secret-token")
            yield ""

    chat["chat"] = Down()
    body, events = _ask(trace_client, "Show me my restaurant dinners please")
    failed = dict(events)["error"]
    assert "Intent routing" in [s["label"] for s in failed["trace"]]
    assert [s for s in failed["trace"] if s["status"] == "error"]
    assert "secret-token" not in body and "Cannot reach" not in body
    _assert_no_leaks(failed["trace"])
