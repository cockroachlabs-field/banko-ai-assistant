"""The per-answer DB Ops trace: every row is something the server really
did for that request, it names the provider the way the persona shows it,
and failures leave a neutral error step instead of the exception."""

import os
import uuid
from datetime import date, timedelta

import pytest
from sqlalchemy import text

from banko_ai.utils import request_trace as rt

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


def test_stored_traces_are_bounded():
    first = rt.store_trace([{"label": "x"}])
    for _ in range(rt.MAX_STORED_TRACES):
        rt.store_trace([{"label": "y"}])
    assert rt.get_stored_trace(first) is None
    assert rt.get_stored_trace("not-an-id") is None


def test_serving_regions_and_index_parse_the_plan():
    plan = ("• scan\n  regions: us-east-1, us-central-1\n"
            "  table: expenses@idx_expenses_user_embedding\n  regions: us-east-1")
    assert rt.serving_regions(plan) == "us-central-1, us-east-1"
    assert rt.plan_indexes(plan) == ["idx_expenses_user_embedding"]
    assert rt.serving_regions("") is None


class _FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def invoke(self, prompt):
        class R:
            content = self.reply
        return R()


@pytest.fixture
def trace_client(monkeypatch):
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
        c.execute(text("DELETE FROM query_cache WHERE scope = :u"), {"u": MAYA})

    client = app.test_client()
    client.post("/login", data={"username": "maya"})
    yield client
    with engine.begin() as c:
        c.execute(text("DELETE FROM expenses WHERE expense_id = ANY(CAST(:ids AS UUID[]))"), {"ids": ids})
    app.extensions["banko_warmup_stop"].set()


def _latest_trace(client):
    with client.session_transaction() as s:
        message = s["chat"][-1]
    assert message["class"] == "Assistant"
    trace = rt.get_stored_trace(message["meta"]["trace_id"])
    assert trace, message
    return trace


def _assert_no_leaks(trace):
    blob = str(trace).lower()
    for word in ("ollama", "granite3.3:8b", "localhost", "postgresql://", "11434"):
        assert word not in blob, word


@needs_db
def test_aggregation_answer_stores_its_trace(trace_client, monkeypatch):
    import banko_ai.agents.llm_factory as lf
    monkeypatch.setattr(lf, "get_llm_for_agent", lambda **kw: _FakeLLM("- Dining is steady."))

    page = trace_client.post("/banko", data={
        "user_input": "How much did I spend on restaurants in the last 30 days?"})
    assert page.status_code == 200
    trace = _latest_trace(trace_client)
    kinds = [s["kind"] for s in trace]
    assert kinds[0] == "route" and "SQL" in trace[0]["detail"]
    assert kinds.index("sql") < kinds.index("llm")
    assert kinds[-1] == "total"
    narrative = next(s for s in trace if s["label"] == "Narrative cache")
    assert narrative["status"] == "miss"
    llm = next(s for s in trace if s["kind"] == "llm")
    assert "IBM watsonx" in llm["detail"] and "ibm/granite-3-3-8b-instruct" in llm["detail"]
    assert llm["ms"] is not None
    assert "Intent routing" in page.get_data(as_text=True)
    _assert_no_leaks(trace)


@needs_db
def test_retrieval_answer_stores_its_trace(trace_client, monkeypatch):
    from banko_ai.ai_providers.ollama_provider import OllamaProvider
    monkeypatch.setattr(OllamaProvider, "_chat", lambda self, *a, **kw: "You had a few dinners.")

    question = f"Show me my restaurant dinners {uuid.uuid4().hex[:6]}"
    trace_client.post("/banko", data={"user_input": question})
    trace = _latest_trace(trace_client)
    kinds = [s["kind"] for s in trace]
    labels = [s["label"] for s in trace]
    assert kinds[0] == "route" and "vector" in trace[0]["detail"].lower()
    assert "Embedding cache" in labels
    assert "Vector search cache" in labels
    assert "Response cache" in labels
    assert labels.index("Response cache") < kinds.index("llm") < labels.index("Response cache write")
    llm = next(s for s in trace if s["kind"] == "llm")
    assert "IBM watsonx" in llm["detail"]
    assert kinds[-1] == "total"
    _assert_no_leaks(trace)


@needs_db
def test_error_path_records_a_neutral_error_step(trace_client, monkeypatch):
    from banko_ai.ai_providers.ollama_provider import OllamaProvider

    def boom(self, *a, **kw):
        raise RuntimeError("Cannot reach Ollama at http://localhost:11434 secret-token")
    monkeypatch.setattr(OllamaProvider, "simple_rag_response", boom)

    trace_client.post("/banko", data={"user_input": "Show me my restaurant dinners please"})
    trace = _latest_trace(trace_client)
    assert trace[0]["kind"] == "route"
    errors = [s for s in trace if s["status"] == "error"]
    assert errors, trace
    assert "secret-token" not in str(trace) and "Cannot reach" not in str(trace)
    _assert_no_leaks(trace)
