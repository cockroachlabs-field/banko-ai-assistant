"""The current chat page keeps working and gains receipt follow-ups."""

import io
import os
from datetime import date

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from PIL import Image
from sqlalchemy import text

from banko_ai.assistant import graph as G
from banko_ai.db.engine import get_engine
from banko_ai.services import receipts
from banko_ai.services.seeding import seed_user
from banko_ai.web import assistant_api
from tests._app import app_client, stop_apps
from tests._chat import ScriptedChat
from tests._money import drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    stop_apps()


def _drop(uid: str) -> None:
    with get_engine().begin() as c:
        for t in ("chat_threads", "receipts", "memories", "assistant_actions", "answer_cache", "search_cache"):
            c.execute(text(f"DELETE FROM {t} WHERE user_id = CAST(:u AS UUID)"), {"u": uid})
    drop_user(uid)


@pytest.fixture(scope="module")
def member():
    uid = make_user("diner")
    seed_user(uid, months=1, today=date.today())
    yield uid
    _drop(uid)


@pytest.fixture
def chat(monkeypatch):
    holder = {"chat": ScriptedChat("- steady")}
    monkeypatch.setattr(assistant_api, "_chat", lambda: holder["chat"])
    monkeypatch.setattr(G, "_graph", G.build_graph(InMemorySaver()))
    return holder


@pytest.fixture
def client(member, monkeypatch, chat):
    c = app_client(monkeypatch, REPORTING_FOLLOWER_READS="false")
    with c.session_transaction() as s:
        s["user_id"] = member
        s["username"] = "tester"
    return c


def test_chat_page_answers_with_meta(client):
    resp = client.post("/banko", data={"user_input": "how much did I spend on restaurants in the last 30 days",
                                       "response_language": "en-US"})
    assert resp.status_code == 200
    page = resp.get_data(as_text=True)
    assert "You spent" in page
    with client.session_transaction() as s:
        last = s["chat"][-1]
        assert last["class"] == "Assistant" and last["meta"]["trace_id"]
        assert "db_ms" in last["meta"]
        assert s["thread_id"]


def test_the_session_cookie_keeps_the_thread(client, chat):
    """A semantic search EXPLAIN runs to 20 KB. Kept in the 4 KB session
    cookie, it made browsers drop the cookie and the new thread id with it."""
    chat["chat"] = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"query": "restaurant dinners"}}]}',
                                '{"steps": []}', "You had a few dinners.")
    resp = client.post("/banko", data={"user_input": "show me my restaurant dinners", "response_language": "en-US"})
    cookie = resp.headers.get("Set-Cookie", "")
    assert cookie and len(cookie) < 4000
    assert "planning time" in resp.get_data(as_text=True)  # the plan still renders with the answer


def test_upload_then_ask_on_the_chat_page(client, chat, monkeypatch):
    fields = {"merchant": "Lantern Books", "amount": 42.00, "date": str(date.today()), "category": "shopping",
              "items": ["Atlas"], "payment_method": "credit card"}
    monkeypatch.setattr(receipts, "default_hooks", lambda model=None, url=None: receipts.ReceiptHooks(
        ocr=lambda p: "LANTERN BOOKS 42.00", extract=lambda t: dict(fields),
        fraud=lambda u, e: "No issues detected", budget=lambda u: "Within budget"))
    buf = io.BytesIO()
    Image.new("RGB", (300, 400), "white").save(buf, "PNG")
    up = client.post("/api/upload-receipt", data={"receipt": (io.BytesIO(buf.getvalue()), "r.png")},
                     content_type="multipart/form-data")
    body = up.get_json()
    assert up.status_code == 200 and body["success"] and body["merchant"] == "Lantern Books"
    assert {"expense_id", "document_id", "fraud_status", "budget_impact"} <= set(body)
    chat["chat"] = ScriptedChat(lambda msgs: "That receipt from Lantern Books came to $42.00.")
    page = client.post("/banko", data={"user_input": "how much was this receipt?",
                                       "response_language": "en-US"}).get_data(as_text=True)
    assert "Lantern Books came to $42.00" in page


def test_template_echo_is_422(client, monkeypatch):
    junk = {"merchant": "store name", "amount": 0.0, "date": "YYYY-MM-DD", "category": "food|transport"}
    monkeypatch.setattr(receipts, "default_hooks", lambda model=None, url=None: receipts.ReceiptHooks(
        ocr=lambda p: "x y z", extract=lambda t: dict(junk), fraud=lambda u, e: "", budget=lambda u: ""))
    buf = io.BytesIO()
    Image.new("RGB", (30, 40), "white").save(buf, "PNG")
    up = client.post("/api/upload-receipt", data={"receipt": (io.BytesIO(buf.getvalue()), "r.png")},
                     content_type="multipart/form-data")
    assert up.status_code == 422 and not up.get_json()["success"]


def test_rag_endpoint_shape(client):
    resp = client.post("/api/rag", json={"query": "what is the capital of France", "language": "en"})
    body = resp.get_json()
    assert body["success"] and body["metadata"]["route"] == "offtopic"


def test_shared_default_history_is_not_written(client):
    with get_engine().connect() as c:
        exists = c.execute(text("SELECT count(*) FROM [SHOW TABLES] WHERE table_name = 'chat_message_store'")).scalar()
        before = c.execute(text("SELECT count(*) FROM chat_message_store WHERE session_id = 'default'")).scalar() if exists else 0
    client.post("/banko", data={"user_input": "how much did I spend on groceries this month"})
    with get_engine().connect() as c:
        after = c.execute(text("SELECT count(*) FROM chat_message_store WHERE session_id = 'default'")).scalar() if exists else 0
    assert after == before


def _hooks(monkeypatch, fields: dict) -> None:
    monkeypatch.setattr(receipts, "default_hooks", lambda model=None, url=None: receipts.ReceiptHooks(
        ocr=lambda p: f"{fields['merchant'].upper()} {fields['amount']:.2f}", extract=lambda t: dict(fields),
        fraud=lambda u, e: "No issues detected", budget=lambda u: "Within budget"))


def _upload(client):
    buf = io.BytesIO()
    Image.new("RGB", (300, 400), "white").save(buf, "PNG")
    return client.post("/api/upload-receipt", data={"receipt": (io.BytesIO(buf.getvalue()), "r.png")},
                       content_type="multipart/form-data")


def test_an_answer_without_database_work_shows_no_latency(client):
    page = client.post("/banko", data={"user_input": "what is the capital of France"}).get_data(as_text=True)
    assert "Nonems" not in page and "how was this answered" in page
    with client.session_transaction() as s:
        assert set(s["chat"][-1]["meta"]) == {"trace_id"}


def test_upload_steps_reach_the_agent_feed_with_their_region(client, monkeypatch):
    app = client.application
    feed = app.extensions["socketio"].test_client(app)
    _hooks(monkeypatch, {"merchant": "Feed Street Bakery", "amount": 7.35, "date": str(date.today()),
                         "category": "groceries", "items": ["Rye loaf"], "payment_method": "debit card"})
    assert _upload(client).status_code == 200
    got = [e["args"][0] for e in feed.get_received() if e["name"] == "agent_activity"]
    assert [a["agent_type"] for a in got] == ["receipt"] * 5 + ["fraud", "budget"]
    assert all(a["region"] and a["timestamp"] for a in got)


def test_a_posted_receipt_reports_even_when_the_thread_cannot_open(client, monkeypatch):
    from banko_ai.services import threads

    _hooks(monkeypatch, {"merchant": "Lone Pine Hardware", "amount": 12.40, "date": str(date.today()),
                         "category": "shopping", "items": ["Hinges"], "payment_method": "debit card"})

    def thread_store_down(*args, **kwargs):
        raise RuntimeError("thread store down")

    monkeypatch.setattr(threads, "create_thread", thread_store_down)
    up = _upload(client)
    assert up.status_code == 200 and up.get_json()["success"] and up.get_json()["expense_id"]


def test_a_duplicate_upload_says_so(client, monkeypatch):
    _hooks(monkeypatch, {"merchant": "Twice Told Tales", "amount": 17.25, "date": str(date.today()),
                         "category": "shopping", "items": ["Paperback"], "payment_method": "credit card"})
    first = _upload(client).get_json()
    again = _upload(client)
    body = again.get_json()
    assert first["duplicate"] is False
    assert again.status_code == 200 and body["duplicate"] is True and body["expense_id"] == first["expense_id"]
    assert body["fraud_status"].startswith("Duplicate") and "nothing posted" in body["budget_impact"]
