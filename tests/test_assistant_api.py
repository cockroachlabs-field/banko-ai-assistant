"""The assistant API: threads, streamed answers, receipts, actions and
memories, always for the signed-in user only."""

import io
import json
import os
from datetime import date

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from PIL import Image
from sqlalchemy import text

from banko_ai.assistant import graph as G
from banko_ai.db.engine import get_engine
from banko_ai.services import memories, receipts
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
    return c


def events(resp) -> list[tuple[str, dict]]:
    out = []
    for block in resp.get_data(as_text=True).strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def _thread(client) -> str:
    resp = client.post("/api/v1/assistant/threads", json={})
    assert resp.status_code == 201
    return resp.get_json()["thread_id"]


def test_test_apps_never_join_the_live_kafka_group(monkeypatch, member):
    """The shell profile exports COACH_KAFKA_ENABLED=true for the live demo;
    a test app must not start a consumer in that group."""
    from banko_ai.config.settings import get_config
    monkeypatch.setenv("COACH_KAFKA_ENABLED", "true")
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9")  # harmless even if the guard regresses
    app_client(monkeypatch, REPORTING_FOLLOWER_READS="false")
    assert get_config().coach_kafka_enabled is False


def test_a_streamed_answer(client):
    tid = _thread(client)
    resp = client.post(f"/api/v1/assistant/threads/{tid}/messages",
                       json={"message": "how much did I spend on restaurants in the last 30 days"})
    assert resp.mimetype == "text/event-stream"
    evs = events(resp)
    kinds = [k for k, _ in evs]
    assert kinds[-2:] == ["answer", "done"] and "token" in kinds and "step" in kinds
    answer = dict(evs)["answer"]
    assert answer["route"] == "aggregate" and "You spent" in answer["answer"]
    got = client.get(f"/api/v1/assistant/threads/{tid}").get_json()
    assert got["message_count"] == 2 and len(got["messages"]) == 2


def test_other_users_thread_is_404(client, monkeypatch):
    other = make_user()
    try:
        c2 = app_client(monkeypatch, REPORTING_FOLLOWER_READS="false")
        with c2.session_transaction() as s:
            s["user_id"] = other
        tid = _thread(c2)
        assert client.get(f"/api/v1/assistant/threads/{tid}").status_code == 404
        assert client.post(f"/api/v1/assistant/threads/{tid}/messages", json={"message": "hi"}).status_code == 404
        assert client.patch(f"/api/v1/assistant/threads/{tid}", json={"title": "x"}).status_code == 404
        assert client.delete(f"/api/v1/assistant/threads/{tid}").status_code == 404
        assert client.get("/api/v1/assistant/threads/not-a-uuid").status_code == 404
    finally:
        _drop(other)


def test_empty_message_is_400(client):
    tid = _thread(client)
    assert client.post(f"/api/v1/assistant/threads/{tid}/messages", json={"message": "  "}).status_code == 400


def test_stream_error_is_neutral_and_closes(client, chat):
    class Broken:
        def complete(self, messages, *, temperature=0.3):
            raise RuntimeError("secret upstream detail 401")

        def stream(self, messages, *, temperature=0.3):
            raise RuntimeError("secret upstream detail 401")
            yield ""

    chat["chat"] = Broken()
    tid = _thread(client)
    evs = events(client.post(f"/api/v1/assistant/threads/{tid}/messages",
                             json={"message": "where could I cut back on dining?"}))
    assert [k for k, _ in evs][-2:] == ["error", "done"]
    assert "secret" not in json.dumps(evs)


def test_confirm_a_proposal(client, chat, member):
    chat["chat"] = ScriptedChat('{"steps": [{"tool": "propose_budget", "args": {"category": "Shopping", "amount": "150"}}]}',
                                '{"steps": []}', "Ready when you are.")
    tid = _thread(client)
    evs = events(client.post(f"/api/v1/assistant/threads/{tid}/messages",
                             json={"message": "set my shopping budget to 150"}))
    proposal = next(d for k, d in evs if k == "proposal")
    ok = client.post(f"/api/v1/actions/{proposal['action_id']}/confirm")
    assert ok.status_code == 200 and ok.get_json()["status"] == "done"
    assert client.post(f"/api/v1/actions/{proposal['action_id']}/confirm").get_json()["status"] == "done"
    assert client.post("/api/v1/actions/00000000-0000-0000-0000-000000000000/confirm").status_code == 404


def test_memories_list_and_delete(client, member):
    m = memories.remember(member, "I'm saving for a canoe")
    listed = client.get("/api/v1/memories").get_json()["memories"]
    assert any(x["memory_id"] == m["memory_id"] for x in listed)
    assert client.delete(f"/api/v1/memories/{m['memory_id']}").status_code == 204
    assert client.delete(f"/api/v1/memories/{m['memory_id']}").status_code == 404


def test_receipt_then_follow_up(client, chat, monkeypatch):
    fields = {"merchant": "Corner Market", "amount": 19.95, "date": str(date.today()), "category": "groceries",
              "items": ["Olive oil"], "payment_method": "debit card"}
    monkeypatch.setattr(receipts, "default_hooks", lambda model=None, url=None: receipts.ReceiptHooks(
        ocr=lambda p: "CORNER MARKET 19.95", extract=lambda t: dict(fields),
        fraud=lambda u, e: "No issues detected", budget=lambda u: "Within budget"))
    buf = io.BytesIO()
    Image.new("RGB", (300, 400), "white").save(buf, "PNG")
    evs = events(client.post("/api/v1/receipts", data={"receipt": (io.BytesIO(buf.getvalue()), "r.png")},
                             content_type="multipart/form-data"))
    receipt = dict(evs)["receipt"]
    assert receipt["status"] == "posted" and receipt["thread_id"]
    chat["chat"] = ScriptedChat(lambda msgs: "It was $19.95 at Corner Market.")
    follow = events(client.post(f"/api/v1/assistant/threads/{receipt['thread_id']}/messages",
                                json={"message": "how much was this receipt?"}))
    answer = dict(follow)["answer"]
    assert answer["route"] == "receipt" and "19.95" in answer["answer"]


def test_receipt_type_is_checked(client):
    resp = client.post("/api/v1/receipts", data={"receipt": (io.BytesIO(b"MZ"), "x.exe")},
                       content_type="multipart/form-data")
    assert resp.status_code == 400


def test_thread_opened_from_a_nudge_keeps_its_context(client):
    resp = client.post("/api/v1/assistant/threads",
                       json={"context": {"kind": "nudge", "id": "00000000-0000-0000-0000-000000000001"}})
    tid = resp.get_json()["thread_id"]
    assert G.thread_state(tid)["context"]["kind"] == "nudge"
    bad = client.post("/api/v1/assistant/threads", json={"context": {"kind": "secrets", "id": "x"}})
    assert bad.status_code == 400


def test_turns_and_receipt_steps_show_in_the_agent_feed(client, chat, monkeypatch):
    app = client.application
    feed = app.extensions["socketio"].test_client(app)

    def lines():
        return [e["args"][0] for e in feed.get_received() if e["name"] == "agent_activity"]

    tid = _thread(client)
    events(client.post(f"/api/v1/assistant/threads/{tid}/messages",
                       json={"message": "how much did I spend on restaurants in the last 30 days"}))
    turn = lines()
    assert [(a["agent_type"], a["status"]) for a in turn] == [("assistant", "processing"), ("assistant", "completed")]
    assert all(a["region"] and a["timestamp"] for a in turn)

    class Down:
        def complete(self, messages, *, temperature=0.3):
            raise RuntimeError("provider down")

        def stream(self, messages, *, temperature=0.3):
            raise RuntimeError("provider down")
            yield ""

    chat["chat"] = Down()
    events(client.post(f"/api/v1/assistant/threads/{tid}/messages", json={"message": "where could I cut back on dining?"}))
    assert [(a["agent_type"], a["status"]) for a in lines()] == [("assistant", "processing"), ("assistant", "failed")]

    fields = {"merchant": "Feed Street Bakery", "amount": 7.35, "date": str(date.today()), "category": "groceries",
              "items": ["Rye loaf"], "payment_method": "debit card"}
    monkeypatch.setattr(receipts, "default_hooks", lambda model=None, url=None: receipts.ReceiptHooks(
        ocr=lambda p: "FEED STREET BAKERY 7.35", extract=lambda t: dict(fields),
        fraud=lambda u, e: "No issues detected", budget=lambda u: "Within budget"))
    buf = io.BytesIO()
    Image.new("RGB", (300, 400), "white").save(buf, "PNG")
    evs = events(client.post("/api/v1/receipts", data={"receipt": (io.BytesIO(buf.getvalue()), "r.png")},
                             content_type="multipart/form-data"))
    steps = [d["step"] for k, d in evs if k == "step" and d["status"] != "start"]
    got = lines()
    assert [a["message"].split(":")[0] for a in got] == steps
    assert [a["agent_type"] for a in got] == ["receipt"] * 5 + ["fraud", "budget"]
    assert {a["status"] for a in got} == {"completed"}


def test_the_answer_carries_the_proposals_it_streamed(client, chat):
    chat["chat"] = ScriptedChat('{"steps": [{"tool": "propose_budget", "args": {"category": "Shopping", "amount": "175"}}]}',
                                '{"steps": []}', "Say the word.")
    tid = _thread(client)
    evs = events(client.post(f"/api/v1/assistant/threads/{tid}/messages",
                             json={"message": "set my shopping budget to 175"}))
    streamed = [d for k, d in evs if k == "proposal"]
    assert streamed and dict(evs)["answer"]["proposals"] == streamed


def test_a_posted_receipt_still_reports_when_its_thread_cannot_open(client, monkeypatch):
    fields = {"merchant": "Lone Pine Hardware", "amount": 12.40, "date": str(date.today()), "category": "shopping",
              "items": ["Hinges"], "payment_method": "debit card"}
    monkeypatch.setattr(receipts, "default_hooks", lambda model=None, url=None: receipts.ReceiptHooks(
        ocr=lambda p: "LONE PINE HARDWARE 12.40", extract=lambda t: dict(fields),
        fraud=lambda u, e: "No issues detected", budget=lambda u: "Within budget"))

    def thread_store_down(*args, **kwargs):
        raise RuntimeError("thread store down")

    monkeypatch.setattr(assistant_api.threads, "create_thread", thread_store_down)
    buf = io.BytesIO()
    Image.new("RGB", (300, 400), "white").save(buf, "PNG")
    evs = events(client.post("/api/v1/receipts", data={"receipt": (io.BytesIO(buf.getvalue()), "r.png")},
                             content_type="multipart/form-data"))
    assert [k for k, _ in evs][-2:] == ["receipt", "done"]
    receipt = dict(evs)["receipt"]
    assert receipt["status"] == "posted" and receipt["thread_id"] is None
