"""The assistant panel ships on every page, and the answer event carries
what the trace needs."""

import json
import os
import re
from datetime import date

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from banko_ai.assistant import background
from banko_ai.assistant import graph as G
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


@pytest.fixture(scope="module")
def member():
    uid = make_user("diner")
    seed_user(uid, months=1, today=date.today())
    yield uid
    drop_user(uid)


@pytest.fixture
def client(member, monkeypatch):
    monkeypatch.setattr(assistant_api, "_chat", lambda: ScriptedChat("- steady"))
    monkeypatch.setattr(G, "_graph", G.build_graph(InMemorySaver()))
    c = app_client(monkeypatch, REPORTING_FOLLOWER_READS="false")
    with c.session_transaction() as s:
        s["user_id"] = member
    return c


def _events(resp):
    out = []
    for block in resp.get_data(as_text=True).strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_panel_is_on_every_page(client):
    html = client.get("/").get_data(as_text=True)
    assert 'x-data="assistant(\'panel\')"' in html
    assert "js/assistant.js" in html


def test_answer_event_carries_the_plan_and_the_whole_trace(client):
    tid = client.post("/api/v1/assistant/threads", json={}).get_json()["thread_id"]
    evs = _events(client.post(f"/api/v1/assistant/threads/{tid}/messages",
                              json={"message": "how much did I spend on restaurants in the last 30 days"}))
    answer = dict(evs)["answer"]
    assert answer["route"] == "aggregate"
    assert "explain" in answer["meta"] and 0 < len(answer["meta"]["explain"]) <= 20000
    trace = answer["meta"]["trace"]
    labels, kinds = [s["label"] for s in trace], [s["kind"] for s in trace]
    assert labels.index("Intent routing") < labels.index("Aggregation SQL")
    assert "llm" in kinds and kinds[-1] == "total"


def test_running_steps_stream_but_never_land_in_the_trace(client):
    tid = client.post("/api/v1/assistant/threads", json={}).get_json()["thread_id"]
    evs = _events(client.post(f"/api/v1/assistant/threads/{tid}/messages",
                              json={"message": "how much did I spend on restaurants in the last 33 days"}))
    running = [d for k, d in evs if k == "step" and d["status"] == "running"]
    assert running and all(d["key"] for d in running)
    assert "Running the numbers in CockroachDB" in [d["label"] for d in running]
    trace = dict(evs)["answer"]["meta"]["trace"]
    assert all(s["status"] != "running" for s in trace)
    assert not {d["label"] for d in running} & {s["label"] for s in trace}


def test_the_answer_says_what_the_model_spent_and_what_the_cache_saved(client):
    tid = client.post("/api/v1/assistant/threads", json={}).get_json()["thread_id"]
    q = "how much did I spend on restaurants in the last 41 days"
    first = dict(_events(client.post(f"/api/v1/assistant/threads/{tid}/messages", json={"message": q})))["answer"]
    (model,) = [s for s in first["meta"]["trace"] if s["kind"] == "llm"]
    # The scripted model reports no usage, so the count is an estimate and says so.
    assert re.search(r", 1 call, about [\d,]+ tokens \([\d,]+ in, [\d,]+ out\)$", model["detail"]), model["detail"]
    tokens = first["meta"]["tokens"]
    assert tokens["estimated"] is True and tokens["input"] > 0 and tokens["output"] > 0 and tokens["saved"] == 0
    assert first["meta"]["model_calls"] == 1 and first["meta"]["answer_ms"] > 0
    background.drain()
    again = dict(_events(client.post(f"/api/v1/assistant/threads/{tid}/messages", json={"message": q})))["answer"]
    assert again["cached"] and again["meta"]["cached"] and again["meta"]["model_calls"] == 0
    (hit,) = [s for s in again["meta"]["trace"] if s["label"] == "Narrative cache"]
    saved = int(re.search(r"about ([\d,]+) tokens saved", hit["detail"]).group(1).replace(",", ""))
    assert hit["status"] == "hit" and saved > 0 and again["meta"]["tokens"]["saved"] == saved
    assert not [s for s in again["meta"]["trace"] if s["kind"] == "llm"]


def test_the_plan_cap_is_60000_characters_and_a_cut_plan_says_so(client, monkeypatch):
    from banko_ai.utils import aggregations
    tid = client.post("/api/v1/assistant/threads", json={}).get_json()["thread_id"]
    monkeypatch.setattr(aggregations, "explain_aggregation", lambda *a, **k: "• group\n" + "│ x\n" * 25000)
    long = dict(_events(client.post(f"/api/v1/assistant/threads/{tid}/messages",
                                    json={"message": "how much did I spend on restaurants in the last 43 days"})))
    assert len(long["answer"]["meta"]["explain"]) == 60000 and long["answer"]["meta"]["explain_cut"] is True
    monkeypatch.setattr(aggregations, "explain_aggregation", lambda *a, **k: "• group\n  regions: us-east-1\n")
    short = dict(_events(client.post(f"/api/v1/assistant/threads/{tid}/messages",
                                     json={"message": "how much did I spend on groceries in the last 43 days"})))
    assert short["answer"]["meta"]["explain"].startswith("• group") and "explain_cut" not in short["answer"]["meta"]


def test_a_failed_turn_keeps_its_trace_and_names_nothing_internal(client, monkeypatch):
    class Down:
        def complete(self, messages, *, temperature=0.3):
            raise RuntimeError("Cannot reach http://localhost:11434 secret-token")

        def stream(self, messages, *, temperature=0.3):
            raise RuntimeError("Cannot reach http://localhost:11434 secret-token")
            yield ""

    monkeypatch.setattr(assistant_api, "_chat", lambda: Down())
    tid = client.post("/api/v1/assistant/threads", json={}).get_json()["thread_id"]
    resp = client.post(f"/api/v1/assistant/threads/{tid}/messages",
                       json={"message": "show me my coffee purchases please"})
    body = resp.get_data(as_text=True)
    for leak in ("secret-token", "11434", "Cannot reach", "localhost"):
        assert leak not in body, leak
    failed = dict(_events(resp))["error"]
    labels = [s["label"] for s in failed["trace"]]
    assert labels[-2:] == ["Answer stopped", "Total request time"]
    assert any(s["kind"] == "llm" and s["status"] == "warning" for s in failed["trace"])


def test_voice_controls_are_feature_detected():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "banko_ai"
    js = (root / "static" / "js" / "assistant.js").read_text()
    chat = (root / "templates" / "components" / "assistant_chat.html").read_text()
    assert "SpeechRecognition" in js and "webkitSpeechRecognition" in js and "speechSynthesis" in js
    assert 'x-show="voiceSupported"' in chat
