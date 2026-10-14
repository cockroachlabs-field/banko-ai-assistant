"""The assistant over HTTP: threads, streamed answers (Server-Sent Events),
receipts, confirmable actions and memories. Every route acts for the
signed-in user, or the default persona without a session. Streams always
end with a done event; failures are one neutral error event."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, Response, current_app, jsonify, request, session, stream_with_context

from ..ai_providers.display import NEUTRAL_FAILED
from ..ai_providers.streaming import ProviderChat, TimedChat
from ..config.settings import get_config
from ..db.regions import agent_home_region
from ..services import actions, memories, receipts, threads
from ..services._sql import jsonable, valid_uuid
from ..services.ledger import NotFound
from ..utils.embeddings import ModelWarmingUp
from ..utils.request_trace import current_trace, reset_trace, start_trace, trace_elapsed_ms, trace_step

assistant_api = Blueprint("assistant_api", __name__, url_prefix="/api/v1")
log = logging.getLogger("banko.assistant")
WARMING = "I'm still warming up. Try that again in a few seconds."
RECONNECTING = "Reconnecting to the database, try that again in a moment."
_AGENT_FOR_STEP = {"fraud": "fraud", "budget": "budget"}
# A semantic search plan carries its query vector twice (about 8,000
# characters each), and a multi-region plan names every region it touched.
EXPLAIN_CAP = 60_000
_RECEIPT_KEYS = ("document_id", "merchant", "amount", "date", "category", "items", "expense_id")


def _graph():
    """The assistant graph module, imported on first use. LangGraph costs
    about half a second, and create_app registers this blueprint before the
    server listens. Tests patch its _graph attribute as before."""
    from ..assistant import graph
    return graph


def _uid() -> str:
    return session.get("user_id") or get_config().coach_default_user_id


@assistant_api.after_request
def _name_the_user(response):
    response.headers["X-Banko-User"] = _uid()
    return response


def _json() -> dict:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _chat():
    current = current_app.extensions.get("banko_ai", {}).get("current_model")
    return ProviderChat(model=current() if current else None)


def user_room(user_id: str) -> str:
    """The Socket.IO room of one user's browsers. coach.join puts a socket in
    its signed-in user's room, so per-user events never reach anyone else."""
    return f"{get_config().coach_socketio_room_prefix}{user_id}"


def _activity(agent_type: str, status: str, message: str, user_id: str) -> None:
    """One line on the user's Agents page. Receipt steps name merchants and
    amounts, so it goes to that user's room only. The page shows "agent
    (region)", so the region rides along."""
    socketio = current_app.extensions.get("socketio")
    if socketio is None:
        return
    try:
        socketio.emit("agent_activity", {"source": "activity", "agent_type": agent_type,
                                         "region": agent_home_region(get_config().database_url),
                                         "status": status, "message": message, "user_id": user_id,
                                         "timestamp": datetime.now(timezone.utc).isoformat()},
                      to=user_room(user_id))
    except Exception:
        log.debug("agent activity not sent", exc_info=True)


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(jsonable(data), default=str)}\n\n"


def _friendly(exc: Exception) -> str:
    from ..db.engine import is_retryable
    if isinstance(exc, ModelWarmingUp):
        return WARMING
    if is_retryable(exc, idempotent=True):
        return RECONNECTING
    return NEUTRAL_FAILED


def _owned(thread_id: str) -> dict | None:
    return threads.get_thread(_uid(), thread_id)


class AnswerMeta:
    """The meta of the answers this process sent (trace, plan, tokens),
    keyed by thread and answer text, so a reloaded thread still shows how
    each answer was made. Bounded and in memory, as the old chat page kept
    its traces: a restart forgets them and those answers reload without."""

    def __init__(self, size: int = 300):
        self.size = size
        self._items: OrderedDict[tuple[str, str], dict] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(thread_id: str, answer: str) -> tuple[str, str]:
        # The checkpoint keeps the answer without citation markers; key on that.
        from ..assistant.answers import strip_citations
        return str(thread_id), hashlib.sha256(strip_citations(answer).encode()).hexdigest()

    def put(self, thread_id: str, answer: str, meta: dict) -> None:
        key = self._key(thread_id, answer)
        with self._lock:
            self._items[key] = meta
            self._items.move_to_end(key)
            while len(self._items) > self.size:
                self._items.popitem(last=False)

    def get(self, thread_id: str, answer: str) -> dict | None:
        key = self._key(thread_id, answer)
        with self._lock:
            meta = self._items.get(key)
            if meta is not None:
                self._items.move_to_end(key)
            return meta

    def forget_thread(self, thread_id: str) -> None:
        with self._lock:
            for key in [k for k in self._items if k[0] == str(thread_id)]:
                del self._items[key]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


answer_meta = AnswerMeta()


# ── threads ──────────────────────────────────────────────────────────────

@assistant_api.get("/assistant/threads")
def list_threads():
    return jsonify({"threads": threads.list_threads(_uid())})


@assistant_api.post("/assistant/threads")
def create_thread():
    body = _json()
    context = body.get("context")
    if context is not None:
        if (not isinstance(context, dict) or context.get("kind") not in ("nudge", "transaction")
                or not valid_uuid(context.get("id"))):
            return jsonify({"error": "context must be a nudge or a transaction id"}), 400
        context = {"kind": context["kind"], "id": str(context["id"])}
    thread = threads.create_thread(_uid(), title=str(body.get("title") or threads.DEFAULT_TITLE))
    if context:
        _graph().set_context(thread["thread_id"], context)
    return jsonify(thread), 201


@assistant_api.get("/assistant/threads/<thread_id>")
def get_thread(thread_id: str):
    thread = _owned(thread_id)
    if thread is None:
        return jsonify({"error": "not_found"}), 404
    state = _graph().thread_state(thread_id)
    messages = []
    for m in state.get("messages") or []:
        meta = answer_meta.get(thread_id, str(m.get("content") or "")) if m.get("role") == "assistant" else None
        messages.append({**m, "meta": meta} if meta else m)
    return jsonify({**thread, "messages": messages, "active_receipt": state.get("active_receipt")})


@assistant_api.patch("/assistant/threads/<thread_id>")
def rename_thread(thread_id: str):
    thread = threads.rename_thread(_uid(), thread_id, str(_json().get("title") or ""))
    return (jsonify(thread), 200) if thread else (jsonify({"error": "not_found"}), 404)


@assistant_api.delete("/assistant/threads/<thread_id>")
def delete_thread(thread_id: str):
    if not threads.delete_thread(_uid(), thread_id):
        return jsonify({"error": "not_found"}), 404
    from ..assistant import checkpoint
    checkpoint.delete_checkpoints(thread_id)
    answer_meta.forget_thread(thread_id)
    return "", 204


def _tokens_text(chat: TimedChat) -> str:
    """", 2,950 tokens (2,600 in, 350 out)", with "about" when any call's
    count was estimated; nothing when the model spent nothing."""
    total = chat.input_tokens + chat.output_tokens
    if not total:
        return ""
    about = "about " if chat.estimated else ""
    return f", {about}{total:,} tokens ({chat.input_tokens:,} in, {chat.output_tokens:,} out)"


def _close_trace(chat: TimedChat, shown, *, stopped: bool = False) -> float | None:
    """End the turn's trace the way the old chat page did: the model step,
    only when the model really ran, then the total. Returns the total in ms."""
    if chat.calls:
        detail = f"{shown.name} · {shown.model}, {chat.calls} call{'s' if chat.calls > 1 else ''}"
        trace_step("llm", "Model", detail + _tokens_text(chat) + (f", {chat.failed} failed" if chat.failed else ""),
                   ms=chat.ms, status="warning" if chat.failed else "success")
    elapsed = trace_elapsed_ms()
    if stopped:
        trace_step("error", "Answer stopped", "The answer failed here; details are in the server log",
                   status="error")
        trace_step("total", "Total request time", ms=elapsed)
    else:
        trace_step("total", "Time to answer", ms=elapsed)
    return elapsed


def _answer_event(thread_id: str, final: dict, chat: TimedChat, shown) -> dict:
    """The SSE answer event from the graph's final event: the trace closes
    here, with the model step and the time to answer. Its meta is kept for
    the thread's next reload."""
    elapsed = _close_trace(chat, shown)
    meta = dict(final.get("meta") or {})
    if meta.get("explain"):
        plan = str(meta["explain"])
        meta["explain"] = plan[:EXPLAIN_CAP]
        if len(plan) > EXPLAIN_CAP:
            meta["explain_cut"] = True  # the block under the answer says "plan shortened"
    meta["trace"] = current_trace()
    meta["tokens"] = {"input": chat.input_tokens, "output": chat.output_tokens, "estimated": chat.estimated,
                      "saved": int(final.get("saved_tokens") or 0)}
    meta["model_calls"] = chat.calls
    meta["answer_ms"] = None if elapsed is None else round(elapsed)
    meta["cached"] = bool(final.get("cached"))
    answer = final.get("answer") or ""
    answer_meta.put(thread_id, answer, jsonable(meta))
    # Shaped like the proposal events, which drop their "type" key on the way out.
    proposals = [{k: v for k, v in p.items() if k != "type"} for p in final.get("proposals") or []]
    return {"type": "answer", "answer": answer, "citations": final.get("citations") or [],
            "proposals": proposals, "cached": bool(final.get("cached")),
            "route": (final.get("route") or {}).get("kind"), "meta": meta}


def _turn_stream(user_id: str, thread_id: str, message: str, language: str | None) -> Iterator[str]:
    """Run the turn on a worker thread and relay its events as they happen.
    The answer goes out the moment respond is done; learn and the checkpoint
    write finish after it, and done closes the stream."""
    events: queue.Queue = queue.Queue()
    chat = TimedChat(_chat())
    # The persona as shown, read here: the worker thread has no app context.
    shown = current_app.extensions["banko_ai"]["shown"]()
    done = object()

    def work() -> None:
        token = start_trace()
        answered = False

        def relay(event: dict) -> None:
            nonlocal answered
            if event.get("type") != "final":
                events.put(event)
            elif not answered:  # one answer per turn, never two
                answer = _answer_event(thread_id, event, chat, shown)
                answered = True
                events.put(answer)

        try:
            out = _graph().run_turn(user_id, thread_id, message, language_code=language, chat=chat,
                                    on_event=relay)
            relay({"type": "final", **out})  # a no-op once respond's final event went out
        except Exception as exc:
            if answered:
                # The user has the answer; only the bookkeeping after it failed.
                log.warning("assistant turn failed after its answer went out", exc_info=True)
            else:
                log.warning("assistant turn failed", exc_info=True)
                _close_trace(chat, shown, stopped=True)
                events.put({"type": "error", "message": _friendly(exc), "trace": current_trace()})
        finally:
            reset_trace(token)
            events.put(done)

    threading.Thread(target=work, name="banko-assistant-turn", daemon=True).start()
    while True:
        item = events.get()
        if item is done:
            break
        # A copy: a proposal event is the same dict the graph keeps in its state.
        payload = dict(item)
        yield sse(payload.pop("type"), payload)
    yield sse("done", {})


@assistant_api.post("/assistant/threads/<thread_id>/messages")
def post_message(thread_id: str):
    if _owned(thread_id) is None:
        return jsonify({"error": "not_found"}), 404
    body = _json()
    message = str(body.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Type a message"}), 400
    user_id = _uid()
    language = body.get("language") if isinstance(body.get("language"), str) else None
    _activity("assistant", "processing", "Thinking about a question", user_id)

    def relay() -> Iterator[str]:
        for chunk in _turn_stream(user_id, thread_id, message, language):
            if chunk.startswith("event: answer"):
                _activity("assistant", "completed", "Answered a question", user_id)
            elif chunk.startswith("event: error"):
                _activity("assistant", "failed", "Could not answer a question", user_id)
            yield chunk

    return Response(stream_with_context(relay()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── actions and memories ─────────────────────────────────────────────────

@assistant_api.post("/actions/<action_id>/confirm")
def confirm_action(action_id: str):
    try:
        return jsonify(actions.confirm(_uid(), action_id))
    except NotFound:
        return jsonify({"error": "not_found"}), 404
    except Exception:
        log.warning("action confirm failed", exc_info=True)
        return jsonify({"error": NEUTRAL_FAILED}), 400


@assistant_api.post("/actions/<action_id>/cancel")
def cancel_action(action_id: str):
    try:
        return jsonify(actions.cancel(_uid(), action_id))
    except NotFound:
        return jsonify({"error": "not_found"}), 404


@assistant_api.get("/memories")
def list_memories():
    return jsonify({"memories": memories.list_memories(_uid())})


@assistant_api.delete("/memories/<memory_id>")
def delete_memory(memory_id: str):
    return ("", 204) if memories.forget(_uid(), memory_id) else (jsonify({"error": "not_found"}), 404)


# ── receipts ─────────────────────────────────────────────────────────────

def _attach_receipt(user_id: str, thread_id: str | None, receipt: dict) -> str | None:
    """Make the receipt the active receipt of its thread (a new one unless
    the upload named one). The purchase has already posted by now, so a
    failure here is logged and the receipt still goes out, without a thread."""
    try:
        if not thread_id:
            thread_id = threads.create_thread(
                user_id, title=f"Receipt from {receipt.get('merchant') or 'a store'}")["thread_id"]
        _graph().set_active_receipt(thread_id, {k: receipt.get(k) for k in _RECEIPT_KEYS})
        return thread_id
    except Exception:
        log.warning("receipt %s posted but not attached to a thread", receipt.get("document_id"), exc_info=True)
        return None


@assistant_api.post("/receipts")
def upload_receipt():
    upload = request.files.get("receipt")
    if upload is None or not upload.filename:
        return jsonify({"error": "Choose a receipt image or PDF"}), 400
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in receipts.ALLOWED_TYPES:
        return jsonify({"error": "Upload a receipt image or PDF"}), 400
    user_id = _uid()
    thread_id = request.form.get("thread_id")
    if thread_id and _owned(thread_id) is None:
        return jsonify({"error": "not_found"}), 404
    # Hooks first: they can raise while the embedding model warms up, and
    # then no temp file is left behind.
    current = current_app.extensions.get("banko_ai", {}).get("current_model")
    hooks = receipts.default_hooks(current() if current else None)
    handle = tempfile.NamedTemporaryFile(prefix="banko_receipt_", suffix=suffix, delete=False)
    handle.close()
    upload.save(handle.name)
    filename = upload.filename

    def run() -> Iterator[str]:
        try:
            for event in receipts.process_receipt(user_id, handle.name, filename, hooks=hooks):
                if event["type"] == "step" and event["status"] != "start":
                    _activity(_AGENT_FOR_STEP.get(event["step"], "receipt"),
                              "completed" if event["status"] == "done" else event["status"],
                              f"{event['step']}: {event['detail']}", user_id)
                if event["type"] == "receipt":
                    event = {**event, "thread_id": _attach_receipt(user_id, thread_id, event)}
                kind = event.pop("type")
                yield sse(kind, event)
        except Exception as exc:
            log.warning("receipt upload failed", exc_info=True)
            yield sse("error", {"message": _friendly(exc)})
        finally:
            try:
                os.remove(handle.name)
            except OSError:
                pass
        yield sse("done", {})

    return Response(stream_with_context(run()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@assistant_api.get("/receipts/<document_id>/thumbnail")
def receipt_thumbnail(document_id: str):
    data = receipts.thumbnail(_uid(), document_id)
    if data is None:
        return jsonify({"error": "not_found"}), 404
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})
