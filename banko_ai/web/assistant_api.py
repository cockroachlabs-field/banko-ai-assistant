"""The assistant over HTTP: threads, streamed answers (Server-Sent Events),
receipts, confirmable actions and memories. Every route acts for the
signed-in user, or the default persona without a session. Streams always
end with a done event; failures are one neutral error event."""

from __future__ import annotations

import json
import logging
import os
import queue
import tempfile
import threading
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from flask import Blueprint, Response, current_app, jsonify, request, session, stream_with_context

from ..ai_providers.display import NEUTRAL_FAILED
from ..ai_providers.streaming import ProviderChat
from ..assistant import checkpoint
from ..assistant import graph as G
from ..config.settings import get_config
from ..db.regions import agent_home_region
from ..services import actions, memories, receipts, threads
from ..services._sql import jsonable, valid_uuid
from ..services.ledger import NotFound
from ..utils.embeddings import ModelWarmingUp
from ..utils.request_trace import reset_trace, start_trace

assistant_api = Blueprint("assistant_api", __name__, url_prefix="/api/v1")
log = logging.getLogger("banko.assistant")
WARMING = "I'm still warming up. Try that again in a few seconds."
RECONNECTING = "Reconnecting to the database, try that again in a moment."
_AGENT_FOR_STEP = {"fraud": "fraud", "budget": "budget"}
_RECEIPT_KEYS = ("document_id", "merchant", "amount", "date", "category", "items", "expense_id")


def _uid() -> str:
    return session.get("user_id") or get_config().coach_default_user_id


def _json() -> dict:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _chat():
    current = current_app.extensions.get("banko_ai", {}).get("current_model")
    return ProviderChat(model=current() if current else None)


def _activity(agent_type: str, status: str, message: str) -> None:
    """One line on the Agents page. The page shows "agent (region)", so the
    region rides along the way every other agent_activity event sends it."""
    socketio = current_app.extensions.get("socketio")
    if socketio is None:
        return
    try:
        socketio.emit("agent_activity", {"agent_type": agent_type,
                                         "region": agent_home_region(get_config().database_url),
                                         "status": status, "message": message,
                                         "timestamp": datetime.now().isoformat()})
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
        G.set_context(thread["thread_id"], context)
    return jsonify(thread), 201


@assistant_api.get("/assistant/threads/<thread_id>")
def get_thread(thread_id: str):
    thread = _owned(thread_id)
    if thread is None:
        return jsonify({"error": "not_found"}), 404
    state = G.thread_state(thread_id)
    return jsonify({**thread, "messages": state.get("messages") or [],
                    "active_receipt": state.get("active_receipt")})


@assistant_api.patch("/assistant/threads/<thread_id>")
def rename_thread(thread_id: str):
    thread = threads.rename_thread(_uid(), thread_id, str(_json().get("title") or ""))
    return (jsonify(thread), 200) if thread else (jsonify({"error": "not_found"}), 404)


@assistant_api.delete("/assistant/threads/<thread_id>")
def delete_thread(thread_id: str):
    if not threads.delete_thread(_uid(), thread_id):
        return jsonify({"error": "not_found"}), 404
    checkpoint.delete_checkpoints(thread_id)
    return "", 204


def _turn_stream(user_id: str, thread_id: str, message: str, language: str | None) -> Iterator[str]:
    """Run the turn on a worker thread and relay its events as they happen."""
    events: queue.Queue = queue.Queue()
    chat = _chat()
    done = object()

    def work() -> None:
        token = start_trace()
        try:
            out = G.run_turn(user_id, thread_id, message, language_code=language, chat=chat,
                             on_event=events.put)
            meta = {k: v for k, v in (out.get("meta") or {}).items() if k != "explain"}
            # Shaped like the proposal events, which drop their "type" key on the way out.
            proposals = [{k: v for k, v in p.items() if k != "type"} for p in out["proposals"]]
            events.put({"type": "answer", "answer": out["answer"], "citations": out["citations"],
                        "proposals": proposals, "cached": out["cached"],
                        "route": (out.get("route") or {}).get("kind"), "meta": meta})
        except Exception as exc:
            log.warning("assistant turn failed", exc_info=True)
            events.put({"type": "error", "message": _friendly(exc)})
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
    _activity("assistant", "processing", "Thinking about a question")

    def relay() -> Iterator[str]:
        for chunk in _turn_stream(user_id, thread_id, message, language):
            if chunk.startswith("event: answer"):
                _activity("assistant", "completed", "Answered a question")
            elif chunk.startswith("event: error"):
                _activity("assistant", "failed", "Could not answer a question")
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
        G.set_active_receipt(thread_id, {k: receipt.get(k) for k in _RECEIPT_KEYS})
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
                              f"{event['step']}: {event['detail']}")
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
