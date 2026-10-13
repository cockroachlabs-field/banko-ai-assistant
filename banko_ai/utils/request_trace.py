"""What the server did to answer one chat request.

The DB Ops panel shows these steps, so every entry has to be something that
really happened, with its real timing. Flask starts a trace per /banko
request; code deeper down (caches, vector search) calls trace_step() and it
lands on that request's list. Background threads never have a trace, so the
same calls are no-ops for the coach, the warm-up and the sample data job.

Steps are short strings written by our own code. Never pass exception text,
connection URLs or the real provider name under a persona into a step.

Finished traces, with the EXPLAIN plan of the answer each one produced, live
in a small in-process LRU keyed by a short id. The chat history sits in
Flask's signed cookie (4 KB) and one semantic search plan runs to 20 KB, so
the cookie keeps only the id.
"""

from __future__ import annotations

import contextvars
import re
import secrets
import threading
import time
from collections import OrderedDict
from typing import Any

STATUSES = ("info", "hit", "miss", "success", "warning", "error")
MAX_STORED_TRACES = 200
_MAX_TEXT = 200
_MAX_STEPS = 60


class _Trace:
    __slots__ = ("steps", "t0")

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self.t0 = time.perf_counter()


_trace: contextvars.ContextVar[_Trace | None] = contextvars.ContextVar("banko_request_trace", default=None)


def start_trace() -> contextvars.Token:
    return _trace.set(_Trace())


def reset_trace(token: contextvars.Token) -> None:
    _trace.reset(token)


def _clip(value: str) -> str:
    value = str(value or "")
    return value if len(value) <= _MAX_TEXT else value[: _MAX_TEXT - 1] + "…"


def _step(trace: _Trace, kind: str, label: str, detail: str, ms: float | None,
          status: str, t: float | None) -> dict[str, Any]:
    return {
        "kind": kind,
        "label": _clip(label),
        "detail": _clip(detail),
        "ms": round(float(ms), 1) if ms is not None else None,
        "status": status if status in STATUSES else "info",
        # When the step finished, in ms since the request started.
        "t": round(t if t is not None else (time.perf_counter() - trace.t0) * 1000, 1),
    }


def trace_step(kind: str, label: str, detail: str = "", ms: float | None = None,
               status: str = "info") -> None:
    trace = _trace.get()
    if trace is None or len(trace.steps) >= _MAX_STEPS:
        return
    trace.steps.append(_step(trace, kind, label, detail, ms, status, None))


def insert_step(index: int, kind: str, label: str, detail: str = "", ms: float | None = None,
                status: str = "info", *, t: float | None = None) -> None:
    """Place a step that was measured from outside at the point it really
    happened, e.g. the model call a provider made between its cache lookup
    and its cache write."""
    trace = _trace.get()
    if trace is None or len(trace.steps) >= _MAX_STEPS:
        return
    trace.steps.insert(index, _step(trace, kind, label, detail, ms, status, t))


def current_trace() -> list[dict[str, Any]]:
    trace = _trace.get()
    return list(trace.steps) if trace is not None else []


def trace_elapsed_ms() -> float | None:
    trace = _trace.get()
    return (time.perf_counter() - trace.t0) * 1000 if trace is not None else None


_store: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
_plans: dict[str, str] = {}
_store_lock = threading.Lock()


def store_trace(steps: list[dict[str, Any]], plan: str | None = None) -> str:
    trace_id = secrets.token_urlsafe(8)
    with _store_lock:
        _store[trace_id] = list(steps)
        if plan:
            _plans[trace_id] = plan
        while len(_store) > MAX_STORED_TRACES:
            evicted, _ = _store.popitem(last=False)
            _plans.pop(evicted, None)
    return trace_id


def get_stored_plan(trace_id: str | None) -> str | None:
    """The EXPLAIN plan stored with a trace, while the trace is kept."""
    if not trace_id:
        return None
    with _store_lock:
        return _plans.get(trace_id)


def get_stored_trace(trace_id: str | None) -> list[dict[str, Any]] | None:
    if not trace_id:
        return None
    with _store_lock:
        steps = _store.get(trace_id)
        if steps is not None:
            _store.move_to_end(trace_id)
        return steps


def serving_regions(explain_text: str | None) -> str | None:
    """The region(s) that actually served a query, parsed from its EXPLAIN
    ANALYZE output. The user's home region is where their rows live, but
    leaseholders fail over, and the plan names the survivors."""
    if not explain_text:
        return None
    found: list[str] = []
    for m in re.finditer(r"regions: ([a-z0-9, -]+)", explain_text):
        for r in m.group(1).split(","):
            r = r.strip()
            if r and r not in found:
                found.append(r)
    return ", ".join(sorted(found)) if found else None


def plan_indexes(explain_text: str | None) -> list[str]:
    """Index names an EXPLAIN plan scanned (the part after table@)."""
    if not explain_text:
        return []
    found: list[str] = []
    for m in re.finditer(r"table: \w+@(\w+)", explain_text):
        if m.group(1) not in found:
            found.append(m.group(1))
    return found
