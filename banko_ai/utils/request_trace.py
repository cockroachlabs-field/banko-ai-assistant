"""What the server did to answer one assistant turn.

The "How this was answered" panel shows these steps, so every entry has to be
something that really happened, with its real timing. The assistant API starts
a trace per turn; code deeper down (caches, vector search, tools) calls
trace_step() and it lands on that turn's list, which goes back to the browser
with the answer. Background threads never have a trace, so the same calls are
no-ops for the coach, the warm-up and the sample data job.

Steps are short strings written by our own code. Never pass exception text,
connection URLs or the real provider name under a persona into a step.
"""

from __future__ import annotations

import contextvars
import re
import time
from typing import Any

STATUSES = ("info", "hit", "miss", "success", "warning", "error")
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
