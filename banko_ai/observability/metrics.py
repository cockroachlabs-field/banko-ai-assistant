"""Per-second metrics with latency percentiles, in memory.

Each series keeps one bucket per second for `retention_s` seconds: counts by
outcome, client retries, error classes, and a log-scale latency histogram
from 0.1 ms to 120 s in steps of 9 percent, so a percentile over any window
is accurate to about 4 percent. Only ok and declined operations carry
latency: an error's time says more about timeouts than about the system.
"""

from __future__ import annotations

import math
import threading
import time
from array import array
from collections.abc import Callable

OUTCOMES = ("ok", "error", "declined", "dropped")

_FLOOR_MS = 0.1
_CEILING_MS = 120_000.0
_RATIO = 1.09
_LOG_RATIO = math.log(_RATIO)
_BUCKETS = math.ceil(math.log(_CEILING_MS / _FLOOR_MS) / _LOG_RATIO) + 1


def _bucket(ms: float) -> int:
    if ms <= _FLOOR_MS:
        return 0
    return min(_BUCKETS - 1, int(math.log(ms / _FLOOR_MS) / _LOG_RATIO))


def _bucket_ms(index: int) -> float:
    """The geometric middle of a bucket."""
    return _FLOOR_MS * _RATIO ** (index + 0.5)


class _Second:
    __slots__ = ("counts", "retries", "classes", "hist", "samples")

    def __init__(self) -> None:
        self.counts = dict.fromkeys(OUTCOMES, 0)
        self.retries = 0
        self.classes: dict[str, int] = {}
        self.hist: array | None = None
        self.samples = 0


def _percentiles(hists: list[array], total: int) -> tuple[float | None, float | None, float | None]:
    if total == 0:
        return None, None, None
    merged = [0] * _BUCKETS
    for h in hists:
        for i, n in enumerate(h):
            if n:
                merged[i] += n
    out = []
    for p in (0.50, 0.95, 0.99):
        rank = max(1, math.ceil(p * total))
        seen = 0
        for i, n in enumerate(merged):
            seen += n
            if seen >= rank:
                out.append(round(_bucket_ms(i), 2))
                break
    return out[0], out[1], out[2]


class Metrics:
    def __init__(self, retention_s: int = 900, clock: Callable[[], float] = time.time):
        self.retention_s = retention_s
        self.clock = clock
        self._series: dict[str, dict[int, _Second]] = {}
        self._lock = threading.Lock()

    def record(self, series: str, *, ms: float | None = None, outcome: str = "ok",
               error_class: str | None = None, retries: int = 0, at: float | None = None) -> None:
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}")
        second = int(self.clock() if at is None else at)
        with self._lock:
            buckets = self._series.setdefault(series, {})
            bucket = buckets.get(second)
            if bucket is None:
                bucket = buckets[second] = _Second()
                if len(buckets) > self.retention_s + 60:
                    cutoff = second - self.retention_s
                    for old in [s for s in buckets if s < cutoff]:
                        del buckets[old]
            bucket.counts[outcome] += 1
            bucket.retries += retries
            if error_class:
                bucket.classes[error_class] = bucket.classes.get(error_class, 0) + 1
            if ms is not None and outcome in ("ok", "declined"):
                if bucket.hist is None:
                    bucket.hist = array("I", bytes(4 * _BUCKETS))
                bucket.hist[_bucket(ms)] += 1
                bucket.samples += 1

    def _window(self, series: str, seconds: int, now: float | None) -> tuple[int, list[_Second | None]]:
        end = int(self.clock() if now is None else now)
        start = end - seconds + 1
        floor = end - self.retention_s
        buckets = self._series.get(series, {})
        return start, [buckets.get(s) if s > floor else None for s in range(start, end + 1)]

    def summary(self, series: str, *, window_s: int = 10, now: float | None = None) -> dict:
        with self._lock:
            _, window = self._window(series, window_s, now)
            totals = dict.fromkeys(OUTCOMES, 0)
            retries, classes, hists, samples = 0, {}, [], 0
            for b in window:
                if b is None:
                    continue
                for k in OUTCOMES:
                    totals[k] += b.counts[k]
                retries += b.retries
                for c, n in b.classes.items():
                    classes[c] = classes.get(c, 0) + n
                if b.hist is not None:
                    hists.append(b.hist)
                    samples += b.samples
            p50, p95, p99 = _percentiles(hists, samples)
        return {"ops_s": round((totals["ok"] + totals["declined"]) / window_s, 2),
                "errors_s": round(totals["error"] / window_s, 2),
                "ok": totals["ok"], "errors": totals["error"], "declined": totals["declined"],
                "dropped": totals["dropped"], "retries": retries,
                "p50": p50, "p95": p95, "p99": p99, "error_classes": classes}

    def timeline(self, series: str, *, seconds: int = 300, now: float | None = None) -> dict:
        with self._lock:
            start, window = self._window(series, seconds, now)
            out: dict[str, list] = {"t": [], "ops": [], "errors": [], "dropped": [], "p50": [], "p95": [], "p99": []}
            for i, b in enumerate(window):
                out["t"].append(start + i)
                if b is None:
                    values: tuple = (0, 0, 0, None, None, None)
                else:
                    values = (b.counts["ok"] + b.counts["declined"], b.counts["error"], b.counts["dropped"],
                              *_percentiles([b.hist] if b.hist is not None else [], b.samples))
                for key, value in zip(("ops", "errors", "dropped", "p50", "p95", "p99"), values):
                    out[key].append(value)
        return out

    def names(self, prefix: str) -> list[str]:
        with self._lock:
            return sorted(n for n in self._series if n.startswith(prefix))

    def clear(self, prefix: str) -> None:
        with self._lock:
            for name in [n for n in self._series if n.startswith(prefix)]:
                del self._series[name]


metrics = Metrics()
