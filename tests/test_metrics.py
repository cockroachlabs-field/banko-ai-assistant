"""In-process metrics: per-second buckets, outcomes, and latency percentiles."""

import threading

import pytest

from banko_ai.observability.metrics import Metrics


def test_percentiles_land_within_bucket_resolution():
    m = Metrics(clock=lambda: 1000.0)
    for ms in range(1, 1001):
        m.record("load", ms=float(ms), at=1000.0)
    s = m.summary("load", window_s=10, now=1000.5)
    assert abs(s["p50"] - 500) / 500 < 0.10 and abs(s["p99"] - 990) / 990 < 0.10
    assert s["ok"] == 1000 and s["ops_s"] == pytest.approx(100.0)


def test_outcomes_classes_and_windows():
    m = Metrics(clock=lambda: 0)
    m.record("load", ms=5, at=100)
    m.record("load", ms=9, outcome="error", error_class="40001", at=101)
    m.record("load", outcome="dropped", at=101)
    m.record("load", ms=7, outcome="declined", retries=2, at=50)  # outside the window
    s = m.summary("load", window_s=10, now=101.5)
    assert (s["ok"], s["errors"], s["dropped"], s["declined"], s["retries"]) == (1, 1, 1, 0, 0)
    assert s["error_classes"] == {"40001": 1}


def test_an_empty_window_has_no_percentiles():
    s = Metrics(clock=lambda: 0).summary("nothing", window_s=10, now=5)
    assert s["ok"] == 0 and s["ops_s"] == 0 and s["p50"] is None and s["p99"] is None


def test_old_buckets_fall_out_and_timeline_has_one_slot_per_second():
    m = Metrics(retention_s=60, clock=lambda: 0)
    m.record("x", ms=1, at=0)
    m.record("x", ms=1, at=100)
    t = m.timeline("x", seconds=5, now=100.2)
    assert t["t"] == [96, 97, 98, 99, 100] and t["ops"] == [0, 0, 0, 0, 1]
    assert t["p50"][:4] == [None] * 4 and t["p50"][-1] is not None
    assert m.summary("x", window_s=200, now=100.2)["ok"] == 1  # the record at 0 is gone


def test_unknown_outcomes_are_rejected():
    with pytest.raises(ValueError):
        Metrics().record("x", outcome="maybe")


def test_names_and_clear_work_by_prefix():
    m = Metrics(clock=lambda: 10)
    for name in ("load", "load.op.purchase", "http"):
        m.record(name, ms=1)
    assert m.names("load") == ["load", "load.op.purchase"]
    m.clear("load")
    assert m.names("") == ["http"]


def test_concurrent_records_all_count():
    m = Metrics()
    threads = [threading.Thread(target=lambda: [m.record("c", ms=1) for _ in range(1000)]) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert m.summary("c", window_s=60)["ok"] == 8000
