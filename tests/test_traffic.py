"""The traffic simulator: open-loop pacing, honest latency under a slow
cluster, error classes, one run at a time, a stop that never hangs, live
changes, and a real run with no errors and no drift."""

import os
import threading
import time

import psycopg
import pytest
from sqlalchemy.exc import OperationalError

from banko_ai.observability.events import EventBus
from banko_ai.observability.metrics import Metrics
from banko_ai.services.load_users import LoadUser
from banko_ai.workers.traffic import LoadConfig, LoadConfigError, SimulatorBusy, TrafficSimulator

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")

USERS = [LoadUser(f"u{i}", f"load-{i:05d}", "saver", r, {})
         for i, r in enumerate(["us-east-1", "us-west-2", "eu-west-1"] * 2)]


def _sim(runner, prepare=None):
    m, ev = Metrics(), EventBus()
    sim = TrafficSimulator(metrics=m, events=ev, runner=runner, prepare=prepare or (lambda n: USERS[:n]),
                           sampler=None, integrity=lambda: None)
    return sim, m, ev


def _cfg(**kw) -> LoadConfig:
    return LoadConfig.from_dict({"users": 3, "rate": 50, "workers": 4, "ramp_s": 0, "duration_s": 3, **kw})


def _wait(sim, state, timeout):
    deadline = time.monotonic() + timeout
    while sim.status()["state"] != state:
        assert time.monotonic() < deadline, sim.status()
        time.sleep(0.02)


def _done(m: Metrics, window_s: int = 30) -> dict:
    """The run's numbers, its last (partial) second included."""
    return m.summary("load", window_s=window_s, now=time.time() + 1)


def _instant(op, user, rng):
    return "ok", 0


def test_config_validates_against_the_caps():
    for bad in ({"rate": "abc"}, {"users": 0}, {"workers": 1000}, {"rate": 10**9}, {"users": True},
                {"profile": "chaos"}, {"profile": "custom", "weights": {"purchase": 0}},
                {"profile": "custom", "weights": {"teleport": 5}}, {"duration_s": -1}, {"ramp_s": 10**6}):
        with pytest.raises(LoadConfigError):
            LoadConfig.from_dict(bad)


def test_config_defaults_come_from_settings():
    cfg = LoadConfig.from_dict({})
    assert (cfg.users, cfg.rate, cfg.workers, cfg.profile, cfg.duration_s, cfg.ramp_s) == (100, 50, 16, "balanced", None, 10)
    assert LoadConfig.from_dict({"profile": "custom", "weights": {"purchase": 3, "search": 1}}).mix() == {
        "purchase": 3, "search": 1}


def test_open_loop_rate_and_a_clean_finish():
    sim, m, ev = _sim(_instant)
    sim.start(_cfg())
    _wait(sim, "idle", 10)
    s = _done(m)
    assert 100 <= s["ok"] <= 200 and s["errors"] == 0 and s["dropped"] == 0
    texts = [e["text"] for e in ev.since(None)]
    assert texts[0].startswith("Load started") and texts[-1].startswith("Load stopped")
    assert set(m.names("load.region.")) == {"load.region.us-east-1", "load.region.us-west-2", "load.region.eu-west-1"}


def test_a_slow_cluster_shows_as_latency_and_drops_not_less_load():
    def slow(op, user, rng):
        time.sleep(0.5)
        return "ok", 0

    sim, m, _ = _sim(slow)
    sim.start(_cfg(rate=20, workers=2, duration_s=2))
    _wait(sim, "idle", 10)
    s = _done(m)
    assert s["dropped"] > 0 and s["p50"] >= 500


def test_errors_are_classed_and_do_not_stop_the_run():
    def failing(op, user, rng):
        raise OperationalError("UPDATE", {}, psycopg.errors.SerializationFailure("restart transaction"))

    sim, m, ev = _sim(failing)
    sim.start(_cfg(rate=40, duration_s=2))
    _wait(sim, "idle", 10)
    s = _done(m)
    assert s["errors"] >= 40 and s["ok"] == 0 and s["error_classes"] == {"40001": s["errors"]}
    assert any(e["level"] == "warn" and "failed" in e["text"] for e in ev.since(None))


def test_two_starts_one_run():
    sim, _, _ = _sim(_instant)
    barrier, outcomes = threading.Barrier(2), []

    def go():
        barrier.wait()
        try:
            sim.start(_cfg(duration_s=1))
            outcomes.append("started")
        except SimulatorBusy:
            outcomes.append("busy")

    threads = [threading.Thread(target=go) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes) == ["busy", "started"]
    _wait(sim, "idle", 10)


def test_stop_returns_while_operations_hang():
    release = threading.Event()

    def hang(op, user, rng):
        release.wait(30)
        return "ok", 0

    sim, _, _ = _sim(hang)
    try:
        sim.start(_cfg(duration_s=None))
        _wait(sim, "running", 10)
        time.sleep(0.3)
        started = time.monotonic()
        assert sim.stop(timeout=2)["state"] == "idle"
        assert time.monotonic() - started < 3
    finally:
        release.set()


def test_apply_changes_the_rate_live():
    sim, m, _ = _sim(_instant)
    sim.start(_cfg(rate=10, duration_s=None))
    try:
        _wait(sim, "running", 10)
        time.sleep(1.0)
        sim.apply(LoadConfig.from_dict({"rate": 80}, base=sim.config))
        assert sim.status()["config"]["rate"] == 80
        time.sleep(2.2)
        # The last whole second: all of it at the new rate, wherever the
        # change fell inside its own second.
        assert m.summary("load", window_s=1)["ok"] >= 50
    finally:
        sim.stop()


def test_apply_needs_a_running_simulator():
    sim, _, _ = _sim(_instant)
    with pytest.raises(LoadConfigError):
        sim.apply(_cfg())


def test_a_failed_preparation_returns_to_idle_with_the_reason():
    def broken(n):
        raise RuntimeError("database unreachable")

    sim, _, ev = _sim(_instant, prepare=broken)
    sim.start(_cfg())
    _wait(sim, "idle", 10)
    assert "could not be prepared" in sim.status()["error"]
    assert ev.since(None)[-1]["level"] == "bad"


def test_a_real_run_has_no_errors_and_no_drift():
    from banko_ai.services.integrity import run_check
    from banko_ai.services.load_users import clear

    clear()
    m, ev = Metrics(), EventBus()
    sim = TrafficSimulator(metrics=m, events=ev)
    try:
        sim.start(LoadConfig.from_dict({"users": 5, "rate": 30, "workers": 4, "ramp_s": 0, "duration_s": 8}))
        _wait(sim, "idle", 120)
        s = _done(m, 60)
        assert s["ok"] + s["declined"] > 150 and s["errors"] == 0, s
        assert run_check().ok
    finally:
        clear()


def test_a_customer_never_has_two_operations_in_flight():
    """A customer is one person: while one of their operations runs, the
    next arrival goes to someone who is free, so busy customers never queue
    on their own account locks."""
    in_flight, peak, lock = {}, {}, threading.Lock()

    def runner(op, user, rng):
        with lock:
            in_flight[user.user_id] = in_flight.get(user.user_id, 0) + 1
            peak[user.user_id] = max(peak.get(user.user_id, 0), in_flight[user.user_id])
        time.sleep(0.01)
        with lock:
            in_flight[user.user_id] -= 1
        return "ok", 0

    sim, m, _ = _sim(runner)
    sim.start(_cfg(users=6, rate=300, workers=4, duration_s=2))
    _wait(sim, "idle", 10)
    assert _done(m)["ok"] > 100 and max(peak.values()) == 1


def test_the_simulator_connects_as_the_database_owner(monkeypatch):
    """Row-level security makes every statement's plan wait on a policy
    lookup and keeps region filters out of the scan (CockroachDB 26.3), so
    the simulator connects as the owner, like the seeder."""
    from banko_ai.db import rls
    from banko_ai.workers import traffic as T

    monkeypatch.setattr(T, "_simulator", None)
    monkeypatch.setattr(rls, "app_url", lambda: "postgresql://banko_app@db:26257/defaultdb?sslmode=disable")
    url = T.get_simulator().url
    assert "banko_app" not in url and "application_name=banko-load" in url


def test_serving_events_mark_leaving_home_and_coming_back_not_each_survivor():
    """During a failover the sampled read lands on different survivors from
    one sample to the next; the feed says once that the region's reads left
    home and once that they are back, each after two samples agree."""
    sim, _, ev = _sim(_instant)
    samples = ["us-west-2", "us-west-2",
               "us-east-1", "eu-west-1, us-east-1", "us-east-1", "eu-west-1, us-east-1",
               "us-east-1, us-west-2", "us-east-1, us-west-2", "us-east-1", "us-west-2", "us-west-2"]
    for served in samples:
        sim._observe_serving("us-west-2", served)
    texts = [(e["level"], e["text"]) for e in ev.since(None)]
    assert texts == [("warn", "Reads for us-west-2 customers now served by eu-west-1, us-east-1"),
                     ("good", "Reads for us-west-2 customers are served in us-west-2 again")]
    assert sim.status()["serving"]["us-west-2"]["served_by"] == "us-west-2"
