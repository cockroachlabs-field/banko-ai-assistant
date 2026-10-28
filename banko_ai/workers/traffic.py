"""The load simulator: an open-loop traffic generator of banking customers.

A scheduler thread draws Poisson arrivals at the target rate (ramping up
over ramp_s) and hands each operation, stamped with when it should start,
to a bounded queue; worker threads run it through the workload. Latency
counts from that intended start, so a slow cluster shows up as latency
and, once the queue is full, as dropped arrivals, never as quietly less
load. Every operation lands in the metrics registry three times: all of
it, its operation, and its customer's home region.

While a run is going the simulator checks money integrity and samples,
per home region, which regions serve its customers' reads. Each process
has one simulator (get_simulator) on a pool of its own, banko-load.
"""

from __future__ import annotations

import logging
import math
import queue
import random
import threading
import time
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from sqlalchemy.exc import DBAPIError

from ..config.settings import get_config
from ..observability.events import EventBus
from ..observability.events import events as default_events
from ..observability.metrics import Metrics
from ..observability.metrics import metrics as default_metrics
from ..services.load_users import LoadUser
from .workload import OPS, PROFILES, OpContext, UserPicker, VectorCache, run_op

log = logging.getLogger("banko.load")

_MAX_DURATION_S = 24 * 3600
_MAX_RAMP_S = 600
_DEFAULT: Any = object()
_ERROR_WORDS = {"40001": "transaction conflicts", "40003": "ambiguous commits", "connection": "lost connections",
                "timeout": "timeouts"}


class LoadConfigError(ValueError):
    """A setting out of range or of the wrong kind; the message says which, in plain words."""


class SimulatorBusy(Exception):
    """A run is already preparing, running or stopping."""


def _number(raw: Any, kind: type, label: str, low: float, high: float) -> Any:
    if isinstance(raw, bool):
        raise LoadConfigError(f"{label} must be a number")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise LoadConfigError(f"{label} must be a number") from None
    if not math.isfinite(value):
        raise LoadConfigError(f"{label} must be a number")
    if kind is int:
        if not value.is_integer():
            raise LoadConfigError(f"{label} must be a whole number")
        value = int(value)
    if not low <= value <= high:
        raise LoadConfigError(f"{label} must be between {low:g} and {high:g}")
    return value


def _weights(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict) or not raw:
        raise LoadConfigError('A custom profile needs weights, for example {"purchase": 50, "search": 50}')
    out = {}
    for op, weight in raw.items():
        if op not in OPS:
            raise LoadConfigError(f"Unknown operation {op!r}; pick from {', '.join(OPS)}")
        out[op] = _number(weight, int, f"The weight of {op}", 0, 1000)
    if sum(out.values()) <= 0:
        raise LoadConfigError("Custom weights must add up to more than 0")
    return {op: w for op, w in out.items() if w > 0}


@dataclass(frozen=True)
class LoadConfig:
    users: int
    rate: float
    workers: int
    profile: str = "balanced"
    duration_s: int | None = None
    ramp_s: int = 10
    weights: dict[str, int] | None = None

    @classmethod
    def from_dict(cls, data: Any, *, base: LoadConfig | None = None) -> LoadConfig:
        """A validated config: data's fields over base's (or the settings'
        defaults), every number checked against the LOAD_MAX_* caps."""
        if not isinstance(data, dict):
            raise LoadConfigError("Send the load settings as a JSON object")
        cfg = get_config()
        start = base or cls(users=cfg.load_default_users, rate=cfg.load_default_rate,
                            workers=cfg.load_default_workers)

        def get(key: str) -> Any:
            return data[key] if key in data else getattr(start, key)

        profile = get("profile")
        if profile not in (*PROFILES, "custom"):
            raise LoadConfigError(f"The profile must be one of {', '.join([*PROFILES, 'custom'])}")
        duration = get("duration_s")
        return cls(
            users=_number(get("users"), int, "Users", 1, cfg.load_max_users),
            rate=_number(get("rate"), float, "The rate", 0.1, cfg.load_max_rate),
            workers=_number(get("workers"), int, "Workers", 1, cfg.load_max_workers),
            profile=profile,
            duration_s=None if duration in (None, 0, "", "0") else _number(duration, int, "The duration", 1,
                                                                           _MAX_DURATION_S),
            ramp_s=_number(get("ramp_s"), int, "The ramp", 0, _MAX_RAMP_S),
            weights=_weights(get("weights")) if profile == "custom" else None,
        )

    def mix(self) -> dict[str, int]:
        return dict(self.weights) if self.profile == "custom" and self.weights else dict(PROFILES[self.profile])

    def as_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        until = f" for {self.duration_s} s" if self.duration_s else ""
        return f"{self.profile}, {self.users} customers, {self.rate:g} ops/s, {self.workers} workers{until}"


def error_class_of(exc: BaseException) -> str:
    """A short name for what went wrong: the SQLSTATE, or connection,
    timeout, or the exception's class."""
    if isinstance(exc, DBAPIError):
        state = str(getattr(getattr(exc, "orig", None), "sqlstate", None) or "")
        if state == "57014":
            return "timeout"
        if exc.connection_invalidated or state.startswith(("08", "57P")):
            return "connection"
        if state:
            return state
    return exc.__class__.__name__


class _Run:
    def __init__(self, config: LoadConfig):
        self.config = config
        self.rng = random.Random()
        self.stop = threading.Event()
        self.queue: queue.Queue = queue.Queue(maxsize=max(4, config.workers * 4))
        self.started: float | None = None
        self.started_wall: float | None = None
        self.users: list[LoadUser] = []
        self.picker: UserPicker | None = None
        self.runner: Callable[[str, LoadUser, random.Random], tuple[str, int]] | None = None
        self.mix: tuple[list[str], list[int]] = ([], [])
        self.threads: list[threading.Thread] = []
        self.workers: dict[int, threading.Thread] = {}
        self.errors: deque[tuple[float, str]] = deque()
        self.last_burst = -1e9
        self.set_mix(config.mix())

    def set_mix(self, mix: dict[str, int]) -> None:
        self.mix = (list(mix), list(mix.values()))


class TrafficSimulator:
    def __init__(self, *, metrics: Metrics | None = None, events: EventBus | None = None, url: str | None = None,
                 runner: Callable[[str, LoadUser, random.Random], tuple[str, int]] | None = None,
                 prepare: Callable[[int], list[LoadUser]] | None = None,
                 sampler: Callable[[str], str | None] | None = _DEFAULT,
                 integrity: Callable[[], Any] | None = _DEFAULT, pool_size: int | None = None,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time):
        self.metrics = metrics or default_metrics
        self._pool_size = pool_size
        self.events = events or default_events
        self.url = url
        self._runner = runner
        self._prepare = prepare or self._prepare_users
        self._sampler = self._sample_serving if sampler is _DEFAULT else sampler
        self._integrity = self._check_integrity if integrity is _DEFAULT else integrity
        self._clock, self._wall = clock, wall
        self._lock = threading.Lock()
        self._run: _Run | None = None
        self._state = "idle"
        self._error: str | None = None
        self._last_config: LoadConfig | None = None
        self._serving: dict[str, dict] = {}
        self._vectors: VectorCache | None = None

    # -- defaults ------------------------------------------------------------------------------

    @staticmethod
    def _prepare_users(count: int) -> list[LoadUser]:
        from ..services.load_users import prepare
        return prepare(count)

    def _sample_serving(self, user_id: str) -> str | None:
        from ..services.platform import sample_serving
        return sample_serving(user_id, url=self.url)

    @staticmethod
    def _check_integrity() -> Any:
        from ..services.integrity import run_check
        return run_check()

    # -- control -------------------------------------------------------------------------------

    @property
    def config(self) -> LoadConfig | None:
        with self._lock:
            return self._run.config if self._run else self._last_config

    def start(self, config: LoadConfig) -> dict:
        with self._lock:
            if self._run is not None:
                raise SimulatorBusy("A load run is already going; stop it first")
            run = self._run = _Run(config)
            self._state, self._error, self._last_config = "preparing", None, config
        threading.Thread(target=self._boot, args=(run,), name="banko-load-boot", daemon=True).start()
        return self.status()

    def apply(self, config: LoadConfig) -> dict:
        with self._lock:
            run = self._run
            if run is None or self._state != "running":
                raise LoadConfigError("Start the load first; there is nothing running to change")
            old, run.config, self._last_config = run.config, config, config
            run.set_mix(config.mix())
        if config.workers > old.workers:
            self._spawn_workers(run)
        if config.users != len(run.users):
            threading.Thread(target=self._resize_users, args=(run, config.users), name="banko-load-users",
                             daemon=True).start()
        self.events.publish(f"Load changed: {config.describe()}", source="load")
        return self.status()

    def stop(self, timeout: float = 5.0, *, run: _Run | None = None) -> dict:
        with self._lock:
            current = self._run
            if current is None or (run is not None and run is not current) or self._state == "stopping":
                return self._status_locked()
            self._state = "stopping"
        current.stop.set()
        deadline = self._clock() + timeout
        me = threading.current_thread()
        for t in list(current.threads):
            if t is not me:
                t.join(max(0.0, deadline - self._clock()))
        span = int(self._clock() - (current.started or self._clock())) + 2
        s = self.metrics.summary("load", window_s=span, now=self.metrics.clock() + 1)
        with self._lock:
            if self._run is current:
                self._run, self._state = None, "idle"
        if current.started is not None:
            elapsed = self._clock() - current.started
            self.events.publish(f"Load stopped after {elapsed:.0f} s: {s['ok'] + s['declined']:,} operations, "
                                f"{s['errors']:,} errors, {s['dropped']:,} dropped", source="load")
        return self.status()

    def status(self) -> dict:
        with self._lock:
            return self._status_locked()

    def _status_locked(self) -> dict:
        run = self._run
        config = run.config if run else self._last_config
        return {"state": self._state, "error": self._error, "config": config.as_dict() if config else None,
                "started_at": run.started_wall if run else None,
                "elapsed_s": round(self._clock() - run.started, 1) if run and run.started else 0.0,
                "users_ready": len(run.users) if run else 0, "serving": dict(self._serving)}

    # -- the run -------------------------------------------------------------------------------

    def _boot(self, run: _Run) -> None:
        try:
            if self.url and self._pool_size:  # the simulator's own pool, created on its first run
                from ..db.engine import get_engine
                get_engine(self.url, pool_size=self._pool_size)
            users = list(self._prepare(run.config.users))[:run.config.users]
            if not users:
                raise RuntimeError("no load users")
            run.rng.shuffle(users)
            if self._runner is None:
                self._vectors = self._vectors or VectorCache.build()
                ctx = OpContext(url=self.url, vectors=self._vectors)
                run.runner = lambda op, user, rng: run_op(op, user, ctx, rng)
            else:
                run.runner = self._runner
            run.users, run.picker = users, UserPicker(users)
        except Exception:
            log.warning("load users could not be prepared", exc_info=True)
            with self._lock:
                if self._run is run:
                    self._run, self._state = None, "idle"
                    self._error = "Load users could not be prepared: check that the database is reachable."
            self.events.publish("Load could not start: the load users could not be prepared.", level="bad",
                                source="load")
            return
        with self._lock:
            if self._run is not run or run.stop.is_set():
                return
            run.started, run.started_wall = self._clock(), self._wall()
            self._state = "running"
        self.events.publish(f"Load started: {run.config.describe()}", source="load")
        for name, target in (("banko-load-scheduler", self._schedule), ("banko-load-integrity", self._watch_integrity),
                             ("banko-load-serving", self._watch_serving)):
            t = threading.Thread(target=target, args=(run,), name=name, daemon=True)
            run.threads.append(t)
            t.start()
        self._spawn_workers(run)

    def _spawn_workers(self, run: _Run) -> None:
        for index in range(run.config.workers):
            t = run.workers.get(index)
            if t is None or not t.is_alive():
                t = threading.Thread(target=self._work, args=(run, index), name=f"banko-load-worker-{index}",
                                     daemon=True)
                run.workers[index] = t
                run.threads.append(t)
                t.start()

    def _resize_users(self, run: _Run, count: int) -> None:
        try:
            users = list(self._prepare(count))[:count] if count > len(run.users) else run.users[:count]
            run.rng.shuffle(users)
            run.users, run.picker = users, UserPicker(users)
        except Exception:
            log.warning("could not resize the load users", exc_info=True)

    def _schedule(self, run: _Run) -> None:
        rng = random.Random(run.rng.random())
        next_at = self._clock()
        while not run.stop.is_set():
            cfg = run.config
            elapsed = self._clock() - (run.started or self._clock())
            if cfg.duration_s is not None and elapsed >= cfg.duration_s:
                self.stop(run=run)
                return
            ramp = min(1.0, 0.1 + 0.9 * elapsed / cfg.ramp_s) if cfg.ramp_s else 1.0
            next_at += rng.expovariate(cfg.rate * ramp)
            delay = next_at - self._clock()
            if delay > 0:
                if run.stop.wait(delay):
                    return
            elif delay < -1.0:  # fell a second behind (a pause): catch up from now, no burst
                next_at = self._clock()
            ops, weights = run.mix
            op, user = rng.choices(ops, weights)[0], run.picker.pick(rng)
            try:
                run.queue.put_nowait((op, user, next_at))
            except queue.Full:
                self._record(op, user, outcome="dropped")

    def _work(self, run: _Run, index: int) -> None:
        rng = random.Random(run.rng.random() + index)
        while not run.stop.is_set() and index < run.config.workers:
            try:
                op, user, intended = run.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            error_class = None
            try:
                outcome, retries = run.runner(op, user, rng)
            except Exception as exc:  # recorded by class; the run goes on
                outcome, retries, error_class = "error", getattr(exc, "client_retries", 0), error_class_of(exc)
                log.debug("load operation %s failed: %s", op, error_class)
            self._record(op, user, outcome=outcome, ms=(self._clock() - intended) * 1000,
                         error_class=error_class, retries=retries)
            if error_class:
                self._note_error(run, error_class)

    def _record(self, op: str, user: LoadUser, *, outcome: str, ms: float | None = None,
                error_class: str | None = None, retries: int = 0) -> None:
        for series in ("load", f"load.op.{op}", f"load.region.{user.region or 'all'}"):
            self.metrics.record(series, ms=ms, outcome=outcome, error_class=error_class, retries=retries)

    def _note_error(self, run: _Run, error_class: str) -> None:
        now = self._clock()
        with self._lock:
            run.errors.append((now, error_class))
            while run.errors and run.errors[0][0] < now - 10:
                run.errors.popleft()
            if len(run.errors) < 5 or now - run.last_burst < 10:
                return
            run.last_burst = now
            count = len(run.errors)
            top = Counter(c for _, c in run.errors).most_common(1)[0][0]
        self.events.publish(f"{count} operations failed in the last 10 s, mostly {_ERROR_WORDS.get(top, top)}",
                            level="warn", source="load")

    def _watch_integrity(self, run: _Run) -> None:
        if self._integrity is None:
            return
        drifted_before = False
        while not run.stop.wait(get_config().load_integrity_interval_seconds):
            try:
                report = self._integrity()
            except Exception:
                log.warning("integrity check during load failed", exc_info=True)
                continue
            drifted = bool(report is not None and getattr(report, "drifted", ()))
            if drifted:
                self.events.publish(f"Money integrity: {len(report.drifted)} accounts drifted", level="bad",
                                    source="integrity")
            elif drifted_before:
                self.events.publish("Money integrity: zero drift again", level="good", source="integrity")
            drifted_before = drifted

    def _watch_serving(self, run: _Run) -> None:
        if self._sampler is None:
            return
        rng = random.Random(run.rng.random())
        while not run.stop.wait(get_config().load_serving_sample_seconds):
            by_region: dict[str, list[LoadUser]] = {}
            for u in run.users:
                if u.region:
                    by_region.setdefault(u.region, []).append(u)
            for region, users in sorted(by_region.items()):
                try:
                    served = self._sampler(rng.choice(users).user_id)
                except Exception:
                    log.debug("serving region sample failed", exc_info=True)
                    continue
                if not served:
                    continue
                with self._lock:
                    before = (self._serving.get(region) or {}).get("served_by")
                    self._serving[region] = {"served_by": served, "at": self._wall()}
                if before and before != served:
                    self.events.publish(f"Reads for {region} customers now served by {served}", level="warn",
                                        source="load")


_simulator: TrafficSimulator | None = None
_simulator_lock = threading.Lock()


def get_simulator() -> TrafficSimulator:
    """The process's simulator, on its own banko-load pool (the app role's
    URL when row-level security is on), sized for LOAD_MAX_WORKERS and
    created on the first run, so asking for its status never connects."""
    global _simulator
    with _simulator_lock:
        if _simulator is None:
            from ..db import rls
            from ..db.engine import with_application_name
            cfg = get_config()
            url = with_application_name(rls.app_url() or cfg.database_url, "banko-load")
            _simulator = TrafficSimulator(url=url, pool_size=cfg.load_max_workers + 2)
        return _simulator


def stop_simulator() -> None:
    """Stop the process's simulator if one ran (app shutdown)."""
    with _simulator_lock:
        sim = _simulator
    if sim is not None:
        sim.stop(timeout=2.0)

