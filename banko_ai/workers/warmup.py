"""Background warm-up.

Everything boot used to do before listening now runs here, after the server
is already accepting requests. Each step reports its state, so /api/health
and the UI show real progress instead of a spinner with nothing behind it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

log = logging.getLogger("banko.warmup")

Step = tuple[str, Callable[[], "str | None"]]

UNFINISHED = ("pending", "running")


@dataclass
class Component:
    # pending | running | ready | degraded | failed. degraded means the
    # step is done and the app works without it (Kafka retrying while the
    # webhook carries signals); it never holds the ready line.
    state: str = "pending"
    detail: str = ""
    updated_at: float = field(default_factory=time.time)


class Readiness:
    """Thread-safe component states for one app."""

    def __init__(self) -> None:
        self._components: dict[str, Component] = {}
        self._background: set[str] = set()
        self._lanes: dict[str, object] = {}   # step -> the warm-up lane that runs it now
        self._changed = threading.Condition()

    def mark_background(self, name: str) -> None:
        """A background component reports progress but never holds the
        ready line (persona seeding can take minutes on a fresh database)."""
        with self._changed:
            self._background.add(name)
            self._changed.notify_all()

    def set(self, name: str, state: str, detail: str = "", *, lane: object | None = None) -> None:
        """lane is the warm-up lane reporting. A step started again (the
        database was rebuilt) belongs to the newer lane from its "pending"
        on, and what an older lane still has to say about it is dropped: a
        first-boot step failing late on the old database would otherwise
        mark as failed what the newer one finished."""
        with self._changed:
            if lane is not None:
                if state == "pending":
                    self._lanes[name] = lane
                elif self._lanes.get(name) is not lane:
                    return
            self._components[name] = Component(state=state, detail=detail)
            self._changed.notify_all()

    def get(self, name: str) -> Component | None:
        with self._changed:
            return self._components.get(name)

    def is_ready(self, name: str) -> bool:
        component = self.get(name)
        return component is not None and component.state == "ready"

    def finished(self) -> bool:
        """True once no foreground step is pending or running."""
        with self._changed:
            return all(c.state not in UNFINISHED for name, c in self._components.items()
                       if name not in self._background)

    def wait_for(self, name: str, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._changed:
            while True:
                component = self._components.get(name)
                if component is not None and component.state == "ready":
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._changed.wait(remaining)

    def snapshot(self) -> dict[str, dict]:
        with self._changed:
            return {name: asdict(c) for name, c in self._components.items()}


def start_warmup(readiness: Readiness, steps: list[Step], *,
                 stop: threading.Event | None = None,
                 name: str = "banko-warmup") -> threading.Thread:
    """Run steps in order on a daemon thread. A failing step is recorded and
    the next one still runs. Setting `stop` ends the lane before its next
    step; steps that retry should watch it too."""
    lane = object()
    for step_name, _ in steps:
        readiness.set(step_name, "pending", lane=lane)

    def run() -> None:
        for step_name, fn in steps:
            if stop is not None and stop.is_set():
                readiness.set(step_name, "failed", "warm-up stopped", lane=lane)
                continue
            readiness.set(step_name, "running", lane=lane)
            started = time.monotonic()
            try:
                detail = fn() or ""
                readiness.set(step_name, "ready", detail, lane=lane)
                log.info("warm-up: %s ready in %.1fs %s", step_name, time.monotonic() - started, detail)
            except Exception as e:
                readiness.set(step_name, "failed", f"{e.__class__.__name__}: {e}", lane=lane)
                log.warning("warm-up: %s failed: %s", step_name, e)

    thread = threading.Thread(target=run, name=name, daemon=True)
    thread.start()
    return thread
