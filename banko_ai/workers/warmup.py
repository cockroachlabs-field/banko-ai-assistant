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


@dataclass
class Component:
    state: str = "pending"  # pending | running | ready | failed
    detail: str = ""
    updated_at: float = field(default_factory=time.time)


class Readiness:
    """Thread-safe component states for one app."""

    def __init__(self) -> None:
        self._components: dict[str, Component] = {}
        self._changed = threading.Condition()

    def set(self, name: str, state: str, detail: str = "") -> None:
        with self._changed:
            self._components[name] = Component(state=state, detail=detail)
            self._changed.notify_all()

    def get(self, name: str) -> Component | None:
        with self._changed:
            return self._components.get(name)

    def is_ready(self, name: str) -> bool:
        component = self.get(name)
        return component is not None and component.state == "ready"

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


def start_warmup(readiness: Readiness, steps: list[Step]) -> threading.Thread:
    """Run steps in order on a daemon thread. A failing step is recorded and
    the next one still runs."""
    for name, _ in steps:
        readiness.set(name, "pending")

    def run() -> None:
        for name, fn in steps:
            readiness.set(name, "running")
            started = time.monotonic()
            try:
                detail = fn() or ""
                readiness.set(name, "ready", detail)
                log.info("warm-up: %s ready in %.1fs %s", name, time.monotonic() - started, detail)
            except Exception as e:
                readiness.set(name, "failed", f"{e.__class__.__name__}: {e}")
                log.warning("warm-up: %s failed: %s", name, e)

    thread = threading.Thread(target=run, name="banko-warmup", daemon=True)
    thread.start()
    return thread
