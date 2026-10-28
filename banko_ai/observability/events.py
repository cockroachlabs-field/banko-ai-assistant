"""The app's event bus: what happened, when, and how bad. The Platform page
lists the events and marks them on its chart; the newest few hundred are
kept in memory."""

from __future__ import annotations

import secrets
import threading
import time
from collections import deque
from collections.abc import Callable

LEVELS = ("info", "good", "warn", "bad")


class EventBus:
    def __init__(self, capacity: int = 500, clock: Callable[[], float] = time.time):
        self.clock = clock
        # Ids restart at 1 in every process; the epoch tells a reader that
        # its cursor belongs to a process that is gone.
        self.epoch = secrets.token_hex(4)
        self._events: deque[dict] = deque(maxlen=capacity)
        self._next_id = 1
        self._lock = threading.Lock()

    def publish(self, text: str, *, level: str = "info", source: str = "app") -> dict:
        if level not in LEVELS:
            raise ValueError(f"unknown event level {level!r}")
        with self._lock:
            event = {"id": self._next_id, "at": self.clock(), "level": level, "source": source, "text": text}
            self._next_id += 1
            self._events.append(event)
        return event

    def since(self, event_id: int | None, *, limit: int = 200) -> list[dict]:
        with self._lock:
            newer = [e for e in self._events if event_id is None or e["id"] > event_id]
        return newer[-limit:]


events = EventBus()
