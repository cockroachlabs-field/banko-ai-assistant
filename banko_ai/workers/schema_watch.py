"""Notices a database that was emptied under a running Banko.

A cluster rebuilt from the chaos stage comes back with an empty database.
The process that was serving it keeps running, and every page fails with
"relation does not exist" until someone with a shell restarts it. The watch
asks, now and then and whenever a statement fails for a missing table,
whether the schema is still there (the one SELECT the boot's fast path
runs). When it is gone, the app runs the boot's database steps again."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

log = logging.getLogger("banko.boot")

MISSING = "missing"
SETTLE_SECONDS = 2.0   # a burst of failing requests is one look, not one each


class SchemaWatch:
    def __init__(self, *, state: Callable[[], str], gate_ready: Callable[[], bool], rebuild: Callable[[], None],
                 interval: float, stop: threading.Event) -> None:
        self._state, self._gate_ready, self._rebuild = state, gate_ready, rebuild
        self._interval, self._stop = interval, stop
        self._wake = threading.Event()

    def poke(self) -> None:
        """Look now. Called when a statement fails because a table is not there."""
        self._wake.set()

    def check(self) -> bool:
        """One look. True when the database was found empty and its setup was started again."""
        if not self._gate_ready():
            return False   # the first boot, or a setup already running
        try:
            state = self._state()
        except Exception:
            return False   # out of reach is not empty: the boot's own retry waits for a late database
        if state != MISSING:
            return False
        log.warning("the database was emptied under this running Banko (its schema is gone); setting it up again")
        self._rebuild()
        return True

    def start(self) -> threading.Thread:
        def run() -> None:
            while not self._stop.is_set():
                self._wake.wait(self._interval)
                self._wake.clear()
                if self._stop.is_set():
                    return
                try:
                    self.check()
                except Exception:
                    log.warning("schema watch: a look failed; trying again later", exc_info=True)
                self._stop.wait(SETTLE_SECONDS)

        thread = threading.Thread(target=run, name="banko-schema-watch", daemon=True)
        thread.start()
        return thread
