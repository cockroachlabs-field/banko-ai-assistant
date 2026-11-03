"""Work a turn starts but does not wait for: the answer cache write, the
narrative cache write and the implicit memory extraction. The answer goes
out first and these land a moment later, on one small pool for the whole
process. Background work never reaches the user; a failure is logged.

drain() waits until everything queued so far has finished. Tests call it
so a cache hit on the next turn is deterministic."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_for

log = logging.getLogger("banko.assistant")
WORKERS = 4

_lock = threading.Lock()
_pool: ThreadPoolExecutor | None = None
_pending: set[Future] = set()


def submit(what: str, fn: Callable[..., object], *args, **kwargs) -> Future:
    """Run fn(*args, **kwargs) on the pool. `what` names it in the log."""
    global _pool

    def run() -> None:
        try:
            fn(*args, **kwargs)
        except Exception:
            log.warning("%s skipped", what, exc_info=True)

    with _lock:
        if _pool is None:
            _pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="banko-assistant-bg")
        future = _pool.submit(run)
        _pending.add(future)
    future.add_done_callback(_finished)
    return future


def _finished(future: Future) -> None:
    with _lock:
        _pending.discard(future)


def drain(timeout: float = 30.0) -> bool:
    """Wait until nothing is queued or running; True when it all finished in time."""
    deadline = time.monotonic() + timeout
    while True:
        with _lock:
            pending = [f for f in _pending if not f.done()]
        left = deadline - time.monotonic()
        if not pending:
            return True
        if left <= 0:
            return False
        wait_for(pending, timeout=left)
