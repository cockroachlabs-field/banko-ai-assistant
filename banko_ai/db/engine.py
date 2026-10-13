"""One engine per database URL for the whole process, plus the two ways
banko talks to CockroachDB: run_transaction for retryable units of work and
user_session for anything that must run with a user context set (row-level
security reads it from banko.user_id)."""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TypeVar

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine, make_url
from sqlalchemy.exc import DBAPIError

from ..config.settings import get_config
from ..utils.db_retry import normalize_db_url

log = logging.getLogger("banko.db")
T = TypeVar("T")

_engines: dict[str, Engine] = {}
_engines_lock = threading.Lock()

RETRYABLE_SQLSTATES = {"40001"}            # serialization failure: always safe to retry
AMBIGUOUS_SQLSTATES = {"40003"}            # result unknown: retry only idempotent work
CONNECTION_SQLSTATE_PREFIXES = ("08", "57P")


def get_engine(url: str | None = None) -> Engine:
    """The process-wide engine for this URL. Never dispose it."""
    cfg = get_config()
    key = normalize_db_url(url or cfg.database_url)
    engine = _engines.get(key)
    if engine is not None:
        return engine
    with _engines_lock:
        engine = _engines.get(key)
        if engine is None:
            engine = create_engine(
                key,
                pool_size=cfg.db_pool_size,
                max_overflow=cfg.db_max_overflow,
                pool_timeout=cfg.db_pool_timeout,
                pool_recycle=cfg.db_pool_recycle,
                pool_pre_ping=cfg.db_pool_pre_ping,
                connect_args={
                    "connect_timeout": cfg.db_connect_timeout,
                    "application_name": "banko",
                    "options": "-c timezone=utc",
                },
            )
            _engines[key] = engine
    return engine


def mask_url(url: str) -> str:
    """The URL with its password hidden, safe for logs and the console."""
    try:
        return make_url(normalize_db_url(url)).render_as_string(hide_password=True)
    except Exception:
        return "(unparseable database URL)"


@dataclass
class TxnStats:
    retries: int = 0
    failures: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_retry(self) -> None:
        with self._lock:
            self.retries += 1

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {"retries": self.retries, "failures": self.failures}


stats = TxnStats()


def _sqlstate(exc: BaseException) -> str:
    orig = getattr(exc, "orig", None)
    return str(getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None) or "")


def is_retryable(exc: BaseException, *, idempotent: bool) -> bool:
    """Whether running the same unit of work again is correct."""
    if not isinstance(exc, DBAPIError):
        return False
    state = _sqlstate(exc)
    if state in RETRYABLE_SQLSTATES:
        return True
    if state in AMBIGUOUS_SQLSTATES or state.startswith(CONNECTION_SQLSTATE_PREFIXES):
        return idempotent
    if exc.connection_invalidated:
        return idempotent
    return False


def _set_user(conn: Connection, user_id: str) -> None:
    conn.execute(text("SELECT set_config('banko.user_id', :uid, true)"), {"uid": str(user_id)})


def run_transaction(fn: Callable[[Connection], T], *, url: str | None = None,
                    user_id: str | None = None, idempotent: bool = True,
                    max_attempts: int | None = None) -> T:
    """Run fn(conn) in one transaction, retrying retryable errors with
    jittered backoff. Keep fn short and free of HTTP or LLM calls: it may
    run more than once."""
    attempts = max_attempts or get_config().db_retry_max_attempts
    delay = 0.05
    for attempt in range(1, attempts + 1):
        try:
            with get_engine(url).begin() as conn:
                if user_id is not None:
                    _set_user(conn, user_id)
                return fn(conn)
        except DBAPIError as exc:
            if attempt < attempts and is_retryable(exc, idempotent=idempotent):
                stats.record_retry()
                log.info("retrying transaction after %s (attempt %d of %d)",
                         _sqlstate(exc) or exc.__class__.__name__, attempt, attempts)
                time.sleep(delay + random.uniform(0, delay))
                delay = min(delay * 2, 2.0)
                continue
            stats.record_failure()
            raise
    raise AssertionError("unreachable")


@contextmanager
def user_session(user_id: str, *, url: str | None = None) -> Iterator[Connection]:
    """A transaction with banko.user_id set, for row-level-security tables."""
    with get_engine(url).begin() as conn:
        _set_user(conn, user_id)
        yield conn
