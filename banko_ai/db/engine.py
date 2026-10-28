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
from typing import Any, TypeVar

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Connection, Engine, make_url
from sqlalchemy.exc import DBAPIError, DisconnectionError, OperationalError

from ..config.settings import get_config
from ..observability.metrics import metrics
from ..utils.db_retry import normalize_db_url

log = logging.getLogger("banko.db")
T = TypeVar("T")

_engines: dict[str, Engine] = {}
_engines_lock = threading.Lock()

RETRYABLE_SQLSTATES = {"40001"}            # serialization failure: always safe to retry
AMBIGUOUS_SQLSTATES = {"40003"}            # result unknown: retry only idempotent work
CONNECTION_SQLSTATE_PREFIXES = ("08", "57P")

_GATEWAY_REGION = "banko_gateway_region"   # URL parameter: the pool's home region, never sent to the server
HOME_ATTEMPTS = 24                         # connections tried for one through the home region
REHOME_AFTER_S = 15.0                      # how long a connection that landed elsewhere waits to try home again
UNREACHABLE_RETRY_S = 300.0                # how often a home region this URL has never reached is tried again
_NO_REGIONS = "no region set"              # gateway_region() on a node started without a region
_now = time.monotonic


def with_application_name(url: str, name: str) -> str:
    """The URL with an application_name, which gives it a pool of its own in
    get_engine and a label in SHOW SESSIONS (the load simulator's banko-load)."""
    return make_url(url).update_query_dict({"application_name": name}).render_as_string(hide_password=False)


def with_gateway_region(url: str, region: str) -> str:
    """The URL with a home region, which gives it a pool of its own in
    get_engine whose connections enter the cluster through a node in that
    region. The load simulator keeps one per region, so each customer is
    served where they live, as a regional deployment of the app would."""
    return make_url(url).update_query_dict({_GATEWAY_REGION: region}).render_as_string(hide_password=False)


def _gateway_of(conn: Any) -> str | None:
    """The region of the node a DBAPI connection entered through, "" on a
    cluster whose nodes have no region, or None when it could not be read."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT gateway_region()")
            row = cur.fetchone()
        return str(row[0]) if row and row[0] else ""
    except Exception as exc:
        return "" if _NO_REGIONS in str(exc) else None
    finally:
        try:
            conn.rollback()
        except Exception:
            pass


def homed_connection(connect: Callable[[], Any], region: str, *,
                     attempts: int = HOME_ATTEMPTS) -> tuple[Any, str | None]:
    """A new DBAPI connection through a node in region, with the region it
    really entered through: None on a cluster without regions, "unknown"
    when that could not be read. Behind a load balancer over every node each
    new connection lands on the next node, so this asks again until one is
    in the region (a connection whose gateway cannot be read is asked again
    too). After attempts misses (no live node there), or when a node
    refuses, it keeps the next connection wherever it lands."""
    for _ in range(attempts):
        try:
            conn = connect()
        except Exception:  # a node is down, or the database is: stop choosing
            break
        gateway = _gateway_of(conn)
        if gateway == region or gateway == "":
            return conn, gateway or None
        conn.close()
    conn = connect()
    return conn, _landed(_gateway_of(conn))


def _landed(gateway: str | None) -> str | None:
    return "unknown" if gateway is None else (gateway or None)


def should_rehome(info: dict, region: str, *, now: float) -> bool:
    """Whether a pooled connection that landed outside its home region (the
    region was down) has waited long enough to try home again."""
    gateway = info.get("gateway")
    return gateway is not None and gateway != region and now - info.get("opened", now) > REHOME_AFTER_S


class _Home:
    """A homed engine's memory: whether its URL has ever reached the home
    region, and when it last missed. After a miss, new connections take the
    first node they land on instead of shopping for the region again, for
    REHOME_AFTER_S when the region was reached before (it is probably down)
    and UNREACHABLE_RETRY_S when it never was (a URL to one region only:
    CockroachDB Cloud, a single node, a sticky load balancer)."""

    def __init__(self, region: str):
        self.region = region
        self.reached = False
        self.missed_at: float | None = None
        self._lock = threading.Lock()

    def _cooling(self, now: float) -> bool:
        if self.missed_at is None:
            return False
        return now - self.missed_at < (REHOME_AFTER_S if self.reached else UNREACHABLE_RETRY_S)

    def connect(self, connect: Callable[[], Any], *, now: float) -> tuple[Any, str | None]:
        with self._lock:
            cooling = self._cooling(now)
        if cooling:
            conn = connect()
            gateway = _landed(_gateway_of(conn))
        else:
            conn, gateway = homed_connection(connect, self.region)
        with self._lock:
            if gateway == self.region:
                self.reached, self.missed_at = True, None
            elif gateway is not None and not cooling:
                self.missed_at = now
        return conn, gateway

    def may_rehome(self, info: dict, *, now: float) -> bool:
        with self._lock:
            cooling = self._cooling(now)
        return not cooling and should_rehome(info, self.region, now=now)


def _home(engine: Engine, region: str) -> None:
    """Every new connection of this engine enters through region when it can;
    one that could not is replaced on checkout once home may be tried again,
    so the pool comes home after an outage."""
    home = _Home(region)

    @event.listens_for(engine, "do_connect")
    def _connect(dialect, conn_rec, cargs, cparams):
        now = _now()
        conn, gateway = home.connect(lambda: dialect.connect(*cargs, **cparams), now=now)
        conn_rec.info.update(gateway=gateway, opened=now)
        return conn

    @event.listens_for(engine, "checkout")
    def _rehome(dbapi_conn, conn_rec, proxy):
        if home.may_rehome(conn_rec.info, now=_now()):
            raise DisconnectionError(f"connection is outside {region}; trying home again")


def get_engine(url: str | None = None, *, pool_size: int | None = None) -> Engine:
    """The process-wide engine for this URL. Never dispose it. pool_size
    applies only when this call creates the engine; an application_name in
    the URL labels its connections, otherwise they say banko, and a home
    region (with_gateway_region) makes them enter the cluster there."""
    cfg = get_config()
    key = normalize_db_url(url or cfg.database_url)
    engine = _engines.get(key)
    if engine is not None:
        return engine
    with _engines_lock:
        engine = _engines.get(key)
        if engine is None:
            parsed = make_url(key)
            name = parsed.query.get("application_name") or "banko"
            home = parsed.query.get(_GATEWAY_REGION)
            engine = create_engine(
                parsed.difference_update_query(["application_name", _GATEWAY_REGION]),
                pool_size=pool_size or cfg.db_pool_size,
                max_overflow=cfg.db_max_overflow,
                pool_timeout=cfg.db_pool_timeout,
                pool_recycle=cfg.db_pool_recycle,
                pool_pre_ping=cfg.db_pool_pre_ping,
                connect_args={
                    "connect_timeout": cfg.db_connect_timeout,
                    "application_name": name if isinstance(name, str) else name[0],
                    "options": "-c timezone=utc",
                },
            )
            if home:
                _home(engine, home if isinstance(home, str) else home[0])
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
    if exc.connection_invalidated or (isinstance(exc, OperationalError) and not state):
        return idempotent  # psycopg reports a connection it could not open without a SQLSTATE
    return False


def _set_user(conn: Connection, user_id: str) -> None:
    conn.execute(text("SELECT set_config('banko.user_id', :uid, true)"), {"uid": str(user_id)})


def run_transaction(fn: Callable[[Connection], T], *, url: str | None = None,
                    user_id: str | None = None, idempotent: bool = True,
                    max_attempts: int | None = None) -> T:
    """Run fn(conn) in one transaction, retrying retryable errors with
    jittered backoff. Keep fn short and free of HTTP or LLM calls: it may
    run more than once. Nothing commits before COMMIT is sent, so a lost
    connection while fn runs is retried even for non-idempotent work; an
    error from COMMIT itself carries at_commit=True, since it may have
    landed."""
    attempts = max_attempts or get_config().db_retry_max_attempts
    delay = 0.05
    for attempt in range(1, attempts + 1):
        ran = False
        try:
            with get_engine(url).begin() as conn:
                if user_id is not None:
                    _set_user(conn, user_id)
                result = fn(conn)
                ran = True
            return result
        except DBAPIError as exc:
            if ran:
                exc.at_commit = True  # type: ignore[attr-defined]
            safe = idempotent or not (ran or _sqlstate(exc) in AMBIGUOUS_SQLSTATES)
            if attempt < attempts and is_retryable(exc, idempotent=safe):
                stats.record_retry()
                metrics.record("db.retries")
                log.info("retrying transaction after %s (attempt %d of %d)",
                         _sqlstate(exc) or exc.__class__.__name__, attempt, attempts)
                time.sleep(delay + random.uniform(0, delay))
                delay = min(delay * 2, 2.0)
                continue
            stats.record_failure()
            metrics.record("db.failures")
            raise
    raise AssertionError("unreachable")


@contextmanager
def user_session(user_id: str, *, url: str | None = None) -> Iterator[Connection]:
    """A transaction with banko.user_id set, for row-level-security tables."""
    with get_engine(url).begin() as conn:
        _set_user(conn, user_id)
        yield conn
