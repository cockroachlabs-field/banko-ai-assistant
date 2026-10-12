"""Row-level security for the money tables.

Policies isolate rows by banko.user_id, which run_transaction and
read_session set at the start of every request transaction. They apply to
the non-admin role banko_app; the admin URL (seeding, migrations, workers)
bypasses them. Vector tables never get a policy: in CockroachDB v26.3 a
policy makes the C-SPANN index fall back to a full scan
(cockroachlabs/cockroach#10432). When the role can't log in, or the cluster
has no RLS, it stays off and the status says why."""

from __future__ import annotations

import logging
import threading
from collections.abc import Collection
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Connection, make_url
from sqlalchemy.exc import DBAPIError

from ..config.settings import get_config
from ..utils.db_retry import normalize_db_url
from .engine import _sqlstate, get_engine

log = logging.getLogger("banko.rls")
RLS_TABLES = ("accounts", "ledger_entries", "transfers", "goals", "user_budgets")
APP_ROLE = "banko_app"
POLICY = "banko_user_isolation"
_USER = "NULLIF(current_setting('banko.user_id', true), '')::UUID"
_INSUFFICIENT_PRIVILEGE = "42501"


@dataclass(frozen=True)
class RlsStatus:
    enabled: bool
    reason: str
    tables: tuple[str, ...] = ()


_status = RlsStatus(False, "not checked yet")
_app_url: str | None = None
_lock = threading.Lock()


def supports_rls(conn: Connection) -> bool:
    try:
        conn.execute(text("SELECT count(*) FROM [SHOW POLICIES FOR banko_meta]"))
        return True
    except DBAPIError:
        if conn.in_transaction():
            conn.rollback()
        return False


def policy_tables(conn: Connection) -> set[str]:
    """The money tables that already carry banko's policy."""
    return {t for t in RLS_TABLES if conn.execute(
        text(f"SELECT count(*) FROM [SHOW POLICIES FOR {t}] WHERE name = :p"), {"p": POLICY}).scalar()}


def rls_statements(table_names: list[str], *, regional: bool = False,
                   existing: Collection[str] = ()) -> list[str]:
    """existing names the tables that already have the policy. Those get an
    ALTER POLICY instead of DROP plus CREATE, so a live banko_app request
    never sees a table with RLS on and no policy. (A transaction cannot hold
    the pair: CockroachDB commits before each DDL by default, and with
    autocommit_before_ddl off it refuses DROP POLICY.)"""
    out = role_statements(table_names, regional=regional)
    for statements in policy_statements(existing).values():
        out += statements
    return out


def role_statements(table_names: list[str], *, regional: bool = False) -> list[str]:
    """CREATE ROLE and the GRANTs. Unlike the policy DDL these run inside an
    explicit transaction, so a fresh schema batches them with its tables."""
    granted = [t for t in table_names if t != "banko_meta"]
    out = [f"CREATE ROLE IF NOT EXISTS {APP_ROLE} WITH LOGIN",
           f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {', '.join(granted)} TO {APP_ROLE}",
           f"GRANT SELECT ON TABLE banko_meta TO {APP_ROLE}"]
    if regional:
        out.append(f"GRANT USAGE ON TYPE crdb_internal_region TO {APP_ROLE}")
    return out


def policy_statements(existing: Collection[str] = ()) -> dict[str, list[str]]:
    """ENABLE ROW LEVEL SECURITY plus the policy, per money table. v26.3
    implements both only in the declarative schema changer, which refuses
    them inside an explicit transaction, so each runs on its own."""
    out = {}
    for table in RLS_TABLES:
        verb = "ALTER" if table in existing else "CREATE"
        out[table] = [f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
                      f"{verb} POLICY {POLICY} ON {table} USING (user_id = {_USER}) WITH CHECK (user_id = {_USER})"]
    return out


def derive_app_url(admin_url: str) -> str | None:
    cfg = get_config()
    if cfg.database_app_url:
        return normalize_db_url(cfg.database_app_url)
    url = make_url(normalize_db_url(admin_url))
    if url.query.get("sslmode") != "disable":
        return None  # a secure cluster needs DATABASE_APP_URL with real credentials
    return url.set(username=APP_ROLE, password=None).render_as_string(hide_password=False)


def refresh_status(admin_url: str | None = None) -> RlsStatus:
    global _status, _app_url
    cfg = get_config()
    admin = admin_url or cfg.database_url
    status, app = RlsStatus(False, "RLS_MODE=off"), None
    if cfg.rls_mode != "off":
        with get_engine(admin).connect() as conn:
            if not supports_rls(conn):
                status = RlsStatus(False, "this CockroachDB version has no row-level security")
            else:
                active = policy_tables(conn)
                candidate = derive_app_url(admin)
                if len(active) != len(RLS_TABLES):
                    status = RlsStatus(False, "RLS is supported but not applied yet; it applies on the next "
                                              "schema change or a forced migrate")
                elif candidate is None:
                    status = RlsStatus(False, "secure cluster: set DATABASE_APP_URL for the banko_app role")
                else:
                    status, app = _probe_app_role(candidate)
    with _lock:
        _status, _app_url = status, app
    log.info("row-level security: %s", status.reason)
    return status


def _probe_app_role(candidate: str) -> tuple[RlsStatus, str | None]:
    """Log in as the app role and read a money table, so a role without
    grants is caught too. Only exception class names reach the status: a
    malformed DATABASE_APP_URL must never put its password in a log."""
    try:
        with get_engine(candidate).connect() as app_conn:
            app_conn.execute(text("SELECT 1 FROM accounts LIMIT 0"))
    except Exception as exc:
        if isinstance(exc, DBAPIError) and _sqlstate(exc) == _INSUFFICIENT_PRIVILEGE:
            return RlsStatus(False, f"the app role in DATABASE_APP_URL lacks grants (it must be {APP_ROLE})"), None
        return RlsStatus(False, f"the {APP_ROLE} role cannot log in ({exc.__class__.__name__})"), None
    return RlsStatus(True, "on for the money tables", RLS_TABLES), candidate


def status() -> RlsStatus:
    with _lock:
        return _status


def app_url() -> str | None:
    with _lock:
        return _app_url


def request_url(url: str | None) -> str | None:
    """The URL a request-path service should use: an explicit one wins,
    otherwise the app role's when RLS is on, otherwise the default."""
    return url or app_url()
