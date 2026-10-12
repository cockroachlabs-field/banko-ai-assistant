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
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Connection, make_url
from sqlalchemy.exc import DBAPIError

from ..config.settings import get_config
from ..utils.db_retry import normalize_db_url
from .engine import get_engine

log = logging.getLogger("banko.rls")
RLS_TABLES = ("accounts", "ledger_entries", "transfers", "goals", "user_budgets")
APP_ROLE = "banko_app"
POLICY = "banko_user_isolation"
_USER = "NULLIF(current_setting('banko.user_id', true), '')::UUID"


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


def rls_statements(table_names: list[str], *, regional: bool = False) -> list[str]:
    granted = [t for t in table_names if t != "banko_meta"]
    out = [f"CREATE ROLE IF NOT EXISTS {APP_ROLE} WITH LOGIN",
           f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE {', '.join(granted)} TO {APP_ROLE}",
           f"GRANT SELECT ON TABLE banko_meta TO {APP_ROLE}"]
    if regional:
        out.append(f"GRANT USAGE ON TYPE crdb_internal_region TO {APP_ROLE}")
    for table in RLS_TABLES:
        out += [f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
                f"DROP POLICY IF EXISTS {POLICY} ON {table}",
                f"CREATE POLICY {POLICY} ON {table} USING (user_id = {_USER}) WITH CHECK (user_id = {_USER})"]
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
                active = [t for t in RLS_TABLES if conn.execute(
                    text(f"SELECT count(*) FROM [SHOW POLICIES FOR {t}]")).scalar()]
                candidate = derive_app_url(admin)
                if len(active) != len(RLS_TABLES):
                    status = RlsStatus(False, "policies are not in place yet (schema pending)")
                elif candidate is None:
                    status = RlsStatus(False, "secure cluster: set DATABASE_APP_URL for the banko_app role")
                else:
                    try:
                        with get_engine(candidate).connect() as app_conn:
                            app_conn.execute(text("SELECT 1"))
                        status, app = RlsStatus(True, "on for the money tables", RLS_TABLES), candidate
                    except DBAPIError as exc:
                        status = RlsStatus(False, f"the banko_app role cannot log in ({exc.__class__.__name__})")
    with _lock:
        _status, _app_url = status, app
    log.info("row-level security: %s", status.reason)
    return status


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
