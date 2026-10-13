"""Row-level security on the money tables: fail closed, scoped, never on vectors."""

import os
from contextlib import contextmanager

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from banko_ai.config.settings import reset_config
from banko_ai.db import rls
from banko_ai.db.engine import get_engine
from banko_ai.db.schema import ensure_schema
from banko_ai.services import accounts as A
from banko_ai.services import ledger as L
from banko_ai.services import transactions as T
from tests._money import drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


_CLEAN = rls.RlsStatus(False, "not checked yet")


@pytest.fixture(scope="module")
def enabled():
    ensure_schema(force=True)
    status = rls.refresh_status()
    app = rls.app_url()
    # Leave the module globals clean; rls_on turns them on per test.
    rls._status, rls._app_url = _CLEAN, None
    if not status.enabled:
        pytest.skip(f"RLS unavailable here: {status.reason}")
    return status, app


@pytest.fixture(autouse=True)
def _isolated_rls_state(monkeypatch):
    """Every test starts with RLS off in this process and puts the globals
    back afterwards, so test order can never leak the app URL."""
    monkeypatch.setattr(rls, "_status", _CLEAN)
    monkeypatch.setattr(rls, "_app_url", None)


@pytest.fixture
def rls_on(enabled, monkeypatch):
    status, app = enabled
    monkeypatch.setattr(rls, "_status", status)
    monkeypatch.setattr(rls, "_app_url", app)
    return app


@pytest.fixture
def two_users(rls_on):
    a, b = make_user(), make_user()
    acct_a = L.open_account(a, kind="checking", name="A", institution="B", mask="1", opening_balance="10")
    acct_b = L.open_account(b, kind="checking", name="B", institution="B", mask="2", opening_balance="20")
    yield a, b, acct_a, acct_b
    drop_user(a)
    drop_user(b)


def _app():
    return get_engine(rls.app_url())


def test_no_user_context_sees_nothing(two_users):
    with _app().connect() as c:
        assert c.execute(text("SELECT count(*) FROM accounts")).scalar() == 0


def test_user_context_sees_only_its_rows(two_users):
    a, _, acct_a, _ = two_users
    with _app().begin() as c:
        c.execute(text("SELECT set_config('banko.user_id', :u, true)"), {"u": a})
        rows = {str(r[0]) for r in c.execute(text("SELECT account_id FROM accounts"))}
    assert rows == {acct_a}


def test_cross_user_insert_is_rejected(two_users):
    a, b, _, _ = two_users
    with pytest.raises(DBAPIError):
        with _app().begin() as c:
            c.execute(text("SELECT set_config('banko.user_id', :u, true)"), {"u": a})
            c.execute(text("INSERT INTO accounts (user_id, kind, name, institution, mask, opened_at) "
                           "VALUES (CAST(:b AS UUID), 'checking', 'X', 'B', '9', current_date)"), {"b": b})


def test_vector_tables_have_no_policy(enabled):
    with get_engine().connect() as c:
        for table in ("expenses", "documents"):
            assert c.execute(text(f"SELECT count(*) FROM [SHOW POLICIES FOR {table}]")).scalar() == 0


def test_ledger_works_through_the_app_role(two_users):
    a, _, acct_a, _ = two_users
    acct_a2 = L.open_account(a, kind="savings", name="A2", institution="B", mask="3", url=rls.app_url())
    assert L.transfer(a, acct_a, acct_a2, "5.00", idempotency_key="rls-1", url=rls.app_url()).status == "ok"


def test_off_mode_reports_why(monkeypatch):
    monkeypatch.setenv("RLS_MODE", "off")
    reset_config()
    status = rls.refresh_status()
    assert not status.enabled and "RLS_MODE=off" in status.reason
    assert rls.app_url() is None


def test_unsupported_version_reports_why(monkeypatch):
    monkeypatch.setattr(rls, "supports_rls", lambda conn: False)
    status = rls.refresh_status()
    assert not status.enabled and "row-level security" in status.reason


def test_secure_cluster_without_app_url_stays_off():
    assert rls.derive_app_url("postgresql://root@host:26257/db?sslmode=verify-full") is None
    assert rls.derive_app_url("postgresql://root@host:26257/db?sslmode=disable").startswith(
        "cockroachdb+psycopg://banko_app@host:26257/db")


def test_cross_user_update_is_rejected(two_users):
    a, b, acct_a, acct_b = two_users
    with _app().begin() as c:
        c.execute(text("SELECT set_config('banko.user_id', :u, true)"), {"u": a})
        hidden = c.execute(text("UPDATE accounts SET name = 'mine now' WHERE account_id = CAST(:b AS UUID)"),
                           {"b": acct_b})
        assert hidden.rowcount == 0
    with pytest.raises(DBAPIError):
        with _app().begin() as c:
            c.execute(text("SELECT set_config('banko.user_id', :u, true)"), {"u": a})
            c.execute(text("UPDATE accounts SET user_id = CAST(:b AS UUID) WHERE account_id = CAST(:a AS UUID)"),
                      {"a": acct_a, "b": b})


def test_a_service_without_a_url_reads_as_the_app_role(two_users, monkeypatch):
    a, _, acct_a, _ = two_users
    real = A.read_session
    roles = []

    @contextmanager
    def spy(user_id, **kwargs):
        with real(user_id, **kwargs) as conn:
            roles.append(conn.execute(text("SELECT current_user")).scalar())
            yield conn

    monkeypatch.setattr(A, "read_session", spy)
    out = A.list_accounts(a)
    assert roles == [rls.APP_ROLE]
    assert [acct["account_id"] for acct in out["accounts"]] == [acct_a]


def test_vector_search_survives_rls_for_the_app_role(two_users):
    a = two_users[0]
    vec = "'[" + ",".join(["0.01"] * 384) + "]'"
    sql = (T.semantic_sql().replace(":qvec", vec).replace(":u", f"'{a}'")
           .replace(":k", "50").replace(":limit", "20"))
    with _app().begin() as c:
        c.execute(text("SELECT set_config('banko.user_id', :u, true)"), {"u": a})
        assert c.execute(text("SELECT current_user")).scalar() == rls.APP_ROLE
        plan = "\n".join(r[0] for r in c.execute(text("EXPLAIN " + sql)))
    assert "vector search" in plan and "idx_expenses_user_embedding" in plan


def test_reapply_alters_existing_policies_instead_of_dropping_them():
    statements = rls.rls_statements(["banko_meta", "accounts", "goals"], existing={"accounts"})
    assert not [s for s in statements if s.startswith("DROP POLICY")]
    assert any(s.startswith(f"ALTER POLICY {rls.POLICY} ON accounts ") for s in statements)
    assert any(s.startswith(f"CREATE POLICY {rls.POLICY} ON goals ") for s in statements)


def test_supported_but_unapplied_says_so(enabled, monkeypatch):
    monkeypatch.setattr(rls, "policy_tables", lambda conn: set())
    status = rls.refresh_status()
    assert not status.enabled and "not applied yet" in status.reason
    assert rls.app_url() is None


def test_app_role_without_grants_is_caught(enabled, monkeypatch):
    class Denied(DBAPIError):
        def __init__(self):
            super().__init__("SELECT", {}, type("Orig", (Exception,), {"sqlstate": "42501"})())

    class Engine:
        def connect(self):
            raise Denied()

    real = rls.get_engine
    monkeypatch.setattr(rls, "get_engine", lambda url=None: Engine() if url == enabled[1] else real(url))
    status = rls.refresh_status()
    assert not status.enabled and "lacks grants" in status.reason


def test_a_bad_app_url_never_leaks_into_the_status(enabled, monkeypatch):
    monkeypatch.setenv("DATABASE_APP_URL", "cockroachdb://banko_app:s3cret@[bad/x")
    reset_config()
    status = rls.refresh_status()
    assert not status.enabled and "s3cret" not in status.reason
