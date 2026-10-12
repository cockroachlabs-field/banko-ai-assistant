"""Row-level security on the money tables: fail closed, scoped, never on vectors."""

import os

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from banko_ai.config.settings import reset_config
from banko_ai.db import rls
from banko_ai.db.engine import get_engine
from banko_ai.db.schema import ensure_schema
from banko_ai.services import ledger as L
from tests._money import drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture(scope="module")
def enabled():
    ensure_schema(force=True)
    status = rls.refresh_status()
    if not status.enabled:
        pytest.skip(f"RLS unavailable here: {status.reason}")
    return status


@pytest.fixture
def two_users(enabled):
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
