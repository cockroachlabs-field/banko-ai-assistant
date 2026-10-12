"""The schema registry: inline DDL, a fingerprint so normal boots run no
DDL, a safe upgrade path, and one applier when processes race."""

import os
import threading
import time

import pytest
from sqlalchemy import event, text
from sqlalchemy.engine import make_url

from banko_ai.db import schema as S
from banko_ai.db.engine import get_engine
from banko_ai.utils.db_retry import normalize_db_url

REGIONAL = S.Topology(regional=True, regions=("eu-de", "us-east", "us-south"))
PLAIN = S.Topology(regional=False)
URL = os.getenv("DATABASE_URL")
MR_URL = os.getenv("BANKO_TEST_MR_URL")


def _sql(topo):
    return {t.name: t.create_sql(topo) for t in S.tables()}


def test_fingerprint_is_stable_and_topology_aware():
    assert S.fingerprint(PLAIN) == S.fingerprint(PLAIN)
    assert S.fingerprint(PLAIN) != S.fingerprint(REGIONAL)


def test_fingerprint_changes_with_rls():
    assert S.fingerprint(PLAIN, rls=True) != S.fingerprint(PLAIN, rls=False)


def test_regional_tables_are_born_with_their_locality():
    sql = _sql(REGIONAL)
    assert sql["expenses"].endswith("LOCALITY REGIONAL BY ROW")
    assert sql["spending_signals"].endswith("LOCALITY REGIONAL BY ROW")
    assert sql["users"].endswith("LOCALITY GLOBAL")
    assert "LOCALITY" not in _sql(PLAIN)["expenses"]


def test_indexes_are_declared_inline():
    sql = _sql(PLAIN)["expenses"]
    assert "VECTOR INDEX idx_expenses_user_embedding (user_id, embedding vector_cosine_ops)" in sql
    assert "INDEX idx_expenses_user_date (user_id ASC, expense_date DESC)" in sql


def test_every_table_has_a_primary_key_and_vector_indexes_use_cosine():
    for table in S.tables():
        assert table.primary_key, table.name
        for index in table.indexes:
            if index.vector:
                assert "vector_cosine_ops" in index.columns, index.name


def test_ttl_comes_from_settings(monkeypatch):
    monkeypatch.setenv("COACH_SIGNAL_TTL_DAYS", "14")
    assert "ttl_expire_after = '14 days'" in _sql(PLAIN)["spending_signals"]


def test_locality_plan_leaves_converted_tables_alone():
    current = {"banko_meta": "GLOBAL", "expenses": "REGIONAL BY ROW", "users": "GLOBAL",
               "spending_signals": "REGIONAL BY ROW", "coach_nudges": "REGIONAL BY ROW"}
    assert S.plan_locality_conversions(current, REGIONAL) == []


def test_locality_plan_converts_only_what_differs():
    current = {"banko_meta": "GLOBAL", "expenses": "REGIONAL BY TABLE IN PRIMARY REGION",
               "users": "REGIONAL BY TABLE IN PRIMARY REGION",
               "spending_signals": "REGIONAL BY ROW", "coach_nudges": "REGIONAL BY ROW"}
    assert S.plan_locality_conversions(current, REGIONAL) == [
        "ALTER TABLE expenses SET LOCALITY REGIONAL BY ROW",
        "ALTER TABLE users SET LOCALITY GLOBAL",
    ]


def test_single_region_never_plans_locality_changes():
    assert S.plan_locality_conversions({"expenses": "anything"}, PLAIN) == []


def test_refresh_lease_raises_when_another_owner_holds_it():
    """_refresh_lease raises RuntimeError when another process holds the lease."""
    if not URL:
        pytest.skip("DATABASE_URL not set")
    url = _scratch(URL, "banko_lease_test")
    try:
        # Create banko_meta and insert a lease held by another owner
        with get_engine(url).connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(text("""
                CREATE TABLE banko_meta (
                    key STRING NOT NULL PRIMARY KEY,
                    value STRING NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """))
            conn.execute(text("""
                INSERT INTO banko_meta (key, value, updated_at)
                VALUES ('schema_lock', 'other-owner', now())
            """))

        # Trying to refresh with our owner should raise
        from banko_ai.db.schema import _refresh_lease
        with get_engine(url).connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            with pytest.raises(RuntimeError, match="lease was taken over"):
                _refresh_lease(conn, "our-owner")

        # Refreshing with the correct owner should succeed
        with get_engine(url).connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            _refresh_lease(conn, "other-owner")  # should not raise
    finally:
        _drop(URL, "banko_lease_test")


def _scratch(base: str, name: str) -> str:
    admin = get_engine(base)
    with admin.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text(f"DROP DATABASE IF EXISTS {name} CASCADE"))
        conn.execute(text(f"CREATE DATABASE {name}"))
    return make_url(normalize_db_url(base)).set(database=name).render_as_string(hide_password=False)


def _drop(base: str, name: str) -> None:
    with get_engine(base).connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text(f"DROP DATABASE IF EXISTS {name} CASCADE"))


@pytest.fixture
def scratch_url():
    if not URL:
        pytest.skip("DATABASE_URL not set")
    url = _scratch(URL, "banko_schema_test")
    yield url
    _drop(URL, "banko_schema_test")


def _tables(url):
    with get_engine(url).connect() as conn:
        return {r[0] for r in conn.execute(text("SELECT table_name FROM [SHOW TABLES] WHERE schema_name = 'public'"))}


def test_fresh_database_gets_every_table_and_a_fingerprint(scratch_url):
    result = S.ensure_schema(scratch_url)
    assert result.changed
    assert {t.name for t in S.tables()} <= _tables(scratch_url)
    with get_engine(scratch_url).connect() as conn:
        stored = conn.execute(text("SELECT value FROM banko_meta WHERE key = 'schema_fingerprint'")).scalar()
    assert result.fingerprint in stored


def test_second_boot_runs_no_ddl(scratch_url):
    S.ensure_schema(scratch_url)
    seen = []
    engine = get_engine(scratch_url)

    def record(conn, cursor, statement, params, context, executemany):
        seen.append(statement.strip().split()[0].upper())

    event.listen(engine, "before_cursor_execute", record)
    try:
        result = S.ensure_schema(scratch_url)
    finally:
        event.remove(engine, "before_cursor_execute", record)
    assert not result.changed
    assert not [s for s in seen if s in ("CREATE", "ALTER", "DROP", "SET")]
    # The fingerprint read, plus at most one statement the driver issues on
    # a fresh pooled connection.
    assert len(seen) <= 2, seen


def test_missing_index_is_restored(scratch_url):
    S.ensure_schema(scratch_url)
    with get_engine(scratch_url).connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text("DROP INDEX expenses@idx_expenses_merchant"))
        conn.execute(text("DELETE FROM banko_meta WHERE key = 'schema_fingerprint'"))
    S.ensure_schema(scratch_url)
    with get_engine(scratch_url).connect() as conn:
        names = {r[0] for r in conn.execute(text("SELECT index_name FROM [SHOW INDEXES FROM expenses]"))}
    assert "idx_expenses_merchant" in names


def test_upgrade_from_v13_schema(scratch_url):
    """Tables as banko 1.3.x left them: indexes created separately, no banko_meta."""
    with get_engine(scratch_url).connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text("""
            CREATE TABLE expenses (
                expense_id UUID PRIMARY KEY DEFAULT gen_random_uuid(), user_id UUID NOT NULL,
                expense_date DATE NOT NULL, expense_amount DECIMAL(10,2) NOT NULL,
                shopping_type STRING NOT NULL, description STRING, merchant STRING,
                payment_method STRING NOT NULL, recurring BOOL DEFAULT false, tags STRING[],
                embedding VECTOR(384), created_at TIMESTAMP DEFAULT now())"""))
        conn.execute(text("""
            CREATE TABLE users (
                user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(), username STRING NOT NULL UNIQUE,
                spending_style STRING NOT NULL, home_region STRING,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(), demo_user BOOL NOT NULL DEFAULT true)
            WITH (ttl_expire_after = '30 days')"""))
        conn.execute(text("INSERT INTO expenses (user_id, expense_date, expense_amount, shopping_type, payment_method) "
                          "VALUES (gen_random_uuid(), current_date, 9.99, 'Coffee', 'Credit Card')"))
    result = S.ensure_schema(scratch_url)
    assert result.changed
    assert {t.name for t in S.tables()} <= _tables(scratch_url)
    with get_engine(scratch_url).connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM expenses")).scalar() == 1
        kinds = {r[0]: r[1] for r in conn.execute(text(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'expenses'"))}
        users_ddl = conn.execute(text("SHOW CREATE TABLE users")).fetchone()[1]
    assert kinds["created_at"] == "timestamp with time zone"
    assert "ttl_expire_after" not in users_ddl


def test_concurrent_boots_apply_schema_once(scratch_url):
    results, errors = [], []

    def boot():
        try:
            results.append(S.ensure_schema(scratch_url))
        except Exception as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=boot) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=300)
    assert not errors, errors
    assert sum(r.changed for r in results) == 1
    assert {t.name for t in S.tables()} <= _tables(scratch_url)


def test_stale_lease_takeover(scratch_url):
    """A stale lease row is taken over instead of timing out."""
    # Create banko_meta and insert a stale lease
    with get_engine(scratch_url).connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS banko_meta (
                key STRING NOT NULL PRIMARY KEY,
                value STRING NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """))
        # Insert a lease that expired 10 minutes ago
        conn.execute(text("""
            INSERT INTO banko_meta (key, value, updated_at)
            VALUES ('schema_lock', 'dead-owner', now() - INTERVAL '10 minutes')
        """))

    # ensure_schema should take over the stale lease and apply promptly
    started = time.monotonic()
    result = S.ensure_schema(scratch_url)
    elapsed = time.monotonic() - started

    assert result.changed
    assert elapsed < 30  # well under the 300s TTL
    assert {t.name for t in S.tables()} <= _tables(scratch_url)


@pytest.mark.skipif(not MR_URL, reason="BANKO_TEST_MR_URL not set")
def test_fresh_multi_region_database_is_born_regional():
    url = _scratch(MR_URL, "banko_schema_mr_test")
    try:
        started = time.monotonic()
        result = S.ensure_schema(url)
        elapsed = time.monotonic() - started
        with get_engine(url).connect() as conn:
            localities = {r[0]: r[1] for r in conn.execute(text(
                "SELECT table_name, locality FROM [SHOW TABLES] WHERE schema_name = 'public'"))}
        assert result.changed and not result.pending_conversions
        assert localities["expenses"] == "REGIONAL BY ROW"
        assert localities["users"] == "GLOBAL"
        print(f"fresh multi-region schema in {elapsed:.1f}s")
        assert elapsed < 120
    finally:
        _drop(MR_URL, "banko_schema_mr_test")


MONEY = ("categories", "accounts", "transfers", "ledger_entries", "goals", "user_data_versions")


def test_money_tables_are_registered_with_their_locality():
    sql = _sql(REGIONAL)
    assert "LOCALITY GLOBAL" in sql["categories"]
    for name in ("accounts", "transfers", "ledger_entries", "goals", "user_data_versions", "user_budgets"):
        assert sql[name].endswith("LOCALITY REGIONAL BY ROW"), name
    plain = _sql(PLAIN)
    assert all("LOCALITY" not in plain[name] for name in MONEY)


def test_money_columns_follow_the_spec():
    sql = _sql(PLAIN)
    assert "balance DECIMAL(14,2) NOT NULL DEFAULT 0" in sql["accounts"]
    assert "CHECK (kind IN ('checking', 'savings', 'credit'))" in sql["accounts"]
    assert "UNIQUE (user_id, idempotency_key)" in sql["transfers"]
    assert "CHECK (amount > 0)" in sql["transfers"]
    assert ("INDEX ledger_user_idx (user_id, posted_at DESC) "
            "STORING (account_id, amount, kind, description, expense_id)") in sql["ledger_entries"]
    assert "INDEX ledger_expense_idx (expense_id)" in sql["ledger_entries"]
    assert "display_name STRING" in sql["users"]
    assert "avatar_color STRING" in sql["users"]


def test_storing_renders_inline_and_standalone():
    idx = S.Index("t_user_idx", "user_id", storing="a, b")
    assert idx.inline() == "INDEX t_user_idx (user_id) STORING (a, b)"
    assert idx.create("t") == "CREATE INDEX IF NOT EXISTS t_user_idx ON t (user_id) STORING (a, b)"


@pytest.mark.skipif(not URL, reason="DATABASE_URL not set")
def test_foundation_database_gains_money_tables_without_losing_data(scratch_url):
    S.ensure_schema(scratch_url)
    eng = get_engine(scratch_url)
    with eng.begin() as c:
        c.execute(text("INSERT INTO users (user_id, username, spending_style) "
                       "VALUES ('00000000-0000-0000-0000-00000000f001', 'keeper', 'saver')"))
        for name in MONEY:
            c.execute(text(f"DROP TABLE {name}"))
        c.execute(text("ALTER TABLE users DROP COLUMN display_name"))
        c.execute(text("ALTER TABLE users DROP COLUMN avatar_color"))
        c.execute(text("DELETE FROM banko_meta WHERE key = 'schema_fingerprint'"))
    result = S.ensure_schema(scratch_url)
    assert result.changed
    assert set(MONEY) <= _tables(scratch_url)
    with eng.connect() as c:
        assert c.execute(text("SELECT username, display_name FROM users "
                              "WHERE user_id = '00000000-0000-0000-0000-00000000f001'")).fetchone() == ("keeper", None)
        from banko_ai.db.regions import has_region_column
        assert has_region_column(c, "accounts") is False  # single node: no crdb_region


def test_region_columns():
    from banko_ai.db.regions import region_columns
    assert region_columns(None) == ("", "")
    assert region_columns("eu-de") == ("crdb_region, ", "CAST(:crdb_region AS crdb_internal_region), ")


def test_has_region_column_cache_never_stores_password():
    from unittest.mock import Mock

    from banko_ai.db import regions

    # Create a stub connection with a URL containing "secret"
    stub_conn = Mock()
    stub_conn.engine.url = make_url("cockroachdb+psycopg://u:secret@h:26257/db")
    stub_conn.execute.return_value.fetchone.return_value = (1,)

    # Call has_region_column, which should cache the result
    result = regions.has_region_column(stub_conn, "test_table")
    assert result is True

    # Verify "secret" doesn't appear in any cache keys
    for url_str, table_name in regions._region_columns_seen:
        assert "secret" not in url_str, f"Password leaked in cache key: {url_str}"
