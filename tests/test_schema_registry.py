"""The schema registry: inline DDL, a fingerprint so normal boots run no
DDL, a safe upgrade path, and one applier when processes race."""

import json
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
        if S.supports_rls(conn) and S.get_config().rls_mode != "off":
            # The fresh path batches the role and grants with the tables
            # and runs the policy DDL per table; every money table ends up
            # with RLS on and the policy, and the app role can read.
            from banko_ai.db import rls
            assert S.policy_tables(conn) == set(rls.RLS_TABLES)
            flagged = {r[0] for r in conn.execute(text(
                "SELECT relname FROM pg_class WHERE relrowsecurity AND relname = ANY(:t)"),
                {"t": list(rls.RLS_TABLES)})}
            assert flagged == set(rls.RLS_TABLES)
            grants = conn.execute(text(
                "SELECT count(*) FROM [SHOW GRANTS ON TABLE accounts] WHERE grantee = :r"),
                {"r": rls.APP_ROLE}).scalar()
            assert grants
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
    assert elapsed < 30  # well under the lease TTL
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
        with get_engine(url).connect() as conn:
            expenses_ddl = conn.execute(text("SELECT create_statement FROM [SHOW CREATE TABLE expenses]")).scalar()
            transfer_keys = _unique_indexes(conn, "transfers")
        # Region-scoped keys: an insert checks uniqueness in its own region only.
        assert "PRIMARY KEY (crdb_region ASC, expense_id ASC)" in expenses_ddl
        assert transfer_keys.get("transfers_pkey") == ["crdb_region", "transfer_id"]
        assert ["crdb_region", "user_id", "idempotency_key"] in transfer_keys.values()
        assert ["user_id", "idempotency_key"] not in transfer_keys.values()
        print(f"fresh multi-region schema in {elapsed:.1f}s")
        # 26 tables, six of them with row-level security, measured 105 to 125 s
        # on a 6-node, 3-region cluster with realistic latency (October 2026).
        # The per-table CREATE INDEX plus SET LOCALITY this replaced took 82 s
        # for a single empty table.
        assert elapsed < 180
    finally:
        _drop(MR_URL, "banko_schema_mr_test")


def _unique_indexes(conn, table: str) -> dict[str, list[str]]:
    """Unique indexes and their explicit key columns (implicit ones left out)."""
    out: dict[str, list[str]] = {}
    for name, column, implicit in conn.execute(text(
            f"SELECT index_name, column_name, implicit FROM [SHOW INDEXES FROM {table}] "
            "WHERE NOT non_unique AND NOT storing ORDER BY index_name, seq_in_index")):
        if not implicit:
            out.setdefault(str(name), []).append(str(column))
    return out


def test_regional_money_keys_lead_with_the_region():
    """On a multi-region database a key without the region is checked in
    every region on each insert, so the expense id and the transfer
    idempotency key lead with crdb_region there."""
    sql = _sql(REGIONAL)
    assert "crdb_region crdb_internal_region NOT VISIBLE NOT NULL" in sql["expenses"]
    assert "PRIMARY KEY (crdb_region, expense_id)" in sql["expenses"]
    assert "UNIQUE (crdb_region, user_id, idempotency_key)" in sql["transfers"]
    assert "UNIQUE (user_id, idempotency_key)" not in sql["transfers"]
    # INSERT ... ON CONFLICT checks a global primary key in every region even
    # when the database generates it, so the transfer id leads with the region too.
    assert "PRIMARY KEY (crdb_region, transfer_id)" in sql["transfers"]


def test_single_region_keys_are_unchanged():
    sql = _sql(PLAIN)
    assert "PRIMARY KEY (expense_id)" in sql["expenses"] and "crdb_region" not in sql["expenses"]
    assert "UNIQUE (user_id, idempotency_key)" in sql["transfers"] and "crdb_region" not in sql["transfers"]


@pytest.mark.skipif(not URL, reason="DATABASE_URL not set")
def test_the_transfer_key_upgrade_leaves_single_region_databases_alone():
    with get_engine(URL).connect() as conn:
        assert S._scoped_transfer_key_upgrade(conn, PLAIN) == []
        assert S._scoped_transfer_key_upgrade(conn, REGIONAL) == []  # no crdb_region on this cluster


@pytest.mark.skipif(not MR_URL, reason="BANKO_TEST_MR_URL not set")
def test_an_existing_regional_transfers_table_gets_the_region_scoped_key():
    """Databases made multi-region by an older release keep a global
    idempotency key; the upgrade swaps it for the region-scoped one in place
    (the expenses primary key cannot be rebuilt in place on CockroachDB 26.3)."""
    url = _scratch(MR_URL, "banko_scoped_key_test")
    try:
        with get_engine(url).connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            regions = [str(r[0]) for r in conn.execute(text("SELECT region FROM [SHOW REGIONS FROM CLUSTER]"))]
            conn.execute(text(f'ALTER DATABASE banko_scoped_key_test SET PRIMARY REGION "{regions[0]}"'))
            conn.execute(text("CREATE TABLE transfers (transfer_id UUID NOT NULL DEFAULT gen_random_uuid(), "
                              "user_id UUID NOT NULL, idempotency_key STRING NOT NULL, PRIMARY KEY (transfer_id), "
                              "UNIQUE (user_id, idempotency_key)) LOCALITY REGIONAL BY ROW"))
            statements = S._scoped_transfer_key_upgrade(conn, S.Topology(regional=True))
            assert statements
            for sql in statements:
                conn.execute(text(sql))
            keys = _unique_indexes(conn, "transfers")
            assert S._scoped_transfer_key_upgrade(conn, S.Topology(regional=True)) == []
        assert ["crdb_region", "user_id", "idempotency_key"] in keys.values()
        assert ["user_id", "idempotency_key"] not in keys.values()
    finally:
        _drop(MR_URL, "banko_scoped_key_test")


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


@pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")
def test_money_region_waits_until_every_money_table_has_crdb_region(monkeypatch):
    """Mid conversion, expenses may carry crdb_region while a money table
    does not yet; pinning the region then would fail the insert."""
    from banko_ai.db import regions
    from tests._money import drop_user, make_user

    monkeypatch.setattr(regions, "regional_tables_ready", lambda url: True)
    uid = make_user(home_region="us-east1")
    try:
        with get_engine().connect() as c:
            assert regions.money_region(c, uid) is None  # single node: no money table has crdb_region
    finally:
        drop_user(uid)


def _meta_with_lease(url: str, owner: str, age_seconds: int) -> None:
    with get_engine(url).connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS banko_meta (
                key STRING NOT NULL PRIMARY KEY,
                value STRING NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """))
        conn.execute(text("UPSERT INTO banko_meta (key, value, updated_at) "
                          "VALUES ('schema_lock', :o, now() - :age * INTERVAL '1 second')"),
                     {"o": owner, "age": age_seconds})


def _lease_age_seconds(url: str) -> float:
    with get_engine(url).connect() as conn:
        return float(conn.execute(text(
            "SELECT extract(epoch FROM now() - updated_at) FROM banko_meta WHERE key = 'schema_lock'")).scalar())


def test_a_lease_a_dead_process_left_is_taken_over_within_a_minute(scratch_url):
    """A boot killed mid-apply leaves its lease behind. The next boot waits
    for it to go stale, so that wait is what a dead process costs."""
    _meta_with_lease(scratch_url, "dead-owner", age_seconds=90)
    with get_engine(scratch_url).connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        assert S._acquire_lock(conn, "me") is True
        assert conn.execute(text("SELECT value FROM banko_meta WHERE key = 'schema_lock'")).scalar() == "me"


def test_the_lease_stays_fresh_while_one_statement_runs_long(scratch_url, monkeypatch):
    """An index backfill on a populated database can run for minutes inside
    one statement. The lease is kept fresh beside it, so a short TTL never
    hands the schema to a second process mid-apply."""
    monkeypatch.setattr(S, "LEASE_HEARTBEAT_SECONDS", 0.2)
    _meta_with_lease(scratch_url, "me", age_seconds=50)
    heartbeat = S.LeaseHeartbeat(get_engine(scratch_url), "me")
    heartbeat.start()
    try:
        time.sleep(1.0)
        assert _lease_age_seconds(scratch_url) < 5
        assert not heartbeat.lost.is_set()
        with get_engine(scratch_url).connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(text("UPDATE banko_meta SET value = 'other' WHERE key = 'schema_lock'"))
        assert heartbeat.lost.wait(5), "the heartbeat never noticed the takeover"
    finally:
        heartbeat.stop()


def test_a_boot_waits_for_a_live_apply_longer_than_the_lease_ttl(scratch_url, monkeypatch):
    """A live apply can outlast the lease TTL many times over (six minutes on
    a three-region cluster). The waiter keeps waiting while the lease is
    fresh, and takes the finished schema when the fingerprint lands."""
    monkeypatch.setattr(S, "LOCK_TTL_SECONDS", 2)
    monkeypatch.setattr(S, "LEASE_HEARTBEAT_SECONDS", 0.5)
    monkeypatch.setattr(S, "APPLY_WAIT_SECONDS", 30)
    _meta_with_lease(scratch_url, "holder", age_seconds=0)
    heartbeat = S.LeaseHeartbeat(get_engine(scratch_url), "holder")
    heartbeat.start()

    def finish():
        time.sleep(5)  # longer than the TTL: only the heartbeat keeps the lease alive
        with get_engine(scratch_url).connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            conn.execute(text("UPSERT INTO banko_meta (key, value, updated_at) VALUES ('schema_fingerprint', :v, now())"),
                         {"v": json.dumps({"fingerprint": "fp-1"})})
        heartbeat.stop()

    threading.Thread(target=finish, daemon=True).start()
    try:
        with get_engine(scratch_url).connect() as conn:
            conn.execution_options(isolation_level="AUTOCOMMIT")
            took_over = S._wait_for(conn, "fp-1", "me")
            assert took_over is False
            assert conn.execute(text("SELECT value FROM banko_meta WHERE key = 'schema_lock'")).scalar() == "holder"
    finally:
        heartbeat.stop()


@pytest.mark.skipif(not URL, reason="DATABASE_URL not set")
def test_a_single_region_database_has_no_region_scoped_keys():
    from banko_ai.db.regions import scoped_key

    with get_engine(URL).connect() as conn:
        assert scoped_key(conn, "expenses") is False and scoped_key(conn, "transfers") is False


def test_reads_name_the_home_region_only_when_that_is_safe_and_useful(monkeypatch):
    """A read may name the user's region when the connection has no
    row-level security (CockroachDB keeps the filter out of an RLS scan),
    the money tables are REGIONAL BY ROW, and all the user's money rows are
    in their home region; otherwise reads stay as they are."""
    from banko_ai.db import regions as R
    from banko_ai.db import rls

    monkeypatch.setattr(R, "is_regional_deployment", lambda url: True)
    monkeypatch.setattr(R, "money_tables_regional", lambda url: True)
    monkeypatch.setattr(R, "user_home_region", lambda user_id, url=None: {"u1": "eu-west-1", "u2": "us-east-1"}.get(user_id))
    monkeypatch.setattr(R, "rows_home", lambda user_id, region, url=None: user_id == "u1")
    monkeypatch.setattr(rls, "app_url", lambda: "postgresql://banko_app@db:26257/defaultdb")
    assert R.read_region("u1", None) == "eu-west-1"
    assert R.read_region("u1", "postgresql://root@db:26257/defaultdb?application_name=banko-load") == "eu-west-1"
    assert R.read_region("u1", "postgresql://banko_app@db:26257/defaultdb") is None  # the RLS path
    assert R.read_region("u2", None) is None    # some rows live elsewhere
    assert R.read_region("u3", None) is None    # no home region
    monkeypatch.setattr(R, "money_tables_regional", lambda url: False)
    assert R.read_region("u1", None) is None
