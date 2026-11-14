"""The assistant's checkpointer rides out a dropped connection. A canceled
session (a node drain, a load balancer, a chaos kill) costs a retry on a
fresh connection, not the chat turn. Its tables' row-level TTL is brought up
to date by keep_ttl(), never by get_saver()."""

import logging
import os
import uuid

import pytest
from langchain_cockroachdb import CockroachDBSaver
from langgraph.checkpoint.base import empty_checkpoint
from sqlalchemy import text

from banko_ai.assistant import checkpoint
from banko_ai.db.engine import get_engine

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


def _checkpointer_sessions() -> int:
    with get_engine().connect() as c:
        return c.execute(text("SELECT count(*) FROM [SHOW CLUSTER SESSIONS] WHERE application_name = :a"),
                         {"a": checkpoint.APPLICATION_NAME}).scalar()


def test_a_canceled_session_costs_a_retry_not_the_checkpoint():
    saver = checkpoint.get_saver()
    config = {"configurable": {"thread_id": str(uuid.uuid4()), "checkpoint_ns": ""}}
    written = saver.put(config, empty_checkpoint(), {"source": "input", "step": -1}, {})
    assert _checkpointer_sessions() > 0, "the checkpointer's connections carry their application name"
    with get_engine().connect() as c:
        c.execute(text("CANCEL SESSIONS (SELECT session_id FROM [SHOW CLUSTER SESSIONS] "
                       "WHERE application_name = :a)"), {"a": checkpoint.APPLICATION_NAME})
    found = saver.get_tuple(written)
    assert found is not None and found.config["configurable"]["thread_id"] == config["configurable"]["thread_id"]
    saver.delete_thread(config["configurable"]["thread_id"])


def test_the_pool_follows_checkpoint_pool_size_and_db_pool_timeout(monkeypatch):
    from banko_ai.config.settings import reset_config

    monkeypatch.setenv("CHECKPOINT_POOL_SIZE", "3")
    monkeypatch.setenv("DB_POOL_TIMEOUT", "7")
    reset_config()
    monkeypatch.setattr(checkpoint, "_saver", None)
    saver = checkpoint.get_saver()
    try:
        assert saver.conn.max_size == 3 and saver.conn.min_size == 1
        assert saver.conn.timeout == 7
    finally:
        saver.conn.close()
        reset_config()


# Row-level TTL on the checkpoint tables, as langchain-cockroachdb 0.6 sets it:
# blob rows outlive their checkpoints by a tenth of the interval, and a run
# that stopped partway leaves TTL paused. keep_ttl() brings the setting up to
# date once and then leaves it alone, since every run is three schema changes
# and the first one waits 15 seconds for other savers.

# How SHOW CREATE TABLE checkpoint_blobs ends, copied from CockroachDB v26.3.2 and v25.4.19.
SET_BY_0_6 = (") WITH (ttl = 'on', ttl_expiration_expression = e'(created_at + \\'7 days\\' + "
              "CAST(\\'7 days\\' AS INTERVAL) / 10)', ttl_job_cron = '@daily', schema_locked = true)")
SET_BY_0_6_ON_25_4 = (") WITH (ttl = 'on', ttl_expiration_expression = e'(created_at + \\'7 days\\' + "
                      "CAST(\\'7 days\\' AS INTERVAL) / 10)', ttl_job_cron = '@daily')")
SET_BY_0_5 = (") WITH (ttl = 'on', ttl_expiration_expression = e'(created_at + \\'7 days\\')', "
              "ttl_job_cron = '@daily', schema_locked = true)")
STOPPED_PARTWAY = (") WITH (ttl = 'on', ttl_expiration_expression = e'(created_at + \\'7 days\\' + "
                   "CAST(\\'7 days\\' AS INTERVAL) / 10)', ttl_job_cron = '@daily', ttl_pause = true, "
                   "schema_locked = true)")
NEVER_SET = ") WITH (schema_locked = true)"
OLD_TTL = "ALTER TABLE {table} SET (ttl_expiration_expression = $$(created_at + '{interval}')$$, ttl_job_cron = '@daily')"
TABLES = ("checkpoints", "checkpoint_blobs", "checkpoint_writes")


@pytest.mark.parametrize("statement, current", [
    (SET_BY_0_6, True), (SET_BY_0_6_ON_25_4, True), (SET_BY_0_5, False), (STOPPED_PARTWAY, False), (NEVER_SET, False),
])
def test_ttl_is_current_only_with_the_grace_and_running(statement, current):
    create = "CREATE TABLE public.checkpoint_blobs (thread_id STRING NOT NULL, " + statement
    assert checkpoint._ttl_current(create, fraction=10) is current


def test_the_grace_is_read_with_the_librarys_own_fraction():
    create = "CREATE TABLE public.checkpoint_blobs (thread_id STRING NOT NULL, " + SET_BY_0_6
    assert checkpoint._ttl_current(create, fraction=5) is False


def test_a_pause_counts_however_it_is_printed():
    """CockroachDB drops ttl_pause from the statement when it is reset, so
    its presence is the pause, quoted or not."""
    quoted = STOPPED_PARTWAY.replace("ttl_pause = true", "ttl_pause = 'true'")
    assert checkpoint._ttl_current("CREATE TABLE public.checkpoint_blobs (" + quoted, fraction=10) is False


@pytest.fixture
def scratch(monkeypatch):
    """A database of its own for the checkpointer, and a way to prepare what
    an earlier boot left in it. The 15 seconds enable_ttl waits for other
    savers are skipped: this database has none."""
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool
    from sqlalchemy.engine import make_url

    from banko_ai.config.settings import get_config, reset_config
    from banko_ai.utils.db_retry import normalize_db_url, to_libpq_url

    name = f"banko_ttl_{uuid.uuid4().hex[:8]}"
    admin = get_engine()
    with admin.connect() as c:
        c.execution_options(isolation_level="AUTOCOMMIT")
        c.execute(text(f"CREATE DATABASE {name}"))
    url = make_url(normalize_db_url(get_config().database_url)).set(database=name).render_as_string(hide_password=False)
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("CHECKPOINT_TTL_DAYS", "7")
    reset_config()
    monkeypatch.setattr(checkpoint, "_saver", None)
    monkeypatch.setattr("langchain_cockroachdb.checkpointer.saver._sleep", lambda seconds: None)
    pools = []

    def earlier_boot(left_by: str) -> None:
        if left_by == "nothing":
            return
        pool = ConnectionPool(to_libpq_url(url), min_size=1, max_size=2, open=True,
                              kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
        pools.append(pool)
        saver = CockroachDBSaver(pool)
        saver.setup()
        with pool.connection() as conn:
            if left_by == "0.5":
                for table in TABLES:
                    conn.execute(OLD_TTL.format(table=table, interval="7 days"))
            elif left_by == "0.5 at 30 days":
                for table in TABLES:
                    conn.execute(OLD_TTL.format(table=table, interval="30 days"))
            elif left_by == "someone else":
                conn.execute("ALTER TABLE checkpoint_blobs SET (ttl_expire_after = '3 days', ttl_job_cron = '@weekly')")
            else:  # a 0.6 run that stopped while TTL was paused, as enable_ttl pauses it: on all three
                saver.enable_ttl(ttl_interval="7 days", cron="@daily")
                for table in TABLES:
                    conn.execute(f"ALTER TABLE {table} SET (ttl_pause = true)")

    yield earlier_boot
    for pool in [*pools, *([checkpoint._saver.conn] if checkpoint._saver is not None else [])]:
        pool.close()
    reset_config()
    with admin.connect() as c:
        c.execution_options(isolation_level="AUTOCOMMIT")
        c.execute(text(f"DROP DATABASE IF EXISTS {name} CASCADE"))


def _shown(saver, table: str = "checkpoint_blobs") -> str:
    with saver.conn.connection() as conn:
        return conn.execute(f"SELECT create_statement FROM [SHOW CREATE TABLE {table}]").fetchone()["create_statement"]


def _count_enable_ttl(monkeypatch) -> list:
    calls, real = [], CockroachDBSaver.enable_ttl

    def counted(self, **kw):
        calls.append(kw)
        return real(self, **kw)

    monkeypatch.setattr(CockroachDBSaver, "enable_ttl", counted)
    return calls


def _next_boot(monkeypatch) -> None:
    checkpoint._saver.conn.close()
    monkeypatch.setattr(checkpoint, "_saver", None)


def _told(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "banko.assistant"]


@pytest.mark.parametrize("left_by", ["0.5", "a run that stopped partway"])
def test_building_the_checkpointer_never_runs_the_ttl_upkeep(scratch, monkeypatch, left_by):
    """A question that finds no checkpointer builds one. It must not sit
    through enable_ttl's 15 seconds, whatever the last boot left."""
    scratch(left_by)
    calls = _count_enable_ttl(monkeypatch)
    saver = checkpoint.get_saver()
    assert saver is checkpoint.get_saver() and calls == []


@pytest.mark.parametrize("left_by", ["nothing", "0.5", "a run that stopped partway"])
def test_ttl_is_brought_up_to_date_once_and_then_left_alone(scratch, monkeypatch, caplog, left_by):
    scratch(left_by)
    calls = _count_enable_ttl(monkeypatch)
    caplog.set_level(logging.INFO, logger="banko.assistant")
    done = checkpoint.keep_ttl()
    assert calls == [{"ttl_interval": "7 days", "cron": "@daily"}] and "7 days" in done
    saver = checkpoint.get_saver()
    assert "/ 10)" in _shown(saver)
    assert not [table for table in TABLES if "ttl_pause" in _shown(saver, table)]
    # The server log says what the 15 seconds are for, and when they are over.
    told = _told(caplog)
    assert len(told) == 2 and "15 seconds" in told[0] and "up to date" in told[1], told
    caplog.clear()
    _next_boot(monkeypatch)
    assert "up to date" in checkpoint.keep_ttl()
    assert len(calls) == 1 and not _told(caplog)


def test_a_database_keeps_the_interval_it_has(scratch, monkeypatch):
    """The setting says 7 days and the database was set up at 30: the upgrade
    adds the grace to the 30 days and deletes nothing a boot ago was safe."""
    scratch("0.5 at 30 days")
    calls = _count_enable_ttl(monkeypatch)
    done = checkpoint.keep_ttl()
    assert calls == [{"ttl_interval": "30 days", "cron": "@daily"}] and "30 days" in done
    saver = checkpoint.get_saver()
    for table in TABLES:
        shown = _shown(saver, table)
        assert "30 days" in shown and "7 days" not in shown, shown
    assert "/ 10)" in _shown(saver)


@pytest.mark.parametrize("left_by, calls_made", [("nothing", []), ("0.5", [{"ttl_interval": "7 days", "cron": "@daily"}])])
def test_ttl_days_of_zero_sets_none_but_keeps_what_a_database_has(scratch, monkeypatch, left_by, calls_made):
    from banko_ai.config.settings import reset_config
    scratch(left_by)
    monkeypatch.setenv("CHECKPOINT_TTL_DAYS", "0")
    reset_config()
    calls = _count_enable_ttl(monkeypatch)
    done = checkpoint.keep_ttl()
    assert calls == calls_made
    shown = _shown(checkpoint.get_saver())
    if left_by == "nothing":
        assert "ttl" not in shown and "off" in done
    else:
        assert "/ 10)" in shown and "ttl_pause" not in shown


def test_a_ttl_setting_banko_did_not_write_is_left_alone(scratch, monkeypatch, caplog):
    scratch("someone else")
    calls = _count_enable_ttl(monkeypatch)
    before = _shown(checkpoint.get_saver())
    caplog.set_level(logging.INFO, logger="banko.assistant")
    done = checkpoint.keep_ttl()
    assert calls == [] and "left alone" in done
    assert _shown(checkpoint.get_saver()) == before and "ttl_expire_after" in before
    assert not _told(caplog)


def test_a_connection_dropped_during_the_check_costs_a_retry_not_the_step(scratch, monkeypatch):
    """The check is Banko's own read on the saver's pool, so it rides out a
    dropped connection the way the saver's calls do."""
    import psycopg
    scratch("0.5")
    checkpoint.keep_ttl()
    real, attempts = checkpoint._blob_table, []

    def dropped_once(pool):
        attempts.append(1)
        if len(attempts) == 1:
            raise psycopg.OperationalError("consuming input failed: server closed the connection unexpectedly")
        return real(pool)

    monkeypatch.setattr(checkpoint, "_blob_table", dropped_once)
    assert "up to date" in checkpoint.keep_ttl() and len(attempts) == 2


def test_a_run_that_leaves_the_setting_behind_is_not_reported_as_done(scratch, monkeypatch, caplog):
    """enable_ttl returns without resuming when another run changed the
    setting meanwhile. Success is what the table shows afterwards."""
    scratch("0.5")
    monkeypatch.setattr(CockroachDBSaver, "enable_ttl", lambda self, **kw: None)
    caplog.set_level(logging.INFO, logger="banko.assistant")
    with pytest.raises(RuntimeError, match="not current"):
        checkpoint.keep_ttl()
    assert not any("up to date" in line for line in _told(caplog))
