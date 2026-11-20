"""A database emptied under a running Banko is set up again without a
restart. On a shared instance anyone at the chaos stage can rebuild the
cluster, which wipes the database, and only someone with a shell could
restart Banko: until then every page failed with "relation does not exist"."""

import os
import threading
import time

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url

from banko_ai.workers.schema_watch import SchemaWatch

DB = os.getenv("DATABASE_URL")


def _watch(state, *, ready=True, interval=3600.0):
    rebuilt, stop = [], threading.Event()
    watch = SchemaWatch(state=state, gate_ready=lambda: ready, rebuild=lambda: rebuilt.append(time.monotonic()),
                        interval=interval, stop=stop)
    return watch, rebuilt, stop


def test_an_emptied_database_is_rebuilt_once():
    watch, rebuilt, _ = _watch(lambda: "missing")
    assert watch.check() is True and len(rebuilt) == 1


@pytest.mark.parametrize("state", ["current", "outdated"])
def test_a_database_that_has_its_schema_is_left_alone(state):
    watch, rebuilt, _ = _watch(lambda: state)
    assert watch.check() is False and rebuilt == []


def test_a_database_that_cannot_be_reached_is_not_taken_for_an_empty_one():
    """A region outage or a restart in progress: the boot's own retry handles
    a database that is late. Nothing here concludes it was emptied."""
    def unreachable():
        raise ConnectionError("connection refused")

    watch, rebuilt, _ = _watch(unreachable)
    assert watch.check() is False and rebuilt == []


def test_nothing_is_rebuilt_while_a_setup_is_still_running():
    """The first boot on an empty database, or a rebuild already under way."""
    watch, rebuilt, _ = _watch(lambda: "missing", ready=False)
    assert watch.check() is False and rebuilt == []


def test_a_poke_checks_now_instead_of_at_the_next_interval():
    watch, rebuilt, stop = _watch(lambda: "missing", interval=3600.0)
    thread = watch.start()
    try:
        time.sleep(0.2)
        assert rebuilt == []            # an hour to go
        watch.poke()
        deadline = time.monotonic() + 5
        while not rebuilt and time.monotonic() < deadline:
            time.sleep(0.02)
        assert len(rebuilt) == 1
    finally:
        stop.set()
        watch.poke()
        thread.join(timeout=5)
    assert not thread.is_alive()


@pytest.mark.skipif(not DB, reason="DATABASE_URL not set")
def test_a_statement_on_a_missing_table_pokes_the_watch():
    from banko_ai.db import engine as E
    poked = []

    class Watch:
        def poke(self):
            poked.append(1)

    watch = Watch()
    E.watch_missing_tables(watch)
    with E.get_engine().connect() as conn:
        conn.execute(text("SELECT 1")).scalar()
        assert poked == []
        with pytest.raises(Exception, match="does not exist"):
            conn.execute(text("SELECT * FROM a_table_nobody_made"))
    assert poked == [1]


def test_a_missing_table_is_found_however_the_error_is_wrapped():
    from sqlalchemy.exc import ProgrammingError
    from werkzeug.exceptions import InternalServerError

    from banko_ai.db.engine import missing_table

    class Undefined(Exception):
        sqlstate = "42P01"

    class Other(Exception):
        sqlstate = "40001"

    wrapped = ProgrammingError("SELECT 1 FROM accounts", {}, Undefined('relation "accounts" does not exist'))
    assert missing_table(wrapped) and missing_table(InternalServerError(original_exception=wrapped))
    try:
        try:
            raise wrapped
        except ProgrammingError as inner:
            raise RuntimeError("the page failed") from inner
    except RuntimeError as outer:
        assert missing_table(outer)
    assert not missing_table(ProgrammingError("UPDATE accounts", {}, Other("restart transaction")))
    assert not missing_table(ValueError("nothing to do with a database")) and not missing_table(None)


@pytest.mark.skipif(not DB, reason="DATABASE_URL not set")
def test_an_app_sets_its_emptied_database_up_again(monkeypatch):
    """End to end, in one process: boot on a scratch database, empty it, and
    watch the app notice, hold pages behind the getting-ready page, and come
    back with the schema, the personas and the checkpointer's tables."""
    from banko_ai.assistant import checkpoint
    from banko_ai.config.settings import reset_config
    from banko_ai.db.schema import schema_state
    from banko_ai.web.app import create_app

    name = "banko_rebuilt_test"
    scratch = str(make_url(DB).set(database=name).render_as_string(hide_password=False))
    admin = create_engine(DB, isolation_level="AUTOCOMMIT")

    def recreate() -> None:
        with admin.connect() as conn:
            conn.execute(text(f"DROP DATABASE IF EXISTS {name} CASCADE"))
            conn.execute(text(f"CREATE DATABASE {name}"))

    class Saver:                       # the real one is process-wide and would outlive this database
        built = 0                      # get_saver() makes the tables when it builds the saver
        setups = 0                     # and they are made again on that saver after a rebuild

        def setup(self):
            Saver.setups += 1

    saver = Saver()

    def get_saver():
        Saver.built += 1
        return saver

    monkeypatch.setattr(checkpoint, "get_saver", get_saver)
    monkeypatch.setattr(checkpoint, "keep_ttl", lambda: "left alone by this test")
    for key, value in {"DATABASE_URL": scratch, "BANKO_SKIP_AUTOSETUP": "0", "SEED_HISTORY_MONTHS": "1",
                       "SCHEMA_WATCH_SECONDS": "1", "AI_SERVICE": "ollama", "OLLAMA_HOST": "http://127.0.0.1:9",
                       "COACH_KAFKA_ENABLED": "false", "INTEGRITY_INTERVAL_SECONDS": "0"}.items():
        monkeypatch.setenv(key, value)
    reset_config()
    recreate()
    app = create_app()
    readiness = app.extensions["banko_readiness"]
    try:
        assert readiness.wait_for("schema", 120) and readiness.wait_for("personas", 120)
        assert readiness.wait_for("assistant", 120) and Saver.built >= 1 and Saver.setups == 0
        client = app.test_client()
        assert client.get("/login").status_code == 200

        # The request that finds the tables gone gets the getting-ready page,
        # not an error page: the setup is about to run again.
        from sqlalchemy.exc import ProgrammingError

        from banko_ai.web import app as app_module

        class Undefined(Exception):
            sqlstate = "42P01"

        def gone(*args, **kwargs):
            raise ProgrammingError("SELECT 1 FROM users", {}, Undefined('relation "users" does not exist'))

        with monkeypatch.context() as patch:
            patch.setattr(app_module, "get_profiles", gone)
            found = client.get("/login")
        assert found.status_code == 503 and b"set up again" in found.data

        recreate()                                                    # the chaos stage rebuilds the cluster
        assert schema_state(scratch) == "missing"
        deadline = time.monotonic() + 30
        while readiness.is_ready("schema") and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not readiness.is_ready("schema"), "the app never noticed its tables were gone"

        assert readiness.wait_for("schema", 120) and readiness.wait_for("personas", 120)
        assert schema_state(scratch) == "current"
        with create_engine(scratch).connect() as conn:
            assert {r[0] for r in conn.execute(text("SELECT username FROM users"))} == {"maya", "sam", "riley"}
        assert readiness.wait_for("assistant", 120) and Saver.setups == 1   # the checkpoint tables went too
        assert readiness.wait_for("persona_data", 300)                      # and the personas have their history back
        assert client.get("/login").status_code == 200
    finally:
        app.extensions["banko_warmup_stop"].set()
        reset_config()
        with admin.connect() as conn:
            conn.execute(text(f"DROP DATABASE IF EXISTS {name} CASCADE"))
        admin.dispose()
