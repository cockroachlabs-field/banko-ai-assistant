"""A report does not read a moment before the schema existed.

Reports read a few seconds in the past, so the nearest replica can serve
them (services/_sql.read_session, follower=True). For the first seconds of
a database's life that moment is before its tables were made, and the read
fails with "relation accounts does not exist". On a first boot that was a
dashboard that failed once. After a cluster is rebuilt under a running
Banko the pages come back within seconds, and the dashboard on screen then
failed exactly there."""

import os
import time

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url

DB = os.getenv("DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB, reason="DATABASE_URL not set")
NAME = "banko_new_schema_test"
SOMEONE = "00000000-0000-0000-0000-0000000000a1"


@pytest.fixture
def new_database(monkeypatch):
    """A database that does not exist yet, and reports that read in the past."""
    from banko_ai.config.settings import reset_config
    monkeypatch.setenv("REPORTING_FOLLOWER_READS", "true")
    reset_config()
    url = str(make_url(DB).set(database=NAME).render_as_string(hide_password=False))
    admin = create_engine(DB, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {NAME} CASCADE"))
        conn.execute(text(f"CREATE DATABASE {NAME}"))
    yield url
    with admin.connect() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {NAME} CASCADE"))
    admin.dispose()


def test_a_report_seconds_after_the_schema_was_made_reads_what_is_there_now(new_database):
    from banko_ai.db.schema import ensure_schema
    from banko_ai.services import accounts
    assert ensure_schema(new_database).changed is True
    # The very next thing a page does. Read in the past, this asked for a
    # table at a moment it did not exist yet.
    history = accounts.net_worth_history(SOMEONE, months=6, url=new_database)
    assert isinstance(history, list)


def test_reports_read_in_the_past_again_once_the_schema_has_settled(new_database, monkeypatch):
    from banko_ai.db import schema
    from banko_ai.services import _sql, accounts
    schema.ensure_schema(new_database)
    in_the_past = []
    real = _sql.follower_snapshot
    monkeypatch.setattr(_sql, "follower_snapshot", lambda conn: (in_the_past.append(1), real(conn))[1])

    accounts.net_worth_history(SOMEONE, months=6, url=new_database)
    assert in_the_past == []                                   # the schema is seconds old: the present
    assert schema.settling(new_database) is True

    later = time.monotonic() + schema.SETTLE_SECONDS + 1
    monkeypatch.setattr(schema, "_now", lambda: later)         # the same process, a little later
    assert schema.settling(new_database) is False
    # (Not run for real here: the read in the past would still be before
    # this database was made. That the statement is issued is the point.)
    monkeypatch.setattr(_sql, "follower_snapshot", lambda conn: in_the_past.append(1))
    accounts.net_worth_history(SOMEONE, months=6, url=new_database)
    assert in_the_past == [1]


def test_a_schema_that_was_already_there_changes_nothing(monkeypatch):
    """The usual boot: the fingerprint matches, nothing is applied, and
    reports read in the past from the first request on."""
    from banko_ai.db import schema
    result = schema.ensure_schema(DB)
    if result.changed:                                         # the suite's own first boot: ask again
        monkeypatch.setattr(schema, "_now", lambda: time.monotonic() + schema.SETTLE_SECONDS + 1)
        result = schema.ensure_schema(DB)
    assert result.changed is False
    monkeypatch.setattr(schema, "_now", lambda: time.monotonic() + schema.SETTLE_SECONDS + 1)
    assert schema.settling(DB) is False


def test_the_app_role_and_the_admin_name_the_same_database():
    """The schema is made with the admin URL and reports read with the app
    role's: both mean the one database."""
    from banko_ai.db import schema
    admin = "postgresql://root@localhost:26258/somewhere?sslmode=disable"
    app = "cockroachdb+psycopg://banko_app@localhost:26258/somewhere?sslmode=disable&application_name=banko"
    other = "postgresql://root@localhost:26258/elsewhere?sslmode=disable"
    schema.note_changed(admin)
    try:
        assert schema.settling(app) is True and schema.settling(other) is False
    finally:
        schema._changed_at.clear()
