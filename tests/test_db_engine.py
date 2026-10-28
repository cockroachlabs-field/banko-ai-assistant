"""The shared engine, the retry policy and the user context."""

import os

import psycopg
import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from banko_ai.db import engine as E

URL = os.getenv("DATABASE_URL")


def _wrapped(exc: Exception) -> OperationalError:
    """The SQLAlchemy wrapper a driver error arrives in."""
    return OperationalError("SELECT 1", {}, exc)


def test_one_engine_per_url_however_it_is_spelled():
    a = E.get_engine("postgresql://root@localhost:26258/defaultdb?sslmode=disable")
    b = E.get_engine("cockroachdb://root@localhost:26258/defaultdb?sslmode=disable")
    assert a is b


def test_mask_url_hides_the_password():
    masked = E.mask_url("postgresql://root:s3cret@db.example.com:26257/defaultdb")
    assert "s3cret" not in masked
    assert "db.example.com:26257" in masked


def test_serialization_failures_are_always_retryable():
    err = _wrapped(psycopg.errors.SerializationFailure("restart transaction"))
    assert E.is_retryable(err, idempotent=False)


def test_ambiguous_results_only_retry_idempotent_work():
    err = _wrapped(psycopg.errors.StatementCompletionUnknown("result is ambiguous"))
    assert E.is_retryable(err, idempotent=True)
    assert not E.is_retryable(err, idempotent=False)


def test_constraint_violations_are_never_retried():
    err = _wrapped(psycopg.errors.UniqueViolation("duplicate key value"))
    assert not E.is_retryable(err, idempotent=True)


def test_run_transaction_retries_then_succeeds(monkeypatch):
    class FakeConn:
        def execute(self, *args, **kwargs):
            return None

    class FakeBegin:
        def __enter__(self):
            return FakeConn()

        def __exit__(self, *exc):
            return False

    class FakeEngine:
        def begin(self):
            return FakeBegin()

    monkeypatch.setattr(E, "get_engine", lambda url=None: FakeEngine())
    monkeypatch.setattr(E.time, "sleep", lambda seconds: None)
    calls = {"n": 0}
    before = E.stats.snapshot()["retries"]

    def work(conn):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _wrapped(psycopg.errors.SerializationFailure("restart transaction"))
        return "ok"

    assert E.run_transaction(work) == "ok"
    assert calls["n"] == 3
    assert E.stats.snapshot()["retries"] == before + 2


def test_run_transaction_gives_up_after_max_attempts(monkeypatch):
    class FakeBegin:
        def __enter__(self):
            raise _wrapped(psycopg.errors.SerializationFailure("restart transaction"))

        def __exit__(self, *exc):
            return False

    class FakeEngine:
        def begin(self):
            return FakeBegin()

    monkeypatch.setattr(E, "get_engine", lambda url=None: FakeEngine())
    monkeypatch.setattr(E.time, "sleep", lambda seconds: None)
    with pytest.raises(OperationalError):
        E.run_transaction(lambda conn: None, max_attempts=2)


@pytest.mark.skipif(not URL, reason="DATABASE_URL not set")
def test_user_session_sets_the_user_context():
    with E.user_session("00000000-0000-0000-0000-0000000000a1") as conn:
        value = conn.execute(text("SELECT current_setting('banko.user_id', true)")).scalar()
    assert value == "00000000-0000-0000-0000-0000000000a1"


class _FakeConn:
    def execute(self, *args, **kwargs):
        return None


def _engine_with(begin):
    class FakeEngine:
        def begin(self):
            return begin()
    return FakeEngine()


def test_a_connection_lost_before_commit_is_retried_even_for_non_idempotent_work(monkeypatch):
    """Nothing commits before COMMIT is sent, so work whose connection died
    while its statements ran is always safe to run again."""
    class FakeBegin:
        def __enter__(self):
            return _FakeConn()

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(E, "get_engine", lambda url=None: _engine_with(FakeBegin))
    monkeypatch.setattr(E.time, "sleep", lambda seconds: None)
    calls = {"n": 0}

    def work(conn):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _wrapped(psycopg.errors.AdminShutdown("server is shutting down"))
        return "ok"

    assert E.run_transaction(work, idempotent=False) == "ok"
    assert calls["n"] == 2


def test_a_failed_commit_is_marked_and_not_retried_for_non_idempotent_work(monkeypatch):
    """A connection lost during COMMIT may or may not have committed: the
    error says where it happened and non-idempotent work does not run again."""
    class FakeBegin:
        def __enter__(self):
            return _FakeConn()

        def __exit__(self, exc_type, exc, tb):
            if exc_type is None:
                raise _wrapped(psycopg.errors.AdminShutdown("server is shutting down"))
            return False

    monkeypatch.setattr(E, "get_engine", lambda url=None: _engine_with(FakeBegin))
    monkeypatch.setattr(E.time, "sleep", lambda seconds: None)
    calls = {"n": 0}

    def work(conn):
        calls["n"] += 1
        return "ok"

    with pytest.raises(OperationalError) as err:
        E.run_transaction(work, idempotent=False)
    assert calls["n"] == 1 and err.value.at_commit is True
