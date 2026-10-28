"""The app measures its own requests and database retries for the Platform view."""

import os
import time

import psycopg
import pytest
from sqlalchemy.exc import OperationalError

from banko_ai.db import engine as E
from banko_ai.observability.metrics import metrics
from tests._app import app_client

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


def test_requests_and_server_errors_are_measured(monkeypatch):
    client = app_client(monkeypatch)
    client.application.add_url_rule("/api/_boom", "_boom", lambda: 1 / 0)
    metrics.clear("http")
    client.get("/api/health/live")
    client.get("/api/_boom")
    s = metrics.summary("http", window_s=60, now=time.time() + 1)
    assert s["ok"] >= 1 and s["errors"] == 1 and s["error_classes"] == {"500": 1}
    assert s["p50"] is not None


def test_static_files_are_not_measured(monkeypatch):
    client = app_client(monkeypatch)
    metrics.clear("http")
    client.get("/static/css/app.css")
    assert metrics.summary("http", window_s=60, now=time.time() + 1)["ok"] == 0


class _FakeConn:
    def execute(self, *args, **kwargs):
        return None


class _FakeBegin:
    def __enter__(self):
        return _FakeConn()

    def __exit__(self, *exc):
        return False


class _FakeEngine:
    def begin(self):
        return _FakeBegin()


def _restart():
    return OperationalError("SELECT 1", {}, psycopg.errors.SerializationFailure("restart transaction"))


def test_transaction_retries_and_failures_are_counted(monkeypatch):
    monkeypatch.setattr(E, "get_engine", lambda url=None: _FakeEngine())
    monkeypatch.setattr(E.time, "sleep", lambda seconds: None)
    metrics.clear("db.")
    calls = {"n": 0}

    def flaky(conn):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _restart()
        return "ok"

    assert E.run_transaction(flaky) == "ok"
    assert metrics.summary("db.retries", window_s=60, now=time.time() + 1)["ok"] == 2

    def always(conn):
        raise _restart()

    with pytest.raises(OperationalError):
        E.run_transaction(always, max_attempts=2)
    assert metrics.summary("db.failures", window_s=60, now=time.time() + 1)["ok"] == 1
