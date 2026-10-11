"""Health endpoints and the warming-up gate."""

import os
import time

import pytest

from banko_ai.web.app import create_app

URL = os.getenv("DATABASE_URL")


def _wait_ready(client, seconds=60):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if client.get("/api/health/ready").status_code == 200:
            return True
        time.sleep(0.5)
    return False


@pytest.mark.skipif(not URL, reason="DATABASE_URL not set")
def test_health_reports_components_once_warm():
    client = create_app().test_client()
    assert client.get("/api/health/live").status_code == 200
    assert _wait_ready(client)
    body = client.get("/api/health").get_json()
    assert body["database"] == "connected"
    assert body["components"]["schema"]["state"] == "ready"
    for key in ("success", "ai_service", "current_model", "ai_provider", "ai_provider_available"):
        assert key in body


def test_app_boots_with_database_down(monkeypatch):
    monkeypatch.setenv("AI_SERVICE", "ollama")  # no provider SDK reaching for the network at init
    monkeypatch.setenv("DATABASE_URL", "postgresql://root@203.0.113.1:26257/defaultdb?sslmode=disable")
    monkeypatch.setenv("DB_CONNECT_TIMEOUT", "1")
    monkeypatch.setenv("WARMUP_WAIT_SECONDS", "1")
    started = time.time()
    app = create_app()
    assert time.time() - started < 5
    client = app.test_client()
    assert client.get("/api/health/live").status_code == 200
    assert client.get("/api/health/ready").status_code == 503
    page = client.get("/login")
    assert page.status_code == 503
    assert b"getting ready" in page.data
    assert client.get("/api/search?q=coffee").status_code == 503
