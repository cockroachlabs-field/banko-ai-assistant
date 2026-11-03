"""Tests for /health/coach. Uses Flask test client; doesn't require LLM."""

import os

import pytest

from banko_ai.web.app import create_app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("FLASK_SECRET_KEY", "test")
    app = create_app()
    app.config["TESTING"] = True
    yield app.test_client()
    app.extensions["banko_warmup_stop"].set()


def test_health_coach_returns_200_with_status_keys(client):
    resp = client.get("/health/coach")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "status" in data
    assert "components" in data
    assert "webhook_secret_configured" in data["components"]
    assert "kafka_enabled" in data["components"]
    assert "last_nudge_at" in data["components"]


def test_health_coach_reports_webhook_secret_state(client, monkeypatch):
    from banko_ai.config.settings import reset_config

    monkeypatch.setenv("CDC_WEBHOOK_HMAC_SECRET", "")
    reset_config()  # Settings are cached; reset after changing env
    resp = client.get("/health/coach")
    data = resp.get_json()
    assert data["components"]["webhook_secret_configured"] is False

    monkeypatch.setenv("CDC_WEBHOOK_HMAC_SECRET", "real-secret")
    reset_config()  # Settings are cached; reset after changing env
    resp = client.get("/health/coach")
    data = resp.get_json()
    assert data["components"]["webhook_secret_configured"] is True
