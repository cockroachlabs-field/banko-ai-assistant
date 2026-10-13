"""Tests for /api/cdc/signals webhook receiver. Uses Flask's test client
with the HMAC secret injected via env."""

import hashlib
import hmac
import json
import os

import pytest

from banko_ai.web.app import create_app

SECRET = "test-secret-hmac-do-not-use-in-prod"


@pytest.fixture(autouse=True)
def _clean_test_signals():
    """Remove the hardcoded test signal_id before each test so reruns are
    deterministic. We use a fixed UUID in the envelope (spec verbatim), so
    leftover rows from prior runs would otherwise turn fresh inserts into
    replays."""
    url = os.getenv("DATABASE_URL")
    if not url:
        yield
        return
    try:
        from sqlalchemy import create_engine
        from sqlalchemy import text as _text
        from sqlalchemy.pool import NullPool
        eng = create_engine(url, poolclass=NullPool)
        with eng.begin() as conn:
            conn.execute(_text(
                "DELETE FROM coach_nudges WHERE signal_id = "
                "'00000000-0000-0000-0000-000000000001'"))
            conn.execute(_text(
                "DELETE FROM spending_signals WHERE signal_id = "
                "'00000000-0000-0000-0000-000000000001'"))
        eng.dispose()
    except Exception:
        pass
    yield


@pytest.fixture
def client(monkeypatch):
    from banko_ai.config.settings import reset_config

    monkeypatch.setenv("CDC_WEBHOOK_HMAC_SECRET", SECRET)
    monkeypatch.setenv("FLASK_SECRET_KEY", "test-flask-secret")
    reset_config()  # Settings are cached; reset after changing env
    app = create_app()
    app.config["TESTING"] = True
    yield app.test_client()
    app.extensions["banko_warmup_stop"].set()


def _sign(body: bytes) -> str:
    return hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


def _envelope(idempotency_key: str = "wh-1") -> dict:
    return {"payload": [{
        "after": {
            "signal_id": "00000000-0000-0000-0000-000000000001",
            "user_id":   "00000000-0000-0000-0000-000000000aaa",
            "signal_type": "budget_threshold",
            "severity": "warn",
            "payload": {"category": "dining", "pct_used": 0.82},
            "idempotency_key": idempotency_key,
        },
        "updated": "1716355200.0000000000"
    }]}


def test_webhook_rejects_missing_signature(client):
    body = json.dumps(_envelope()).encode()
    resp = client.post("/api/cdc/signals", data=body,
                       content_type="application/json")
    assert resp.status_code == 401


def test_webhook_rejects_bad_signature(client):
    body = json.dumps(_envelope()).encode()
    resp = client.post("/api/cdc/signals", data=body,
                       content_type="application/json",
                       headers={"X-Banko-Signature": "deadbeef"})
    assert resp.status_code == 401


def test_webhook_rejects_malformed_payload(client):
    body = b"{not json"
    resp = client.post("/api/cdc/signals", data=body,
                       content_type="application/json",
                       headers={"X-Banko-Signature": _sign(body)})
    assert resp.status_code == 400


def test_webhook_accepts_valid_signed_envelope(client):
    body = json.dumps(_envelope("wh-valid")).encode()
    resp = client.post("/api/cdc/signals", data=body,
                       content_type="application/json",
                       headers={"X-Banko-Signature": _sign(body),
                                "X-Idempotency-Key": "wh-valid"})
    assert resp.status_code in (200, 202)
    data = resp.get_json()
    assert data["status"] in ("queued", "delivered")


def test_webhook_returns_replayed_on_duplicate(client):
    body = json.dumps(_envelope("wh-dup")).encode()
    headers = {"X-Banko-Signature": _sign(body),
               "X-Idempotency-Key": "wh-dup"}
    first = client.post("/api/cdc/signals", data=body,
                        content_type="application/json", headers=headers)
    second = client.post("/api/cdc/signals", data=body,
                         content_type="application/json", headers=headers)
    assert first.status_code in (200, 202)
    assert second.status_code == 200
    assert second.get_json()["replayed"] is True
