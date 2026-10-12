"""Health endpoints and the warming-up gate."""

import os
import threading
import time
import uuid

import pytest

from banko_ai.web.app import create_app

URL = os.getenv("DATABASE_URL")


@pytest.fixture
def make_app():
    """create_app, with every warm-up lane stopped at teardown so no thread
    keeps retrying a database for the rest of the session."""
    apps = []

    def _make():
        app = create_app()
        apps.append(app)
        return app

    yield _make
    for app in apps:
        app.extensions["banko_warmup_stop"].set()


def _wait_ready(client, seconds=60):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if client.get("/api/health/ready").status_code == 200:
            return True
        time.sleep(0.5)
    return False


@pytest.mark.skipif(not URL, reason="DATABASE_URL not set")
def test_health_reports_components_once_warm(make_app):
    client = make_app().test_client()
    assert client.get("/api/health/live").status_code == 200
    assert _wait_ready(client)
    body = client.get("/api/health").get_json()
    assert body["database"] == "connected"
    assert body["components"]["schema"]["state"] == "ready"
    assert body["components"]["personas"]["state"] == "ready"
    for key in ("success", "ai_service", "current_model", "ai_provider", "ai_provider_available"):
        assert key in body


def test_app_boots_with_database_down(monkeypatch, make_app):
    monkeypatch.setenv("AI_SERVICE", "ollama")  # no provider SDK reaching for the network at init
    monkeypatch.setenv("DATABASE_URL", "postgresql://root@203.0.113.1:26257/defaultdb?sslmode=disable")
    monkeypatch.setenv("DB_CONNECT_TIMEOUT", "1")
    monkeypatch.setenv("WARMUP_WAIT_SECONDS", "1")
    started = time.time()
    app = make_app()
    assert time.time() - started < 5
    client = app.test_client()
    assert client.get("/api/health/live").status_code == 200
    assert client.get("/api/health/ready").status_code == 503
    page = client.get("/login")
    assert page.status_code == 503
    assert b"getting ready" in page.data
    assert client.get("/api/search?q=coffee").status_code == 503
    health = client.get("/api/health")
    assert health.status_code == 503
    assert health.get_json()["error"].startswith("database unreachable")


@pytest.mark.skipif(not URL, reason="DATABASE_URL not set")
def test_ready_waits_for_the_personas(monkeypatch, make_app):
    """A presenter typing "maya" before the personas exist would create an
    ordinary user the backfill never repairs, so /login waits for them."""
    from banko_ai.web.auth import UserManager

    release = threading.Event()
    real_backfill = UserManager.backfill_personas

    def slow_backfill(self, *args, **kwargs):
        release.wait(30)
        return real_backfill(self, *args, **kwargs)

    monkeypatch.setattr(UserManager, "backfill_personas", slow_backfill)
    monkeypatch.setenv("WARMUP_WAIT_SECONDS", "1")
    app = make_app()
    readiness = app.extensions["banko_readiness"]
    client = app.test_client()
    try:
        assert readiness.wait_for("schema", 60)
        assert client.get("/api/health/ready").status_code == 503
        assert client.get("/login").status_code == 503
    finally:
        release.set()
    assert _wait_ready(client)
    assert client.get("/login").status_code == 200


def _wait_finished(readiness, seconds=120):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if readiness.finished():
            return True
        time.sleep(0.5)
    return False


@pytest.mark.skipif(not URL, reason="DATABASE_URL not set")
def test_unreachable_kafka_is_degraded_and_does_not_hold_warmup(monkeypatch, make_app):
    monkeypatch.setenv("AI_SERVICE", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")  # refused at once
    monkeypatch.setenv("COACH_KAFKA_ENABLED", "true")
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9")
    app = make_app()
    readiness = app.extensions["banko_readiness"]
    # Warm-up finishes while kafka is still on its first attempt: a missing
    # broker never holds the ready line.
    assert _wait_finished(readiness)
    snap = app.test_client().get("/api/health").get_json()["components"]
    assert "kafka" in snap
    assert snap["kafka"]["state"] == "degraded"
    assert not readiness.is_ready("kafka")
    # Once the first attempt fails, the detail says why.
    deadline = time.time() + 90
    while time.time() < deadline and "unreachable" not in readiness.get("kafka").detail:
        time.sleep(0.5)
    kafka = readiness.get("kafka")
    assert kafka.state == "degraded"
    assert "brokers unreachable at 127.0.0.1:9" in kafka.detail
    assert snap["coach"]["state"] == "ready"
    assert snap["coach"]["detail"] == "webhook active, kafka enabled"


@pytest.mark.skipif(not URL, reason="DATABASE_URL not set")
def test_a_loading_model_answers_503_warming_up(monkeypatch, make_app):
    from banko_ai.utils import cache_manager, embeddings

    def still_loading(*args, **kwargs):
        raise embeddings.ModelWarmingUp("embedding model all-MiniLM-L6-v2 is still loading")

    monkeypatch.setattr(cache_manager, "load_embedding_model", still_loading)
    client = make_app().test_client()
    assert _wait_ready(client)
    # A query nobody has embedded before, so the embedding cache misses.
    resp = client.post("/api/search", json={"query": f"coffee {uuid.uuid4()}"})
    assert resp.status_code == 503
    assert resp.get_json()["error"] == "warming_up"
