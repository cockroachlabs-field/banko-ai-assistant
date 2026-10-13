"""A test client on a real app: Ollama as the serving provider (its init
never reaches for the network), no autosetup, and no Kafka consumer (the
shell profile turns Kafka on for the live demo, and a test app must never
join that consumer group)."""

_apps = []


def app_client(monkeypatch, **env):
    from banko_ai.config.settings import reset_config
    from banko_ai.web.app import create_app

    monkeypatch.setenv("BANKO_SKIP_AUTOSETUP", "1")
    env.setdefault("AI_SERVICE", "ollama")
    env.setdefault("COACH_KAFKA_ENABLED", "false")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    reset_config()
    app = create_app()
    _apps.append(app)
    readiness = app.extensions["banko_readiness"]
    assert readiness.wait_for("schema", 60)
    assert readiness.wait_for("personas", 60)
    return app.test_client()


def stop_apps() -> None:
    while _apps:
        _apps.pop().extensions["banko_warmup_stop"].set()
