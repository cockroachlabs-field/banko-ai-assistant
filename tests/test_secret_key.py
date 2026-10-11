"""Regression tests for SECRET_KEY handling in production vs dev."""
import pytest

from banko_ai.config.settings import Config


def _config_with_db(monkeypatch):
    """Helper: build a Config with a database_url set so validate() reaches
    the secret_key block instead of bailing early on DATABASE_URL check."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://nowhere/db")
    return Config()


def test_validate_raises_in_prod_when_secret_key_missing(monkeypatch):
    """In FLASK_ENV=production, an unset SECRET_KEY must raise at startup
    rather than silently generating a per-worker random value (which breaks
    Flask sessions across gunicorn workers)."""
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.setenv("FLASK_ENV", "production")

    cfg = _config_with_db(monkeypatch)
    with pytest.raises(RuntimeError, match="SECRET_KEY must be set"):
        cfg.validate()


def test_validate_raises_under_gunicorn_when_secret_key_missing(monkeypatch):
    """Running under gunicorn (detected via SERVER_SOFTWARE) also triggers
    the prod-mode SECRET_KEY requirement, even if FLASK_ENV isn't set."""
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("FLASK_ENV", raising=False)
    monkeypatch.setenv("SERVER_SOFTWARE", "gunicorn/23.0.0")

    cfg = _config_with_db(monkeypatch)
    with pytest.raises(RuntimeError, match="SECRET_KEY must be set"):
        cfg.validate()


def test_validate_generates_random_secret_in_dev(monkeypatch):
    """Dev mode (no FLASK_ENV=production, no gunicorn) keeps the
    random-fallback ergonomics — single-process Flask sessions still work."""
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("FLASK_ENV", raising=False)
    monkeypatch.delenv("SERVER_SOFTWARE", raising=False)

    cfg = _config_with_db(monkeypatch)
    cfg.validate()
    assert cfg.secret_key
    assert len(cfg.secret_key) >= 32
