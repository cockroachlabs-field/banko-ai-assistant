"""The provider persona: run on one provider, show another everywhere a
user can see, and keep the truth where engineers look."""

import os

import pytest

from banko_ai.ai_providers import display as D
from banko_ai.config.settings import Settings


def _cfg(**env):
    return Settings(**env)


def test_no_persona_shows_the_real_provider():
    actual = D.actual_identity("ollama", "granite3.3:8b")
    assert D.display_identity(_cfg(ai_service="ollama"), actual) == actual
    assert actual.name == "Ollama"


def test_persona_replaces_name_icon_and_model():
    cfg = _cfg(ai_service="ollama", ai_display_provider="watsonx", ai_display_model="ibm/granite-3-3-8b-instruct")
    shown = D.display_identity(cfg, D.actual_identity("ollama", "granite3.3:8b"))
    assert (shown.provider, shown.name, shown.model) == ("watsonx", "IBM watsonx", "ibm/granite-3-3-8b-instruct")
    assert shown.icon_file == "watsonx-icon.svg"


def test_persona_model_list_and_selection():
    cfg = _cfg(ai_service="ollama", ai_display_provider="watsonx",
               ai_display_models="ibm/granite-3-3-8b-instruct, meta-llama/llama-3-3-70b-instruct")
    assert D.display_models(cfg, ["granite3.3:8b"]) == ["ibm/granite-3-3-8b-instruct", "meta-llama/llama-3-3-70b-instruct"]
    shown = D.display_identity(cfg, D.actual_identity("ollama", "granite3.3:8b"),
                               selected_model="meta-llama/llama-3-3-70b-instruct")
    assert shown.model == "meta-llama/llama-3-3-70b-instruct"


def test_stored_provider_labels_follow_the_persona():
    cfg = _cfg(ai_service="ollama", ai_display_provider="watsonx", ai_display_model="ibm/granite-3-3-8b-instruct")
    assert D.display_label(cfg, "ollama") == "watsonx"
    assert D.display_label(cfg, "fallback") == "fallback"
    assert D.display_label(_cfg(ai_service="ollama"), "ollama") == "ollama"


def test_neutral_errors_never_name_a_provider_or_host():
    message = D.neutral_error(RuntimeError("Cannot reach Ollama at http://localhost:11434. Is `ollama serve` running?"))
    lowered = message.lower()
    for word in ("ollama", "watsonx", "openai", "bedrock", "gemini", "localhost", "http"):
        assert word not in lowered


needs_db = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture
def persona_client(monkeypatch):
    monkeypatch.setenv("AI_SERVICE", "ollama")
    monkeypatch.setenv("AI_DISPLAY_PROVIDER", "watsonx")
    monkeypatch.setenv("AI_DISPLAY_MODEL", "ibm/granite-3-3-8b-instruct")
    from banko_ai.web.app import create_app
    from banko_ai.web.auth import UserManager
    app = create_app()
    app.extensions["banko_readiness"].wait_for("schema", 60)
    UserManager(os.environ["DATABASE_URL"]).backfill_personas()
    client = app.test_client()
    client.post("/login", data={"username": "maya"})
    yield client
    app.extensions["banko_warmup_stop"].set()


@needs_db
def test_no_page_or_api_leaks_the_real_provider(persona_client):
    for path in ("/", "/settings", "/api/models"):
        body = persona_client.get(path).get_data(as_text=True)
        assert "ollama" not in body.lower(), path
    assert "IBM watsonx" in persona_client.get("/").get_data(as_text=True)


@needs_db
def test_models_api_offers_the_persona_and_keeps_the_real_model(persona_client):
    models = persona_client.get("/api/models").get_json()
    assert models["provider"] == "watsonx"
    assert models["current"] == "ibm/granite-3-3-8b-instruct"
    assert models["models"] == ["ibm/granite-3-3-8b-instruct"]
    switched = persona_client.post("/api/models", json={"model": "ibm/granite-3-3-8b-instruct"}).get_json()
    assert switched["success"] is True


@needs_db
def test_health_keeps_the_truth_for_engineers(persona_client):
    health = persona_client.get("/api/health").get_json()
    assert health["ai_service"] == "watsonx"
    assert health["display"]["provider"] == "watsonx"
    assert health["actual"]["provider"] == "ollama"


@needs_db
def test_rag_api_metadata_shows_the_persona(monkeypatch):
    """The /api/rag endpoint must not leak the real provider in metadata."""
    from dataclasses import dataclass
    from unittest.mock import MagicMock, patch

    @dataclass
    class FakeRAGResponse:
        response: str
        sources: list
        metadata: dict

    monkeypatch.setenv("AI_SERVICE", "ollama")
    monkeypatch.setenv("AI_DISPLAY_PROVIDER", "watsonx")
    monkeypatch.setenv("AI_DISPLAY_MODEL", "ibm/granite-3-3-8b-instruct")

    from banko_ai.web.app import create_app
    from banko_ai.web.auth import UserManager

    fake_response = FakeRAGResponse(
        response="Here's your answer",
        sources=[],
        metadata={"provider": "ollama", "model": "granite3.3:8b", "cached": False}
    )

    # Patch the provider's generate_rag_response at creation time
    with patch('banko_ai.ai_providers.ollama_provider.OllamaProvider.generate_rag_response', return_value=fake_response):
        app = create_app()
        app.extensions["banko_readiness"].wait_for("schema", 60)
        UserManager(os.environ["DATABASE_URL"]).backfill_personas()

        client = app.test_client()
        client.post("/login", data={"username": "maya"})

        # Call /api/rag
        result = client.post("/api/rag", json={"query": "how much did I spend on groceries?"}).get_json()

        # The metadata must show watsonx, not ollama
        assert result["success"] is True
        assert result["metadata"]["provider"] == "watsonx"
        assert result["metadata"]["model"] == "ibm/granite-3-3-8b-instruct"
        assert "ollama" not in str(result).lower()

        app.extensions["banko_warmup_stop"].set()


@needs_db
def test_no_persona_keeps_the_real_provider(monkeypatch):
    """With no persona set, everything shows the real provider."""
    from unittest.mock import MagicMock, patch

    monkeypatch.setenv("AI_SERVICE", "ollama")
    # No AI_DISPLAY_PROVIDER set

    from banko_ai.web.app import create_app
    from banko_ai.web.auth import UserManager

    with patch('banko_ai.ai_providers.ollama_provider.OllamaProvider.set_model', return_value=True) as mock_set_model:
        app = create_app()
        app.extensions["banko_readiness"].wait_for("schema", 60)
        UserManager(os.environ["DATABASE_URL"]).backfill_personas()

        client = app.test_client()
        client.post("/login", data={"username": "maya"})

        # /api/health shows ollama everywhere
        health = client.get("/api/health").get_json()
        assert health["ai_service"] == "ollama"
        assert health["display"]["provider"] == "ollama"
        assert health["actual"]["provider"] == "ollama"
        assert health["display"] == health["actual"]

        # /api/models shows ollama
        models = client.get("/api/models").get_json()
        assert models["provider"] == "ollama"

        # POST /api/models reaches the real provider's set_model
        result = client.post("/api/models", json={"model": "granite3.3:8b"}).get_json()
        assert result["success"] is True
        mock_set_model.assert_called_once_with("granite3.3:8b")

        app.extensions["banko_warmup_stop"].set()
