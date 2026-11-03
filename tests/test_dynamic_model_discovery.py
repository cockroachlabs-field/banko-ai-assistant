"""Tests for WatsonxProvider.get_available_models() — cache behaviour and
fallback when the foundation-model-specs API is unreachable.

Background: the receipt-upload smoke on 2026-05-21 showed that picking
an unsuitable watsonx model (the dropdown is built from the live IBM API)
produces a 500. Task #22 (Pydantic gate) handles the bad-output case;
this test pins the discovery layer so we don't regress to (a) one API
call per `/api/models` page load or (b) a one-element fallback that
hides the option to recover by switching models.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from banko_ai.ai_providers.watsonx_provider import WatsonxProvider


@pytest.fixture
def provider(monkeypatch):
    """A WatsonxProvider that never actually talks to IBM. Token fetch is
    stubbed so __init__ and the discovery call both stay offline."""
    monkeypatch.delenv("WATSONX_MODELS", raising=False)
    p = WatsonxProvider({
        "api_key": "fake-key",
        "project_id": "fake-project",
        "model": "openai/gpt-oss-120b",
    })
    monkeypatch.setattr(p, "_get_access_token", lambda: "fake-token")
    return p


def test_discovery_caches_after_first_call(provider):
    """A second call must NOT hit the network."""
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "resources": [
            {"model_id": "meta-llama/llama-3-3-70b-instruct",
             "tasks": [{"id": "chat"}]},
            {"model_id": "ibm/granite-3-1-8b-base",
             "tasks": [{"id": "generation"}]},
        ],
    }
    with patch("banko_ai.ai_providers.watsonx_provider.requests.get",
               return_value=fake_response) as mock_get:
        first = provider.get_available_models()
        second = provider.get_available_models()

    assert first == second
    assert "meta-llama/llama-3-3-70b-instruct" in first
    assert mock_get.call_count == 1, (
        "Second call should hit the cache, not the network"
    )


def test_fallback_stub_is_non_trivial_on_discovery_failure(provider):
    """If IBM is unreachable, dropdown must still offer multiple known-good
    chat models so the user can recover. A single-element stub hides the
    fix from the user."""
    with patch("banko_ai.ai_providers.watsonx_provider.requests.get",
               side_effect=RuntimeError("simulated network failure")):
        models = provider.get_available_models()

    assert isinstance(models, list)
    assert len(models) >= 2, (
        f"Fallback stub must offer multiple models; got {models}"
    )
    assert all(isinstance(m, str) and m for m in models)
    assert not any("code-instruct" in m for m in models), (
        "Fallback must not seed the dropdown with code-tuned models — those "
        "echo prompt templates on JSON extraction tasks (see receipt agent)"
    )


def test_env_override_skips_discovery(provider, monkeypatch):
    """WATSONX_MODELS env var must short-circuit the API call so operators
    can pin a known-good set in airgap/restricted environments."""
    from banko_ai.config.settings import reset_config

    monkeypatch.setenv(
        "WATSONX_MODELS",
        "meta-llama/llama-3-3-70b-instruct, ibm/granite-3-1-8b-base",
    )
    reset_config()  # Settings are cached; reset after changing env
    with patch("banko_ai.ai_providers.watsonx_provider.requests.get") as mock_get:
        models = provider.get_available_models()

    assert models == [
        "meta-llama/llama-3-3-70b-instruct",
        "ibm/granite-3-1-8b-base",
    ]
    assert mock_get.call_count == 0, (
        "Env override must bypass the network call entirely"
    )


def test_env_override_not_cached_across_changes(monkeypatch):
    """A fresh provider instance must see the current env, not a cached
    value from the prior process state. (lru_cache traps that bit teams in
    the past.)"""
    from banko_ai.config.settings import reset_config

    monkeypatch.setenv("WATSONX_MODELS", "model-a")
    reset_config()  # Settings are cached; reset after changing env
    p1 = WatsonxProvider({"api_key": "k", "project_id": "p", "model": "model-a"})
    monkeypatch.setattr(p1, "_get_access_token", lambda: "tok")
    assert p1.get_available_models() == ["model-a"]

    monkeypatch.setenv("WATSONX_MODELS", "model-b")
    reset_config()  # Settings are cached; reset after changing env
    p2 = WatsonxProvider({"api_key": "k", "project_id": "p", "model": "model-b"})
    monkeypatch.setattr(p2, "_get_access_token", lambda: "tok")
    assert p2.get_available_models() == ["model-b"]
