"""Provider health checks: honest about credentials, and bounded in time."""

from __future__ import annotations

from unittest.mock import MagicMock

from botocore.exceptions import TokenRetrievalError

from banko_ai.ai_providers.aws_provider import AWSProvider


def _aws(monkeypatch) -> AWSProvider:
    monkeypatch.delenv("AWS_MODELS", raising=False)
    return AWSProvider({
        "access_key_id": "fake-key",
        "secret_access_key": "fake-secret",
        "region": "us-east-1",
        "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    })


class _ExpiredSSOCreds:
    def get_frozen_credentials(self):
        raise TokenRetrievalError(provider="sso", error_msg="Token has expired and refresh failed")


def test_aws_expired_sso_token_is_not_reachable(monkeypatch):
    provider = _aws(monkeypatch)
    provider.session = MagicMock()
    provider.session.get_credentials.return_value = _ExpiredSSOCreds()
    assert provider.test_connection() is False


def test_aws_no_credentials_is_not_reachable(monkeypatch):
    provider = _aws(monkeypatch)
    provider.session = MagicMock()
    provider.session.get_credentials.return_value = None
    assert provider.test_connection() is False


def test_aws_fresh_credentials_are_reachable(monkeypatch):
    provider = _aws(monkeypatch)
    creds = MagicMock()
    provider.session = MagicMock()
    provider.session.get_credentials.return_value = creds
    assert provider.test_connection() is True
    creds.get_frozen_credentials.assert_called_once()


def test_openai_check_uses_a_short_timeout_and_no_retries(monkeypatch):
    from banko_ai.ai_providers.openai_provider import OpenAIProvider

    provider = OpenAIProvider({"api_key": "sk-test", "model": "gpt-4o-mini"})
    client = MagicMock()
    provider.client = client
    assert provider.test_connection() is True
    client.with_options.assert_called_once_with(timeout=5, max_retries=0)
    client.with_options.return_value.models.list.assert_called_once()


def test_watsonx_check_bounds_the_token_call(monkeypatch):
    from banko_ai.ai_providers import watsonx_provider

    seen = {}

    def fake_post(url, **kwargs):
        seen["timeout"] = kwargs.get("timeout")
        response = MagicMock(status_code=200)
        response.json.return_value = {"access_token": "t"}
        return response

    monkeypatch.setattr(watsonx_provider.requests, "post", fake_post)
    provider = watsonx_provider.WatsonxProvider({"api_key": "k", "project_id": "p"})
    assert provider.test_connection() is True
    assert seen["timeout"] is not None and seen["timeout"] <= 5


def test_a_hanging_provider_check_cannot_hold_warmup(monkeypatch):
    """Black-holed network: the check gives up within a few seconds."""
    import time

    from banko_ai.web.app import bounded_connection_test

    class _Stuck:
        def test_connection(self):
            time.sleep(30)
            return True

    started = time.monotonic()
    assert bounded_connection_test(_Stuck(), seconds=0.5) is False
    assert time.monotonic() - started < 2
