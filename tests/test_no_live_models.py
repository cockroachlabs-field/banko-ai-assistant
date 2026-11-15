"""Tests never reach a real model on their own. The suite runs the way CI
does, whatever the shell exports: OpenAI as the provider with no key, no
other provider setting, and the local Ollama out of reach (tests/_env.py,
applied by conftest.py before anything reads the settings).

It used to run with the shell's provider. With one configured and its key
exported, a run of the suite sent 21 chat calls, about 10,800 tokens, and
20 model-list requests to it: older script-style files that build a model
themselves, the coach behind the webhook tests, and every app a test booted
checking that its provider answered. BANKO_LIVE_MODEL_TESTS=1 leaves the
shell alone, for the files that need a real model."""

import os

import pytest

from banko_ai.config.settings import Settings, field_meta
from tests._env import NOWHERE, OPT_IN, without_providers


def test_a_shell_full_of_provider_settings_reaches_no_test():
    shell = {
        "AI_SERVICE": "watsonx", "WATSONX_API_KEY": "key", "WATSONX_PROJECT_ID": "project",
        "WATSONX_URL": "https://eu-de.ml.cloud.ibm.com",          # an old name that still works
        "OPENAI_API_KEY": "sk-key", "OPENAI_BASE_URL": "http://localhost:1234/v1",  # a keyless local endpoint
        "openai_model": "gpt-4o",                                  # settings read names in any case
        "GEMINI_API_KEY": "key", "GOOGLE_APPLICATION_CREDENTIALS": "/home/me/adc.json",
        "AWS_PROFILE": "work", "AWS_ACCESS_KEY_ID": "id", "AWS_SECRET_ACCESS_KEY": "secret",
        "OLLAMA_HOST": "http://127.0.0.1:11434",                  # needs no key at all
        "BANKO_CONFIG": "/home/me/banko.yaml",                    # a file can carry any of the above
        "DATABASE_URL": "postgresql://root@localhost:26258/defaultdb", "CDC_WEBHOOK_HMAC_SECRET": "dev",
    }
    without_providers(shell)
    assert shell == {"AI_SERVICE": "openai", "OLLAMA_HOST": NOWHERE,
                     "DATABASE_URL": "postgresql://root@localhost:26258/defaultdb", "CDC_WEBHOOK_HMAC_SECRET": "dev"}


def test_every_provider_setting_is_covered_as_new_ones_are_added():
    """The list is read from the settings themselves, so a provider setting
    added later is removed too, without anyone remembering this file."""
    shell = {name: "set" for field in Settings.model_fields.values()
             if field_meta(field).group == "ai" for name in (field_meta(field).env, *field_meta(field).aliases)}
    assert len(shell) > 25  # the scan found the provider settings
    without_providers(shell)
    assert shell == {"AI_SERVICE": "openai", "OLLAMA_HOST": NOWHERE}


def test_asking_for_live_models_leaves_the_shell_alone():
    shell = {OPT_IN: "1", "AI_SERVICE": "watsonx", "WATSONX_API_KEY": "key", "OLLAMA_HOST": "http://127.0.0.1:11434"}
    before = dict(shell)
    without_providers(shell)
    assert shell == before


@pytest.mark.skipif(os.environ.get(OPT_IN) == "1", reason="this run asked for live models")
def test_this_run_sees_no_provider():
    """conftest.py applied it to this process: what an app built in a test would read."""
    cfg = Settings()
    assert cfg.ai_service == "openai" and cfg.ollama_host == NOWHERE and cfg.openai_base_url == ""
    credentials = [name for name, field in Settings.model_fields.items()
                   if field_meta(field).group == "ai" and field_meta(field).secret and getattr(cfg, name)]
    assert credentials == []
    assert not cfg.watsonx_project_id and not cfg.aws_profile and not cfg.google_application_credentials
