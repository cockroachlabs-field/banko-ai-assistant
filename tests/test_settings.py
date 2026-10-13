"""Settings: one typed model, env over file over defaults, old names still
honored with a warning, and bad persona configuration rejected at startup."""

import logging
import os

import pytest
from click.testing import CliRunner
from pydantic import ValidationError

from banko_ai.config import settings as S


def _all_env_names() -> set[str]:
    names = set(S.REMOVED_SETTINGS) | {"BANKO_CONFIG"}
    for field in S.Settings.model_fields.values():
        extra = field.json_schema_extra or {}
        names.add(extra["env"])
        names.update(extra.get("aliases", []))
    return names


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    names = _all_env_names()
    for key in list(os.environ):
        if key.upper() in names:
            monkeypatch.delenv(key, raising=False)
    S.reset_config()
    S._warned.clear()
    yield
    S.reset_config()


def test_defaults_match_the_documented_values():
    cfg = S.Settings()
    assert cfg.ai_service == "watsonx"
    assert cfg.db_pool_size == 20
    assert cfg.embedding_device == "cpu"
    assert cfg.port == 5000
    assert cfg.ai_display_provider == ""


def test_environment_overrides_defaults(monkeypatch):
    monkeypatch.setenv("AI_SERVICE", "Ollama")
    monkeypatch.setenv("DB_POOL_SIZE", "7")
    cfg = S.Settings()
    assert cfg.ai_service == "ollama"
    assert cfg.db_pool_size == 7


def test_bedrock_is_accepted_as_aws(monkeypatch):
    monkeypatch.setenv("AI_SERVICE", "bedrock")
    assert S.Settings().ai_service == "aws"


def test_yaml_file_is_read_and_env_wins(tmp_path, monkeypatch):
    config_file = tmp_path / "banko.yaml"
    config_file.write_text("ai_service: openai\nollama_model: llama3.2:3b\nPORT: 6000\n")
    monkeypatch.setenv("BANKO_CONFIG", str(config_file))
    monkeypatch.setenv("AI_SERVICE", "ollama")
    cfg = S.Settings()
    assert cfg.ai_service == "ollama"          # env beats the file
    assert cfg.ollama_model == "llama3.2:3b"   # file beats the default
    assert cfg.port == 6000                    # env-style keys work in the file too


def test_deprecated_aliases_warn_and_apply(monkeypatch, caplog):
    monkeypatch.setenv("WATSONX_MODEL_ID", "ibm/granite-3-3-8b-instruct")
    monkeypatch.setenv("CACHE_THRESHOLD", "0.8")
    with caplog.at_level(logging.WARNING, logger="banko.config"):
        cfg = S.get_config()
        S.reset_config()
        S.get_config()
    assert cfg.watsonx_model == "ibm/granite-3-3-8b-instruct"
    assert cfg.cache_similarity_threshold == 0.8
    messages = [r.getMessage() for r in caplog.records]
    assert sum("WATSONX_MODEL_ID is deprecated" in m for m in messages) == 1


def test_new_name_beats_old_name(monkeypatch):
    monkeypatch.setenv("WATSONX_MODEL", "new/model")
    monkeypatch.setenv("WATSONX_MODEL_ID", "old/model")
    assert S.Settings().watsonx_model == "new/model"


def test_removed_settings_warn_once(monkeypatch, caplog):
    monkeypatch.setenv("DATABASE_HOST", "db.example.com")
    with caplog.at_level(logging.WARNING, logger="banko.config"):
        S.get_config()
        S.reset_config()
        S.get_config()
    messages = [r.getMessage() for r in caplog.records]
    assert sum("DATABASE_HOST is no longer used" in m for m in messages) == 1


def test_persona_settings_validate(monkeypatch):
    monkeypatch.setenv("AI_DISPLAY_PROVIDER", "ibm")
    monkeypatch.setenv("AI_DISPLAY_MODEL", "ibm/granite-3-3-8b-instruct")
    with pytest.raises(ValidationError, match="AI_DISPLAY_PROVIDER"):
        S.Settings()
    monkeypatch.delenv("AI_DISPLAY_PROVIDER")
    with pytest.raises(ValidationError, match="AI_DISPLAY_PROVIDER"):
        S.Settings()
    monkeypatch.delenv("AI_DISPLAY_MODEL")
    monkeypatch.setenv("AI_DISPLAY_PROVIDER", "watsonx")
    with pytest.raises(ValidationError, match="AI_DISPLAY_MODEL"):
        S.Settings()
    monkeypatch.setenv("AI_DISPLAY_MODEL", "ibm/granite-3-3-8b-instruct")
    assert S.Settings().ai_display_provider == "watsonx"


def test_unknown_ai_service_is_rejected(monkeypatch):
    monkeypatch.setenv("AI_SERVICE", "skynet")
    with pytest.raises(ValidationError, match="AI_SERVICE"):
        S.Settings()


def test_watsonx_base_url_works_for_chat_and_bare_urls(monkeypatch):
    assert S.Settings().watsonx_base_url == "https://us-south.ml.cloud.ibm.com"
    monkeypatch.setenv("WATSONX_URL", "https://eu-de.ml.cloud.ibm.com")
    cfg = S.Settings()
    assert cfg.watsonx_base_url == "https://eu-de.ml.cloud.ibm.com"
    assert cfg.watsonx_api_url.startswith("https://eu-de.ml.cloud.ibm.com/ml/v1/text/chat")


def test_get_ai_config_shape_is_unchanged():
    ai = S.Settings().get_ai_config()
    assert set(ai) == {"service", "openai", "aws", "ollama", "watsonx", "gemini"}
    assert set(ai["openai"]) == {"api_key", "base_url", "model"}
    assert set(ai["aws"]) == {"access_key_id", "secret_access_key", "profile_name", "region", "model"}
    assert set(ai["ollama"]) == {"host", "model"}
    assert set(ai["watsonx"]) == {"api_key", "project_id", "model"}
    assert set(ai["gemini"]) == {"project_id", "location", "model"}


def test_describe_masks_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-very-secret")
    monkeypatch.setenv("DATABASE_URL", "postgresql://banko:pa55word@db.example.com:26257/defaultdb")
    info = {i.env: i for i in S.Settings().describe()}
    assert "sk-very-secret" not in info["OPENAI_API_KEY"].value
    assert info["OPENAI_API_KEY"].source == "env"
    assert "pa55word" not in info["DATABASE_URL"].value
    assert "db.example.com" in info["DATABASE_URL"].value
    assert info["PORT"].source == "default"


def test_config_docs_cover_every_setting():
    docs = S.render_config_docs()
    for field in S.Settings.model_fields.values():
        assert f"`{field.json_schema_extra['env']}`" in docs
    assert "—" not in docs


def test_cli_config_command():
    from banko_ai.cli import cli

    result = CliRunner().invoke(cli, ["config"])
    assert result.exit_code == 0, result.output
    assert "AI_SERVICE" in result.output
    docs = CliRunner().invoke(cli, ["config", "--docs"])
    assert docs.exit_code == 0 and "| `AI_SERVICE` |" in docs.output
