"""OllamaProvider unit tests with a mocked daemon. The live smoke happens
in the provider matrix, not here."""

from unittest.mock import patch

from banko_ai.ai_providers.factory import AIProviderFactory


def _cfg():
    return {"host": "http://localhost:11434", "model": "granite3.3:8b"}


@patch("banko_ai.ai_providers.ollama_provider.requests.get")
def test_dynamic_model_discovery(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "models": [{"name": "granite3.3:8b"}, {"name": "llama3.2:latest"}]}
    p = AIProviderFactory.create_provider("ollama", _cfg())
    models = p.get_available_models()
    assert "granite3.3:8b" in models
    assert "llama3.2:latest" in models


@patch("banko_ai.ai_providers.ollama_provider.requests.post")
def test_chat_roundtrip(mock_post):
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {
        "message": {"content": "You spent $10."}}
    p = AIProviderFactory.create_provider("ollama", _cfg())
    out = p._chat([{"role": "user", "content": "hi"}])
    assert "$10" in out


@patch("banko_ai.ai_providers.ollama_provider.requests.get")
def test_connection_failure_is_clean(mock_get):
    import requests as real_requests
    mock_get.side_effect = real_requests.exceptions.ConnectionError("boom")
    p = AIProviderFactory.create_provider("ollama", _cfg())
    assert p.test_connection() is False


@patch("banko_ai.ai_providers.ollama_provider.requests.post")
def test_chat_sends_context_size_and_keep_alive(mock_post, monkeypatch):
    monkeypatch.setenv("OLLAMA_NUM_CTX", "4096")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "0")
    mock_post.return_value.status_code = 200
    mock_post.return_value.json.return_value = {"message": {"content": "ok"}}
    p = AIProviderFactory.create_provider("ollama", _cfg())
    p._chat([{"role": "user", "content": "hi"}])
    body = mock_post.call_args.kwargs["json"]
    assert body["options"] == {"num_ctx": 4096}
    assert body["keep_alive"] == 0


def _tags(*names):
    return {"models": [{"name": n} for n in names]}


@patch("banko_ai.ai_providers.ollama_provider.requests.get")
def test_connection_fails_when_the_model_is_missing(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = _tags("llama3.2:latest")
    p = AIProviderFactory.create_provider("ollama", _cfg())
    assert p.test_connection() is False


@patch("banko_ai.ai_providers.ollama_provider.requests.get")
def test_connection_matches_the_latest_tag_both_ways(mock_get):
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = _tags("granite3.3:8b", "llama3.2:latest")
    assert AIProviderFactory.create_provider("ollama", _cfg()).test_connection() is True
    bare = {"host": "http://localhost:11434", "model": "llama3.2"}
    assert AIProviderFactory.create_provider("ollama", bare).test_connection() is True


def test_agent_llm_uses_native_ollama_with_the_same_context(monkeypatch):
    """The agent path must load the model with the same num_ctx as the
    provider, or Ollama reloads it whenever traffic switches paths."""
    from langchain_ollama import ChatOllama

    from banko_ai.agents.llm_factory import get_llm_for_agent

    monkeypatch.setenv("AI_SERVICE", "ollama")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "8192")
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "5m")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    llm = get_llm_for_agent(temperature=0.2)
    assert isinstance(llm, ChatOllama)
    assert llm.num_ctx == 8192
    assert llm.keep_alive == "5m"
    assert llm.base_url == "http://localhost:11434"
