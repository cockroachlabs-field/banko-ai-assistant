"""Some models take only their default temperature and refuse any other
with a 400. Banko sends one on every request, so those models could not be
used at all. Which models do is learned from the refusal, not listed: the
request goes out again without a temperature, and the model is remembered."""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from banko_ai.ai_providers import openai_temperature as T

# As the gateway relays it, nested, for invoke and for a stream.
REFUSAL = ('Error code: 400 - {"error": {"message": "{\\n  \\"error\\": {\\n    \\"message\\": \\"Unsupported value: '
           "'temperature' does not support 0.3 with this model. Only the default (1) value is supported.\\\","
           '\\n    \\"param\\": \\"temperature\\",\\n    \\"code\\": \\"unsupported_value\\"\\n  }\\n}"}}')


@pytest.fixture(autouse=True)
def _nothing_learned_yet():
    T.forget()
    yield
    T.forget()


def test_a_refusal_for_the_temperature_is_recognised_whatever_wraps_it():
    assert T.rejects_temperature(RuntimeError(REFUSAL))
    assert T.rejects_temperature(RuntimeError("Unsupported parameter: 'temperature' is not supported with this model."))
    assert not T.rejects_temperature(RuntimeError("Unsupported parameter: 'max_tokens' is not supported with this model."))
    assert not T.rejects_temperature(RuntimeError("Error code: 401 - Incorrect API key provided"))
    assert not T.rejects_temperature(RuntimeError("Rate limit reached; temperature of the queue is high"))


def _strict(monkeypatch):
    """Make the real ChatOpenAI behave like a model that takes no temperature, with no network."""
    from langchain_openai import ChatOpenAI
    sent: list[float | None] = []

    def generate(self, messages, stop=None, run_manager=None, **kwargs):
        sent.append(self.temperature)
        if self.temperature is not None:
            raise RuntimeError(REFUSAL)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="OK"))])

    def stream(self, messages, stop=None, run_manager=None, **kwargs):
        sent.append(self.temperature)
        if self.temperature is not None:
            raise RuntimeError(REFUSAL)
        yield ChatGenerationChunk(message=AIMessageChunk(content="O"))
        yield ChatGenerationChunk(message=AIMessageChunk(content="K"))

    monkeypatch.setattr(ChatOpenAI, "_generate", generate)
    monkeypatch.setattr(ChatOpenAI, "_stream", stream)
    return sent


def test_a_refused_request_goes_out_again_without_a_temperature(monkeypatch):
    sent = _strict(monkeypatch)
    llm = T.chat_model(model="luna", api_key="k", base_url=None, temperature=0.3)
    assert llm.invoke("Reply OK").content == "OK"
    assert sent == [0.3, None]
    assert T.default_only("luna") and not T.default_only("gpt-4o-mini")


def test_a_refused_stream_starts_again_without_a_temperature(monkeypatch):
    sent = _strict(monkeypatch)
    llm = T.chat_model(model="luna", api_key="k", base_url=None, temperature=0)
    assert "".join(chunk.content for chunk in llm.stream("Reply OK")) == "OK"
    assert sent == [0, None]


def test_a_model_that_refused_once_is_asked_without_a_temperature_from_then_on(monkeypatch):
    sent = _strict(monkeypatch)
    T.chat_model(model="luna", api_key="k", base_url=None, temperature=0.7).invoke("Reply OK")
    del sent[:]
    again = T.chat_model(model="luna", api_key="k", base_url=None, temperature=0.7)
    assert again.temperature is None
    again.invoke("Reply OK")
    assert sent == [None]                                                  # one request, not two
    assert T.chat_model(model="gpt-4o-mini", api_key="k", base_url=None, temperature=0.7).temperature == 0.7


def test_any_other_failure_is_not_retried(monkeypatch):
    from langchain_openai import ChatOpenAI
    calls = []

    def generate(self, messages, stop=None, run_manager=None, **kwargs):
        calls.append(self.temperature)
        raise RuntimeError("Error code: 401 - Incorrect API key provided")

    monkeypatch.setattr(ChatOpenAI, "_generate", generate)
    with pytest.raises(RuntimeError, match="401"):
        T.chat_model(model="luna", api_key="k", base_url=None, temperature=0.3).invoke("Reply OK")
    assert calls == [0.3] and not T.default_only("luna")


def test_the_model_factory_builds_openai_models_this_way(monkeypatch):
    from banko_ai.agents.llm_factory import get_llm_for_agent
    from banko_ai.config.settings import reset_config
    monkeypatch.setenv("AI_SERVICE", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    reset_config()
    sent = _strict(monkeypatch)
    assert get_llm_for_agent(0.2, model_override="luna").invoke("Reply OK").content == "OK"
    assert sent == [0.2, None]
    assert get_llm_for_agent(0.2, model_override="luna").temperature is None


def test_the_provider_asks_again_without_a_temperature():
    """The older answer path calls the SDK itself."""
    from types import SimpleNamespace

    asked = []

    def create(**kwargs):
        asked.append("temperature" in kwargs)
        if "temperature" in kwargs:
            raise RuntimeError(REFUSAL)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])

    reply = T.create_chat_completion(create, model="luna", messages=[{"role": "user", "content": "hi"}],
                                     max_tokens=1000, temperature=0.7)
    assert reply.choices[0].message.content == "OK" and asked == [True, False]
    T.create_chat_completion(create, model="luna", messages=[], max_tokens=1000, temperature=0.7)
    assert asked == [True, False, False]
