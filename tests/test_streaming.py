"""Token streaming with a non-streaming fallback, on whatever chat model
the active provider builds."""

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from banko_ai.agents import llm_factory
from banko_ai.ai_providers import streaming as S
from tests._chat import ScriptedChat


class StreamingLLM:
    def __init__(self, pieces):
        self.pieces = pieces
        self.seen = None

    def stream(self, messages):
        self.seen = messages
        for p in self.pieces:
            yield AIMessageChunk(content=p)

    def invoke(self, messages):
        return AIMessage(content="".join(self.pieces))


class NoStreamLLM:
    def stream(self, messages):
        raise NotImplementedError

    def invoke(self, messages):
        return AIMessage(content="whole answer")


class BrokenMidStream:
    def stream(self, messages):
        yield AIMessageChunk(content="half ")
        raise RuntimeError("connection reset")

    def invoke(self, messages):
        raise AssertionError("must not retry after tokens went out")


def _use(monkeypatch, llm, seen=None):
    def fake(temperature=0.7, model_override=None):
        if seen is not None:
            seen.append((temperature, model_override))
        return llm
    monkeypatch.setattr(llm_factory, "get_llm_for_agent", fake)


def test_stream_yields_tokens_and_maps_roles(monkeypatch):
    llm = StreamingLLM(["Hel", "lo"])
    seen = []
    _use(monkeypatch, llm, seen)
    out = list(S.ProviderChat(model="m1").stream(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
         {"role": "assistant", "content": "a"}], temperature=0.2))
    assert out == ["Hel", "lo"]
    assert [type(m).__name__ for m in llm.seen] == ["SystemMessage", "HumanMessage", "AIMessage"]
    assert seen == [(0.2, "m1")]


def test_stream_falls_back_to_invoke(monkeypatch):
    _use(monkeypatch, NoStreamLLM())
    assert list(S.ProviderChat().stream([{"role": "user", "content": "q"}])) == ["whole answer"]


def test_failure_after_tokens_is_raised_not_retried(monkeypatch):
    _use(monkeypatch, BrokenMidStream())
    got = []
    with pytest.raises(RuntimeError):
        for piece in S.ProviderChat().stream([{"role": "user", "content": "q"}]):
            got.append(piece)
    assert got == ["half "]


def test_complete_returns_stripped_text(monkeypatch):
    _use(monkeypatch, StreamingLLM(["  done  "]))
    assert S.ProviderChat().complete([{"role": "user", "content": "q"}]) == "done"


def test_chunk_text_handles_part_lists():
    assert S.chunk_text(AIMessageChunk(content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])) == "ab"
    assert S.chunk_text("plain") == "plain"


def test_scripted_chat_repeats_its_last_reply():
    chat = ScriptedChat("one", lambda msgs: msgs[-1]["content"].upper())
    assert chat.complete([{"role": "user", "content": "x"}]) == "one"
    assert "".join(chat.stream([{"role": "user", "content": "hey"}])) == "HEY"
    assert "".join(chat.stream([{"role": "user", "content": "again"}])) == "AGAIN"
