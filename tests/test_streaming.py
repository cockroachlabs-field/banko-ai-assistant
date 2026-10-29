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


def _usage(i, o):
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}


class UsageLLM:
    """A chat model that reports usage the way LangChain hands it over:
    usage_metadata on the invoke result, and on stream chunks (some
    providers send it on the last chunk only, others a share per chunk)."""

    def __init__(self, chunks, whole=None):
        self.chunks, self.whole = chunks, whole

    def stream(self, messages):
        yield from self.chunks

    def invoke(self, messages):
        return self.whole


def test_provider_chat_keeps_the_usage_the_provider_reports(monkeypatch):
    _use(monkeypatch, UsageLLM([AIMessageChunk(content="Hel"), AIMessageChunk(content="lo"),
                                AIMessageChunk(content="", usage_metadata=_usage(50, 7))]))
    chat = S.ProviderChat()
    assert "".join(chat.stream([{"role": "user", "content": "q"}])) == "Hello"
    assert chat.last_usage == {"input": 50, "output": 7}

    _use(monkeypatch, UsageLLM([AIMessageChunk(content="a", usage_metadata=_usage(40, 1)),
                                AIMessageChunk(content="b", usage_metadata=_usage(0, 1))]))
    assert "".join(chat.stream([{"role": "user", "content": "q"}])) == "ab"
    assert chat.last_usage == {"input": 40, "output": 2}

    _use(monkeypatch, UsageLLM([], whole=AIMessage(content=" whole ", usage_metadata=_usage(12, 3))))
    assert chat.complete([{"role": "user", "content": "q"}]) == "whole"
    assert chat.last_usage == {"input": 12, "output": 3}
    assert list(chat.stream([{"role": "user", "content": "q"}])) == ["whole"]  # the one-call fallback
    assert chat.last_usage == {"input": 12, "output": 3}

    _use(monkeypatch, StreamingLLM(["no", " usage"]))
    assert "".join(chat.stream([{"role": "user", "content": "q"}])) == "no usage"
    assert chat.last_usage is None


class ReportingChat:
    """A Chat that reports usage like ProviderChat does."""

    def __init__(self):
        self.last_usage = None

    def complete(self, messages, *, temperature=0.3):
        self.last_usage = {"input": 120, "output": 30}
        return "planned"

    def stream(self, messages, *, temperature=0.3):
        self.last_usage = None
        yield "an "
        yield "answer"
        self.last_usage = {"input": 200, "output": 2}


def test_timed_chat_adds_up_reported_tokens():
    chat = S.TimedChat(ReportingChat())
    chat.complete([{"role": "user", "content": "plan this"}])
    assert "".join(chat.stream([{"role": "user", "content": "answer this"}])) == "an answer"
    assert (chat.calls, chat.input_tokens, chat.output_tokens, chat.estimated) == (2, 320, 32, False)


def test_timed_chat_estimates_tokens_a_provider_did_not_report():
    chat = S.TimedChat(ScriptedChat("x" * 400, "y" * 80))
    chat.complete([{"role": "system", "content": "s" * 399}, {"role": "user", "content": "u" * 400}])
    assert (chat.input_tokens, chat.output_tokens, chat.estimated) == (200, 100, True)
    assert "".join(chat.stream([{"role": "user", "content": "q" * 40}])) == "y" * 80
    assert (chat.calls, chat.input_tokens, chat.output_tokens) == (2, 210, 120)


def test_one_estimated_call_marks_the_whole_count_estimated():
    reporting = ReportingChat()
    chat = S.TimedChat(reporting)
    chat.complete([{"role": "user", "content": "q"}])

    def silent(messages, *, temperature=0.3):  # this call reports nothing
        reporting.last_usage = None
        return "x" * 40
    reporting.complete = silent
    chat.complete([{"role": "user", "content": "q" * 80}])
    assert (chat.input_tokens, chat.output_tokens, chat.estimated) == (140, 40, True)


class RateLimitedStream:
    def __init__(self):
        self.invoked = 0

    def stream(self, messages):
        raise RuntimeError('Failure during chat. Status code: 429, body: {"errors":[{"code":'
                           '"consumption_limit_reached","message":"The usage limit for the current plan '
                           'has been reached"}]}')
        yield  # pragma: no cover

    def invoke(self, messages):
        self.invoked += 1
        raise AssertionError("a rate-limited request must not be sent again at once")


def test_a_rate_limited_stream_is_raised_not_sent_again(monkeypatch):
    """A provider SDK retries a 429 with backoff before it gives up (watsonx's
    for about a minute); asking again at once only doubles the wait for an
    answer that comes back the same."""
    llm = RateLimitedStream()
    _use(monkeypatch, llm)
    with pytest.raises(RuntimeError):
        list(S.ProviderChat().stream([{"role": "user", "content": "q"}]))
    assert llm.invoked == 0
