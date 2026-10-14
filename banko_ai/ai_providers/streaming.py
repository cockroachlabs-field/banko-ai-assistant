"""Token streaming for every provider.

Each provider already has a LangChain chat model (llm_factory builds
ChatWatsonx, ChatOpenAI, ChatBedrock, ChatGoogleGenerativeAI or ChatOllama),
and all of them implement stream(). A model that cannot stream, or fails
before its first token, gets one plain call instead. A failure after tokens
went out is raised: retrying would repeat half an answer.

Token counts come from the provider when LangChain hands them over
(usage_metadata on the reply, or on stream chunks); TimedChat estimates
the calls that report nothing and says so."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any, Protocol

log = logging.getLogger("banko.assistant")


class Chat(Protocol):
    def stream(self, messages: list[dict], *, temperature: float = 0.3) -> Iterator[str]: ...

    def complete(self, messages: list[dict], *, temperature: float = 0.3) -> str: ...


def chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # Gemini and Bedrock send content parts
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content or "")


def _usage(message: Any) -> dict | None:
    """The tokens a provider reported on one LangChain message or chunk."""
    meta = getattr(message, "usage_metadata", None)
    if not isinstance(meta, dict):
        return None
    try:
        return {"input": int(meta.get("input_tokens") or 0), "output": int(meta.get("output_tokens") or 0)}
    except (TypeError, ValueError):
        return None


def _plus(total: dict | None, more: dict | None) -> dict | None:
    """Chunk usage adds up, the way LangChain merges chunks: a provider that
    reports on the last chunk only and one that reports a share per chunk
    both come out right."""
    if more is None:
        return total
    if total is None:
        return dict(more)
    return {"input": total["input"] + more["input"], "output": total["output"] + more["output"]}


def _lc(messages: list[dict]) -> list:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    kinds = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
    return [kinds.get(str(m.get("role", "")), HumanMessage)(content=m.get("content", "")) for m in messages]


class ProviderChat:
    """The active provider's chat model, with the model picked in Settings."""

    def __init__(self, model: str | None = None):
        self.model = model
        # What the provider reported for the latest call, {"input", "output"}
        # tokens, or None when it reported nothing.
        self.last_usage: dict | None = None

    def _llm(self, temperature: float):
        from ..agents import llm_factory
        return llm_factory.get_llm_for_agent(temperature=temperature, model_override=self.model)

    def complete(self, messages: list[dict], *, temperature: float = 0.3) -> str:
        self.last_usage = None
        reply = self._llm(temperature).invoke(_lc(messages))
        self.last_usage = _usage(reply)
        return chunk_text(reply).strip()

    def stream(self, messages: list[dict], *, temperature: float = 0.3) -> Iterator[str]:
        self.last_usage = None
        llm = self._llm(temperature)
        produced, usage = False, None
        try:
            for chunk in llm.stream(_lc(messages)):
                usage = _plus(usage, _usage(chunk))
                piece = chunk_text(chunk)
                if piece:
                    produced = True
                    yield piece
        except Exception:
            if produced:
                self.last_usage = usage
                raise
        if produced:
            self.last_usage = usage
            return
        reply = llm.invoke(_lc(messages))
        self.last_usage = _usage(reply)
        text = chunk_text(reply).strip()
        if text:
            yield text


class TimedChat:
    """A turn's model, counted, timed and metered, so the trace names the
    model only when the turn really called it, with the tokens it spent."""

    def __init__(self, chat: Chat):
        self.chat = chat
        self.calls = 0
        self.failed = 0
        self.ms = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.estimated = False  # at least one call reported no usage, so part of the count is a guess

    def _ran(self, t0: float, ok: bool) -> None:
        self.calls += 1
        self.failed += 0 if ok else 1
        self.ms += (time.perf_counter() - t0) * 1000

    def _metered(self, messages: list[dict], output: str) -> None:
        try:
            usage = getattr(self.chat, "last_usage", None)
            if isinstance(usage, dict) and (usage.get("input") or usage.get("output")):
                self.input_tokens += int(usage.get("input") or 0)
                self.output_tokens += int(usage.get("output") or 0)
                return
            from ..services.caches import estimate_tokens
            self.input_tokens += estimate_tokens("\n".join(str(m.get("content") or "") for m in messages))
            self.output_tokens += estimate_tokens(output)
            self.estimated = True
        except Exception:  # a meter that fails never costs the answer
            log.debug("token count skipped", exc_info=True)

    def complete(self, messages: list[dict], *, temperature: float = 0.3) -> str:
        t0, ok = time.perf_counter(), False
        try:
            text_out = self.chat.complete(messages, temperature=temperature)
            ok = True
            self._metered(messages, text_out)
            return text_out
        finally:
            self._ran(t0, ok)

    def stream(self, messages: list[dict], *, temperature: float = 0.3) -> Iterator[str]:
        t0, ok = time.perf_counter(), False
        parts: list[str] = []
        try:
            for piece in self.chat.stream(messages, temperature=temperature):
                parts.append(piece)
                yield piece
            ok = True
            self._metered(messages, "".join(parts))
        finally:
            self._ran(t0, ok)
