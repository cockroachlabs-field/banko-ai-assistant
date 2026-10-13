"""Token streaming for every provider.

Each provider already has a LangChain chat model (llm_factory builds
ChatWatsonx, ChatOpenAI, ChatBedrock, ChatGoogleGenerativeAI or ChatOllama),
and all of them implement stream(). A model that cannot stream, or fails
before its first token, gets one plain call instead. A failure after tokens
went out is raised: retrying would repeat half an answer."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol


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


def _lc(messages: list[dict]) -> list:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    kinds = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
    return [kinds.get(m.get("role"), HumanMessage)(content=m.get("content", "")) for m in messages]


class ProviderChat:
    """The active provider's chat model, with the model picked in Settings."""

    def __init__(self, model: str | None = None):
        self.model = model

    def _llm(self, temperature: float):
        from ..agents import llm_factory
        return llm_factory.get_llm_for_agent(temperature=temperature, model_override=self.model)

    def complete(self, messages: list[dict], *, temperature: float = 0.3) -> str:
        return chunk_text(self._llm(temperature).invoke(_lc(messages))).strip()

    def stream(self, messages: list[dict], *, temperature: float = 0.3) -> Iterator[str]:
        llm = self._llm(temperature)
        produced = False
        try:
            for chunk in llm.stream(_lc(messages)):
                piece = chunk_text(chunk)
                if piece:
                    produced = True
                    yield piece
        except Exception:
            if produced:
                raise
        if not produced:
            text = chunk_text(llm.invoke(_lc(messages))).strip()
            if text:
                yield text
