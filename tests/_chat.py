"""A model that says what the test scripted. Implements the Chat protocol
of banko_ai.ai_providers.streaming, so no test ever reaches a provider."""

from __future__ import annotations

from collections.abc import Callable, Iterator


class ScriptedChat:
    def __init__(self, *replies: str | Callable[[list[dict]], str]):
        self.replies = list(replies) or [""]
        self.calls: list[tuple[str, list[dict]]] = []

    def _next(self, messages: list[dict]) -> str:
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return reply(messages) if callable(reply) else reply

    def complete(self, messages: list[dict], *, temperature: float = 0.3) -> str:
        self.calls.append(("complete", messages))
        return self._next(messages)

    def stream(self, messages: list[dict], *, temperature: float = 0.3) -> Iterator[str]:
        self.calls.append(("stream", messages))
        text = self._next(messages)
        for i in range(0, len(text), 8):
            yield text[i:i + 8]

    def prompts(self) -> list[str]:
        """Every message content the model saw, flattened, for assertions."""
        return [m["content"] for _, msgs in self.calls for m in msgs]


def reworded(messages: list[dict], words: dict[str, str] | None = None) -> str:
    """A translation as a model gives it: the text it was sent, its words
    changed and every figure kept. The assistant does not show a translation
    that changed, added or lost a figure, so a scripted one has to keep
    them as a real one does."""
    text = messages[-1]["content"].split("\n\n", 1)[1]
    for was, now in (words or SPANISH).items():
        text = text.replace(was, now)
    return text


SPANISH = {"You spent": "Gastaste", "You went to": "Fuiste a", "Your biggest purchase": "Tu compra más grande",
           "transactions": "transacciones", "- Steady.": "- Estable."}

