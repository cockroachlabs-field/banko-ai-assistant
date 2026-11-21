"""Models that take only their default temperature.

Some models behind the OpenAI API refuse any temperature but their default
with a 400: "Unsupported value: 'temperature' does not support 0.3 with this
model. Only the default (1) value is supported." Banko sets a temperature on
every request, so those models could not be used at all.

Which models do this is not listed here, and a gateway can name them
anything. It is learned: a request refused in words about the temperature
goes out again without one, and when that one is answered the model is
remembered for the life of the process, so later requests leave it out from
the start. A model is remembered only then: an error that merely mentions
the temperature comes back the same without it, and nothing was learned."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

log = logging.getLogger("banko.ai")

_lock = threading.Lock()
_default_only: set[str] = set()
_chat_class: type | None = None


# How providers and gateways put it: "Unsupported value: 'temperature' does not
# support 0.3", "models don't support temperature=0.3. Only temperature=1 is
# supported", "`temperature` may only be set to 1", "`temperature` is
# deprecated for this model".
_REFUSALS = ("unsupported", "not support", "n't support", "only the default", "only temperature",
             "may only be", "deprecated")


def rejects_temperature(exc: BaseException) -> bool:
    """Whether this failure reads like the provider refusing the temperature,
    which makes it worth one more try without. Read from the message: a
    gateway relays the provider's error as text, and the status and parameter
    fields the SDK would carry are gone by then. A loose reading costs one
    request that fails the same way."""
    said = str(exc).lower()
    return "temperature" in said and any(phrase in said for phrase in _REFUSALS)


def default_only(model: str) -> bool:
    return model in _default_only


def remember(model: str) -> None:
    with _lock:
        new = model not in _default_only
        _default_only.add(model)
    if new:
        log.info("%s takes only its default temperature; requests to it leave the temperature out", model)


def forget() -> None:
    """Start over, as a new process would (tests)."""
    with _lock:
        _default_only.clear()


def create_chat_completion(create: Callable[..., Any], *, model: str, temperature: float, **kwargs: Any) -> Any:
    """Call the SDK's chat.completions.create with a temperature, or without
    one for a model that refuses it."""
    if default_only(model):
        return create(model=model, **kwargs)
    try:
        return create(model=model, temperature=temperature, **kwargs)
    except Exception as exc:
        if not rejects_temperature(exc):
            raise
        answer = create(model=model, **kwargs)
        remember(model)
        return answer


def _class() -> type:
    global _chat_class
    if _chat_class is None:
        from langchain_openai import ChatOpenAI

        class TemperatureOptional(ChatOpenAI):
            """ChatOpenAI that sends a request again without its temperature
            when the model refuses it."""

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[no-untyped-def]
                try:
                    return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                except Exception as exc:
                    if self.temperature is None or not rejects_temperature(exc):
                        raise
                    asked, self.temperature = self.temperature, None
                    try:
                        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                    except Exception:
                        self.temperature = asked     # it was something else
                        raise
                    remember(self.model_name)
                    return result

            def _stream(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[no-untyped-def]
                started = False
                try:
                    for chunk in super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
                        started = True
                        yield chunk
                    return
                except Exception as exc:
                    # A refusal comes before the first piece. Anything after
                    # one is a different failure, and half an answer is out.
                    if started or self.temperature is None or not rejects_temperature(exc):
                        raise
                    asked, self.temperature = self.temperature, None
                try:
                    for chunk in super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs):
                        if not started:
                            started = True
                            remember(self.model_name)
                        yield chunk
                except Exception:
                    if not started:
                        self.temperature = asked         # it was something else
                    raise

        _chat_class = TemperatureOptional
    return _chat_class


def chat_model(*, model: str, api_key: str | None, base_url: str | None, temperature: float) -> Any:
    """The LangChain chat model for an OpenAI-compatible endpoint."""
    return _class()(model=model, api_key=api_key, base_url=base_url,
                    temperature=None if default_only(model) else temperature)
