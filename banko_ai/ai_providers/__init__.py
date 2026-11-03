"""AI provider implementations for Banko AI Assistant.

Concrete providers load on first attribute access, so importing this package
does not import every vendor SDK.
"""

import importlib

from .base import AIProvider, AIProviderError
from .factory import AIProviderFactory

_LAZY = {
    "OpenAIProvider": "openai_provider",
    "AWSProvider": "aws_provider",
    "WatsonxProvider": "watsonx_provider",
    "GeminiProvider": "gemini_provider",
    "OllamaProvider": "ollama_provider",
}

__all__ = ["AIProvider", "AIProviderError", "AIProviderFactory", *_LAZY]


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
