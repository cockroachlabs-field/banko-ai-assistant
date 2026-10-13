"""AI provider factory.

Provider modules are imported only when that provider is created. Importing
all five pulled in google-genai, openai, boto3 and the IBM SDK on every boot,
about a second of import time for SDKs that one run never uses.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from .base import AIProvider, AIProviderError

_BUILTIN: dict[str, str] = {
    "openai": "banko_ai.ai_providers.openai_provider:OpenAIProvider",
    "aws": "banko_ai.ai_providers.aws_provider:AWSProvider",
    "watsonx": "banko_ai.ai_providers.watsonx_provider:WatsonxProvider",
    "gemini": "banko_ai.ai_providers.gemini_provider:GeminiProvider",
    "ollama": "banko_ai.ai_providers.ollama_provider:OllamaProvider",
}


class AIProviderFactory:
    """Factory for creating AI provider instances."""

    _registered: dict[str, type[AIProvider]] = {}

    @classmethod
    def _resolve(cls, name: str) -> type[AIProvider]:
        if name in cls._registered:
            return cls._registered[name]
        module_name, _, class_name = _BUILTIN[name].partition(":")
        return getattr(importlib.import_module(module_name), class_name)

    @classmethod
    def create_provider(cls, service_name: str, config: dict[str, Any], cache_manager=None) -> AIProvider:
        service_name = service_name.lower()
        available = cls.get_available_providers()
        if service_name not in available:
            raise AIProviderError(f"Unsupported AI service: {service_name}. Available services: {', '.join(available)}")
        try:
            provider_class = cls._resolve(service_name)
        except ImportError as e:
            raise AIProviderError(f"{service_name} provider is unavailable: {e}")
        try:
            if "cache_manager" in inspect.signature(provider_class.__init__).parameters:
                return provider_class(config, cache_manager)
            return provider_class(config)
        except Exception as e:
            raise AIProviderError(f"Failed to create {service_name} provider: {e}")

    @classmethod
    def get_available_providers(cls) -> list[str]:
        return list(_BUILTIN) + [name for name in cls._registered if name not in _BUILTIN]

    @classmethod
    def register_provider(cls, name: str, provider_class: type[AIProvider]) -> None:
        cls._registered[name.lower()] = provider_class
