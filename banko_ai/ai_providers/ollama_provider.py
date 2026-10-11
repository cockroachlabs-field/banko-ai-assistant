"""Ollama provider: local models as a first-class AI_SERVICE.

Talks plain HTTP to an Ollama daemon (default http://localhost:11434), the
same requests-only pattern the watsonx provider uses. No SDK dependency.
Embeddings stay on sentence-transformers like every other provider, so the
whole app runs with the network cable unplugged.
"""

from __future__ import annotations

from typing import Any

import requests

from ..config.settings import get_config
from .base import (
    AIConnectionError,
    AIProvider,
    RAGResponse,
    SearchResult,
)
from .rag_prompts import build_banko_rag_prompt


class OllamaProvider(AIProvider):
    """Chat and RAG through a local (or LAN) Ollama daemon."""

    def __init__(self, config: dict[str, Any] = None, cache_manager=None):
        config = config or {}
        cfg = get_config()
        self.host = (config.get("host") or cfg.ollama_host).rstrip("/")
        self.current_model = config.get("model") or cfg.ollama_model
        self.cache_manager = cache_manager
        self.embedding_model_name = cfg.embedding_model
        self.timeout = cfg.ollama_timeout_seconds
        super().__init__(config)

    def _validate_config(self) -> None:
        # Nothing secret to validate; reachability is checked lazily so the
        # app can boot before the daemon (airgap compose start order).
        return

    def get_default_model(self) -> str:
        return self.current_model

    def get_provider_name(self) -> str:
        return "ollama"

    # ── daemon plumbing ────────────────────────────────────────────────

    def _chat(self, messages: list[dict[str, str]]) -> str:
        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json={"model": self.current_model, "messages": messages,
                      "stream": False},
                timeout=self.timeout)
        except requests.exceptions.ConnectionError as e:
            raise AIConnectionError(
                f"Cannot reach Ollama at {self.host}. "
                f"Is `ollama serve` running?") from e
        if resp.status_code != 200:
            raise AIConnectionError(
                f"Ollama returned {resp.status_code}: {resp.text[:200]}")
        return resp.json().get("message", {}).get("content", "")

    def get_available_models(self) -> list[str]:
        """Dynamic discovery from the daemon, never a hardcoded list."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=10)
            if resp.status_code != 200:
                return [self.current_model]
            return [m["name"] for m in resp.json().get("models", [])]
        except requests.exceptions.RequestException:
            return [self.current_model]

    def set_model(self, model_id: str) -> bool:
        self.current_model = model_id
        return True

    def test_connection(self) -> bool:
        try:
            return requests.get(f"{self.host}/api/tags",
                                timeout=5).status_code == 200
        except requests.exceptions.RequestException:
            return False

    # ── embeddings (local, same as every provider) ─────────────────────

    def generate_embedding(self, text: str) -> list[float]:
        try:
            from banko_ai.utils.embeddings import load_embedding_model
            model = load_embedding_model(self.embedding_model_name)
            return model.encode([text])[0].tolist()
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return []

    # ── search (delegates to the shared engine) ────────────────────────

    def search_expenses(self, query: str, user_id: str | None = None,
                        limit: int = 10) -> list[SearchResult]:
        from ..vector_search.search import VectorSearchEngine
        cfg = get_config()
        engine = VectorSearchEngine(cfg.database_url,
                                    cache_manager=self.cache_manager)
        results = engine.search_expenses(query=query, user_id=user_id,
                                         limit=limit)
        return [SearchResult(
            expense_id=r.expense_id, user_id=r.user_id,
            description=r.description, merchant=r.merchant,
            amount=r.amount, date=r.date,
            similarity_score=r.similarity_score, metadata=r.metadata)
            for r in results]

    # ── RAG ────────────────────────────────────────────────────────────

    def simple_rag_response(self, prompt: str,
                            search_results: list[dict[str, Any]],
                            language: str = "English") -> str:
        print("\n🦙 OLLAMA RAG:")
        print(f"1. Query: '{prompt[:60]}...' (model={self.current_model})")
        if self.cache_manager:
            cached = self.cache_manager.get_cached_response(
                prompt, search_results, "ollama", language=language)
            if cached:
                est = int(len(cached.split()) * 1.3)
                print(f"2. ✅ Response cache HIT (est. {est} tokens saved)")
                return cached
            print("2. ❌ Response cache MISS, generating locally")
        expense_lines = []
        for r in (search_results or [])[:15]:
            expense_lines.append(
                f"- {r.get('expense_date', '?')} {r.get('merchant', '?')} "
                f"${float(r.get('expense_amount', 0)):.2f} "
                f"({r.get('shopping_type', '?')})")
        rag_prompt = build_banko_rag_prompt(
            question=prompt,
            expense_data="\n".join(expense_lines) or "(no matching expenses)",
            language=language)
        answer = self._chat([{"role": "user", "content": rag_prompt}])
        if self.cache_manager and answer:
            try:
                self.cache_manager.cache_response(
                    prompt, answer, search_results, "ollama",
                    language=language)
                prompt_tokens = len(rag_prompt.split()) * 1.3
                response_tokens = len(answer.split()) * 1.3
                print(f"3. ✅ Cached response "
                      f"(est. {int(prompt_tokens + response_tokens)} tokens)")
            except Exception:
                pass
        return answer

    def generate_rag_response(self, query: str,
                              context: list[Any] | None = None,
                              user_id: str | None = None,
                              language: str | None = "English"
                              ) -> RAGResponse:
        dicts = []
        for item in context or []:
            if isinstance(item, dict):
                dicts.append(item)
            else:
                dicts.append({
                    "expense_date": getattr(item, "date", "?"),
                    "merchant": getattr(item, "merchant", "?"),
                    "expense_amount": getattr(item, "amount", 0),
                    "shopping_type": getattr(item, "metadata", {}).get(
                        "shopping_type", "?")
                    if hasattr(item, "metadata") else "?",
                })
        text = self.simple_rag_response(query, dicts,
                                        language=language or "English")
        return RAGResponse(response=text, sources=[],
                           metadata={"model": self.current_model,
                                     "provider": "ollama",
                                     "context_rows": len(dicts)})
