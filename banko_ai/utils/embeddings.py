"""One place to load the sentence-transformers embedding model.

There is exactly one model per process. Fourteen call sites used to build
their own (base_agent.py built one per call), each holding its own copy of
the weights. encode() is serialized because Hugging Face fast tokenizers
raise "Already borrowed" when two threads tokenize at once; a short query
encodes in a few milliseconds, so the lock never shows up in latency.

Offline first: when the model is already in the local Hugging Face cache it
loads with local_files_only, so no network request fires. The airgap and
wifi-off demos depend on this. Only a cold cache falls through to a normal
download.

The model runs on CPU unless EMBEDDING_DEVICE says otherwise. Left to pick,
sentence-transformers chooses MPS on Apple Silicon, which reserves about
1 GB of Metal memory for this 22M-parameter model and makes the first encode
over a second slower.
"""

from __future__ import annotations

import threading
from typing import Any

from ..config.settings import get_config

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class ModelWarmingUp(RuntimeError):
    """The embedding model is still loading; try again shortly."""


class SharedEmbeddingModel:
    def __init__(self, model: Any) -> None:
        self._model = model
        self._lock = threading.Lock()

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            return self._model.encode(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


_models: dict[tuple[str, str], SharedEmbeddingModel] = {}
_load_lock = threading.Lock()


def _load(model_name: str, device: str) -> Any:
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_name, local_files_only=True, device=device)
    except Exception:
        # Not cached yet (first run, or a new model name): this one
        # legitimately needs the network.
        return SentenceTransformer(model_name, device=device)


def load_embedding_model(model_name: str | None = None, *,
                         wait_seconds: float | None = None) -> SharedEmbeddingModel:
    """The shared model. Waits up to WARMUP_WAIT_SECONDS if another thread is
    loading it, then raises ModelWarmingUp."""
    cfg = get_config()
    name = model_name or cfg.embedding_model
    key = (name, cfg.embedding_device)
    model = _models.get(key)
    if model is not None:
        return model
    timeout = cfg.warmup_wait_seconds if wait_seconds is None else wait_seconds
    if not _load_lock.acquire(timeout=timeout):
        raise ModelWarmingUp(f"embedding model {name} is still loading")
    try:
        model = _models.get(key)
        if model is None:
            model = SharedEmbeddingModel(_load(name, cfg.embedding_device))
            _models[key] = model
        return model
    finally:
        _load_lock.release()
