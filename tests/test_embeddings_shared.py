"""One embedding model per process, safe to share across request threads."""

import threading
import time

import pytest
import sentence_transformers

from banko_ai.utils import embeddings as E


@pytest.fixture(autouse=True)
def fresh_models(monkeypatch):
    monkeypatch.setattr(E, "_models", {})


def test_one_model_per_process(monkeypatch):
    built = []

    class Fake:
        def __init__(self, name, **kwargs):
            built.append((name, kwargs.get("device")))

        def encode(self, text, **kwargs):
            return [0.0] * 384

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", Fake)
    first = E.load_embedding_model()
    second = E.load_embedding_model("all-MiniLM-L6-v2")
    assert first is second
    assert built == [("all-MiniLM-L6-v2", "cpu")]


def test_encode_never_runs_twice_at_once(monkeypatch):
    active = {"now": 0, "most": 0}

    class Fake:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, text, **kwargs):
            active["now"] += 1
            active["most"] = max(active["most"], active["now"])
            time.sleep(0.01)
            active["now"] -= 1
            return [0.0] * 384

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", Fake)
    model = E.load_embedding_model()
    threads = [threading.Thread(target=model.encode, args=("coffee",)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert active["most"] == 1


def test_a_busy_loader_raises_model_warming_up():
    E._load_lock.acquire()
    try:
        with pytest.raises(E.ModelWarmingUp):
            E.load_embedding_model(wait_seconds=0.05)
    finally:
        E._load_lock.release()
