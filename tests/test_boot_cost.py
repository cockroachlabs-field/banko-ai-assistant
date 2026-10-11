"""The embedding model must not reserve Metal memory when tests can
run on CPU.

It landed on Apple's MPS device, which reserves about 1 GB of Metal memory
for a 22M-parameter model.
"""

import sentence_transformers

from banko_ai.utils.embeddings import load_embedding_model


def test_embedding_model_loads_on_cpu(monkeypatch):
    calls = []

    def fake_model(*args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise OSError("not in the local cache")
        return object()

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", fake_model)
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)
    load_embedding_model()
    # Both the cached load and the cold-cache download must pin CPU.
    assert [c.get("device") for c in calls] == ["cpu", "cpu"]
