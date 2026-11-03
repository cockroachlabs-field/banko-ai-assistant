"""A sentence-transformers upgrade must not move the vectors.

Every expense, receipt and cached query in CockroachDB was embedded with the
version recorded in the fixture. If an upgrade changes the output, stored
vectors and new query vectors stop being comparable and search quietly gets
worse. This test catches that before it ships.
"""

import json
from pathlib import Path

import numpy as np

FIXTURE = Path(__file__).parent / "fixtures" / "embedding_parity.json"


def test_embeddings_match_the_recorded_fixture():
    from sentence_transformers import SentenceTransformer

    data = json.loads(FIXTURE.read_text())
    model = SentenceTransformer(data["model"], device="cpu")
    got = np.asarray(model.encode(data["corpus"]))
    want = np.asarray(data["vectors"])

    # 1e-4 absorbs float noise between torch builds; a real model or
    # tokenizer change moves the vectors far more than that.
    assert np.abs(got - want).max() < 1e-4
    cosine = (got * want).sum(axis=1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(want, axis=1))
    assert cosine.min() > 0.9999
