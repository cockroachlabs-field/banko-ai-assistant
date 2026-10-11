"""Write the embedding parity fixture.

Encodes a fixed corpus with the sentence-transformers version that is
installed right now, on CPU. Run it before upgrading sentence-transformers;
tests/test_embedding_parity.py then proves the new version produces the same
vectors, which means nothing already stored in CockroachDB needs re-embedding.
"""

import json
import sys
from importlib.metadata import version
from pathlib import Path

from sentence_transformers import SentenceTransformer

CORPUS = [
    "Spent $84.12 at Whole Foods Market for Groceries on 2026-10-11",
    "coffee near the office",
    "Netflix subscription went up to 22.99",
    "Acme Corp payroll deposit",
    "Uber ride to the airport",
    "how much did I spend on dining last month",
    "Transfer to Travel Fund for the Japan trip",
    "¿Cuánto gasté en comestibles este mes?",
    "Electric bill from City Power",
    "Malbec, salmon fillet, Greek yogurt, organic bananas",
]


def main(out: str) -> None:
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu", local_files_only=True)
    vectors = model.encode(CORPUS)
    payload = {
        "sentence_transformers": version("sentence-transformers"),
        "model": "all-MiniLM-L6-v2",
        "corpus": CORPUS,
        "vectors": [[round(float(x), 6) for x in vector] for vector in vectors],
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, ensure_ascii=False))
    print(f"wrote {out} with sentence-transformers {payload['sentence_transformers']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/embedding_parity.json")
