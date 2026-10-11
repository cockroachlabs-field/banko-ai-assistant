"""CockroachDB VectorStore wrapper using langchain-cockroachdb.

Provides a LangChain-compatible VectorStore backed by CockroachDB's native
VECTOR type with C-SPANN indexes, replacing hand-rolled SQL queries.
"""

from typing import Any

from langchain_cockroachdb import (
    CockroachDBVectorStore,
    CSPANNIndex,
    DistanceStrategy,
)
from langchain_core.embeddings import Embeddings

from banko_ai.utils.embeddings import load_embedding_model

from ..config.settings import get_config
from ..utils.crdb_engine import get_crdb_engine


class _SentenceTransformerEmbeddings(Embeddings):
    """Wraps SentenceTransformer as a LangChain Embeddings object."""

    def __init__(self, model_name: str | None = None):
        self._model = load_embedding_model(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [e.tolist() for e in self._model.encode(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_vectorstore: CockroachDBVectorStore | None = None
_embeddings: _SentenceTransformerEmbeddings | None = None

TABLE_NAME = "expense_vectors"
VECTOR_DIM = 384


def get_embeddings() -> _SentenceTransformerEmbeddings:
    """Return the shared embeddings model."""
    global _embeddings
    if _embeddings is None:
        model_name = get_config().embedding_model
        _embeddings = _SentenceTransformerEmbeddings(model_name)
    return _embeddings


def get_vectorstore(database_url: str | None = None) -> CockroachDBVectorStore:
    """Get or create the singleton CockroachDBVectorStore.

    On first call this creates the ``expense_vectors`` table (IF NOT EXISTS)
    and applies a C-SPANN cosine index.
    """
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    engine = get_crdb_engine(database_url)

    # Create table (idempotent)
    engine.init_vectorstore_table(
        table_name=TABLE_NAME,
        vector_dimension=VECTOR_DIM,
    )

    embeddings = get_embeddings()

    _vectorstore = CockroachDBVectorStore(
        engine=engine,
        embeddings=embeddings,
        collection_name=TABLE_NAME,
        distance_strategy=DistanceStrategy.COSINE,
    )

    # Apply C-SPANN index (idempotent -- CREATE INDEX IF NOT EXISTS)
    index = CSPANNIndex(distance_strategy=DistanceStrategy.COSINE)
    _vectorstore.apply_vector_index(index)

    return _vectorstore


# ---------------------------------------------------------------------------
# Convenience helpers used by the rest of the application
# ---------------------------------------------------------------------------

def search_expenses_via_vectorstore(
    query: str,
    limit: int = 5,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Semantic search over expense documents.

    Returns a list of dicts compatible with the existing search API so
    callers do not need to change.
    """
    vs = get_vectorstore()
    results = vs.similarity_search_with_score(
        query, k=limit, filter=metadata_filter
    )

    search_results: list[dict[str, Any]] = []
    for doc, score in results:
        meta = doc.metadata or {}
        search_results.append({
            "description": doc.page_content,
            "merchant": meta.get("merchant", ""),
            "shopping_type": meta.get("shopping_type", ""),
            "expense_amount": meta.get("expense_amount", 0.0),
            "expense_date": meta.get("expense_date", ""),
            "expense_id": meta.get("expense_id", doc.id or ""),
            "user_id": meta.get("user_id", ""),
            "similarity_score": 1.0 - score,  # cosine distance -> similarity
        })
    return search_results


def index_expense_document(
    expense_id: str,
    description: str,
    metadata: dict[str, Any],
    doc_id: str | None = None,
) -> str:
    """Add or update a single expense in the vectorstore.

    Returns the document id (a UUID).
    """
    import uuid as _uuid

    vs = get_vectorstore()
    # The vectorstore table uses UUID primary keys.
    # If the caller provides a valid UUID we reuse it, otherwise generate one.
    _id = doc_id or expense_id
    try:
        _uuid.UUID(_id)
    except (ValueError, AttributeError):
        _id = str(_uuid.uuid4())

    ids = vs.add_texts(
        texts=[description],
        metadatas=[{**metadata, "expense_id": expense_id}],
        ids=[_id],
    )
    return ids[0]


def bulk_index_expenses(
    expenses: list[dict[str, Any]],
) -> list[str]:
    """Bulk-insert expense documents into the vectorstore."""
    import uuid as _uuid

    if not expenses:
        return []

    vs = get_vectorstore()
    texts = [e.get("description", "") for e in expenses]
    metadatas = [
        {
            "expense_id": e.get("expense_id", ""),
            "user_id": e.get("user_id", ""),
            "merchant": e.get("merchant", ""),
            "shopping_type": e.get("shopping_type", ""),
            "expense_amount": float(e.get("expense_amount", 0)),
            "expense_date": str(e.get("expense_date", "")),
            "payment_method": e.get("payment_method", ""),
        }
        for e in expenses
    ]
    # Ensure all IDs are valid UUIDs
    ids = []
    for e in expenses:
        eid = e.get("expense_id", "")
        try:
            _uuid.UUID(eid)
            ids.append(eid)
        except (ValueError, AttributeError):
            ids.append(str(_uuid.uuid4()))

    return vs.add_texts(texts=texts, metadatas=metadatas, ids=ids)
