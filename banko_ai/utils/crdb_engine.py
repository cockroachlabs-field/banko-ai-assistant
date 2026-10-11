"""Singleton CockroachDBEngine for the entire application.

Provides a single async connection pool backed by langchain-cockroachdb
that all components (vectorstore, chat history, checkpointer) share.
"""


from langchain_cockroachdb import CockroachDBEngine

from .db_retry import get_database_url, normalize_db_url


def normalize_crdb_url(url: str) -> str:
    """Convert any CockroachDB URL to the psycopg driver langchain-cockroachdb uses."""
    return normalize_db_url(url)

_engine: CockroachDBEngine | None = None


def get_crdb_engine(database_url: str | None = None) -> CockroachDBEngine:
    """Get or create the singleton CockroachDBEngine.

    Args:
        database_url: CockroachDB connection string.  Falls back to
            DATABASE_URL env var, then localhost default.

    Returns:
        Shared CockroachDBEngine instance.
    """
    global _engine
    if _engine is not None:
        return _engine

    url = normalize_crdb_url(get_database_url(database_url))

    _engine = CockroachDBEngine.from_connection_string(
        url,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
        pool_recycle=3600,
    )
    return _engine


def close_crdb_engine() -> None:
    """Dispose the singleton engine (call on app shutdown)."""
    global _engine
    if _engine is not None:
        _engine.close()
        _engine = None
