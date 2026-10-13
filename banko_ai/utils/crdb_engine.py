"""Singleton CockroachDBEngine for the entire application.

Provides a single async connection pool backed by langchain-cockroachdb
that all components (vectorstore, chat history, checkpointer) share.
"""


from langchain_cockroachdb import CockroachDBEngine

from ..config.settings import get_config
from .db_retry import get_database_url, normalize_db_url


def normalize_crdb_url(url: str) -> str:
    """Convert any CockroachDB URL to the psycopg driver langchain-cockroachdb uses."""
    return normalize_db_url(url)

_engine: CockroachDBEngine | None = None


def get_crdb_engine(database_url: str | None = None) -> CockroachDBEngine:
    """Get or create the singleton CockroachDBEngine.

    Args:
        database_url: CockroachDB connection string. Falls back to
            the DATABASE_URL setting.

    Returns:
        Shared CockroachDBEngine instance.
    """
    global _engine
    if _engine is not None:
        return _engine

    url = normalize_crdb_url(get_database_url(database_url))

    # langchain-cockroachdb wants an async engine, so this can't reuse the
    # sync pool in banko_ai.db.engine. It honors the same DB_POOL_* settings.
    cfg = get_config()
    _engine = CockroachDBEngine.from_connection_string(
        url,
        pool_size=cfg.db_pool_size,
        max_overflow=cfg.db_max_overflow,
        pool_pre_ping=cfg.db_pool_pre_ping,
        pool_recycle=cfg.db_pool_recycle,
        pool_timeout=cfg.db_pool_timeout,
        retry_max_attempts=cfg.db_retry_max_attempts,
    )
    return _engine


def close_crdb_engine() -> None:
    """Dispose the singleton engine (call on app shutdown)."""
    global _engine
    if _engine is not None:
        _engine.close()
        _engine = None
