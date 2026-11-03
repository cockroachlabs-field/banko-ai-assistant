"""
Database retry utilities for handling transient connection failures.

This module provides decorators and utilities to automatically retry database
operations that fail due to transient errors like connection blips, HAProxy
resets, or network issues.
"""

import functools
import time
from collections.abc import Callable

import psycopg
from sqlalchemy.exc import DBAPIError, OperationalError

from ..config.settings import get_config

# Spellings people already have in env files and compose files. All of them
# land on the CockroachDB dialect with the psycopg 3 driver.
_DRIVER_PREFIXES = (
    "postgresql+psycopg2://",
    "postgres+psycopg2://",
    "cockroachdb+psycopg2://",
    "postgresql+psycopg://",
    "cockroachdb://",
    "postgresql://",
    "postgres://",
)
_SQLALCHEMY_PREFIX = "cockroachdb+psycopg://"


def normalize_db_url(url: str) -> str:
    """Return the SQLAlchemy URL banko uses everywhere: the CockroachDB
    dialect on psycopg 3. Already-normalized and asyncpg URLs pass through."""
    for prefix in _DRIVER_PREFIXES:
        if url.startswith(prefix):
            return _SQLALCHEMY_PREFIX + url[len(prefix):]
    return url


def to_libpq_url(url: str) -> str:
    """A plain postgresql:// URL, for libraries that open psycopg
    connections themselves (the LangGraph checkpointer)."""
    normalized = normalize_db_url(url)
    if normalized.startswith(_SQLALCHEMY_PREFIX):
        return "postgresql://" + normalized[len(_SQLALCHEMY_PREFIX):]
    return normalized


def get_database_url(url: str | None = None) -> str:
    """Get and normalize the database URL from argument or DATABASE_URL env var."""
    raw = url or get_config().database_url
    return normalize_db_url(raw)


# Transient errors that should trigger a retry
TRANSIENT_ERRORS = (
    OperationalError,
    psycopg.OperationalError,
    psycopg.InterfaceError,
    DBAPIError,
)

# Error messages that indicate transient failures
TRANSIENT_ERROR_MESSAGES = (
    "server closed the connection unexpectedly",
    "connection already closed",
    "SSL connection has been closed unexpectedly",
    "could not receive data from server",
    "connection timed out",
    "Connection refused",
    "connection reset by peer",
    "broken pipe",
    "restart transaction",
    "TransactionRetryError",
    "SerializationFailure",
    "StatementCompletionUnknown",  # Multi-region: result is ambiguous
    "result is ambiguous",  # Multi-region failover scenario
    "failed to connect",  # Node/region unavailable
    "no such host",  # DNS lookup failure during region failover
    "initial connection heartbeat failed",  # Region heartbeat failure
    "sending to all replicas failed",  # All replicas in region unavailable
    "40001",  # PostgreSQL/CockroachDB serialization failure code
    "40003",  # Statement completion unknown code
)


def is_transient_error(error: Exception) -> bool:
    """
    Check if an error is transient and should be retried.
    
    Args:
        error: The exception to check
        
    Returns:
        True if the error is transient and retryable
    """
    if not isinstance(error, TRANSIENT_ERRORS):
        return False
    
    error_msg = str(error).lower()
    return any(msg.lower() in error_msg for msg in TRANSIENT_ERROR_MESSAGES)


def db_retry(
    max_attempts: int = 3,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 10.0,
    exceptions: tuple[type[Exception], ...] = TRANSIENT_ERRORS
):
    """
    Decorator to retry database operations on transient failures.
    
    Uses exponential backoff with configurable parameters:
    - Retry delay: initial_delay * (backoff_factor ** attempt)
    - Capped at max_delay
    
    Args:
        max_attempts: Maximum number of retry attempts (default: 3)
        initial_delay: Initial retry delay in seconds (default: 0.5)
        backoff_factor: Multiplier for exponential backoff (default: 2.0)
        max_delay: Maximum delay between retries (default: 10.0)
        exceptions: Tuple of exception types to catch and retry
        
    Example:
        @db_retry(max_attempts=5, initial_delay=1.0)
        def query_database():
            # Your database operation here
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            delay = initial_delay
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                    
                except exceptions as e:
                    attempt += 1
                    
                    # Check if error is truly transient
                    if not is_transient_error(e):
                        print(f"❌ Non-transient error in {func.__name__}: {e}")
                        raise
                    
                    if attempt >= max_attempts:
                        print(f"❌ Max retry attempts ({max_attempts}) reached for {func.__name__}")
                        raise
                    
                    # Calculate next delay with exponential backoff
                    current_delay = min(delay * (backoff_factor ** (attempt - 1)), max_delay)
                    
                    print(f"⚠️  Transient DB error in {func.__name__} (attempt {attempt}/{max_attempts}): {e}")
                    print(f"🔄 Retrying in {current_delay:.2f}s...")
                    
                    time.sleep(current_delay)
                    
            # This should never be reached, but just in case
            raise Exception(f"Unexpected state in retry logic for {func.__name__}")
        
        return wrapper
    return decorator


def db_retry_context(max_attempts: int = 3, initial_delay: float = 0.5):
    """
    Context manager for database retry logic.
    
    Usage:
        with db_retry_context(max_attempts=5):
            # Your database operations here
            conn.execute(query)
    """
    class RetryContext:
        def __init__(self, max_attempts, initial_delay):
            self.max_attempts = max_attempts
            self.initial_delay = initial_delay
            self.attempt = 0
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type and issubclass(exc_type, TRANSIENT_ERRORS):
                if is_transient_error(exc_val):
                    self.attempt += 1
                    if self.attempt < self.max_attempts:
                        delay = self.initial_delay * (2 ** (self.attempt - 1))
                        print(f"⚠️  Transient DB error (attempt {self.attempt}/{self.max_attempts}): {exc_val}")
                        print(f"🔄 Retrying in {delay:.2f}s...")
                        time.sleep(delay)
                        return True  # Suppress exception and retry
            return False  # Don't suppress exception
    
    return RetryContext(max_attempts, initial_delay)


def create_resilient_engine(database_url: str, **kwargs):
    """Older name for banko_ai.db.engine.get_engine. Pool settings come from
    Settings; per-call overrides are ignored so every caller shares one pool."""
    from ..db.engine import get_engine

    return get_engine(database_url)
