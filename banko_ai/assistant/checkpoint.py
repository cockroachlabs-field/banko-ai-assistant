"""The process-wide LangGraph checkpointer. Threads are checkpointed in
CockroachDB, so a follow-up still knows its receipt after a restart or a
region failure. This module owns the one psycopg pool outside the shared
SQLAlchemy engine: the saver speaks psycopg directly."""

from __future__ import annotations

import logging
import threading

from ..config.settings import get_config

log = logging.getLogger("banko.assistant")
_lock = threading.Lock()
_saver = None


def _ttl_already_on(pool) -> bool:
    with pool.connection() as conn:
        row = conn.execute("SELECT create_statement FROM [SHOW CREATE TABLE checkpoints]").fetchone()
    stmt = (row["create_statement"] if isinstance(row, dict) else row[0]) if row else ""
    return "ttl" in stmt


def get_saver():
    global _saver
    with _lock:
        if _saver is not None:
            return _saver
        from langchain_cockroachdb import CockroachDBSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        from ..utils.db_retry import to_libpq_url
        cfg = get_config()
        pool = ConnectionPool(to_libpq_url(cfg.database_url), min_size=1, max_size=8, open=True,
                              kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
        saver = CockroachDBSaver(pool)
        saver.setup()
        if cfg.checkpoint_ttl_days > 0 and not _ttl_already_on(pool):
            # Only once per database: ALTER TABLE ... SET ttl is a schema change job.
            saver.enable_ttl(ttl_interval=f"{cfg.checkpoint_ttl_days} days", cron="@daily")
        _saver = saver
        return _saver


def delete_checkpoints(thread_id: str) -> None:
    try:
        get_saver().delete_thread(thread_id)
    except Exception:
        log.warning("could not delete checkpoints for thread %s", thread_id, exc_info=True)
