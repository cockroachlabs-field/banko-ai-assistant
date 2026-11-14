"""The process-wide LangGraph checkpointer. Threads are checkpointed in
CockroachDB, so a follow-up still knows its receipt after a restart or a
region failure. This module owns the one psycopg pool outside the shared
SQLAlchemy engine: the saver speaks psycopg directly."""

from __future__ import annotations

import logging
import re
import threading

from ..config.settings import get_config

log = logging.getLogger("banko.assistant")
_lock = threading.Lock()
_saver = None
# The pool's connections show up under this name in SHOW SESSIONS.
APPLICATION_NAME = "banko-checkpointer"
# In SHOW CREATE TABLE: any row-level TTL setting, and the interval of an
# expression enable_ttl() wrote. 0.5 wrote (created_at + '7 days'), and 0.6
# writes the same with a grace after it.
_TTL_SETTING = re.compile(r"\bttl\w* = ")
_TTL_INTERVAL = re.compile(r"\(created_at \+ \\?'([^'\\]+)")


def get_saver():
    """The checkpointer, built on first use. Its tables' TTL is keep_ttl()'s
    job, so a turn that builds the checkpointer never waits on that."""
    global _saver
    with _lock:
        if _saver is not None:
            return _saver
        from langchain_cockroachdb import CockroachDBSaver
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        from ..utils.db_retry import to_libpq_url
        cfg = get_config()
        # Checkpoint reads and writes run concurrently, one pooled connection
        # each. A turn that finds the pool full waits up to the timeout, then
        # fails with PoolTimeout, which the saver does not retry.
        pool = ConnectionPool(to_libpq_url(cfg.database_url), min_size=1, max_size=cfg.checkpoint_pool_size,
                              timeout=cfg.db_pool_timeout, open=True,
                              kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row,
                                      "application_name": APPLICATION_NAME})
        saver = CockroachDBSaver(pool)
        saver.setup()
        _saver = saver
        return _saver


def _blob_table(pool) -> str:
    """checkpoint_blobs as SHOW CREATE TABLE prints it. The 0.6 TTL setting shows there."""
    with pool.connection() as conn:
        row = conn.execute("SELECT create_statement FROM [SHOW CREATE TABLE checkpoint_blobs]").fetchone()
    return (row["create_statement"] if isinstance(row, dict) else row[0]) if row else ""


def _ttl_current(statement: str, fraction: int) -> bool:
    """Whether the table has TTL as langchain-cockroachdb 0.6 sets it, and
    running. 0.6 lets blob rows outlive their checkpoints by the interval
    divided by the saver's TTL_REFRESH_FRACTION, shown as "/ 10)"; 0.5's
    setting lacks it. A run that stopped partway leaves ttl_pause = true, and
    CockroachDB drops the entry when the pause is reset."""
    return f"/ {fraction})" in statement and "ttl_pause" not in statement


def keep_ttl() -> str:
    """Bring row-level TTL on the checkpoint tables up to date, and say what
    was done. enable_ttl() runs on a database with no TTL yet, for
    CHECKPOINT_TTL_DAYS, and on one that langchain-cockroachdb 0.5 set up or
    whose last run stopped partway, for the interval it already has, so that
    nothing an earlier boot kept is deleted by this one. The run pauses TTL,
    waits about 15 seconds for every saver to read the new setting, refreshes
    what live threads still use, and resumes. With the setting in place it
    would still be three schema changes, so then it is not called at all."""
    saver = get_saver()
    fraction = saver.TTL_REFRESH_FRACTION

    def shown_now() -> str:
        # Banko's own read on the saver's pool, retried as the saver retries its calls.
        shown: str = saver.retry_policy.run(lambda: _blob_table(saver.conn), label="read checkpoint TTL")
        return shown

    shown = shown_now()
    found = _TTL_INTERVAL.search(shown)
    if found:
        interval = found.group(1)
        if _ttl_current(shown, fraction):
            return f"up to date ({interval})"
    elif _TTL_SETTING.search(shown):
        return "left alone: a TTL setting Banko does not recognize"
    else:
        days = get_config().checkpoint_ttl_days
        if days <= 0:
            return "off (CHECKPOINT_TTL_DAYS=0)"
        interval = f"{days} days"
    # While it runs, a turn in this process makes the saver warn that TTL is
    # paused. These two lines say whose run that is.
    log.info("checkpoint TTL: updating the setting on this database, once. It takes about 15 seconds, "
             "and questions are answered meanwhile.")
    saver.enable_ttl(ttl_interval=interval, cron="@daily")
    if not _ttl_current(shown_now(), fraction):
        # enable_ttl() leaves the resume to another run that changed the setting meanwhile.
        raise RuntimeError("the checkpoint TTL setting is not current after enable_ttl()")
    log.info("checkpoint TTL: up to date. A conversation's state expires once it has been quiet for %s.", interval)
    return f"set to {interval}"


def delete_checkpoints(thread_id: str) -> None:
    try:
        get_saver().delete_thread(thread_id)
    except Exception:
        log.warning("could not delete checkpoints for thread %s", thread_id, exc_info=True)
