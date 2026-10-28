"""Removing users. Every cleanup (load users, clear-demo-users, the demo
user expiry) goes through delete_users, so none of them can leave a user's
rows behind in a table it forgot about: the table list comes from the
schema registry."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from sqlalchemy import text

from ..config.settings import get_config
from ..db.engine import get_engine, run_transaction
from ..db.schema import tables
from ..utils.db_retry import normalize_db_url

log = logging.getLogger("banko.users")


def _user_id_types() -> dict[str, str]:
    """Registry table -> the SQL type of its user_id column (two legacy
    tables keep it as STRING), children before parents: the registry lists
    foreign key targets first."""
    out = {}
    for t in reversed(tables()):
        col = next((c for c in t.columns if c.name == "user_id"), None)
        if col is not None and t.name != "users":
            out[t.name] = col.definition.split()[0]
    return out


def user_tables() -> list[str]:
    """Every registry table with a user_id column except users, children
    before parents."""
    return list(_user_id_types())


def _present(url: str) -> set[str]:
    with get_engine(url).connect() as conn:
        return {str(r[0]) for r in conn.execute(text(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))}


def _forget_conversations(ids: list[str], url: str) -> None:
    """Their threads' LangGraph checkpoints. The checkpointer serves the
    configured database only; checkpoints elsewhere expire by their TTL."""
    if normalize_db_url(url) != normalize_db_url(get_config().database_url):
        return
    with get_engine(url).connect() as conn:
        threads = [str(r[0]) for r in conn.execute(text(
            "SELECT thread_id FROM chat_threads WHERE user_id = ANY(CAST(:ids AS UUID[]))"), {"ids": ids})]
    if threads:
        from ..assistant.checkpoint import delete_checkpoints
        for thread_id in threads:
            delete_checkpoints(thread_id)


def delete_users(ids: Sequence[str], *, url: str | None = None, batch: int = 1000) -> int:
    """Delete these users and every row they own, in bounded set-based
    batches, each its own short transaction. Tables a database does not
    have (an older schema, a scratch database) are skipped. Returns the
    number of users removed."""
    ids = [str(i) for i in ids]
    if not ids:
        return 0
    url = url or get_config().database_url  # the admin role: these deletes cross every user's rows
    present = _present(url)
    removed = 0
    for start in range(0, len(ids), batch):
        chunk = ids[start:start + batch]
        if "chat_threads" in present:
            try:
                _forget_conversations(chunk, url)
            except Exception:  # their checkpoints expire on their own; never block the delete
                log.warning("could not delete checkpoints while removing users", exc_info=True)
        for table, kind in _user_id_types().items():
            if table not in present:
                continue
            sql = text(f"DELETE FROM {table} WHERE user_id = ANY(CAST(:ids AS {kind}[])) LIMIT {batch}")

            def delete_batch(conn, sql=sql, chunk=chunk) -> int:
                return int(conn.execute(sql, {"ids": chunk}).rowcount)

            while run_transaction(delete_batch, url=url) >= batch:
                pass
        if "banko_meta" in present:
            keys = [f"seed:{u}" for u in chunk] + [f"seed_lock:{u}" for u in chunk]
            run_transaction(lambda conn: conn.execute(text(
                "DELETE FROM banko_meta WHERE key = ANY(CAST(:keys AS STRING[]))"), {"keys": keys}), url=url)
        removed += run_transaction(lambda conn: conn.execute(text(
            "DELETE FROM users WHERE user_id = ANY(CAST(:ids AS UUID[]))"), {"ids": chunk}).rowcount, url=url)
    return removed
