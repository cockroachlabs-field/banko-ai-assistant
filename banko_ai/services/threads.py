"""Chat threads. The messages themselves live in LangGraph checkpoints keyed
by thread id; this table is the list a user sees. chat_threads carries
row-level security, so every statement runs with banko.user_id set."""

from __future__ import annotations

from sqlalchemy import text

from ..db.engine import run_transaction
from ..db.regions import money_region, region_columns
from ..db.rls import request_url
from ._sql import jsonable, valid_uuid

DEFAULT_TITLE = "New chat"
TITLE_MAX = 48
_COLS = "thread_id, title, message_count, created_at, last_message_at"


def _row(r) -> dict:
    return jsonable({"thread_id": str(r[0]), "title": r[1], "message_count": int(r[2]),
                     "created_at": r[3], "last_message_at": r[4]})


def title_from(message: str) -> str:
    words = " ".join((message or "").split())
    if not words:
        return DEFAULT_TITLE
    words = words[0].upper() + words[1:]
    if len(words) <= TITLE_MAX:
        return words
    cut = words[:TITLE_MAX].rsplit(" ", 1)[0].rstrip(",.;:!?")
    return cut + "…"


def create_thread(user_id: str, *, title: str = DEFAULT_TITLE, url: str | None = None) -> dict:
    url = request_url(url)
    title = (title or DEFAULT_TITLE).strip()[:120] or DEFAULT_TITLE

    def fn(conn) -> dict:
        region = money_region(conn, user_id, url)
        rcol, rval = region_columns(region)
        return _row(conn.execute(text(f"""
            INSERT INTO chat_threads ({rcol}user_id, title) VALUES ({rval}CAST(:u AS UUID), :t)
            RETURNING {_COLS}
        """), {"u": user_id, "t": title, "crdb_region": region}).fetchone())

    return run_transaction(fn, url=url, user_id=user_id, idempotent=False)


def list_threads(user_id: str, *, limit: int = 50, url: str | None = None) -> list[dict]:
    url = request_url(url)
    rows = run_transaction(lambda conn: conn.execute(text(f"""
        SELECT {_COLS} FROM chat_threads WHERE user_id = CAST(:u AS UUID)
        ORDER BY last_message_at DESC LIMIT :n
    """), {"u": user_id, "n": max(1, min(int(limit), 200))}).fetchall(), url=url, user_id=user_id)
    return [_row(r) for r in rows]


def get_thread(user_id: str, thread_id: str, *, url: str | None = None) -> dict | None:
    if not valid_uuid(thread_id):
        return None
    url = request_url(url)
    row = run_transaction(lambda conn: conn.execute(text(
        f"SELECT {_COLS} FROM chat_threads WHERE thread_id = CAST(:t AS UUID) AND user_id = CAST(:u AS UUID)"),
        {"t": thread_id, "u": user_id}).fetchone(), url=url, user_id=user_id)
    return _row(row) if row else None


def rename_thread(user_id: str, thread_id: str, title: str, *, url: str | None = None) -> dict | None:
    title = (title or "").strip()[:120]
    if not valid_uuid(thread_id) or not title:
        return None
    url = request_url(url)
    row = run_transaction(lambda conn: conn.execute(text(f"""
        UPDATE chat_threads SET title = :title
        WHERE thread_id = CAST(:t AS UUID) AND user_id = CAST(:u AS UUID) RETURNING {_COLS}
    """), {"title": title, "t": thread_id, "u": user_id}).fetchone(), url=url, user_id=user_id)
    return _row(row) if row else None


def delete_thread(user_id: str, thread_id: str, *, url: str | None = None) -> bool:
    if not valid_uuid(thread_id):
        return False
    url = request_url(url)
    return bool(run_transaction(lambda conn: conn.execute(text(
        "DELETE FROM chat_threads WHERE thread_id = CAST(:t AS UUID) AND user_id = CAST(:u AS UUID)"),
        {"t": thread_id, "u": user_id}).rowcount, url=url, user_id=user_id))


def touch_thread(user_id: str, thread_id: str, *, added: int = 2, title: str | None = None,
                 url: str | None = None) -> None:
    """Count the turn's messages and title a thread still called New chat."""
    if not valid_uuid(thread_id):
        return
    url = request_url(url)
    run_transaction(lambda conn: conn.execute(text("""
        UPDATE chat_threads
        SET message_count = message_count + :n, last_message_at = now(),
            title = CASE WHEN title = :default AND CAST(:title AS STRING) IS NOT NULL
                         THEN CAST(:title AS STRING) ELSE title END
        WHERE thread_id = CAST(:t AS UUID) AND user_id = CAST(:u AS UUID)
    """), {"n": int(added), "default": DEFAULT_TITLE, "title": title, "t": thread_id, "u": user_id}),
        url=url, user_id=user_id, idempotent=False)
