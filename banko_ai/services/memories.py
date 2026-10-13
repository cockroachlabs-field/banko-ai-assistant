"""Long-term memories the user can see and delete. Stored with a per-user
vector index, recalled by meaning, deduplicated on write. Adding or deleting
one advances the user's data version in the same transaction, so a cached
answer that leaned on a memory never outlives it."""

from __future__ import annotations

import re

from sqlalchemy import text

from ..config.settings import get_config
from ..db.engine import get_engine, run_transaction
from ..db.regions import money_region, region_columns
from ..db.rls import request_url
from ._sql import jsonable, valid_uuid
from ._vector import embed, nearest
from .ledger import bump_data_version

DUPLICATE_DISTANCE = 0.08   # closer than this is the same memory said again
FORGET_DISTANCE = 0.55      # "forget the Japan trip" matches within this
_GOAL = re.compile(r"\b(saving for|save for|my goal|goal is|want to (buy|save|pay off)|paying off)\b", re.I)
_PREF = re.compile(r"\b(i prefer|i like|i usually|i always|i never|i'd rather)\b", re.I)

_SEARCH = dict(table="memories", index="memories_user_embedding", exact_index="memories_user_idx",
               id_col="memory_id", emb_col="embedding")


def kind_for(content: str) -> str:
    if _GOAL.search(content):
        return "goal"
    if _PREF.search(content):
        return "preference"
    return "fact"


def remember(user_id: str, content: str, *, kind: str | None = None, source_thread_id: str | None = None,
             url: str | None = None) -> dict:
    content = " ".join((content or "").split())[:500]
    if not content:
        raise ValueError("Nothing to remember")
    kind = kind or kind_for(content)
    if kind not in ("fact", "preference", "goal"):
        raise ValueError("Unknown memory kind")
    url = request_url(url)
    qvec = embed(content)  # outside the transaction: CPU work never holds a connection

    def fn(conn) -> dict:
        near = nearest(conn, user_id=user_id, qvec=qvec, k=1, **_SEARCH)
        if near and near[0][1] <= DUPLICATE_DISTANCE:
            mid = near[0][0]
            conn.execute(text("UPDATE memories SET use_count = use_count + 1, last_used_at = now() "
                              "WHERE memory_id = CAST(:m AS UUID) AND user_id = CAST(:u AS UUID)"),
                         {"m": mid, "u": user_id})
            row = conn.execute(text("SELECT content, kind FROM memories WHERE memory_id = CAST(:m AS UUID)"),
                               {"m": mid}).fetchone()
            return {"memory_id": mid, "content": row[0], "kind": row[1], "created": False}
        region = money_region(conn, user_id, url)
        rcol, rval = region_columns(region)
        mid = conn.execute(text(f"""
            INSERT INTO memories ({rcol}user_id, kind, content, embedding, source_thread_id)
            VALUES ({rval}CAST(:u AS UUID), :k, :c, :e, CAST(:t AS UUID))
            RETURNING memory_id
        """), {"u": user_id, "k": kind, "c": content, "e": qvec,
               "t": source_thread_id if valid_uuid(source_thread_id) else None,
               "crdb_region": region}).scalar()
        bump_data_version(conn, user_id, region)
        return {"memory_id": str(mid), "content": content, "kind": kind, "created": True}

    return run_transaction(fn, url=url, idempotent=False)


def recall(user_id: str, query: str, *, k: int | None = None, url: str | None = None) -> list[dict]:
    query = (query or "").strip()
    if not query:
        return []
    cfg = get_config()
    url = request_url(url)
    qvec = embed(query)

    def fn(conn) -> list[dict]:
        near = [(m, d) for m, d in nearest(conn, user_id=user_id, qvec=qvec,
                                           k=k or cfg.assistant_memory_recall_k, **_SEARCH)
                if d <= cfg.assistant_memory_max_distance]
        if not near:
            return []
        ids = [m for m, _ in near]
        rows = {str(r[0]): r for r in conn.execute(text("""
            UPDATE memories SET use_count = use_count + 1, last_used_at = now()
            WHERE user_id = CAST(:u AS UUID) AND memory_id = ANY(CAST(:ids AS UUID[]))
            RETURNING memory_id, kind, content
        """), {"u": user_id, "ids": ids})}
        return [{"memory_id": m, "kind": rows[m][1], "content": rows[m][2], "distance": round(d, 4)}
                for m, d in near if m in rows]

    return run_transaction(fn, url=url)


def list_memories(user_id: str, *, url: str | None = None) -> list[dict]:
    url = request_url(url)
    with get_engine(url).connect() as conn:
        rows = conn.execute(text("""
            SELECT memory_id, kind, content, use_count, created_at, last_used_at
            FROM memories@memories_user_idx WHERE user_id = CAST(:u AS UUID) ORDER BY created_at DESC
        """), {"u": user_id}).fetchall()
    return jsonable([{"memory_id": str(r[0]), "kind": r[1], "content": r[2], "use_count": r[3],
                      "created_at": r[4], "last_used_at": r[5]} for r in rows])


def forget(user_id: str, memory_id: str, *, url: str | None = None) -> bool:
    if not valid_uuid(memory_id):
        return False
    url = request_url(url)

    def fn(conn) -> bool:
        deleted = conn.execute(text(
            "DELETE FROM memories WHERE memory_id = CAST(:m AS UUID) AND user_id = CAST(:u AS UUID)"),
            {"m": memory_id, "u": user_id}).rowcount
        if deleted:
            bump_data_version(conn, user_id, money_region(conn, user_id, url))
        return bool(deleted)

    return run_transaction(fn, url=url)


def forget_matching(user_id: str, phrase: str, *, url: str | None = None) -> list[str]:
    phrase = (phrase or "").strip()
    if not phrase:
        return []
    url = request_url(url)
    qvec = embed(phrase)

    def fn(conn) -> list[str]:
        near = nearest(conn, user_id=user_id, qvec=qvec, k=1, **_SEARCH)
        if not near or near[0][1] > FORGET_DISTANCE:
            return []
        row = conn.execute(text("DELETE FROM memories WHERE memory_id = CAST(:m AS UUID) "
                                "AND user_id = CAST(:u AS UUID) RETURNING content"),
                           {"m": near[0][0], "u": user_id}).fetchone()
        if row is None:
            return []
        bump_data_version(conn, user_id, money_region(conn, user_id, url))
        return [row[0]]

    return run_transaction(fn, url=url)
