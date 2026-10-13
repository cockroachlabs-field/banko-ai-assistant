"""One way to ask a per-user vector index for the nearest rows.

The index is pinned with a hint inside a MATERIALIZED CTE, so no other
predicate can make the optimizer skip it (cockroachlabs/cockroach#10432) and
a user seeded seconds ago is still served by the index. On CockroachDB v26.3
a REGIONAL BY ROW table can return nothing from the index for a user whose
rows are there (cockroachlabs/cockroach#13756); then the user's rows are
ranked exactly through a plain user_id index and the request trace says so."""

from __future__ import annotations

import logging
import time

from sqlalchemy import text
from sqlalchemy.engine import Connection

from ..utils.request_trace import trace_step
from ._sql import vector_text

log = logging.getLogger("banko.vector")
_logged: set[str] = set()


def embed(text_value: str) -> str:
    """The query vector as the text CockroachDB binds to VECTOR(384)."""
    from ..utils.embeddings import load_embedding_model
    return vector_text(load_embedding_model().encode([text_value])[0])


def nearest(conn: Connection, *, table: str, index: str, exact_index: str, id_col: str, emb_col: str,
            user_id: str, qvec: str, k: int) -> list[tuple[str, float]]:
    """(id, distance) for the user's k nearest rows, nearest first. Table,
    index and column names are code constants, never user input."""
    params = {"u": user_id, "qvec": qvec, "k": max(1, int(k))}
    rows = conn.execute(text(f"""
        WITH nn AS MATERIALIZED (
            SELECT {id_col} AS id, {emb_col} <=> :qvec AS distance
            FROM {table}@{index}
            WHERE user_id = CAST(:u AS UUID)
            ORDER BY {emb_col} <=> :qvec LIMIT :k
        )
        SELECT id, distance FROM nn WHERE distance IS NOT NULL ORDER BY distance
    """), params).fetchall()
    if rows:
        return [(str(r[0]), float(r[1])) for r in rows]
    has_rows = conn.execute(text(
        f"SELECT EXISTS (SELECT 1 FROM {table}@{exact_index} WHERE user_id = CAST(:u AS UUID) AND {emb_col} IS NOT NULL)"
    ), params).scalar()
    if not has_rows:
        return []
    t0 = time.perf_counter()
    rows = conn.execute(text(f"""
        SELECT {id_col}, {emb_col} <=> :qvec AS distance
        FROM {table}@{exact_index}
        WHERE user_id = CAST(:u AS UUID) AND {emb_col} IS NOT NULL
        ORDER BY distance LIMIT :k
    """), params).fetchall()
    trace_step("vector_search", "Vector index fallback",
               f"{index} returned no rows; exact scan of this user's {table}",
               ms=(time.perf_counter() - t0) * 1000, status="warning")
    if table not in _logged:
        _logged.add(table)
        log.warning("The per-user vector index %s returned no rows for a user who has rows in %s; "
                    "ranking that user's rows exactly (cockroachlabs/cockroach#13756). Logged once per table.",
                    index, table)
    return [(str(r[0]), float(r[1])) for r in rows]
