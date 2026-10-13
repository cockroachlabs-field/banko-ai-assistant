"""The assistant's caches. Both are keyed on the user and the user's data
version, so a cached answer or search result never crosses users and never
outlives a write that changed the user's data (a new receipt, a
recategorization, a memory added or deleted). Hits and misses land in
cache_stats, so /cache-stats keeps its shape."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from sqlalchemy import text

from ..config.settings import get_config
from ..db.engine import get_engine, run_transaction
from ..db.regions import money_region, region_columns
from ..db.rls import request_url
from ._sql import jsonable
from ._vector import nearest

log = logging.getLogger("banko.caches")


def estimate_tokens(text_value: str) -> int:
    """About four characters per token; the trace calls it an estimate."""
    return len(text_value or "") // 4


def record_stat(cache_type: str, operation: str, tokens_saved: int = 0) -> None:
    try:
        with get_engine().begin() as conn:
            conn.execute(text("INSERT INTO cache_stats (cache_type, operation, tokens_saved) "
                              "VALUES (:t, :o, :n)"), {"t": cache_type, "o": operation, "n": int(tokens_saved)})
    except Exception:  # stats never break an answer
        log.debug("cache stat not recorded", exc_info=True)


def search_key(user_id: str, version: int, payload: dict) -> str:
    body = json.dumps({"u": str(user_id), "v": int(version), "p": jsonable(payload)}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


def get_search(user_id: str, key: str, *, url: str | None = None, stat: str = "search") -> list | None:
    url = request_url(url)

    def fn(conn) -> list | None:
        row = conn.execute(text("""
            UPDATE search_cache SET hits = hits + 1
            WHERE cache_key = :k AND user_id = CAST(:u AS UUID) AND expires_at > now()
            RETURNING results
        """), {"k": key, "u": user_id}).fetchone()
        return None if row is None else (row[0] if isinstance(row[0], list) else json.loads(row[0]))

    results = run_transaction(fn, url=url)
    record_stat(stat, "hit" if results is not None else "miss")
    return results


def put_search(user_id: str, key: str, results: list, *, url: str | None = None,
               stat: str = "search") -> None:
    url = request_url(url)
    minutes = get_config().search_cache_ttl_minutes

    def fn(conn) -> None:
        region = money_region(conn, user_id, url)
        rcol, rval = region_columns(region)
        conn.execute(text(f"""
            UPSERT INTO search_cache ({rcol}cache_key, user_id, results, hits, created_at, expires_at)
            VALUES ({rval}:k, CAST(:u AS UUID), CAST(:r AS JSONB), 0, now(), now() + :m * INTERVAL '1 minute')
        """), {"k": key, "u": user_id, "r": json.dumps(jsonable(results)), "m": minutes, "crdb_region": region})

    run_transaction(fn, url=url)
    record_stat(stat, "write")


_PUNCT = re.compile(r"[^\w\s]")


def normalize_question(question: str) -> str:
    """The question as the answer cache compares it: lowercase, no
    punctuation, single spaces."""
    return " ".join(_PUNCT.sub("", (question or "").lower()).split())


def get_answer(user_id: str, *, version: int, language: str, query: str, qvec: str,
               url: str | None = None) -> dict | None:
    """A cached answer to this exact question. The vector index finds the
    candidates, but "bills due next week" and "bills due this week" sit
    0.017 apart, so a hit also needs the same normalized question text."""
    url = request_url(url)
    max_distance = 1.0 - float(get_config().cache_similarity_threshold)
    want = normalize_question(query)

    def fn(conn) -> dict | None:
        near = [(cache_id, distance) for cache_id, distance in nearest(
            conn, table="answer_cache", index="answer_cache_user_embedding", exact_index="answer_cache_user_idx",
            id_col="cache_id", emb_col="query_embedding", user_id=user_id, qvec=qvec, k=10)
            if distance <= max_distance]
        if not near:
            return None
        asked = {str(r[0]): r[1] for r in conn.execute(text("""
            SELECT cache_id, query_text FROM answer_cache
            WHERE user_id = CAST(:u AS UUID) AND cache_id = ANY(CAST(:ids AS UUID[]))
        """), {"u": user_id, "ids": [c for c, _ in near]})}
        for cache_id, distance in near:
            if normalize_question(asked.get(cache_id) or "") != want:
                continue
            row = conn.execute(text("""
                UPDATE answer_cache SET hits = hits + 1
                WHERE cache_id = CAST(:c AS UUID) AND user_id = CAST(:u AS UUID)
                  AND data_version = :v AND language = :l AND expires_at > now()
                RETURNING answer, tokens
            """), {"c": cache_id, "u": user_id, "v": int(version), "l": language}).fetchone()
            if row is not None:
                return {"cache_id": cache_id, "answer": row[0], "tokens": int(row[1]), "distance": distance}
        return None

    hit = run_transaction(fn, url=url)
    record_stat("answer", "hit" if hit else "miss", hit["tokens"] if hit else 0)
    return hit


def put_answer(user_id: str, *, version: int, language: str, query: str, qvec: str, answer: str,
               tokens: int, url: str | None = None) -> None:
    url = request_url(url)
    hours = get_config().cache_ttl_hours

    def fn(conn) -> None:
        region = money_region(conn, user_id, url)
        rcol, rval = region_columns(region)
        conn.execute(text(
            "DELETE FROM answer_cache WHERE user_id = CAST(:u AS UUID) AND data_version < :v"
        ), {"u": user_id, "v": int(version)})
        conn.execute(text(f"""
            INSERT INTO answer_cache ({rcol}user_id, data_version, language, query_text, query_embedding,
                                      answer, tokens, expires_at)
            VALUES ({rval}CAST(:u AS UUID), :v, :l, :q, :qvec, :a, :t, now() + :h * INTERVAL '1 hour')
        """), {"u": user_id, "v": int(version), "l": language, "q": query[:2000], "qvec": qvec,
               "a": answer, "t": int(tokens), "h": hours, "crdb_region": region})

    run_transaction(fn, url=url, idempotent=False)
    record_stat("answer", "write")


def cache_summary(user_id: str, *, url: str | None = None) -> dict[str, Any]:
    """Counts for the Settings page and the trace."""
    url = request_url(url)
    with get_engine(url).connect() as conn:
        answers = conn.execute(text("SELECT count(*) FROM answer_cache WHERE user_id = CAST(:u AS UUID)"),
                               {"u": user_id}).scalar()
        searches = conn.execute(text("SELECT count(*) FROM search_cache WHERE user_id = CAST(:u AS UUID)"),
                                {"u": user_id}).scalar()
    return {"answers": int(answers or 0), "searches": int(searches or 0)}
