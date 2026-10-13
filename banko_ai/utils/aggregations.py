"""Deterministic SQL aggregation for spending questions.

The LLM narrates these numbers; it never computes them. That is the whole
point: every provider returns the same figure because the figure comes from
one SQL statement against the user's own rows.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

from sqlalchemy import text

from .intent_classifier import AggregationIntent


def _to_pyformat(sql: str) -> str:
    """Convert :name placeholders to %(name)s format for psycopg's client-side binding.

    Word boundaries ensure :start and :start_date convert independently.
    Negative lookbehind (?<!:) ensures :: type casts are left alone.
    """
    return re.sub(r"(?<!:):(\w+)\b", r"%(\1)s", sql)


@dataclass(frozen=True)
class AggregationResult:
    operation: str
    category: str | None
    window_start: date
    window_end: date
    total: float
    count: int
    average: float
    first_date: date | None
    last_date: date | None
    rows: list[dict] = field(default_factory=list)
    merchant: str | None = None


@lru_cache(maxsize=4)
def _distinct_categories(database_url: str) -> tuple[str, ...]:
    from ..db.engine import get_engine
    eng = get_engine(database_url)
    with eng.connect() as c:
        rows = c.execute(
            text("SELECT DISTINCT shopping_type FROM expenses")).fetchall()
    return tuple(r[0] for r in rows if r[0])


def resolve_category(subject: str | None, database_url: str) -> str | None:
    """Fuzzy map a question phrase ("restaurants") to a real shopping_type
    ("Restaurant"). None means no confident match; the caller aggregates
    across all categories and says so."""
    if not subject:
        return None
    cats = _distinct_categories(database_url)
    if not cats:
        return None
    subject_n = subject.lower().rstrip("s")
    for cat in cats:
        if cat.lower().rstrip("s") == subject_n:
            return cat
    close = difflib.get_close_matches(
        subject_n, [c.lower() for c in cats], n=1, cutoff=0.75)
    if close:
        return next(c for c in cats if c.lower() == close[0])
    return None


def resolve_merchant(subject: str | None, user_id: str, database_url: str) -> str | None:
    """Map a question phrase ("starbucks", "blue bottle coffee") to one of
    this user's merchants, or None. Exact match first, then containment for
    phrases of four letters or more, then a close fuzzy match."""
    if not subject:
        return None
    from ..db.engine import get_engine
    with get_engine(database_url).connect() as c:
        names = [r[0] for r in c.execute(text(
            "SELECT DISTINCT merchant FROM expenses WHERE user_id = CAST(:u AS UUID) AND merchant IS NOT NULL"),
            {"u": user_id}) if r[0]]
    want = subject.lower().replace("'", "").strip()
    flat = {n.lower().replace("'", ""): n for n in names}
    if want in flat:
        return flat[want]
    if len(want) >= 4:
        for low, name in flat.items():
            if want in low or low in want:
                return name
    close = difflib.get_close_matches(want, list(flat), n=1, cutoff=0.8)
    return flat[close[0]] if close else None


def _where_and_params(intent: AggregationIntent, user_id: str,
                      category: str | None,
                      region: str | None,
                      merchant: str | None = None) -> tuple[str, dict]:
    """One WHERE builder for the query and its EXPLAIN so they can never
    drift apart. The region predicate prunes the scan to the user's home
    partition on REGIONAL BY ROW tables; single-region callers pass None
    and the SQL never mentions crdb_region."""
    where = ("WHERE user_id = :u AND expense_date >= :s "
             "AND expense_date < :e")
    params: dict = {"u": user_id, "s": intent.window_start,
                    "e": intent.window_end}
    if merchant:
        where += " AND lower(merchant) = lower(:m)"
        params["m"] = merchant
    elif category:
        where += " AND shopping_type = :c"
        params["c"] = category
    if region:
        where += " AND crdb_region = :r"
        params["r"] = region
    return where, params


def run_aggregation(intent: AggregationIntent, user_id: str,
                    database_url: str,
                    region: str | None = None,
                    merchant: str | None = None) -> AggregationResult:
    from ..db.engine import get_engine
    category = None if merchant else resolve_category(intent.subject, database_url)
    where, params = _where_and_params(intent, user_id, category, region, merchant)

    eng = get_engine(database_url)
    with eng.connect() as c:
        agg = c.execute(text(f"""
            SELECT COALESCE(SUM(expense_amount), 0),
                   COUNT(*),
                   MIN(expense_date), MAX(expense_date)
            FROM expenses {where}
        """), params).fetchone()
        detail = c.execute(text(f"""
            SELECT expense_date, merchant, expense_amount
            FROM expenses {where}
            ORDER BY expense_date DESC
            LIMIT 25
        """), params).fetchall()

    total = float(agg[0] or 0)
    count = int(agg[1] or 0)
    return AggregationResult(
        operation=intent.operation,
        category=category,
        window_start=intent.window_start,
        window_end=intent.window_end,
        total=round(total, 2),
        count=count,
        average=round(total / count, 2) if count else 0.0,
        first_date=agg[2],
        last_date=agg[3],
        rows=[{"date": str(r[0]), "merchant": r[1],
               "amount": float(r[2])} for r in detail],
        merchant=merchant,
    )


def explain_aggregation(intent: AggregationIntent, user_id: str,
                        database_url: str,
                        region: str | None = None,
                        merchant: str | None = None) -> str:
    """Run EXPLAIN ANALYZE of the aggregation query and return the plan as text.

    CockroachDB does not accept bound placeholders in EXPLAIN under server-side binding,
    so this statement binds parameters client-side with psycopg.ClientCursor."""
    import psycopg

    category = None if merchant else resolve_category(intent.subject, database_url)
    where, params = _where_and_params(intent, user_id, category, region, merchant)

    # Convert :name placeholders to %(name)s for client-side binding
    where_pyformat = _to_pyformat(where)

    sql = f"""
        EXPLAIN ANALYZE
        SELECT COALESCE(SUM(expense_amount), 0),
               COUNT(*),
               MIN(expense_date), MAX(expense_date)
        FROM expenses {where_pyformat}
    """

    from ..db.engine import get_engine
    eng = get_engine(database_url)
    with eng.connect() as c:
        # Get the raw psycopg connection for client-side binding
        raw_conn = c.connection.driver_connection
        with psycopg.ClientCursor(raw_conn) as cur:
            cur.execute(sql, params)
            plan_rows = cur.fetchall()

    return "\n".join(str(row[0]) for row in plan_rows)
