"""Deterministic SQL aggregation for spending questions.

The LLM narrates these numbers; it never computes them. That is the whole
point: every provider returns the same figure because the figure comes from
one SQL statement against the user's own rows.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

from sqlalchemy import create_engine, text

from .intent_classifier import AggregationIntent


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


@lru_cache(maxsize=4)
def _distinct_categories(database_url: str) -> tuple[str, ...]:
    eng = create_engine(database_url)
    try:
        with eng.connect() as c:
            rows = c.execute(
                text("SELECT DISTINCT shopping_type FROM expenses")).fetchall()
        return tuple(r[0] for r in rows if r[0])
    finally:
        eng.dispose()


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


def _where_and_params(intent: AggregationIntent, user_id: str,
                      category: str | None,
                      region: str | None) -> tuple[str, dict]:
    """One WHERE builder for the query and its EXPLAIN so they can never
    drift apart. The region predicate prunes the scan to the user's home
    partition on REGIONAL BY ROW tables; single-region callers pass None
    and the SQL never mentions crdb_region."""
    where = ("WHERE user_id = :u AND expense_date >= :s "
             "AND expense_date < :e")
    params: dict = {"u": user_id, "s": intent.window_start,
                    "e": intent.window_end}
    if category:
        where += " AND shopping_type = :c"
        params["c"] = category
    if region:
        where += " AND crdb_region = :r"
        params["r"] = region
    return where, params


def run_aggregation(intent: AggregationIntent, user_id: str,
                    database_url: str,
                    region: str | None = None) -> AggregationResult:
    category = resolve_category(intent.subject, database_url)
    where, params = _where_and_params(intent, user_id, category, region)

    eng = create_engine(database_url)
    try:
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
    finally:
        eng.dispose()

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
    )


def explain_aggregation(intent: AggregationIntent, user_id: str,
                        database_url: str,
                        region: str | None = None) -> str:
    """Run EXPLAIN ANALYZE of the aggregation query and return the plan as text.

    EXPLAIN does not support placeholders with psycopg3, so we inline parameters."""
    category = resolve_category(intent.subject, database_url)
    where, params = _where_and_params(intent, user_id, category, region)

    # Build WHERE clause with inlined values (EXPLAIN doesn't support placeholders)
    inlined_where = where
    for key, value in params.items():
        if isinstance(value, str):
            # Quote and escape strings
            escaped = value.replace("'", "''")
            inlined_where = inlined_where.replace(f":{key}", f"'{escaped}'")
        elif isinstance(value, (int, float)):
            inlined_where = inlined_where.replace(f":{key}", str(value))
        else:
            # For dates and other types, convert to string representation
            inlined_where = inlined_where.replace(f":{key}", f"'{str(value)}'")

    eng = create_engine(database_url)
    try:
        with eng.connect() as c:
            plan_rows = c.execute(text(f"""
                EXPLAIN ANALYZE
                SELECT COALESCE(SUM(expense_amount), 0),
                       COUNT(*),
                       MIN(expense_date), MAX(expense_date)
                FROM expenses {inlined_where}
            """)).fetchall()
    finally:
        eng.dispose()

    return "\n".join(str(row[0]) for row in plan_rows)
