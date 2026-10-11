"""Coach insights — aggregation tools that turn raw expenses into the
analyses a real PFM agent needs to ground its nudges in evidence:

  - get_monthly_summary: total + by-category + top merchants for a month
  - get_spending_velocity: pace vs. budget, projected end-of-month
  - get_top_merchants: ranked merchants by spend over a window
  - detect_subscriptions: recurring-merchant + amount-drift detector

All functions accept `database_url` explicitly so the MCP server and
tests can pin the DB independent of process env.

Returns are JSON-serializable so the MCP layer can pass them through.
"""

from __future__ import annotations

import calendar
import statistics
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text


def _engine(database_url: str):
    from ..db.engine import get_engine
    return get_engine(database_url)


def get_monthly_summary(user_id: str, year: int, month: int,
                        database_url: str,
                        top_merchants_k: int = 5) -> dict[str, Any]:
    """Total spend, per-category breakdown, and top merchants for a given
    calendar month. The Coach uses this to ground budget-threshold nudges
    in concrete numbers."""
    first_day = date(year, month, 1)
    last_day = date(year, month, calendar.monthrange(year, month)[1])

    eng = _engine(database_url)
    with eng.connect() as conn:
        rows = conn.execute(text("""
            SELECT shopping_type AS category,
                   COALESCE(merchant, 'Unknown') AS merchant,
                   expense_amount
            FROM expenses
            WHERE user_id = :u
              AND expense_date >= :start
              AND expense_date <= :end
        """), {"u": user_id, "start": first_day, "end": last_day}).fetchall()

    if not rows:
        return {
            "year": year, "month": month, "total": 0.0,
            "transaction_count": 0,
            "by_category": [], "top_merchants": [],
        }

    total = 0.0
    by_cat: dict[str, float] = defaultdict(float)
    by_merch: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"amount": 0.0, "count": 0})
    for cat, merch, amount in rows:
        a = float(amount)
        total += a
        by_cat[cat] += a
        by_merch[merch]["amount"] += a
        by_merch[merch]["count"] += 1

    cats = [{"category": c, "amount": round(a, 2),
             "pct_of_total": round(a / total, 4) if total else 0.0}
            for c, a in sorted(by_cat.items(), key=lambda x: -x[1])]
    merchs = [{"merchant": m, "amount": round(v["amount"], 2),
               "transaction_count": v["count"]}
              for m, v in sorted(by_merch.items(),
                                  key=lambda x: -x[1]["amount"])][:top_merchants_k]

    return {
        "year": year, "month": month,
        "total": round(total, 2),
        "transaction_count": len(rows),
        "by_category": cats,
        "top_merchants": merchs,
    }


def get_spending_velocity(user_id: str, category: str,
                          monthly_budget: float,
                          database_url: str) -> dict[str, Any]:
    """How fast the user is burning this category's budget this month, and
    a linear projection of end-of-month spend. The Coach uses this to say
    things like 'at this pace you'll hit 110% by month-end.'"""
    today = date.today()
    first_day = date(today.year, today.month, 1)
    days_in_month = calendar.monthrange(today.year, today.month)[1]
    days_into_month = (today - first_day).days + 1

    eng = _engine(database_url)
    with eng.connect() as conn:
        row = conn.execute(text("""
            SELECT COALESCE(SUM(expense_amount), 0) AS spent,
                   COUNT(*) AS n
            FROM expenses
            WHERE user_id = :u
              AND shopping_type = :c
              AND expense_date >= :start
              AND expense_date <= :today
        """), {"u": user_id, "c": category, "start": first_day,
               "today": today}).fetchone()

    spent = float(row[0])
    txn_count = int(row[1])
    daily_rate = spent / days_into_month if days_into_month > 0 else 0.0
    projected_eom = daily_rate * days_in_month
    pct_of_budget = spent / monthly_budget if monthly_budget > 0 else 0.0
    projected_pct = projected_eom / monthly_budget if monthly_budget > 0 else 0.0

    return {
        "category": category,
        "monthly_budget": round(monthly_budget, 2),
        "spent_so_far": round(spent, 2),
        "transaction_count": txn_count,
        "days_into_month": days_into_month,
        "days_in_month": days_in_month,
        "daily_burn_rate": round(daily_rate, 2),
        "projected_eom": round(projected_eom, 2),
        "pct_of_budget": round(pct_of_budget, 4),
        "projected_pct_of_budget": round(projected_pct, 4),
        "on_track": projected_eom <= monthly_budget,
    }


def get_top_merchants(user_id: str, database_url: str,
                      days: int = 30, k: int = 5,
                      category: str | None = None) -> list[dict[str, Any]]:
    """Ranked list of merchants by spend over the last `days`, optionally
    filtered to a single category. Useful for the Coach to identify which
    merchants are driving a category overshoot."""
    cutoff = date.today() - timedelta(days=days)
    eng = _engine(database_url)
    with eng.connect() as conn:
        if category:
            rows = conn.execute(text("""
                SELECT COALESCE(merchant, 'Unknown') AS merchant,
                       SUM(expense_amount) AS total,
                       COUNT(*) AS n
                FROM expenses
                WHERE user_id = :u
                  AND shopping_type = :c
                  AND expense_date >= :cutoff
                GROUP BY merchant
                ORDER BY total DESC
                LIMIT :k
            """), {"u": user_id, "c": category, "cutoff": cutoff,
                   "k": k}).fetchall()
        else:
            rows = conn.execute(text("""
                SELECT COALESCE(merchant, 'Unknown') AS merchant,
                       SUM(expense_amount) AS total,
                       COUNT(*) AS n
                FROM expenses
                WHERE user_id = :u
                  AND expense_date >= :cutoff
                GROUP BY merchant
                ORDER BY total DESC
                LIMIT :k
            """), {"u": user_id, "cutoff": cutoff, "k": k}).fetchall()
    return [{
        "merchant": r[0],
        "total": round(float(r[1]), 2),
        "transaction_count": int(r[2]),
    } for r in rows]


def detect_subscriptions(user_id: str, database_url: str,
                         lookback_days: int = 120,
                         min_occurrences: int = 3
                         ) -> list[dict[str, Any]]:
    """Heuristic recurring-charge detector: a merchant appearing
    `min_occurrences`+ times in `lookback_days` with relatively stable
    cadence and amount. Returns each candidate with typical amount,
    occurrence count, and an `amount_drift` flag when the most recent
    charge differs meaningfully (>10%) from the prior typical amount —
    that's exactly the situation Plan 2-A's RECURRING_DRIFT signal would
    fire on, and the Coach can cite the change."""
    cutoff = date.today() - timedelta(days=lookback_days)
    eng = _engine(database_url)
    with eng.connect() as conn:
        rows = conn.execute(text("""
            SELECT COALESCE(merchant, 'Unknown') AS merchant,
                   expense_date,
                   expense_amount
            FROM expenses
            WHERE user_id = :u
              AND expense_date >= :cutoff
              AND merchant IS NOT NULL
            ORDER BY merchant, expense_date
        """), {"u": user_id, "cutoff": cutoff}).fetchall()

    by_merchant: dict[str, list[tuple[date, float]]] = defaultdict(list)
    for merch, d, amt in rows:
        by_merchant[merch].append((d, float(amt)))

    candidates = []
    for merch, charges in by_merchant.items():
        if len(charges) < min_occurrences:
            continue
        charges.sort(key=lambda x: x[0])
        amounts = [a for _, a in charges]
        latest_amount = amounts[-1]
        prior_amounts = amounts[:-1]
        typical_amount = statistics.median(prior_amounts) if prior_amounts \
            else latest_amount

        amount_drift = False
        if typical_amount > 0:
            drift_pct = abs(latest_amount - typical_amount) / typical_amount
            amount_drift = drift_pct > 0.10

        gaps_days = [(charges[i+1][0] - charges[i][0]).days
                     for i in range(len(charges) - 1)]
        avg_gap = statistics.mean(gaps_days) if gaps_days else None

        candidates.append({
            "merchant": merch,
            "occurrence_count": len(charges),
            "typical_amount": round(typical_amount, 2),
            "latest_amount": round(latest_amount, 2),
            "amount_drift": amount_drift,
            "avg_gap_days": round(avg_gap, 1) if avg_gap is not None else None,
            "first_seen": charges[0][0].isoformat(),
            "last_seen": charges[-1][0].isoformat(),
        })

    candidates.sort(key=lambda c: -c["typical_amount"])
    return candidates


INSIGHTS_TOOLS = {
    "get_monthly_summary": get_monthly_summary,
    "get_spending_velocity": get_spending_velocity,
    "get_top_merchants": get_top_merchants,
    "detect_subscriptions": detect_subscriptions,
}
