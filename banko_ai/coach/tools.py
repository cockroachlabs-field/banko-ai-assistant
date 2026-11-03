"""Coach tools — the agent's read/write contract with CockroachDB.

The MCP server in Plan 2-B wraps this same module. Keep return types
JSON-serializable (dicts of primitives, lists of dicts, ISO timestamps
as strings) so the MCP layer can pass results through unchanged.

All functions accept `database_url` explicitly rather than reading env
inside so tests can pin to a known DB and the MCP layer can override.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import text

from .insights import (
    detect_subscriptions,
    get_monthly_summary,
    get_spending_velocity,
    get_top_merchants,
)

_DEFAULT_BUDGETS_USD = {
    "dining": 400.0,
    "groceries": 600.0,
    "travel": 500.0,
    "transport": 200.0,
    "entertainment": 200.0,
    "shopping": 300.0,
    "utilities": 250.0,
    "other": 300.0,
}


def _engine(database_url: str):
    from ..db.engine import get_engine
    return get_engine(database_url)


def get_user_budget(user_id: str, category: str,
                    database_url: str) -> dict[str, Any]:
    """Return the user's monthly budget for `category`. Falls back to a
    sensible default if the user has not set one. The Coach UI also calls
    this to seed the budget editor."""
    eng = _engine(database_url)
    with eng.connect() as conn:
        row = conn.execute(text("""
            SELECT amount FROM user_budgets
            WHERE user_id = :u AND category = :c
        """), {"u": user_id, "c": category}).fetchone()
    if row is not None:
        return {
            "user_id": user_id,
            "category": category,
            "monthly_budget": float(row[0]),
            "source": "user_override",
        }
    return {
        "user_id": user_id,
        "category": category,
        "monthly_budget": _DEFAULT_BUDGETS_USD.get(category.lower(), 300.0),
        "source": "default",
    }


def set_budget(user_id: str, category: str, amount: float,
               database_url: str) -> dict[str, Any]:
    """Insert or update the user's budget for `category`. Creates the
    `user_budgets` table on first call (idempotent) so this works on a
    fresh DB."""
    eng = _engine(database_url)
    with eng.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_budgets (
              user_id   UUID NOT NULL,
              category  STRING NOT NULL,
              amount    DECIMAL NOT NULL,
              updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
              PRIMARY KEY (user_id, category)
            )
        """))
        conn.execute(text("""
            UPSERT INTO user_budgets (user_id, category, amount, updated_at)
            VALUES (:u, :c, :a, now())
        """), {"u": user_id, "c": category, "a": amount})
        conn.commit()
    return {"user_id": user_id, "category": category,
            "monthly_budget": float(amount), "source": "user_override"}


def get_recent_signals(user_id: str, database_url: str,
                       limit: int = 20) -> list[dict[str, Any]]:
    """Return the user's most recent spending_signals, newest first."""
    eng = _engine(database_url)
    with eng.connect() as conn:
        rows = conn.execute(text("""
            SELECT signal_id, signal_type, severity, payload,
                   produced_at, consumed_at
            FROM spending_signals
            WHERE user_id = :u
            ORDER BY produced_at DESC
            LIMIT :l
        """), {"u": user_id, "l": limit}).fetchall()
    return [{
        "signal_id": str(r[0]),
        "signal_type": r[1],
        "severity": r[2],
        "payload": r[3] if isinstance(r[3], dict) else json.loads(r[3] or "{}"),
        "produced_at": r[4].isoformat() if r[4] else None,
        "consumed_at": r[5].isoformat() if r[5] else None,
    } for r in rows]


def get_recent_transactions(user_id: str, database_url: str,
                            limit: int | None = None,
                            category: str | None = None,
                            days: int | None = None) -> list[dict[str, Any]]:
    """Return the user's recent expense rows, newest first. Optional
    category filter and lookback window. Maps real expenses columns
    (`expense_id`, `expense_amount`, `shopping_type`) to the names the
    agent and MCP layer expect (`id`, `amount`, `category`).

    Defaults for `limit` and `days` come from Config when not supplied,
    so deployments can tune for their data volumes without code changes.
    """
    if limit is None or days is None:
        from ..config.settings import get_config
        cfg = get_config()
        if limit is None:
            limit = cfg.coach_tx_default_limit
        if days is None:
            days = cfg.coach_agg_lookback_days
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    eng = _engine(database_url)
    with eng.connect() as conn:
        if category:
            rows = conn.execute(text("""
                SELECT expense_id, description, expense_amount,
                       shopping_type, expense_date
                FROM expenses
                WHERE user_id = :u
                  AND shopping_type = :c
                  AND expense_date >= :cutoff
                ORDER BY expense_date DESC
                LIMIT :l
            """), {"u": user_id, "c": category, "cutoff": cutoff,
                   "l": limit}).fetchall()
        else:
            rows = conn.execute(text("""
                SELECT expense_id, description, expense_amount,
                       shopping_type, expense_date
                FROM expenses
                WHERE user_id = :u
                  AND expense_date >= :cutoff
                ORDER BY expense_date DESC
                LIMIT :l
            """), {"u": user_id, "cutoff": cutoff, "l": limit}).fetchall()
    return [{
        "id": str(r[0]),
        "description": r[1],
        "amount": float(r[2]),
        "category": r[3],
        "expense_date": r[4].isoformat() if r[4] else None,
    } for r in rows]


def explain_nudge(nudge_id: str, database_url: str) -> dict[str, Any]:
    """Return the full record for a nudge: message, tool trace, provider,
    correlation IDs. Used by the UI's 'show evidence' panel and the MCP
    tool of the same name."""
    eng = _engine(database_url)
    with eng.connect() as conn:
        row = conn.execute(text("""
            SELECT nudge_id, signal_id, user_id, message, tool_trace,
                   provider_used, trace_id, created_at
            FROM coach_nudges
            WHERE nudge_id = :n
        """), {"n": nudge_id}).fetchone()
    if row is None:
        return {}
    return {
        "nudge_id": str(row[0]),
        "signal_id": str(row[1]) if row[1] else None,
        "user_id": str(row[2]),
        "message": row[3],
        "tool_trace": row[4] if isinstance(row[4], list)
                      else (json.loads(row[4]) if row[4] else []),
        "provider_used": row[5],
        "trace_id": row[6],
        "created_at": row[7].isoformat() if row[7] else None,
    }


COACH_TOOLS = {
    "get_user_budget": get_user_budget,
    "set_budget": set_budget,
    "get_recent_signals": get_recent_signals,
    "get_recent_transactions": get_recent_transactions,
    "explain_nudge": explain_nudge,
    "get_monthly_summary": get_monthly_summary,
    "get_spending_velocity": get_spending_velocity,
    "get_top_merchants": get_top_merchants,
    "detect_subscriptions": detect_subscriptions,
}
