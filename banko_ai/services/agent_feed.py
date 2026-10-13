"""What the agents did: decisions for one expense (the transaction drawer)
and, from Task 12, the activity feed and stats for the Agents page.
agent_decisions has no user column; callers check the expense belongs to
the user before asking about it."""

from __future__ import annotations

import json

from sqlalchemy import text

from ..db.engine import get_engine
from ._sql import jsonable, valid_uuid


def _obj(value):
    return value if isinstance(value, (dict, list)) or value is None else json.loads(value)


def decisions_for_expense(expense_id: str, *, url: str | None = None) -> list[dict]:
    if not valid_uuid(expense_id):
        return []
    with get_engine(url).connect() as conn:
        rows = conn.execute(text("""
            SELECT decision_id, decision_type, confidence, reasoning, action, created_at
            FROM agent_decisions WHERE context->>'expense_id' = :x
            ORDER BY created_at DESC LIMIT 5
        """), {"x": str(expense_id).lower()}).fetchall()
    return jsonable([{"decision_id": str(r[0]), "decision_type": r[1], "confidence": r[2], "reasoning": r[3],
                      "action": _obj(r[4]), "created_at": r[5]} for r in rows])
