"""What the agents did: decisions for one expense (the transaction drawer)
and, from Task 12, the activity feed and stats for the Agents page.
agent_decisions has no user column; callers check the expense belongs to
the user before asking about it."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import text

from ..db.engine import get_engine
from ..db.rls import request_url
from ._sql import jsonable, read_session, valid_uuid


def _obj(value):
    return value if isinstance(value, (dict, list)) or value is None else json.loads(value)


_AGENT_FOR_DECISION = {"fraud": "fraud", "budget": "budget", "receipt": "receipt", "coach": "coach"}


def _agent_of(decision_type: str) -> str:
    for prefix, agent in _AGENT_FOR_DECISION.items():
        if (decision_type or "").startswith(prefix):
            return agent
    return "assistant"


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


def activity(user_id: str, *, limit: int = 60, url: str | None = None) -> list[dict]:
    """This user's receipts and nudges, plus agent decisions about this user's
    recent expenses. agent_decisions is shared, so it is filtered through the
    user's own expense ids."""
    if not valid_uuid(user_id):
        return []
    limit = max(1, min(int(limit), 200))
    url = request_url(url)
    with read_session(user_id, url=url) as conn:
        receipts = conn.execute(text("""
            SELECT document_id, status, extracted->>'merchant', extracted->>'amount', error, created_at
            FROM receipts@receipts_user_idx WHERE user_id = CAST(:u AS UUID) ORDER BY created_at DESC LIMIT :n
        """), {"u": user_id, "n": limit}).fetchall()
        nudges = conn.execute(text("""
            SELECT nudge_id, message, created_at FROM coach_nudges
            WHERE user_id = CAST(:u AS UUID) ORDER BY created_at DESC LIMIT :n
        """), {"u": user_id, "n": limit}).fetchall()
        expense_ids = [str(r[0]) for r in conn.execute(text("""
            SELECT expense_id FROM expenses@idx_expenses_user_date WHERE user_id = CAST(:u AS UUID)
            ORDER BY expense_date DESC LIMIT 500
        """), {"u": user_id}).fetchall()]
    with get_engine(url).connect() as conn:
        decisions = conn.execute(text("""
            SELECT d.decision_id, d.decision_type, d.reasoning, d.confidence, d.created_at, s.region
            FROM agent_decisions AS d LEFT JOIN agent_state AS s ON s.agent_id = d.agent_id
            WHERE d.context->>'expense_id' = ANY(CAST(:ids AS STRING[]))
            ORDER BY d.created_at DESC LIMIT :n
        """), {"ids": expense_ids, "n": limit}).fetchall() if expense_ids else []
    items = [{"id": f"receipt:{r[0]}", "agent": "receipt", "kind": r[1],
              "title": (r[2] or "Receipt") + (f" · ${float(r[3]):,.2f}" if r[3] else ""),
              "detail": r[4] or {"posted": "Posted through the ledger", "duplicate": "Matched a purchase you already had",
                                 "processing": "Working on it", "failed": "Could not be read"}.get(r[1], r[1]),
              "confidence": None, "region": None, "at": r[5]} for r in receipts]
    items += [{"id": f"nudge:{r[0]}", "agent": "coach", "kind": "nudge", "title": "Coach nudge", "detail": r[1],
               "confidence": None, "region": None, "at": r[2]} for r in nudges]
    items += [{"id": f"decision:{r[0]}", "agent": _agent_of(r[1]), "kind": r[1], "title": r[1].replace("_", " ").capitalize(),
               "detail": r[2] or "", "confidence": float(r[3]) if r[3] is not None else None, "region": r[5], "at": r[4]}
              for r in decisions]
    items.sort(key=lambda i: i["at"], reverse=True)
    return jsonable(items[:limit])


def stats(user_id: str, *, url: str | None = None) -> dict:
    with get_engine(url).connect() as conn:
        agents = conn.execute(text("""
            SELECT agent_type, region, status, last_heartbeat FROM agent_state ORDER BY agent_type, region
        """)).fetchall()
    today: dict[str, int] = {}
    utc_today = datetime.now(timezone.utc).date().isoformat()
    for item in activity(user_id, limit=200, url=url):
        if str(item["at"])[:10] == utc_today:
            today[item["agent"]] = today.get(item["agent"], 0) + 1
    return jsonable({"agents": [{"agent_type": a[0], "region": a[1], "status": a[2], "last_heartbeat": a[3]} for a in agents],
                     "today": today})
