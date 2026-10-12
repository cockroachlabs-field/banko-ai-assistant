"""Savings goals: progress, pace from the last 90 days, projected completion."""

from __future__ import annotations

import calendar
import math
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import text

from . import ledger
from ._sql import jsonable, read_session


def _add_months(day: date, months: int) -> date:
    total = day.year * 12 + day.month - 1 + months
    year, month = total // 12, total % 12 + 1
    return date(year, month, min(day.day, calendar.monthrange(year, month)[1]))


def list_goals(user_id: str, *, today: date | None = None, url: str | None = None) -> list[dict]:
    today = today or datetime.now(timezone.utc).date()
    since = datetime.combine(today - timedelta(days=90), datetime.min.time(), tzinfo=timezone.utc)
    with read_session(user_id, url=url) as conn:
        rows = conn.execute(text("""
            SELECT g.goal_id, g.name, g.icon, g.target_amount, g.target_date, g.account_id,
                   COALESCE(sum(l.amount) FILTER (WHERE l.amount > 0), 0) AS saved,
                   COALESCE(sum(l.amount) FILTER (WHERE l.amount > 0 AND l.posted_at >= :since), 0) AS recent
            FROM goals AS g
            LEFT JOIN ledger_entries AS l ON l.goal_id = g.goal_id AND l.account_id = g.account_id
            WHERE g.user_id = CAST(:u AS UUID)
            GROUP BY g.goal_id, g.name, g.icon, g.target_amount, g.target_date, g.account_id
            ORDER BY g.name
        """), {"u": user_id, "since": since}).fetchall()
    out = []
    for goal_id, name, icon, target, target_date, account_id, saved, recent in rows:
        pace = recent / 3
        remaining = max(target - saved, Decimal("0"))
        projected = None
        if remaining == 0:
            projected = today
        elif pace > 0:
            projected = _add_months(today, math.ceil(remaining / pace))
        out.append({"goal_id": str(goal_id), "name": name, "icon": icon, "target_amount": target,
                    "target_date": target_date, "account_id": str(account_id), "saved": saved,
                    "remaining": remaining, "progress": min(saved / target, Decimal("1")),
                    "monthly_pace": pace, "projected_completion": projected,
                    "on_track": bool(projected and target_date and projected <= target_date)})
    return jsonable(out)


def contribute(user_id: str, goal_id: str, from_account: str, amount: Any, *, idempotency_key: str,
               url: str | None = None) -> ledger.TransferResult:
    return ledger.goal_contribution(user_id, goal_id, from_account, amount,
                                    idempotency_key=idempotency_key, url=url)
