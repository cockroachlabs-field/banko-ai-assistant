"""Recurring bills and subscriptions, detected from spending, no extra table.

Per merchant, all three must hold:
- the gaps between charges (lag() over dates) settle on a weekly, monthly
  or yearly cadence with little scatter;
- the amounts are steady (at most two distinct values, or a small spread);
- there are enough charges.

A price change is reported only for a clean step from one price to another,
so a variable utility bill never counts as one."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import text

from ..config.catalog import get_catalog
from ..config.settings import get_config
from ._sql import jsonable, read_session

_CADENCES = (("weekly", 5, 9, 2), ("monthly", 26, 35, 4), ("yearly", 350, 380, 10))


@dataclass(frozen=True)
class Recurring:
    merchant: str
    category: str
    cadence: str
    amount: Decimal
    previous_amount: Decimal
    price_change_pct: float | None
    last_date: date
    next_due: date
    occurrences: int


def _cadence(gaps: list[int]) -> str | None:
    if not gaps:
        return None
    median = statistics.median(gaps)
    scatter = statistics.median(abs(g - median) for g in gaps)
    for name, low, high, tolerance in _CADENCES:
        if low <= median <= high and scatter <= tolerance:
            return name
    return None


def _steady(amounts: list[Decimal]) -> bool:
    if len(set(amounts)) <= 2:
        return True
    mean = statistics.mean(float(a) for a in amounts)
    return mean > 0 and statistics.pstdev(float(a) for a in amounts) / mean <= 0.15


def detect_recurring(user_id: str, *, today: date | None = None, url: str | None = None) -> list[Recurring]:
    cfg = get_config()
    today = today or datetime.now(timezone.utc).date()
    with read_session(user_id, url=url, follower=True) as conn:
        rows = conn.execute(text("""
            SELECT lower(merchant) AS m, merchant, shopping_type, expense_date, expense_amount,
                   expense_date - lag(expense_date) OVER (PARTITION BY lower(merchant) ORDER BY expense_date) AS gap
            FROM expenses
            WHERE user_id = CAST(:u AS UUID) AND merchant IS NOT NULL AND expense_date >= :since
            ORDER BY m, expense_date
        """), {"u": user_id, "since": today - timedelta(days=400)}).fetchall()
    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(row[0], []).append(row)
    found = []
    for charges in groups.values():
        if len(charges) < cfg.recurring_min_occurrences:
            continue
        gaps = [int(c[5]) for c in charges if c[5] is not None and int(c[5]) > 0]
        cadence = _cadence(gaps)
        amounts = [c[4] for c in charges]
        if cadence is None or not _steady(amounts):
            continue
        last = amounts[-1]
        change = None
        previous = amounts[-2]
        if len(set(amounts)) == 2:
            previous = next(a for a in reversed(amounts) if a != last)
            pct = float((last - previous) / previous * 100) if previous else 0.0
            if abs(pct) >= cfg.recurring_price_change_pct:
                change = round(pct, 2)
        found.append(Recurring(
            merchant=charges[-1][1], category=get_catalog().category_for(charges[-1][2]).label,
            cadence=cadence, amount=last, previous_amount=previous, price_change_pct=change,
            last_date=charges[-1][3], next_due=charges[-1][3] + timedelta(days=round(statistics.median(gaps))),
            occurrences=len(charges)))
    return sorted(found, key=lambda r: r.next_due)


def recurring_json(items: list[Recurring]) -> list[dict]:
    out = []
    for r in items:
        cat = get_catalog().category_for(r.category)
        out.append({**asdict(r), "icon": cat.icon, "color": cat.color})
    return jsonable(out)


def upcoming_bills(user_id: str, *, days: int = 14, today: date | None = None, url: str | None = None) -> list[dict]:
    today = today or datetime.now(timezone.utc).date()
    horizon = today + timedelta(days=days)
    return recurring_json([r for r in detect_recurring(user_id, today=today, url=url)
                           if today <= r.next_due <= horizon])
