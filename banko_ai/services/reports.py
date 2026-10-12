"""Cash flow, spending pace and the dashboard, composed from the other services."""

from __future__ import annotations

import calendar
from datetime import date, datetime, timezone
from decimal import Decimal

from sqlalchemy import text

from ..config.catalog import get_catalog
from . import accounts, budgets, recurring, transactions
from ._sql import jsonable, read_session


def _month_labels(today: date, months: int) -> list[str]:
    labels, year, mon = [], today.year, today.month
    for _ in range(months):
        labels.append(f"{year:04d}-{mon:02d}")
        year, mon = (year - 1, 12) if mon == 1 else (year, mon - 1)
    return labels[::-1]


def cashflow(user_id: str, *, months: int = 6, today: date | None = None, url: str | None = None) -> dict:
    today = today or datetime.now(timezone.utc).date()
    labels = _month_labels(today, months)
    start = date.fromisoformat(f"{labels[0]}-01")
    with read_session(user_id, url=url, follower=True) as conn:
        income = dict(conn.execute(text("""
            SELECT to_char(posted_at AT TIME ZONE 'UTC', 'YYYY-MM'), sum(amount) FROM ledger_entries
            WHERE user_id = CAST(:u AS UUID) AND kind = 'income' AND posted_at >= :s GROUP BY 1
        """), {"u": user_id, "s": start}).fetchall())
        spending = dict(conn.execute(text("""
            SELECT to_char(expense_date, 'YYYY-MM'), sum(expense_amount) FROM expenses
            WHERE user_id = CAST(:u AS UUID) AND expense_date >= :s GROUP BY 1
        """), {"u": user_id, "s": start}).fetchall())
        cats = conn.execute(text("""
            SELECT shopping_type, sum(expense_amount), count(*) FROM expenses
            WHERE user_id = CAST(:u AS UUID) AND expense_date >= :s GROUP BY 1
        """), {"u": user_id, "s": start}).fetchall()
        merchants = conn.execute(text("""
            SELECT merchant, sum(expense_amount), count(*) FROM expenses
            WHERE user_id = CAST(:u AS UUID) AND expense_date >= :s AND merchant IS NOT NULL
            GROUP BY 1 ORDER BY 2 DESC LIMIT 8
        """), {"u": user_id, "s": start}).fetchall()
    catalog = get_catalog()
    folded: dict[str, list] = {}
    for label, total, count in cats:
        cat = catalog.category_for(label)
        entry = folded.setdefault(cat.key, [cat, Decimal("0"), 0])
        entry[1] += total
        entry[2] += count
    month_rows = []
    for label in labels:
        inc, spent = income.get(label, Decimal("0")), spending.get(label, Decimal("0"))
        month_rows.append({"month": label, "income": inc, "spending": spent, "net": inc - spent,
                           "savings_rate": ((inc - spent) / inc) if inc else None})
    return jsonable({
        "months": month_rows,
        "categories": sorted(({"category": c.label, "key": c.key, "icon": c.icon, "color": c.color,
                               "group": c.grp, "total": t, "count": n} for c, t, n in folded.values()),
                             key=lambda x: x["total"], reverse=True),
        "merchants": [{"merchant": m, "total": t, "count": n} for m, t, n in merchants],
    })


def spending_pace(user_id: str, *, today: date | None = None, url: str | None = None) -> dict:
    today = today or datetime.now(timezone.utc).date()
    this_first = today.replace(day=1)
    last_first = date(this_first.year - (this_first.month == 1),
                      12 if this_first.month == 1 else this_first.month - 1, 1)
    with read_session(user_id, url=url, follower=True) as conn:
        rows = conn.execute(text("""
            SELECT expense_date, sum(expense_amount) FROM expenses
            WHERE user_id = CAST(:u AS UUID) AND expense_date >= :s AND expense_date <= :t GROUP BY 1
        """), {"u": user_id, "s": last_first, "t": today}).fetchall()
    daily = dict(rows)

    def cumulative(first: date, last_day: int) -> list[dict]:
        total, out = Decimal("0"), []
        for d in range(1, last_day + 1):
            total += daily.get(first.replace(day=d), Decimal("0"))
            out.append({"day": d, "total": total})
        return out

    this_month = cumulative(this_first, today.day)
    last_month = cumulative(last_first, calendar.monthrange(last_first.year, last_first.month)[1])
    same_day = last_month[min(today.day, len(last_month)) - 1]["total"]
    return jsonable({"this_month": this_month, "last_month": last_month,
                     "this_total": this_month[-1]["total"], "last_same_day": same_day})


def latest_insight(user_id: str, *, url: str | None = None) -> dict | None:
    with read_session(user_id, url=url) as conn:
        row = conn.execute(text("SELECT message, created_at FROM coach_nudges WHERE user_id = CAST(:u AS UUID) "
                                "ORDER BY created_at DESC LIMIT 1"), {"u": user_id}).fetchone()
    return jsonable({"message": row[0], "created_at": row[1]}) if row else None


def dashboard(user_id: str, *, today: date | None = None, url: str | None = None) -> dict:
    return {
        "net_worth": accounts.list_accounts(user_id, url=url)["net_worth"],
        "net_worth_history": accounts.net_worth_history(user_id, months=6, today=today, url=url),
        "spending": spending_pace(user_id, today=today, url=url),
        "budgets": budgets.budget_progress(user_id, today=today, url=url),
        "upcoming_bills": recurring.upcoming_bills(user_id, days=14, today=today, url=url),
        "recent": transactions.list_transactions(user_id, limit=6, url=url)["items"],
        "insight": latest_insight(user_id, url=url),
    }
