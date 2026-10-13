"""Cash flow, spending pace and the dashboard, composed from the other services."""

from __future__ import annotations

import calendar
from datetime import date, datetime, timezone
from decimal import Decimal

from sqlalchemy import text

from ..config.catalog import get_catalog
from ..db.rls import request_url
from . import accounts, budgets, recurring, transactions
from ._sql import jsonable, month_labels, read_session, valid_uuid


def cashflow(user_id: str, *, months: int = 6, today: date | None = None, url: str | None = None) -> dict:
    url = request_url(url)
    today = today or datetime.now(timezone.utc).date()
    labels = month_labels(today, months)
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
                           "savings_rate": float(round((inc - spent) / inc, 4)) if inc else None})
    return jsonable({
        "months": month_rows,
        "categories": sorted(({"category": c.label, "key": c.key, "icon": c.icon, "color": c.color,
                               "group": c.grp, "total": t, "count": n} for c, t, n in folded.values()),
                             key=lambda x: x["total"], reverse=True),
        "merchants": [{"merchant": m, "total": t, "count": n} for m, t, n in merchants],
    })


def spending_pace(user_id: str, *, today: date | None = None, url: str | None = None) -> dict:
    url = request_url(url)
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
    url = request_url(url)
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


def nudge_detail(user_id: str, nudge_id: str, *, url: str | None = None) -> dict | None:
    """One of the user's coach nudges with the signal that raised it, for a
    conversation opened from "Ask Banko about this"."""
    if not valid_uuid(nudge_id):
        return None
    url = request_url(url)
    with read_session(user_id, url=url) as conn:
        r = conn.execute(text("""
            SELECT n.nudge_id, n.message, n.created_at, s.signal_type
            FROM coach_nudges AS n LEFT JOIN spending_signals AS s ON s.signal_id = n.signal_id
            WHERE n.nudge_id = CAST(:n AS UUID) AND n.user_id = CAST(:u AS UUID)
        """), {"n": nudge_id, "u": user_id}).fetchone()
    return jsonable({"nudge_id": str(r[0]), "message": r[1], "created_at": r[2], "signal_type": r[3]}) if r else None
