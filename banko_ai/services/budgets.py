"""Monthly budgets: progress against this month's spending and month-end pace."""

from __future__ import annotations

import calendar
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from sqlalchemy import text

from ..config.catalog import get_catalog
from ..db.engine import run_transaction
from ..db.regions import has_region_column, money_region, region_columns
from ..db.rls import request_url
from ._sql import jsonable, read_session
from .ledger import LedgerError, bump_data_version, money


def budget_progress(user_id: str, *, today: date | None = None, url: str | None = None) -> dict:
    url = request_url(url)
    today = today or datetime.now(timezone.utc).date()
    first = today.replace(day=1)
    days = calendar.monthrange(today.year, today.month)[1]
    nxt = date(today.year + (today.month == 12), 1 if today.month == 12 else today.month + 1, 1)
    catalog = get_catalog()
    with read_session(user_id, url=url) as conn:
        budgets = conn.execute(text("SELECT category, amount FROM user_budgets WHERE user_id = CAST(:u AS UUID)"),
                               {"u": user_id}).fetchall()
        spent_rows = conn.execute(text("""
            SELECT shopping_type, sum(expense_amount) FROM expenses
            WHERE user_id = CAST(:u AS UUID) AND expense_date >= :first AND expense_date < :next
            GROUP BY shopping_type
        """), {"u": user_id, "first": first, "next": nxt}).fetchall()
    spent: dict[str, Decimal] = {}
    for label, total in spent_rows:
        key = catalog.category_for(label).key
        spent[key] = spent.get(key, Decimal("0")) + total
    items = []
    for label, amount in budgets:
        cat = catalog.category_for(label)
        used = spent.get(cat.key, Decimal("0"))
        projected = used / today.day * days
        status = "over" if used > amount else "watch" if projected > amount else "ok"
        items.append({"category": cat.label, "key": cat.key, "icon": cat.icon, "color": cat.color,
                      "budget": amount, "spent": used, "pct": float(round(used / amount, 4)) if amount else 0.0,
                      "projected": projected, "status": status, "_sort": cat.sort})
    items.sort(key=lambda i: i.pop("_sort"))
    return jsonable({"month": first.strftime("%Y-%m"), "day": today.day, "days_in_month": days,
                     "items": items,
                     "total_budget": sum((i["budget"] for i in items), Decimal("0")),
                     "total_spent": sum((i["spent"] for i in items), Decimal("0"))})


def set_budget(user_id: str, category: str, amount: Any, *, url: str | None = None) -> dict:
    url = request_url(url)
    cat = get_catalog().category_for(category)
    if cat.key == "other" and (category or "").strip().lower() != "other":
        raise LedgerError("Unknown category")
    if cat.grp not in ("needs", "wants"):
        raise LedgerError("Budgets can be set on spending categories only")
    value = money(amount)
    if value < 0:
        raise LedgerError("A budget can't be negative")

    def fn(conn) -> None:
        region = money_region(conn, user_id, url)
        if value == 0:
            conn.execute(text("DELETE FROM user_budgets WHERE user_id = CAST(:u AS UUID) AND category = :c"),
                         {"u": user_id, "c": cat.label})
        else:
            pinned = region if has_region_column(conn, "user_budgets") else None
            rcol, rval = region_columns(pinned)
            conn.execute(text(f"""
                INSERT INTO user_budgets ({rcol}user_id, category, amount, updated_at)
                VALUES ({rval}CAST(:u AS UUID), :c, :a, now())
                ON CONFLICT (user_id, category) DO UPDATE SET amount = excluded.amount, updated_at = now()
            """), {"u": user_id, "c": cat.label, "a": value, "crdb_region": pinned})
        bump_data_version(conn, user_id, region)

    run_transaction(fn, url=url, user_id=user_id)
    return jsonable({"category": cat.label, "amount": value})
