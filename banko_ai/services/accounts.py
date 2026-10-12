"""Accounts, net worth and monthly statements, scoped to one user."""

from __future__ import annotations

import csv
import io
import re
from datetime import date, datetime, timezone
from decimal import Decimal

from sqlalchemy import text

from ._sql import jsonable, read_session, valid_uuid

_KIND_ORDER = {"checking": 0, "savings": 1, "credit": 2}
_MONTH = re.compile(r"(\d{4})-(0[1-9]|1[0-2])")
_ACCOUNT_COLS = "account_id, kind, name, institution, mask, balance, credit_limit, apr, opened_at"


def _account_dict(r) -> dict:
    return {"account_id": str(r[0]), "kind": r[1], "name": r[2], "institution": r[3], "mask": r[4],
            "balance": r[5], "credit_limit": r[6], "apr": r[7], "opened_at": r[8]}


def list_accounts(user_id: str, *, url: str | None = None) -> dict:
    with read_session(user_id, url=url) as conn:
        rows = conn.execute(text(f"SELECT {_ACCOUNT_COLS} FROM accounts WHERE user_id = CAST(:u AS UUID)"),
                            {"u": user_id}).fetchall()
    accounts = sorted((_account_dict(r) for r in rows), key=lambda a: (_KIND_ORDER[a["kind"]], a["name"]))
    cash = [a for a in accounts if a["kind"] != "credit"]
    credit = [a for a in accounts if a["kind"] == "credit"]
    limits = sum((a["credit_limit"] or Decimal("0") for a in credit), Decimal("0"))
    owed = -sum((a["balance"] for a in credit), Decimal("0"))
    return jsonable({
        "accounts": accounts,
        "groups": {"cash": cash, "credit": credit},
        "cash_total": sum((a["balance"] for a in cash), Decimal("0")),
        "credit_total": -owed,
        "net_worth": sum((a["balance"] for a in accounts), Decimal("0")),
        "utilization": float(round(owed / limits, 4)) if limits else 0.0,
    })


def get_account(user_id: str, account_id: str, *, url: str | None = None) -> dict | None:
    if not valid_uuid(account_id):
        return None
    with read_session(user_id, url=url) as conn:
        row = conn.execute(text(f"SELECT {_ACCOUNT_COLS} FROM accounts WHERE account_id = CAST(:a AS UUID) "
                                "AND user_id = CAST(:u AS UUID)"), {"a": account_id, "u": user_id}).fetchone()
    return jsonable(_account_dict(row)) if row else None


def _month_bounds(month: str) -> tuple[datetime, datetime]:
    match = _MONTH.fullmatch(month or "")
    if not match:
        raise ValueError("month must look like 2026-09")
    year, mon = int(match.group(1)), int(match.group(2))
    start = datetime(year, mon, 1, tzinfo=timezone.utc)
    end = datetime(year + (mon == 12), 1 if mon == 12 else mon + 1, 1, tzinfo=timezone.utc)
    return start, end


def _labels(today: date, months: int) -> list[str]:
    out, year, mon = [], today.year, today.month
    for _ in range(months):
        out.append(f"{year:04d}-{mon:02d}")
        year, mon = (year - 1, 12) if mon == 1 else (year, mon - 1)
    return out[::-1]


def net_worth_history(user_id: str, *, months: int = 12, today: date | None = None,
                      url: str | None = None) -> list[dict]:
    today = today or datetime.now(timezone.utc).date()
    months = max(1, int(months))
    with read_session(user_id, url=url, follower=True) as conn:
        opening = conn.execute(text("SELECT COALESCE(sum(opening_balance), 0) FROM accounts "
                                    "WHERE user_id = CAST(:u AS UUID)"), {"u": user_id}).scalar()
        rows = conn.execute(text("""
            SELECT to_char(posted_at AT TIME ZONE 'UTC', 'YYYY-MM') AS m, sum(amount)
            FROM ledger_entries WHERE user_id = CAST(:u AS UUID)
            GROUP BY m ORDER BY m
        """), {"u": user_id}).fetchall()
    by_month = {m: total for m, total in rows}
    labels = _labels(today, months)
    running = opening + sum((v for m, v in by_month.items() if m < labels[0]), Decimal("0"))
    out = []
    for label in labels:
        running += by_month.get(label, Decimal("0"))
        out.append({"month": label, "net_worth": running})
    # The follower snapshot is a few seconds old. The current month's point
    # comes from live balances so it matches the accounts page.
    if labels[-1] == datetime.now(timezone.utc).strftime("%Y-%m"):
        with read_session(user_id, url=url) as conn:
            out[-1]["net_worth"] = conn.execute(text("SELECT COALESCE(sum(balance), 0) FROM accounts "
                                                     "WHERE user_id = CAST(:u AS UUID)"),
                                                {"u": user_id}).scalar()
    return jsonable(out)


def statement_months(user_id: str, account_id: str, *, url: str | None = None) -> list[str]:
    if not valid_uuid(account_id):
        return []
    with read_session(user_id, url=url) as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT to_char(posted_at AT TIME ZONE 'UTC', 'YYYY-MM') AS m
            FROM ledger_entries WHERE account_id = CAST(:a AS UUID) AND user_id = CAST(:u AS UUID)
            ORDER BY m DESC
        """), {"a": account_id, "u": user_id}).fetchall()
    return [r[0] for r in rows]


def statement(user_id: str, account_id: str, month: str, *, url: str | None = None) -> dict | None:
    start, end = _month_bounds(month)
    if not valid_uuid(account_id):
        return None
    # Ownership is checked on current data so a just-opened account never
    # 404s. Only months that have already closed read from a follower.
    with read_session(user_id, url=url) as conn:
        acct = conn.execute(text(f"SELECT {_ACCOUNT_COLS}, opening_balance FROM accounts "
                                 "WHERE account_id = CAST(:a AS UUID) AND user_id = CAST(:u AS UUID)"),
                            {"a": account_id, "u": user_id}).fetchone()
    if acct is None:
        return None
    this_month = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    params = {"a": account_id, "u": user_id, "s": start, "e": end}
    with read_session(user_id, url=url, follower=end <= this_month) as conn:
        before = conn.execute(text("SELECT COALESCE(sum(amount), 0) FROM ledger_entries "
                                   "WHERE account_id = CAST(:a AS UUID) AND user_id = CAST(:u AS UUID) "
                                   "AND posted_at < :s"), params).scalar()
        entries = conn.execute(text("""
            SELECT posted_at, description, kind, amount FROM ledger_entries
            WHERE account_id = CAST(:a AS UUID) AND user_id = CAST(:u AS UUID)
              AND posted_at >= :s AND posted_at < :e
            ORDER BY posted_at, entry_id
        """), params).fetchall()
    opening = acct[9] + before
    charges = sum((r[3] for r in entries if r[3] < 0), Decimal("0"))
    credits = sum((r[3] for r in entries if r[3] > 0), Decimal("0"))
    return jsonable({
        "account": _account_dict(acct[:9]), "month": month,
        "opening": opening, "charges": charges, "credits": credits,
        "closing": opening + charges + credits,
        "entries": [{"posted_at": r[0], "description": r[1], "kind": r[2], "amount": r[3]} for r in entries],
    })


def _csv_text(value) -> str:
    """Spreadsheets run a cell that starts with = + - or @ as a formula."""
    cell = "" if value is None else str(value)
    return "'" + cell if cell[:1] in ("=", "+", "-", "@") else cell


def statement_csv(st: dict) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["Date", "Description", "Kind", "Amount"])
    writer.writerow(["Opening balance", "", "", f"{st['opening']:.2f}"])
    for e in st["entries"]:
        writer.writerow([e["posted_at"][:10], _csv_text(e["description"]), _csv_text(e["kind"]),
                         f"{e['amount']:.2f}"])
    writer.writerow(["Closing balance", "", "", f"{st['closing']:.2f}"])
    return buf.getvalue()
