"""Changes the assistant proposes. A proposal is stored pending and runs
only when the user clicks Confirm; confirming twice runs it once."""

from __future__ import annotations

import json

from sqlalchemy import text

from ..config.settings import get_config
from ..db.engine import run_transaction
from ..db.regions import money_region, region_columns
from ..db.rls import request_url
from . import budgets, transactions
from ._sql import valid_uuid
from .ledger import NotFound

KINDS = ("recategorize", "budget")


def _out(action_id: str, kind: str, summary: str, status: str) -> dict:
    return {"action_id": action_id, "kind": kind, "summary": summary, "status": status}


def propose(user_id: str, kind: str, args: dict, summary: str, *, thread_id: str | None = None,
            url: str | None = None) -> dict:
    if kind not in KINDS:
        raise ValueError("Unknown action")
    url = request_url(url)
    hours = get_config().assistant_action_ttl_hours

    def fn(conn) -> dict:
        region = money_region(conn, user_id, url)
        rcol, rval = region_columns(region)
        aid = conn.execute(text(f"""
            INSERT INTO assistant_actions ({rcol}user_id, thread_id, kind, args, summary, expires_at)
            VALUES ({rval}CAST(:u AS UUID), CAST(:t AS UUID), :k, CAST(:a AS JSONB), :s,
                    now() + :h * INTERVAL '1 hour')
            RETURNING action_id
        """), {"u": user_id, "t": thread_id if valid_uuid(thread_id) else None, "k": kind,
               "a": json.dumps(args), "s": summary[:300], "h": hours, "crdb_region": region}).scalar()
        return _out(str(aid), kind, summary[:300], "pending")

    return run_transaction(fn, url=url, idempotent=False)


def _claim(user_id: str, action_id: str, to_status: str, url: str | None):
    """Move a pending action to to_status, or report where it already is."""
    if not valid_uuid(action_id):
        raise NotFound("That action was not found")

    def fn(conn):
        row = conn.execute(text("""
            SELECT kind, args, summary, status FROM assistant_actions
            WHERE action_id = CAST(:a AS UUID) AND user_id = CAST(:u AS UUID) AND expires_at > now()
            FOR UPDATE
        """), {"a": action_id, "u": user_id}).fetchone()
        if row is None:
            raise NotFound("That action was not found")
        if row[3] == "pending":
            conn.execute(text("UPDATE assistant_actions SET status = :s WHERE action_id = CAST(:a AS UUID)"),
                         {"s": to_status, "a": action_id})
        args = row[1] if isinstance(row[1], dict) else json.loads(row[1])
        return row[0], args, row[2], row[3]

    return run_transaction(fn, url=url)


def confirm(user_id: str, action_id: str, *, url: str | None = None) -> dict:
    url = request_url(url)
    kind, args, summary, was = _claim(user_id, action_id, "done", url)
    if was != "pending":
        return _out(action_id, kind, summary, was)
    try:
        if kind == "budget":
            budgets.set_budget(user_id, args["category"], args["amount"], url=url)
        else:
            transactions.recategorize(user_id, args["expense_id"], args["category"], url=url)
    except Exception:
        # Put it back so the user can try again; the error surfaces neutrally upstream.
        run_transaction(lambda conn: conn.execute(text(
            "UPDATE assistant_actions SET status = 'pending' WHERE action_id = CAST(:a AS UUID)"),
            {"a": action_id}), url=url)
        raise
    return _out(action_id, kind, summary, "done")


def cancel(user_id: str, action_id: str, *, url: str | None = None) -> dict:
    url = request_url(url)
    kind, _, summary, was = _claim(user_id, action_id, "cancelled", url)
    return _out(action_id, kind, summary, "cancelled" if was == "pending" else was)
