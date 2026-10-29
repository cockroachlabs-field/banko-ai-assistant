"""Every money movement in banko, one function each.

Receipts, seeding, the Move money dialog, assistant actions and the traffic
simulator all write through here, so balance = opening_balance + the sum of
an account's ledger entries holds for every account, always. Each function
is one short transaction through run_transaction: no HTTP or LLM call runs
inside it, and every function is safe to retry."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from ..config.settings import get_config
from ..db.engine import get_engine, routed, run_transaction
from ..db.regions import SCOPED_REGION_VALUE, home_region, money_region, region_columns, scoped_key
from ..db.rls import request_url
from ._sql import vector_text

CENTS = Decimal("0.01")
ACCOUNT_KINDS = ("checking", "savings", "credit")
TRANSFER_KINDS = ("transfer", "card_payment", "goal_contribution")
_ENTER_AMOUNT = "Enter an amount like 12.34"


class LedgerError(Exception):
    """A movement the ledger refuses. The message is safe to show a user."""


class InvalidAmount(LedgerError):
    pass


class InsufficientFunds(LedgerError):
    pass


class NotFound(LedgerError):
    pass


@dataclass(frozen=True)
class TransferResult:
    status: str  # ok | insufficient_funds | replay | key_conflict | not_found
    transfer_id: str | None = None


def money(value: Any) -> Decimal:
    """Exact two-decimal money from str, int, float or Decimal.

    Strings with more than two decimals are refused (a user typed them);
    floats are rounded to cents (code produced them)."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise InvalidAmount(_ENTER_AMOUNT)
    try:
        amount = Decimal(str(value).strip())
    except InvalidOperation:
        raise InvalidAmount(_ENTER_AMOUNT) from None
    if not amount.is_finite():
        raise InvalidAmount(_ENTER_AMOUNT)
    if amount.adjusted() > 11:
        raise InvalidAmount("That amount is too large")
    try:
        rounded = amount.quantize(CENTS, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        raise InvalidAmount(_ENTER_AMOUNT) from None
    if not isinstance(value, float) and rounded != amount:
        raise InvalidAmount("Amounts have at most two decimal places")
    return rounded


def _positive(value: Any) -> Decimal:
    amount = money(value)
    ceiling = Decimal(str(get_config().max_transfer_amount))
    if amount <= 0:
        raise InvalidAmount("The amount must be more than zero")
    if amount > ceiling:
        raise InvalidAmount(f"The amount can be at most {ceiling:,.2f}")
    return amount


def _can_debit(kind: str, balance: Decimal, limit: Decimal | None, amount: Decimal) -> bool:
    after = balance - amount
    if kind == "credit":
        return limit is None or after >= -limit
    if kind == "checking":
        return after >= -Decimal(str(get_config().checking_overdraft_limit))
    return after >= 0


def _uuid(value: Any) -> str:
    """Normalize and validate a UUID, returning its canonical lowercase form."""
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError):
        raise NotFound("That account was not found") from None


def bump_data_version(conn: Connection, user_id: str, region: str | None) -> None:
    """Advance the user's data version in the same transaction as the write,
    so caches keyed on it can never serve an answer from before the write."""
    rcol, rval = region_columns(region)
    conn.execute(text(f"""
        INSERT INTO user_data_versions ({rcol}user_id, version, updated_at)
        VALUES ({rval}CAST(:u AS UUID), 1, now())
        ON CONFLICT (user_id) DO UPDATE
        SET version = user_data_versions.version + 1, updated_at = now()
    """), {"u": user_id, "crdb_region": region})


def data_version(user_id: str, url: str | None = None) -> int:
    url = request_url(url)
    with get_engine(routed(url, user_id)).connect() as conn:
        value = conn.execute(text("SELECT version FROM user_data_versions WHERE user_id = CAST(:u AS UUID)"),
                             {"u": user_id}).scalar()
    return int(value or 0)


def open_account(user_id: str, *, kind: str, name: str, institution: str, mask: str,
                 opening_balance: Any = 0, credit_limit: Any = None, apr: Any = None,
                 payment_method: str | None = None, opened_at: date | None = None,
                 url: str | None = None) -> str:
    url = request_url(url)
    if kind not in ACCOUNT_KINDS:
        raise LedgerError("Pick checking, savings or credit")
    opening = money(opening_balance)
    limit = money(credit_limit) if credit_limit is not None else None
    rate = None
    if apr is not None:
        try:
            rate = Decimal(str(apr))
            if not (0 <= rate < 1000):
                raise ValueError
        except (InvalidOperation, ValueError):
            raise InvalidAmount("APR must be a number between 0 and 999.99") from None

    def fn(conn: Connection) -> str:
        region = money_region(conn, user_id, url)
        rcol, rval = region_columns(region)
        row = conn.execute(text(f"""
            INSERT INTO accounts ({rcol}user_id, kind, name, institution, mask, payment_method,
                                  opening_balance, balance, credit_limit, apr, opened_at)
            VALUES ({rval}CAST(:u AS UUID), :kind, :name, :inst, :mask, :pm,
                    :opening, :opening, :limit, :apr, :opened)
            RETURNING account_id
        """), {"u": user_id, "kind": kind, "name": name, "inst": institution, "mask": mask,
               "pm": payment_method, "opening": opening, "limit": limit, "apr": rate,
               "opened": opened_at or datetime.now(timezone.utc).date(), "crdb_region": region}).fetchone()
        bump_data_version(conn, user_id, region)
        return str(row[0])

    return run_transaction(fn, url=url, user_id=user_id, idempotent=False)


def create_goal(user_id: str, *, name: str, icon: str, target_amount: Any, account_id: str,
                target_date: date | None = None, url: str | None = None) -> str:
    url = request_url(url)
    target = _positive(target_amount)
    acct = _uuid(account_id)

    def fn(conn: Connection) -> str:
        owned = conn.execute(text("SELECT 1 FROM accounts WHERE account_id = CAST(:a AS UUID) "
                                  "AND user_id = CAST(:u AS UUID)"), {"a": acct, "u": user_id}).fetchone()
        if owned is None:
            raise NotFound("That account was not found")
        region = money_region(conn, user_id, url)
        rcol, rval = region_columns(region)
        row = conn.execute(text(f"""
            INSERT INTO goals ({rcol}user_id, name, icon, target_amount, target_date, account_id)
            VALUES ({rval}CAST(:u AS UUID), :name, :icon, :target, :tdate, CAST(:a AS UUID))
            RETURNING goal_id
        """), {"u": user_id, "name": name, "icon": icon, "target": target, "tdate": target_date,
               "a": acct, "crdb_region": region}).fetchone()
        bump_data_version(conn, user_id, region)
        return str(row[0])

    return run_transaction(fn, url=url, user_id=user_id, idempotent=False)


def _lock_account(conn: Connection, user_id: str, account_id: str):
    row = conn.execute(text("""
        SELECT kind, balance, credit_limit, payment_method FROM accounts
        WHERE account_id = CAST(:a AS UUID) AND user_id = CAST(:u AS UUID)
        FOR UPDATE
    """), {"a": account_id, "u": user_id}).fetchone()
    if row is None:
        raise NotFound("That account was not found")
    return row


def post_purchase(user_id: str, account_id: str, *, amount: Any, merchant: str, category: str,
                  description: str | None = None, payment_method: str | None = None,
                  expense_date: date | None = None, embedding: Sequence[float] | None = None,
                  expense_id: str | None = None, url: str | None = None) -> str:
    """Record a purchase: the expense, its ledger entry and the balance,
    together. Without an expense_id the database picks one: on a REGIONAL BY
    ROW table a key the client chose costs a uniqueness check in every region."""
    url = request_url(url)
    amt = _positive(amount)
    acct = _uuid(account_id)
    id_col, id_val = ("expense_id, ", "CAST(:eid AS UUID), ") if expense_id else ("", "")
    today = datetime.now(timezone.utc).date()
    day = expense_date or today

    def fn(conn: Connection) -> str:
        kind, balance, limit, account_pm = _lock_account(conn, user_id, acct)
        if not _can_debit(kind, balance, limit, amt):
            raise InsufficientFunds("Not enough available on that account")
        pay = payment_method or account_pm or ("Credit Card" if kind == "credit" else "Debit Card")
        desc = description or f"Spent ${amt:.2f} on {category.lower()} at {merchant} using {pay}."
        region = money_region(conn, user_id, url)
        rcol, rval = region_columns(region)
        params = {"u": user_id, "a": acct, "eid": expense_id, "d": day, "amt": amt, "neg": -amt,
                  "cat": category, "desc": desc, "merchant": merchant, "pay": pay,
                  "tags": [category.lower(), merchant.lower().replace(" ", "_")],
                  "emb": vector_text(embedding) if embedding is not None else None,
                  "crdb_region": region,
                  "posted": datetime.now(timezone.utc) if day == today
                  else datetime.combine(day, time(12, 0), tzinfo=timezone.utc)}
        params["eid"] = str(conn.execute(text(f"""
            INSERT INTO expenses ({rcol}{id_col}user_id, expense_date, expense_amount, shopping_type,
                                  description, merchant, payment_method, recurring, tags, embedding)
            VALUES ({rval}{id_val}CAST(:u AS UUID), :d, :amt, :cat,
                    :desc, :merchant, :pay, false, :tags, :emb)
            RETURNING expense_id
        """), params).scalar())
        conn.execute(text(f"""
            INSERT INTO ledger_entries ({rcol}user_id, account_id, posted_at, amount, kind, description, expense_id)
            VALUES ({rval}CAST(:u AS UUID), CAST(:a AS UUID), :posted, :neg, 'purchase', :desc, CAST(:eid AS UUID))
        """), params)
        conn.execute(text("UPDATE accounts SET balance = balance - :amt "
                          "WHERE account_id = CAST(:a AS UUID) AND user_id = CAST(:u AS UUID)"), params)
        bump_data_version(conn, user_id, region)
        return str(params["eid"])

    return run_transaction(fn, url=url, user_id=user_id, idempotent=False)


def record_income(user_id: str, account_id: str, amount: Any, *, description: str = "Income",
                  posted_at: datetime | None = None, entry_id: str | None = None, url: str | None = None) -> str:
    """Credit an account. A caller that picks entry_id up front can replay
    the deposit safely: the second attempt hits the primary key. Without one
    the database picks it, which needs no uniqueness check in other regions."""
    url = request_url(url)
    amt = _positive(amount)
    acct = _uuid(account_id)
    id_col, id_val = ("entry_id, ", "CAST(:eid AS UUID), ") if entry_id else ("", "")

    def fn(conn: Connection) -> str:
        _lock_account(conn, user_id, acct)
        region = money_region(conn, user_id, url)
        rcol, rval = region_columns(region)
        row = conn.execute(text(f"""
            INSERT INTO ledger_entries ({rcol}{id_col}user_id, account_id, posted_at, amount, kind, description)
            VALUES ({rval}{id_val}CAST(:u AS UUID), CAST(:a AS UUID), :posted, :amt, 'income', :desc)
            RETURNING entry_id
        """), {"u": user_id, "a": acct, "amt": amt, "desc": description, "crdb_region": region,
               "eid": entry_id, "posted": posted_at or datetime.now(timezone.utc)}).fetchone()
        conn.execute(text("UPDATE accounts SET balance = balance + :amt "
                          "WHERE account_id = CAST(:a AS UUID) AND user_id = CAST(:u AS UUID)"),
                     {"amt": amt, "a": acct, "u": user_id})
        bump_data_version(conn, user_id, region)
        return str(row[0])

    return run_transaction(fn, url=url, user_id=user_id, idempotent=False)


def transfer(user_id: str, from_account: str, to_account: str, amount: Any, *, idempotency_key: str,
             kind: str = "transfer", goal_id: str | None = None, url: str | None = None) -> TransferResult:
    """Move money between two of the user's accounts, once per idempotency key."""
    url = request_url(url)
    if kind not in TRANSFER_KINDS:
        raise LedgerError("Pick transfer, card payment or goal contribution")
    from_acct = _uuid(from_account)
    to_acct = _uuid(to_account) if kind != "goal_contribution" else from_acct
    if from_acct == to_acct and kind != "goal_contribution":
        raise LedgerError("Pick two different accounts")
    if not idempotency_key or len(idempotency_key) > 128:
        raise LedgerError("An idempotency key of at most 128 characters is required")
    amt = _positive(amount)
    goal = _uuid(goal_id) if goal_id else None
    out_kind, in_kind = ("card_payment", "card_payment") if kind == "card_payment" else ("transfer_out", "transfer_in")
    label = {"transfer": "Transfer", "card_payment": "Card payment", "goal_contribution": "Goal contribution"}[kind]

    def fn(conn: Connection) -> TransferResult:
        region = money_region(conn, user_id, url)
        rcol, rval = region_columns(region)
        actual_to_acct = to_acct
        if goal:
            goal_row = conn.execute(text("SELECT account_id FROM goals WHERE goal_id = CAST(:g AS UUID) "
                                         "AND user_id = CAST(:u AS UUID)"), {"g": goal, "u": user_id}).fetchone()
            if goal_row is None:
                raise NotFound("That account was not found")
            goal_acct = str(goal_row[0])
            if kind == "goal_contribution":
                if goal_acct == from_acct:
                    # One account on both legs nets to zero in the ledger
                    # while the balance update would still debit it.
                    raise LedgerError("Pick an account other than the goal's")
                actual_to_acct = goal_acct
            elif goal_acct != to_acct:
                raise NotFound("That account was not found")
        params = {"u": user_id, "f": from_acct, "t": actual_to_acct, "amt": amt, "neg": -amt,
                  "kind": kind, "g": goal, "key": idempotency_key, "desc": label,
                  "ok": out_kind, "ik": in_kind, "crdb_region": region}
        # With the key scoped to the region, the row's region must not depend
        # on the gateway, or a retry through another region would pass twice.
        tcol, tval, arbiter = rcol, rval, "user_id, idempotency_key"
        if scoped_key(conn, "transfers"):
            tcol, tval, arbiter = "crdb_region, ", SCOPED_REGION_VALUE, "crdb_region, user_id, idempotency_key"
            params["scoped_region"] = region or home_region(conn, user_id) or ""
        row = conn.execute(text(f"""
            INSERT INTO transfers ({tcol}user_id, kind, from_account, to_account, amount, goal_id, idempotency_key)
            VALUES ({tval}CAST(:u AS UUID), :kind, CAST(:f AS UUID), CAST(:t AS UUID), :amt, CAST(:g AS UUID), :key)
            ON CONFLICT ({arbiter}) DO NOTHING
            RETURNING transfer_id
        """), params).fetchone()
        if row is None:
            prior = conn.execute(text("SELECT transfer_id, from_account, to_account, amount, kind FROM transfers "
                                      "WHERE user_id = CAST(:u AS UUID) AND idempotency_key = :key"),
                                 params).fetchone()
            if prior and (str(prior[1]) != from_acct or str(prior[2]) != actual_to_acct or prior[3] != amt or prior[4] != kind):
                return TransferResult("key_conflict", str(prior[0]))
            return TransferResult("replay", str(prior[0]) if prior else None)
        transfer_id = str(row[0])
        params["tid"] = transfer_id
        # One lock per account, in a fixed order: a lookup by one key reads the
        # home region first, while one lock over both keys reads every region.
        accounts = {}
        for acct in sorted({from_acct, actual_to_acct}):
            row = conn.execute(text("""
                SELECT account_id, kind, balance, credit_limit FROM accounts
                WHERE account_id = CAST(:a AS UUID) AND user_id = CAST(:u AS UUID)
                FOR UPDATE
            """), {"a": acct, "u": user_id}).fetchone()
            if row is not None:
                accounts[str(row[0])] = row
        if from_acct not in accounts or actual_to_acct not in accounts:
            raise NotFound("That account was not found")
        _, src_kind, src_balance, src_limit = accounts[from_acct]
        _, dst_kind, dst_balance, _ = accounts[actual_to_acct]
        if not _can_debit(src_kind, src_balance, src_limit, amt):
            raise InsufficientFunds("Not enough available on that account")
        if dst_kind == "credit" and dst_balance + amt > 0:
            raise InvalidAmount("That pays more than the card owes")
        # One update per account: a lookup by one key reads the home region
        # first, while one update over both keys reads every region.
        for acct_param, delta in (("f", "- :amt"), ("t", "+ :amt")):
            conn.execute(text(f"UPDATE accounts SET balance = balance {delta} "
                              f"WHERE account_id = CAST(:{acct_param} AS UUID) AND user_id = CAST(:u AS UUID)"),
                         params)
        conn.execute(text(f"""
            INSERT INTO ledger_entries ({rcol}user_id, account_id, posted_at, amount, kind, description, transfer_id, goal_id)
            VALUES ({rval}CAST(:u AS UUID), CAST(:f AS UUID), now(), :neg, :ok, :desc, CAST(:tid AS UUID), CAST(:g AS UUID)),
                   ({rval}CAST(:u AS UUID), CAST(:t AS UUID), now(), :amt, :ik, :desc, CAST(:tid AS UUID), CAST(:g AS UUID))
        """), params)
        bump_data_version(conn, user_id, region)
        return TransferResult("ok", transfer_id)

    try:
        return run_transaction(fn, url=url, user_id=user_id, idempotent=True)
    except InsufficientFunds:
        return TransferResult("insufficient_funds")
    except NotFound:
        return TransferResult("not_found")


def card_payment(user_id: str, from_account: str, card_account: str, amount: Any, *,
                 idempotency_key: str, url: str | None = None) -> TransferResult:
    return transfer(user_id, from_account, card_account, amount, idempotency_key=idempotency_key,
                    kind="card_payment", url=url)


def goal_contribution(user_id: str, goal_id: str, from_account: str, amount: Any, *,
                      idempotency_key: str, url: str | None = None) -> TransferResult:
    url = request_url(url)
    return transfer(user_id, from_account, from_account, amount, idempotency_key=idempotency_key,
                    kind="goal_contribution", goal_id=goal_id, url=url)


@dataclass(frozen=True)
class HistoryRow:
    """One seeded ledger entry, with the purchase's expense when it is one.

    expense keys: expense_id, expense_date, expense_amount, shopping_type,
    merchant, payment_method, recurring, tags (two strings), embedding
    (vector text or None)."""

    entry_id: str
    account_id: str
    posted_at: datetime
    amount: Decimal
    kind: str
    description: str
    expense: dict | None = None
    goal_id: str | None = None
    transfer_id: str | None = None


def post_history(user_id: str, rows: Sequence[HistoryRow], *, url: str | None = None) -> int:
    """Write one bounded batch of seeded history and move the balances of the
    accounts it touched, in one transaction, so the invariant holds after
    every batch. Ids are deterministic: only rows actually inserted count
    toward a balance, so a retried or repeated batch changes nothing.
    Returns the number of ledger entries inserted."""
    if not rows:
        return 0
    purchases = [r for r in rows if r.expense]

    def fn(conn: Connection) -> int:
        region = money_region(conn, user_id, url)
        rcol, rval = region_columns(region)
        if purchases:
            exp = [r.expense for r in purchases if r.expense is not None]
            # See transfer(): a region-scoped key needs a region that never
            # depends on the gateway, or a repeated batch could write twice.
            ecol, evals, arbiter, scoped_region = rcol, rval, "expense_id", None
            if scoped_key(conn, "expenses"):
                ecol, evals, arbiter = "crdb_region, ", SCOPED_REGION_VALUE, "crdb_region, expense_id"
                scoped_region = region or home_region(conn, user_id) or ""
            conn.execute(text(f"""
                INSERT INTO expenses ({ecol}expense_id, user_id, expense_date, expense_amount, shopping_type,
                                      description, merchant, payment_method, recurring, tags, embedding)
                SELECT {evals}t.id, CAST(:u AS UUID), t.d, t.amt, t.cat, t.descr, t.merch, t.pay, t.rec,
                       ARRAY[t.tag1, t.tag2], CAST(t.emb AS VECTOR(384))
                FROM unnest(CAST(:ids AS UUID[]), CAST(:ds AS DATE[]), CAST(:amts AS DECIMAL[]),
                            CAST(:cats AS STRING[]), CAST(:descrs AS STRING[]), CAST(:merchs AS STRING[]),
                            CAST(:pays AS STRING[]), CAST(:recs AS BOOL[]), CAST(:tag1s AS STRING[]),
                            CAST(:tag2s AS STRING[]), CAST(:embs AS STRING[]))
                     AS t(id, d, amt, cat, descr, merch, pay, rec, tag1, tag2, emb)
                ON CONFLICT ({arbiter}) DO NOTHING
            """), {
                "u": user_id, "crdb_region": region, "scoped_region": scoped_region,
                "ids": [e["expense_id"] for e in exp],
                "ds": [e["expense_date"] for e in exp],
                "amts": [e["expense_amount"] for e in exp],
                "cats": [e["shopping_type"] for e in exp],
                "descrs": [r.description for r in purchases],
                "merchs": [e["merchant"] for e in exp],
                "pays": [e["payment_method"] for e in exp],
                "recs": [bool(e["recurring"]) for e in exp],
                "tag1s": [e["tags"][0] for e in exp],
                "tag2s": [e["tags"][1] for e in exp],
                "embs": [e["embedding"] for e in exp],
            })
        inserted = conn.execute(text(f"""
            INSERT INTO ledger_entries ({rcol}entry_id, user_id, account_id, posted_at, amount, kind,
                                        description, expense_id, transfer_id, goal_id)
            SELECT {rval}t.id, CAST(:u AS UUID), t.acct, t.posted, t.amt, t.kind, t.descr, t.eid, t.tid, t.gid
            FROM unnest(CAST(:ids AS UUID[]), CAST(:accts AS UUID[]), CAST(:posted AS TIMESTAMPTZ[]),
                        CAST(:amts AS DECIMAL[]), CAST(:kinds AS STRING[]), CAST(:descrs AS STRING[]),
                        CAST(:eids AS UUID[]), CAST(:tids AS UUID[]), CAST(:gids AS UUID[]))
                 AS t(id, acct, posted, amt, kind, descr, eid, tid, gid)
            ON CONFLICT (entry_id) DO NOTHING
            RETURNING account_id, amount
        """), {
            "u": user_id, "crdb_region": region,
            "ids": [r.entry_id for r in rows],
            "accts": [r.account_id for r in rows],
            "posted": [r.posted_at for r in rows],
            "amts": [r.amount for r in rows],
            "kinds": [r.kind for r in rows],
            "descrs": [r.description for r in rows],
            "eids": [r.expense["expense_id"] if r.expense else None for r in rows],
            "tids": [r.transfer_id for r in rows],
            "gids": [r.goal_id for r in rows],
        }).fetchall()
        if not inserted:
            return 0
        deltas: dict[str, Decimal] = {}
        for account_id, amount in inserted:
            deltas[str(account_id)] = deltas.get(str(account_id), Decimal("0")) + amount
        conn.execute(text("""
            UPDATE accounts AS a SET balance = a.balance + t.delta
            FROM unnest(CAST(:accts AS UUID[]), CAST(:deltas AS DECIMAL[])) AS t(id, delta)
            WHERE a.account_id = t.id AND a.user_id = CAST(:u AS UUID)
        """), {"u": user_id, "accts": list(deltas), "deltas": list(deltas.values())})
        bump_data_version(conn, user_id, region)
        return len(inserted)

    return run_transaction(fn, url=url, user_id=user_id, idempotent=True)


def rebalance(user_id: str, *, url: str | None = None) -> None:
    """Set every account of the user to opening balance plus its ledger.
    post_history already keeps balances right batch by batch; seeding still
    calls this at the end as a cheap safety net."""

    def fn(conn: Connection) -> None:
        conn.execute(text("""
            UPDATE accounts
            SET balance = opening_balance + COALESCE(
                (SELECT sum(l.amount) FROM ledger_entries AS l WHERE l.account_id = accounts.account_id), 0)
            WHERE user_id = CAST(:u AS UUID)
        """), {"u": user_id})
        bump_data_version(conn, user_id, money_region(conn, user_id, url))

    run_transaction(fn, url=url, user_id=user_id, idempotent=True)
