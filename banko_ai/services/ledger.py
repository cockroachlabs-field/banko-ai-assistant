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
from ..db.engine import get_engine, run_transaction
from ..db.regions import money_region, region_columns

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
    status: str  # ok | insufficient_funds | replay | not_found
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
    rounded = amount.quantize(CENTS, rounding=ROUND_HALF_UP)
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


def _vector_text(embedding: Sequence[float] | None) -> str | None:
    if embedding is None:
        return None
    return "[" + ",".join(repr(float(x)) for x in embedding) + "]"


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
    with get_engine(url).connect() as conn:
        value = conn.execute(text("SELECT version FROM user_data_versions WHERE user_id = CAST(:u AS UUID)"),
                             {"u": user_id}).scalar()
    return int(value or 0)


def open_account(user_id: str, *, kind: str, name: str, institution: str, mask: str,
                 opening_balance: Any = 0, credit_limit: Any = None, apr: Any = None,
                 payment_method: str | None = None, opened_at: date | None = None,
                 url: str | None = None) -> str:
    if kind not in ACCOUNT_KINDS:
        raise LedgerError(f"Unknown account kind '{kind}'")
    opening = money(opening_balance)
    limit = money(credit_limit) if credit_limit is not None else None
    rate = Decimal(str(apr)) if apr is not None else None

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
    target = _positive(target_amount)

    def fn(conn: Connection) -> str:
        owned = conn.execute(text("SELECT 1 FROM accounts WHERE account_id = CAST(:a AS UUID) "
                                  "AND user_id = CAST(:u AS UUID)"), {"a": account_id, "u": user_id}).fetchone()
        if owned is None:
            raise NotFound("That account was not found")
        region = money_region(conn, user_id, url)
        rcol, rval = region_columns(region)
        row = conn.execute(text(f"""
            INSERT INTO goals ({rcol}user_id, name, icon, target_amount, target_date, account_id)
            VALUES ({rval}CAST(:u AS UUID), :name, :icon, :target, :tdate, CAST(:a AS UUID))
            RETURNING goal_id
        """), {"u": user_id, "name": name, "icon": icon, "target": target, "tdate": target_date,
               "a": account_id, "crdb_region": region}).fetchone()
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
    """Record a purchase: the expense, its ledger entry and the balance, together."""
    amt = _positive(amount)
    eid = expense_id or str(uuid.uuid4())
    today = datetime.now(timezone.utc).date()
    day = expense_date or today

    def fn(conn: Connection) -> str:
        kind, balance, limit, account_pm = _lock_account(conn, user_id, account_id)
        if not _can_debit(kind, balance, limit, amt):
            raise InsufficientFunds("Not enough available on that account")
        pay = payment_method or account_pm or ("Credit Card" if kind == "credit" else "Debit Card")
        desc = description or f"Spent ${amt:.2f} on {category.lower()} at {merchant} using {pay}."
        region = money_region(conn, user_id, url)
        rcol, rval = region_columns(region)
        params = {"u": user_id, "a": account_id, "eid": eid, "d": day, "amt": amt, "neg": -amt,
                  "cat": category, "desc": desc, "merchant": merchant, "pay": pay,
                  "tags": [category.lower(), merchant.lower().replace(" ", "_")],
                  "emb": _vector_text(embedding), "crdb_region": region,
                  "posted": datetime.now(timezone.utc) if day == today
                  else datetime.combine(day, time(12, 0), tzinfo=timezone.utc)}
        conn.execute(text(f"""
            INSERT INTO expenses ({rcol}expense_id, user_id, expense_date, expense_amount, shopping_type,
                                  description, merchant, payment_method, recurring, tags, embedding)
            VALUES ({rval}CAST(:eid AS UUID), CAST(:u AS UUID), :d, :amt, :cat,
                    :desc, :merchant, :pay, false, :tags, :emb)
        """), params)
        conn.execute(text(f"""
            INSERT INTO ledger_entries ({rcol}user_id, account_id, posted_at, amount, kind, description, expense_id)
            VALUES ({rval}CAST(:u AS UUID), CAST(:a AS UUID), :posted, :neg, 'purchase', :desc, CAST(:eid AS UUID))
        """), params)
        conn.execute(text("UPDATE accounts SET balance = balance - :amt "
                          "WHERE account_id = CAST(:a AS UUID) AND user_id = CAST(:u AS UUID)"), params)
        bump_data_version(conn, user_id, region)
        return eid

    return run_transaction(fn, url=url, user_id=user_id, idempotent=False)


def record_income(user_id: str, account_id: str, amount: Any, *, description: str = "Income",
                  posted_at: datetime | None = None, url: str | None = None) -> str:
    amt = _positive(amount)

    def fn(conn: Connection) -> str:
        _lock_account(conn, user_id, account_id)
        region = money_region(conn, user_id, url)
        rcol, rval = region_columns(region)
        row = conn.execute(text(f"""
            INSERT INTO ledger_entries ({rcol}user_id, account_id, posted_at, amount, kind, description)
            VALUES ({rval}CAST(:u AS UUID), CAST(:a AS UUID), :posted, :amt, 'income', :desc)
            RETURNING entry_id
        """), {"u": user_id, "a": account_id, "amt": amt, "desc": description, "crdb_region": region,
               "posted": posted_at or datetime.now(timezone.utc)}).fetchone()
        conn.execute(text("UPDATE accounts SET balance = balance + :amt "
                          "WHERE account_id = CAST(:a AS UUID) AND user_id = CAST(:u AS UUID)"),
                     {"amt": amt, "a": account_id, "u": user_id})
        bump_data_version(conn, user_id, region)
        return str(row[0])

    return run_transaction(fn, url=url, user_id=user_id, idempotent=False)


def transfer(user_id: str, from_account: str, to_account: str, amount: Any, *, idempotency_key: str,
             kind: str = "transfer", goal_id: str | None = None, url: str | None = None) -> TransferResult:
    """Move money between two of the user's accounts, once per idempotency key."""
    if kind not in TRANSFER_KINDS:
        raise LedgerError(f"Unknown transfer kind '{kind}'")
    if from_account == to_account:
        raise LedgerError("Pick two different accounts")
    if not idempotency_key or len(idempotency_key) > 128:
        raise LedgerError("An idempotency key of at most 128 characters is required")
    amt = _positive(amount)
    out_kind, in_kind = ("card_payment", "card_payment") if kind == "card_payment" else ("transfer_out", "transfer_in")
    label = {"transfer": "Transfer", "card_payment": "Card payment", "goal_contribution": "Goal contribution"}[kind]

    def fn(conn: Connection) -> TransferResult:
        region = money_region(conn, user_id, url)
        rcol, rval = region_columns(region)
        params = {"u": user_id, "f": from_account, "t": to_account, "amt": amt, "neg": -amt,
                  "kind": kind, "g": goal_id, "key": idempotency_key, "desc": label,
                  "ok": out_kind, "ik": in_kind, "crdb_region": region}
        row = conn.execute(text(f"""
            INSERT INTO transfers ({rcol}user_id, kind, from_account, to_account, amount, goal_id, idempotency_key)
            VALUES ({rval}CAST(:u AS UUID), :kind, CAST(:f AS UUID), CAST(:t AS UUID), :amt, CAST(:g AS UUID), :key)
            ON CONFLICT (user_id, idempotency_key) DO NOTHING
            RETURNING transfer_id
        """), params).fetchone()
        if row is None:
            prior = conn.execute(text("SELECT transfer_id FROM transfers WHERE user_id = CAST(:u AS UUID) "
                                      "AND idempotency_key = :key"), params).fetchone()
            return TransferResult("replay", str(prior[0]) if prior else None)
        params["tid"] = str(row[0])
        accounts = {str(r[0]): r for r in conn.execute(text("""
            SELECT account_id, kind, balance, credit_limit FROM accounts
            WHERE account_id IN (CAST(:f AS UUID), CAST(:t AS UUID)) AND user_id = CAST(:u AS UUID)
            FOR UPDATE
        """), params)}
        if from_account not in accounts or to_account not in accounts:
            raise NotFound("That account was not found")
        _, src_kind, src_balance, src_limit = accounts[from_account]
        _, dst_kind, dst_balance, _ = accounts[to_account]
        if not _can_debit(src_kind, src_balance, src_limit, amt):
            raise InsufficientFunds("Not enough available on that account")
        if dst_kind == "credit" and dst_balance + amt > 0:
            raise InvalidAmount("That pays more than the card owes")
        # One statement moves both balances: CockroachDB rejects two
        # mutations of the same table in one statement.
        conn.execute(text("""
            UPDATE accounts
            SET balance = CASE WHEN account_id = CAST(:f AS UUID) THEN balance - :amt ELSE balance + :amt END
            WHERE account_id IN (CAST(:f AS UUID), CAST(:t AS UUID)) AND user_id = CAST(:u AS UUID)
        """), params)
        conn.execute(text(f"""
            INSERT INTO ledger_entries ({rcol}user_id, account_id, posted_at, amount, kind, description, transfer_id, goal_id)
            VALUES ({rval}CAST(:u AS UUID), CAST(:f AS UUID), now(), :neg, :ok, :desc, CAST(:tid AS UUID), CAST(:g AS UUID)),
                   ({rval}CAST(:u AS UUID), CAST(:t AS UUID), now(), :amt, :ik, :desc, CAST(:tid AS UUID), CAST(:g AS UUID))
        """), params)
        bump_data_version(conn, user_id, region)
        return TransferResult("ok", params["tid"])

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
    try:
        uuid.UUID(str(goal_id))
    except ValueError:
        return TransferResult("not_found")
    with get_engine(url).connect() as conn:
        row = conn.execute(text("SELECT account_id FROM goals WHERE goal_id = CAST(:g AS UUID) "
                                "AND user_id = CAST(:u AS UUID)"), {"g": goal_id, "u": user_id}).fetchone()
    if row is None:
        return TransferResult("not_found")
    return transfer(user_id, from_account, str(row[0]), amount, idempotency_key=idempotency_key,
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
    """Write one bounded batch of seeded history. Ids are deterministic, so a
    retried or repeated batch changes nothing. Balances are not touched:
    call rebalance once the whole history is in."""
    if not rows:
        return 0
    purchases = [r for r in rows if r.expense]

    def fn(conn: Connection) -> int:
        region = money_region(conn, user_id, url)
        rcol, rval = region_columns(region)
        if purchases:
            exp = [r.expense for r in purchases]
            conn.execute(text(f"""
                INSERT INTO expenses ({rcol}expense_id, user_id, expense_date, expense_amount, shopping_type,
                                      description, merchant, payment_method, recurring, tags, embedding)
                SELECT {rval}t.id, CAST(:u AS UUID), t.d, t.amt, t.cat, t.descr, t.merch, t.pay, t.rec,
                       ARRAY[t.tag1, t.tag2], CAST(t.emb AS VECTOR(384))
                FROM unnest(CAST(:ids AS UUID[]), CAST(:ds AS DATE[]), CAST(:amts AS DECIMAL[]),
                            CAST(:cats AS STRING[]), CAST(:descrs AS STRING[]), CAST(:merchs AS STRING[]),
                            CAST(:pays AS STRING[]), CAST(:recs AS BOOL[]), CAST(:tag1s AS STRING[]),
                            CAST(:tag2s AS STRING[]), CAST(:embs AS STRING[]))
                     AS t(id, d, amt, cat, descr, merch, pay, rec, tag1, tag2, emb)
                ON CONFLICT (expense_id) DO NOTHING
            """), {
                "u": user_id, "crdb_region": region,
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
        conn.execute(text(f"""
            INSERT INTO ledger_entries ({rcol}entry_id, user_id, account_id, posted_at, amount, kind,
                                        description, expense_id, transfer_id, goal_id)
            SELECT {rval}t.id, CAST(:u AS UUID), t.acct, t.posted, t.amt, t.kind, t.descr, t.eid, t.tid, t.gid
            FROM unnest(CAST(:ids AS UUID[]), CAST(:accts AS UUID[]), CAST(:posted AS TIMESTAMPTZ[]),
                        CAST(:amts AS DECIMAL[]), CAST(:kinds AS STRING[]), CAST(:descrs AS STRING[]),
                        CAST(:eids AS UUID[]), CAST(:tids AS UUID[]), CAST(:gids AS UUID[]))
                 AS t(id, acct, posted, amt, kind, descr, eid, tid, gid)
            ON CONFLICT (entry_id) DO NOTHING
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
        })
        return len(rows)

    return run_transaction(fn, url=url, user_id=user_id, idempotent=True)


def rebalance(user_id: str, *, url: str | None = None) -> None:
    """Set every account of the user to opening balance plus its ledger,
    which makes the invariant true by construction after seeding."""

    def fn(conn: Connection) -> None:
        conn.execute(text("""
            UPDATE accounts
            SET balance = opening_balance + COALESCE(
                (SELECT sum(l.amount) FROM ledger_entries AS l WHERE l.account_id = accounts.account_id), 0)
            WHERE user_id = CAST(:u AS UUID)
        """), {"u": user_id})
        bump_data_version(conn, user_id, money_region(conn, user_id, url))

    run_transaction(fn, url=url, user_id=user_id, idempotent=True)
