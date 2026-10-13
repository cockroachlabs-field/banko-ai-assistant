"""Helpers for tests that touch money tables: a throwaway user, a full
cleanup, and the ledger invariant."""

import uuid

from sqlalchemy import text

from banko_ai.db.engine import get_engine, run_transaction

MONEY_TABLES = ("ledger_entries", "transfers", "goals", "accounts", "user_budgets",
                "user_data_versions", "expenses")


def make_user(style: str = "saver", home_region: str | None = None) -> str:
    user_id = str(uuid.uuid4())
    with get_engine().begin() as c:
        c.execute(text("INSERT INTO users (user_id, username, spending_style, home_region, demo_user) "
                       "VALUES (CAST(:u AS UUID), :name, :style, :region, true)"),
                  {"u": user_id, "name": f"t-{user_id[:8]}", "style": style, "region": home_region})
    return user_id


def drop_user(user_id: str) -> None:
    # Retried: deleting expenses touches C-SPANN partition metadata, which
    # can lose a 40001 race with another user's delete right next to it.
    def fn(c) -> None:
        for table in MONEY_TABLES:
            c.execute(text(f"DELETE FROM {table} WHERE user_id = CAST(:u AS UUID)"), {"u": user_id})
        c.execute(text("DELETE FROM banko_meta WHERE key IN (:k, :lock)"),
                  {"k": f"seed:{user_id}", "lock": f"seed_lock:{user_id}"})
        c.execute(text("DELETE FROM users WHERE user_id = CAST(:u AS UUID)"), {"u": user_id})

    run_transaction(fn)


def invariant_violations(user_id: str) -> list[tuple]:
    """Accounts whose balance differs from opening balance plus ledger sum."""
    with get_engine().connect() as c:
        return c.execute(text("""
            SELECT a.account_id, a.balance, a.opening_balance + COALESCE(sum(l.amount), 0)
            FROM accounts AS a LEFT JOIN ledger_entries AS l ON l.account_id = a.account_id
            WHERE a.user_id = CAST(:u AS UUID)
            GROUP BY a.account_id, a.balance, a.opening_balance
            HAVING a.balance <> a.opening_balance + COALESCE(sum(l.amount), 0)
        """), {"u": user_id}).fetchall()
