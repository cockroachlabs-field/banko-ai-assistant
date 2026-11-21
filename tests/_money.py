"""Helpers for tests that touch money tables: a throwaway user, a full
cleanup, and the ledger invariant."""

import uuid
from datetime import date

from sqlalchemy import text

from banko_ai.db.engine import get_engine, run_transaction

MONEY_TABLES = ("ledger_entries", "transfers", "goals", "accounts", "user_budgets",
                "user_data_versions", "expenses")


def make_user(style: str = "saver", home_region: str | None = None, user_id: str | None = None) -> str:
    user_id = user_id or str(uuid.uuid4())
    with get_engine().begin() as c:
        c.execute(text("INSERT INTO users (user_id, username, spending_style, home_region, demo_user) "
                       "VALUES (CAST(:u AS UUID), :name, :style, :region, true)"),
                  {"u": user_id, "name": f"t-{user_id[:8]}", "style": style, "region": home_region})
    return user_id


def seeded_diner(months: int, *, today: date | None = None, user_id: str | None = None) -> str:
    """A diner with this many months of seeded history, and one grocery
    purchase today. Seeding draws each day's purchases at random per user:
    about one diner in a thousand buys no groceries in 30 days, and a test
    that totals groceries for that user reads "No matching transactions"."""
    from banko_ai.services import ledger
    from banko_ai.services.accounts import list_accounts
    from banko_ai.services.seeding import seed_user
    uid = make_user("diner", user_id=user_id)
    seed_user(uid, months=months, today=today or date.today())
    card = next(a for a in list_accounts(uid)["accounts"] if a["kind"] == "credit")
    ledger.post_purchase(uid, card["account_id"], amount="23.45", merchant="Local Market", category="Groceries")
    return uid


def card_purchase(user_id: str, merchant: str, amount: str, category: str = "Coffee") -> str:
    """A card purchase posted the way the app posts one: with an embedding
    (a receipt, the load simulator and seeding all store one), here of the
    description, as seeding does."""
    from banko_ai.services import ledger
    from banko_ai.services._sql import vector_floats
    from banko_ai.services._vector import embed
    from banko_ai.services.accounts import list_accounts
    card = next(a for a in list_accounts(user_id)["accounts"] if a["kind"] == "credit")
    description = f"Spent ${float(amount):.2f} on {category.lower()} at {merchant} using Credit Card."
    return ledger.post_purchase(user_id, card["account_id"], amount=amount, merchant=merchant, category=category,
                                description=description, payment_method="Credit Card",
                                embedding=vector_floats(embed(description)))


def splurge(user_id: str) -> str:
    """An amount three standard deviations above what this user usually
    spends, whatever the seed drew for them."""
    with get_engine().connect() as c:
        mean, spread = c.execute(text("SELECT avg(expense_amount), stddev(expense_amount) FROM expenses "
                                      "WHERE user_id = CAST(:u AS UUID)"), {"u": user_id}).one()
    return f"{mean + 3 * spread:.2f}"


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
