"""One deletion path for users: every row they own goes, in every table."""

import os
from datetime import date

import pytest
from sqlalchemy import text

from banko_ai.db.engine import get_engine
from banko_ai.services import memories, threads
from banko_ai.services.seeding import seed_user
from banko_ai.services.users import delete_users, user_tables
from tests._money import make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


def _count(table: str, user_id: str) -> int:
    with get_engine().connect() as c:
        # As text: two legacy tables keep user_id as STRING.
        return c.execute(text(f"SELECT count(*) FROM {table} WHERE CAST(user_id AS STRING) = :u"),
                         {"u": user_id}).scalar()


def test_user_tables_cover_money_and_assistant_tables_children_first():
    tables = user_tables()
    assert {"accounts", "ledger_entries", "transfers", "goals", "expenses", "chat_threads", "memories",
            "receipts", "answer_cache", "search_cache", "assistant_actions", "user_data_versions"} <= set(tables)
    assert "users" not in tables
    assert tables.index("ledger_entries") < tables.index("accounts")
    assert tables.index("coach_nudges") < tables.index("spending_signals")


def test_delete_users_empties_every_user_table():
    uid = make_user("saver")
    seed_user(uid, months=1, today=date(2026, 10, 11))
    memories.remember(uid, "I am saving for a trip to Japan")
    threads.create_thread(uid)
    assert _count("ledger_entries", uid) > 0 and _count("memories", uid) == 1 and _count("chat_threads", uid) == 1
    assert delete_users([uid]) == 1
    for table in user_tables():
        assert _count(table, uid) == 0, table
    assert _count("users", uid) == 0


def test_deleting_nobody_is_a_no_op():
    assert delete_users([]) == 0
