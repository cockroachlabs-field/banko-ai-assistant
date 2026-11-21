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


def test_what_the_agents_wrote_about_a_users_purchases_goes_with_them():
    """agent_decisions has no user column: a decision names its user, or the
    purchase it was about, inside its context. The fraud agent's reasoning
    names the merchant and the amount, so it is the user's as much as the
    purchase is."""
    import json
    import uuid
    gone, stays = make_user("saver"), make_user("saver")
    seed_user(gone, months=1, today=date(2026, 10, 11))
    with get_engine().begin() as c:
        expense = str(c.execute(text("SELECT expense_id FROM expenses WHERE user_id = CAST(:u AS UUID) LIMIT 1"),
                                {"u": gone}).scalar())
        ids = {"by purchase": (str(uuid.uuid4()), {"expense_id": expense}),
               "by user": (str(uuid.uuid4()), {"user_id": gone, "action": "budget_check"}),
               "someone else's": (str(uuid.uuid4()), {"user_id": stays, "expense_id": str(uuid.uuid4())})}
        for decision_id, context in ids.values():
            c.execute(text("INSERT INTO agent_decisions (decision_id, decision_type, context, reasoning) "
                           "VALUES (CAST(:d AS UUID), 'fraud_analysis', CAST(:c AS JSONB), "
                           "'The $9.75 purchase at Blue Bottle looks normal.')"),
                      {"d": decision_id, "c": json.dumps(context)})
    try:
        assert delete_users([gone]) == 1
        with get_engine().connect() as c:
            left = {str(r[0]) for r in c.execute(text(
                "SELECT decision_id FROM agent_decisions WHERE decision_id = ANY(CAST(:ids AS UUID[]))"),
                {"ids": [d for d, _ in ids.values()]})}
        assert left == {ids["someone else's"][0]}
    finally:
        delete_users([stays])
        with get_engine().begin() as c:
            c.execute(text("DELETE FROM agent_decisions WHERE decision_id = ANY(CAST(:ids AS UUID[]))"),
                      {"ids": [d for d, _ in ids.values()]})
