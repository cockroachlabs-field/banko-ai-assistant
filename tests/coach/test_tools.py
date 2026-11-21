"""Unit tests for Coach tools. Uses a real local CRDB via DATABASE_URL.
Each test seeds its own rows and cleans up after itself."""

import os

import pytest
from sqlalchemy import create_engine, text

from banko_ai.coach.tools import (
    explain_nudge,
    get_recent_signals,
    get_recent_transactions,
    get_user_budget,
    set_budget,
)
from banko_ai.utils.migration import DatabaseMigration

TEST_USER = "00000000-0000-0000-0000-000000000fff"


@pytest.fixture(scope="module")
def db_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set")
    DatabaseMigration(database_url=url).migrate_to_coach_v1()
    # Ensure expenses table exists for get_recent_transactions tests; the
    # main app creates a richer schema via DatabaseManager, but the tool
    # only needs these columns.
    eng = create_engine(url)
    with eng.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS expenses (
              expense_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
              user_id UUID NOT NULL,
              expense_date DATE NOT NULL,
              expense_amount DECIMAL(10,2) NOT NULL,
              shopping_type STRING NOT NULL,
              description STRING,
              merchant STRING,
              payment_method STRING NOT NULL DEFAULT 'Credit Card'
            )
        """))
        conn.execute(text(
            "ALTER TABLE expenses ADD COLUMN IF NOT EXISTS payment_method STRING NOT NULL DEFAULT 'Credit Card'"))
    eng.dispose()
    return url


@pytest.fixture(autouse=True)
def _cleanup(db_url):
    def _purge():
        # Per-table transactions so a missing table or constraint error on
        # one delete cannot poison the others (CRDB aborts the whole tx on
        # error).
        eng = create_engine(db_url)
        for table in ("coach_nudges", "spending_signals", "expenses",
                      "user_budgets"):
            try:
                with eng.begin() as conn:
                    conn.execute(
                        text(f"DELETE FROM {table} WHERE user_id = :u"),
                        {"u": TEST_USER})
            except Exception:
                pass
        eng.dispose()

    # Purge on both sides: before for a clean slate even after a crashed
    # run, after so the module's last test leaves nothing on a shared DB.
    _purge()
    yield
    _purge()


def test_get_user_budget_default_when_no_override(db_url):
    result = get_user_budget(user_id=TEST_USER, category="dining",
                             database_url=db_url)
    assert "monthly_budget" in result
    assert result["category"] == "dining"
    assert result["source"] in {"default", "user_override"}


def test_set_budget_then_get_user_budget_returns_override(db_url):
    set_budget(user_id=TEST_USER, category="dining", amount=450.0,
               database_url=db_url)
    result = get_user_budget(user_id=TEST_USER, category="dining",
                             database_url=db_url)
    assert result["monthly_budget"] == 450.0
    assert result["source"] == "user_override"


def test_get_recent_signals_empty(db_url):
    result = get_recent_signals(user_id=TEST_USER, limit=10,
                                database_url=db_url)
    assert result == []


def test_get_recent_signals_returns_seeded_row(db_url):
    eng = create_engine(db_url)
    with eng.begin() as conn:
        conn.execute(text("""
            INSERT INTO spending_signals
              (user_id, signal_type, severity, payload, idempotency_key)
            VALUES (:u, 'anomaly', 'critical', '{"merchant":"Uber"}'::JSONB,
                    'seed-1')
        """), {"u": TEST_USER})
    eng.dispose()
    result = get_recent_signals(user_id=TEST_USER, limit=10,
                                database_url=db_url)
    assert len(result) == 1
    assert result[0]["signal_type"] == "anomaly"
    assert result[0]["payload"]["merchant"] == "Uber"


def test_get_recent_transactions_empty(db_url):
    result = get_recent_transactions(user_id=TEST_USER, limit=5,
                                     database_url=db_url)
    assert result == []


def test_explain_nudge_returns_record(db_url):
    eng = create_engine(db_url)
    with eng.begin() as conn:
        sig = conn.execute(text("""
            INSERT INTO spending_signals
              (user_id, signal_type, severity, payload, idempotency_key)
            VALUES (:u, 'budget_threshold', 'warn',
                    CAST(:payload AS JSONB), 'explain-seed-1')
            RETURNING signal_id
        """), {"u": TEST_USER,
               "payload": '{"category":"dining","pct_used":0.82}'}).fetchone()
        signal_id = str(sig[0])
        nudge = conn.execute(text("""
            INSERT INTO coach_nudges
              (signal_id, user_id, message, tool_trace, provider_used)
            VALUES (:sig, :u, 'You are at 82% of dining budget',
                    CAST(:trace AS JSONB), 'watsonx')
            RETURNING nudge_id
        """), {"sig": signal_id, "u": TEST_USER,
               "trace": '[{"tool":"get_user_budget"}]'}).fetchone()
        nudge_id = str(nudge[0])
    eng.dispose()

    result = explain_nudge(user_id=TEST_USER, nudge_id=nudge_id, database_url=db_url)
    assert result["message"] == "You are at 82% of dining budget"
    assert result["tool_trace"][0]["tool"] == "get_user_budget"
    assert result["provider_used"] == "watsonx"
