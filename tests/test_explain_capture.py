"""Test EXPLAIN capture for aggregation and vector search queries."""

import os
import uuid
from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine, text

from banko_ai.utils.aggregations import explain_aggregation
from banko_ai.utils.intent_classifier import AggregationIntent
from banko_ai.vector_search.search import VectorSearchEngine

DB = os.getenv("DATABASE_URL")
USER = "00000000-0000-0000-0000-00000000e99a"
pytestmark = pytest.mark.skipif(not DB, reason="DATABASE_URL not set")


@pytest.fixture(autouse=True)
def seed():
    zero_vec = "[" + ",".join(["0.0"] * 384) + "]"
    eng = create_engine(DB)
    with eng.begin() as c:
        c.execute(text("DELETE FROM expenses WHERE user_id = :u"), {"u": USER})
        for i, (amount, days_ago, cat) in enumerate([
                (50.00, 10, "Restaurant"), (72.59, 30, "Restaurant"),
                (100.00, 45, "Groceries"), (25.00, 5, "Groceries")]):
            c.execute(text(f"""
                INSERT INTO expenses (expense_id, user_id, expense_date,
                  expense_amount, shopping_type, description, merchant,
                  payment_method, embedding)
                VALUES (:id, :u, :d, :a, :c, 'Test purchase', :m, 'Credit Card',
                  '{zero_vec}'::VECTOR(384))
            """), {"id": str(uuid.uuid4()), "u": USER,
                   "d": date.today() - timedelta(days=days_ago),
                   "a": amount, "c": cat, "m": f"Merchant{i}"})
    yield
    with eng.begin() as c:
        c.execute(text("DELETE FROM expenses WHERE user_id = :u"), {"u": USER})
    eng.dispose()


def test_explain_aggregation_contains_plan():
    end = date.today() + timedelta(days=1)
    intent = AggregationIntent("sum", "restaurants",
                               end - timedelta(days=61), end)
    plan = explain_aggregation(intent, USER, DB)
    assert isinstance(plan, str)
    assert len(plan) > 0
    plan_lower = plan.lower()
    assert "group" in plan_lower or "aggregat" in plan_lower


def test_search_expenses_explain_mentions_expenses():
    engine = VectorSearchEngine(database_url=DB, cache_manager=None)
    result = engine.search_expenses(
        query="restaurant purchases",
        user_id=USER,
        limit=5,
        capture_explain=True
    )
    assert isinstance(result, tuple)
    assert len(result) == 2
    results, explain_text = result
    assert isinstance(results, list)
    assert isinstance(explain_text, str)
    assert len(explain_text) > 0
    # The fallback text also names the table, so rule it out explicitly.
    assert "explain unavailable" not in explain_text
    assert "execution time" in explain_text.lower()
    assert "expenses" in explain_text.lower()
