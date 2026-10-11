"""SQL aggregation correctness: the window excludes old rows, the user
filter excludes other users, and the fuzzy category map finds real
shopping_type values."""

import os
import uuid
from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine, text

from banko_ai.utils.aggregations import _to_pyformat, resolve_category, run_aggregation
from banko_ai.utils.intent_classifier import AggregationIntent

DB = os.getenv("DATABASE_URL")
USER = "00000000-0000-0000-0000-00000000a99e"
pytestmark = pytest.mark.skipif(not DB, reason="DATABASE_URL not set")


@pytest.fixture(autouse=True)
def seed():
    eng = create_engine(DB)
    with eng.begin() as c:
        c.execute(text("DELETE FROM expenses WHERE user_id = :u"), {"u": USER})
        for i, (amount, days_ago, cat) in enumerate([
                (50.00, 10, "Restaurant"), (72.59, 30, "Restaurant"),
                (100.00, 45, "Restaurant"), (999.99, 200, "Restaurant"),
                (25.00, 5, "Groceries")]):
            c.execute(text("""
                INSERT INTO expenses (expense_id, user_id, expense_date,
                  expense_amount, shopping_type, description, merchant,
                  payment_method)
                VALUES (:id, :u, :d, :a, :c, '', :m, 'Credit Card')
            """), {"id": str(uuid.uuid4()), "u": USER,
                   "d": date.today() - timedelta(days=days_ago),
                   "a": amount, "c": cat, "m": f"M{i}"})
    yield
    with eng.begin() as c:
        c.execute(text("DELETE FROM expenses WHERE user_id = :u"), {"u": USER})
    eng.dispose()


def test_resolve_category_fuzzy():
    assert resolve_category("restaurants", DB) == "Restaurant"
    assert resolve_category("grocery", DB) == "Groceries"
    assert resolve_category("zzzunknownzzz", DB) is None


def test_sum_respects_window_and_user():
    end = date.today() + timedelta(days=1)
    intent = AggregationIntent("sum", "restaurants",
                               end - timedelta(days=61), end)
    r = run_aggregation(intent, USER, DB)
    assert r.count == 3
    assert round(r.total, 2) == 222.59
    assert r.category == "Restaurant"
    assert len(r.rows) == 3


def test_count_all_categories():
    end = date.today() + timedelta(days=1)
    intent = AggregationIntent("count", None, end - timedelta(days=61), end)
    r = run_aggregation(intent, USER, DB)
    assert r.count == 4


def test_to_pyformat_converts_placeholders():
    """Test that :name placeholders are converted to %(name)s format."""
    # Simple case
    assert _to_pyformat("WHERE a = :u") == "WHERE a = %(u)s"
    # Multiple placeholders
    assert _to_pyformat("WHERE a = :u AND b = :s") == "WHERE a = %(u)s AND b = %(s)s"
    # Word boundary: :start and :start_date should not interfere
    assert _to_pyformat("WHERE a = :start AND b = :start_date") == "WHERE a = %(start)s AND b = %(start_date)s"
    # No placeholders
    assert _to_pyformat("WHERE a = 1") == "WHERE a = 1"
    # Repeated placeholder
    assert _to_pyformat("WHERE a = :x OR b = :x") == "WHERE a = %(x)s OR b = %(x)s"
