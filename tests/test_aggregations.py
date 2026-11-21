"""SQL aggregation correctness: the window excludes old rows, the user
filter excludes other users, and the fuzzy category map finds real
shopping_type values."""

import os
import uuid
from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine, text

from banko_ai.utils.aggregations import _to_pyformat, explain_aggregation, resolve_category, run_aggregation
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


def test_resolve_category_goes_through_the_catalog_first():
    """The README asks "what did I spend on dining last month?"; every alias
    in the catalog names its category, and an unknown word names none."""
    for phrase, label in (("dining", "Restaurant"), ("food", "Restaurant"), ("dining out", "Restaurant"),
                          ("eating out", "Restaurant"), ("takeout", "Restaurant"), ("gas", "Fuel"),
                          ("cafe", "Coffee"), ("rideshare", "Transport"), ("streaming", "Subscription"),
                          ("my groceries", "Groceries"), ("Restaurant", "Restaurant")):
        assert resolve_category(phrase, DB) == label, phrase
    assert resolve_category("zorbing", DB) is None


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
    # PostgreSQL :: casts should not be treated as placeholders
    assert _to_pyformat("WHERE d::DATE >= :start AND n = :name") == "WHERE d::DATE >= %(start)s AND n = %(name)s"


@pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")
def test_merchant_filter_matches_sql():
    from datetime import date, timedelta

    from sqlalchemy import text as sql_text

    from banko_ai.config.settings import get_config
    from banko_ai.db.engine import get_engine
    from banko_ai.services.seeding import seed_user
    from banko_ai.utils.aggregations import resolve_merchant
    from banko_ai.utils.intent_classifier import AggregationIntent
    from tests._money import drop_user, make_user

    uid = make_user("diner")
    try:
        seed_user(uid, months=2, today=date.today())
        url = get_config().database_url
        with get_engine().connect() as c:
            merchant, n = c.execute(sql_text(
                "SELECT merchant, count(*) FROM expenses WHERE user_id = CAST(:u AS UUID) "
                "GROUP BY merchant ORDER BY count(*) DESC LIMIT 1"), {"u": uid}).fetchone()
        assert resolve_merchant(merchant.lower(), uid, url) == merchant
        assert resolve_merchant("zzzz nothing", uid, url) is None
        intent = AggregationIntent("count", merchant.lower(), date.today() - timedelta(days=90),
                                   date.today() + timedelta(days=1))
        agg = run_aggregation(intent, uid, url, merchant=merchant)
        assert agg.merchant == merchant and agg.category is None and agg.count == n
        assert "scan" in explain_aggregation(intent, uid, url, merchant=merchant)
    finally:
        drop_user(uid)


def test_the_largest_purchases_come_back_in_order_of_amount():
    """The rows a total carries are the most recent ones, which is how a
    month's largest purchase went missing: it was not among the last ten."""
    end = date.today() + timedelta(days=1)
    r = run_aggregation(AggregationIntent("largest", None, end - timedelta(days=61), end, limit=3), USER, DB)
    assert [(row["amount"], row["merchant"], row["category"]) for row in r.rows] == [
        (100.00, "M2", "Restaurant"), (72.59, "M1", "Restaurant"), (50.00, "M0", "Restaurant")]
    assert r.rows[0]["date"] == str(date.today() - timedelta(days=45))   # the oldest row in the window, and the largest
    assert r.count == 4 and round(r.total, 2) == 247.59                    # still out of every purchase in the window


def test_the_smallest_purchases_come_back_smallest_first():
    end = date.today() + timedelta(days=1)
    r = run_aggregation(AggregationIntent("smallest", None, end - timedelta(days=61), end, limit=2), USER, DB)
    assert [row["amount"] for row in r.rows] == [25.00, 50.00]


def test_the_largest_purchase_in_one_category_or_window():
    end = date.today() + timedelta(days=1)
    groceries = run_aggregation(AggregationIntent("largest", "groceries", end - timedelta(days=61), end), USER, DB)
    assert [row["amount"] for row in groceries.rows] == [25.00] and groceries.category == "Groceries"
    ever = run_aggregation(AggregationIntent("largest", None, end - timedelta(days=365), end), USER, DB)
    assert ever.rows[0]["amount"] == 999.99
    none = run_aggregation(AggregationIntent("largest", None, end - timedelta(days=2), end), USER, DB)
    assert none.rows == [] and none.count == 0


def test_the_plan_of_a_size_question_is_the_plan_of_the_statement_that_ordered_the_rows():
    """The trace shows how the answer was computed. For the largest
    purchases that is the ORDER BY amount statement, not the totals."""
    end = date.today() + timedelta(days=1)
    largest = AggregationIntent("largest", None, end - timedelta(days=61), end, limit=3)
    plan = explain_aggregation(largest, USER, DB)
    assert "expense_amount" in plan and ("top-k" in plan or "sort" in plan), plan
    assert "group (scalar)" not in plan
    totals = explain_aggregation(AggregationIntent("sum", None, end - timedelta(days=61), end), USER, DB)
    assert "group (scalar)" in totals


def test_a_size_question_lists_ten_at_most_and_keeps_what_was_asked():
    end = date.today() + timedelta(days=1)
    r = run_aggregation(AggregationIntent("largest", None, end - timedelta(days=365), end, limit=50), USER, DB)
    assert len(r.rows) == 5 and r.limit == 50            # all five there are; the card says what was asked
    eng = create_engine(DB)
    with eng.begin() as c:
        for i in range(12):
            c.execute(text("INSERT INTO expenses (expense_id, user_id, expense_date, expense_amount, shopping_type, "
                           "description, merchant, payment_method) VALUES (:id, :u, :d, :a, 'Coffee', '', :m, "
                           "'Credit Card')"), {"id": str(uuid.uuid4()), "u": USER, "d": date.today(),
                                               "a": 3 + i, "m": f"C{i}"})
    eng.dispose()
    r = run_aggregation(AggregationIntent("largest", None, end - timedelta(days=365), end, limit=50), USER, DB)
    assert len(r.rows) == 10 and r.count == 17 and r.limit == 50


def test_purchases_of_the_same_amount_come_back_in_one_order():
    """Latest first, then by id: the same question gives the same card."""
    day = date.today() - timedelta(days=3)
    ids = sorted(str(uuid.uuid4()) for _ in range(3))
    eng = create_engine(DB)
    with eng.begin() as c:
        for expense_id, when in ((ids[2], day), (ids[0], day), (ids[1], day - timedelta(days=1))):
            c.execute(text("INSERT INTO expenses (expense_id, user_id, expense_date, expense_amount, shopping_type, "
                           "description, merchant, payment_method) VALUES (:id, :u, :d, 4321.00, 'Travel', '', :m, "
                           "'Credit Card')"), {"id": expense_id, "u": USER, "d": when, "m": f"T-{expense_id[:4]}"})
    eng.dispose()
    end = date.today() + timedelta(days=1)
    intent = AggregationIntent("largest", None, end - timedelta(days=30), end, limit=3)
    first = run_aggregation(intent, USER, DB).rows
    assert [row["merchant"] for row in first] == [f"T-{ids[0][:4]}", f"T-{ids[2][:4]}", f"T-{ids[1][:4]}"]
    assert run_aggregation(intent, USER, DB).rows == first


def _buy(conn, merchant: str, amount: float, days_ago: int, category: str) -> None:
    conn.execute(text("""
        INSERT INTO expenses (expense_id, user_id, expense_date, expense_amount, shopping_type, description,
                              merchant, payment_method)
        VALUES (:id, :u, :d, :a, :c, '', :m, 'Credit Card')
    """), {"id": str(uuid.uuid4()), "u": USER, "d": date.today() - timedelta(days=days_ago), "a": amount,
           "c": category, "m": merchant})


def test_a_total_is_broken_down_over_the_whole_window_not_over_the_purchases_it_lists():
    """The observations under a total named "top merchants" added up from
    the latest purchases the card lists. Over three months that is a sample:
    the landlord, paid on the first, was not in it at all, and the pizza
    place was credited with one week's orders."""
    eng = create_engine(DB)
    with eng.begin() as c:
        for days_ago in (85, 55, 25):
            _buy(c, "Greenway Apartments", 1850.00, days_ago, "Housing")
        for days_ago in range(1, 13):                      # the latest twelve purchases, none of them the rent
            _buy(c, "Corner Cafe", 5.00, days_ago, "Coffee")
    end = date.today() + timedelta(days=1)
    r = run_aggregation(AggregationIntent("sum", None, end - timedelta(days=91), end), USER, DB)
    assert r.count == 19 and round(r.total, 2) == 5857.59   # 3 rents, 12 coffees and the fixture's four in the window
    assert len(r.rows) == 10 and {row["merchant"] for row in r.rows} == {"Corner Cafe", "M4"}   # the latest ten, no rent
    assert [(m["merchant"], m["total"], m["count"], m["category"]) for m in r.top_merchants[:3]] == [
        ("Greenway Apartments", 5550.0, 3, "Housing"), ("M2", 100.0, 1, "Restaurant"), ("M1", 72.59, 1, "Restaurant")]
    with eng.begin() as c:
        _buy(c, "Greenway Apartments", 40.00, 3, "Utilities")       # the landlord also bills the water
    mixed = run_aggregation(AggregationIntent("sum", None, end - timedelta(days=91), end), USER, DB).top_merchants[0]
    assert (mixed["merchant"], mixed["total"], mixed["category"]) == ("Greenway Apartments", 5590.0, None)   # no one category
    assert [(g["category"], g["total"], g["count"]) for g in r.top_categories] == [
        ("Housing", 5550.0, 3), ("Restaurant", 222.59, 3), ("Coffee", 60.0, 12), ("Groceries", 25.0, 1)]

    dining = run_aggregation(AggregationIntent("sum", "restaurants", end - timedelta(days=91), end), USER, DB)
    assert dining.top_categories == []                      # one category was asked about: its merchants
    assert [(m["merchant"], m["total"]) for m in dining.top_merchants] == [("M2", 100.0), ("M1", 72.59), ("M0", 50.0)]
    one = run_aggregation(AggregationIntent("sum", None, end - timedelta(days=91), end), USER, DB, merchant="Corner Cafe")
    assert one.top_merchants == [] and one.top_categories == [] and one.count == 12     # one merchant: nothing to break down
    eng.dispose()
