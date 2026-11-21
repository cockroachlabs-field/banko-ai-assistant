"""Derived views: recurring bills, goals, cash flow and the dashboard."""

import os
from datetime import date, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import text

from banko_ai.config.settings import reset_config
from banko_ai.db.engine import get_engine
from banko_ai.services import goals as G
from banko_ai.services import recurring as R
from banko_ai.services import reports as P
from banko_ai.services.accounts import list_accounts
from banko_ai.services.seeding import seed_user
from tests._money import drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")
TODAY = date(2026, 10, 11)


@pytest.fixture(autouse=True)
def _current_reads(monkeypatch):
    monkeypatch.setenv("REPORTING_FOLLOWER_READS", "false")
    reset_config()


# A shop visited on no schedule and for no steady amount. The seed draws each
# user's groceries at random, and now and then the draw looks weekly: the
# test that said "Whole Foods is no subscription" failed for that user.
BODEGA = "Corner Bodega"
_VISITS = ((97, "12.10"), (94, "87.40"), (75, "5.25"), (73, "44.00"), (32, "23.80"))   # days before TODAY, amount


@pytest.fixture(scope="module")
def subscriber():
    uid = make_user("subscriber")
    seed_user(uid, months=4, today=TODAY)
    with get_engine().begin() as c:
        for days_ago, amount in _VISITS:
            c.execute(text("INSERT INTO expenses (expense_id, user_id, expense_date, expense_amount, shopping_type, "
                           "description, merchant, payment_method) "
                           "VALUES (gen_random_uuid(), CAST(:u AS UUID), :d, :a, 'Groceries', '', :m, 'Credit Card')"),
                      {"u": uid, "d": TODAY - timedelta(days=days_ago), "a": amount, "m": BODEGA})
    yield uid
    drop_user(uid)


def test_recurring_finds_monthly_bills_and_the_price_change(subscriber):
    found = {r.merchant: r for r in R.detect_recurring(subscriber, today=TODAY)}
    assert {"Netflix", "Spotify", "Electric Company", "Internet Provider"} <= set(found)
    assert found["Netflix"].cadence == found["Spotify"].cadence == "monthly"
    assert found["Netflix"].price_change_pct == pytest.approx(43.78, abs=0.1)
    assert found["Netflix"].previous_amount == Decimal("15.99")
    assert found["Spotify"].price_change_pct is None
    assert found["Electric Company"].price_change_pct is None  # variable bill, not a price change
    assert BODEGA not in found  # five visits on no schedule are not a subscription


def test_upcoming_bills_are_in_the_window(subscriber):
    bills = R.upcoming_bills(subscriber, days=14, today=TODAY)
    assert bills and all(TODAY.isoformat() <= b["next_due"] <= "2026-10-25" for b in bills)


def test_goals_show_progress_and_take_contributions(subscriber):
    (goal,) = G.list_goals(subscriber, today=TODAY)
    assert goal["name"] == "New laptop" and goal["saved"] > 0 and 0 < goal["progress"] < 1
    assert goal["monthly_pace"] > 0 and goal["projected_completion"]
    checking = next(a for a in list_accounts(subscriber)["accounts"] if a["kind"] == "checking")
    result = G.contribute(subscriber, goal["goal_id"], checking["account_id"], "25.00", idempotency_key="test-c1")
    assert result.status == "ok"
    assert G.list_goals(subscriber, today=TODAY)[0]["saved"] == pytest.approx(goal["saved"] + 25.0)


def test_cashflow_months_add_up(subscriber):
    out = P.cashflow(subscriber, months=3, today=TODAY)
    assert [m["month"] for m in out["months"]] == ["2026-08", "2026-09", "2026-10"]
    for m in out["months"]:
        assert m["net"] == pytest.approx(m["income"] - m["spending"])
    assert out["categories"][0]["total"] >= out["categories"][-1]["total"]
    assert out["merchants"] and out["categories"][0]["icon"]


def test_dashboard_has_every_card(subscriber):
    out = P.dashboard(subscriber, today=TODAY)
    assert set(out) == {"net_worth", "net_worth_history", "spending", "budgets", "upcoming_bills",
                        "recent", "insight"}
    assert out["spending"]["this_month"] and out["spending"]["last_month"]
    assert len(out["recent"]) == 6
