"""Seed plans are pure and deterministic: no database involved."""

from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal

from banko_ai.config.catalog import get_catalog
from banko_ai.services.seed_plan import build_plan, history_start

MAYA = "00000000-0000-0000-0000-0000000000a1"
SAM = "00000000-0000-0000-0000-0000000000a2"
END = date(2026, 10, 11)
START = history_start(END, 6)


def _plan(user=MAYA, style="diner", start=START, end=END):
    cat = get_catalog()
    return build_plan(user, cat.profiles[style], cat, start=start, end=end)


def test_history_start_is_months_back_and_clamped():
    assert history_start(date(2026, 10, 11), 6) == date(2026, 4, 11)
    assert history_start(date(2026, 3, 31), 1) == date(2026, 2, 28)
    assert history_start(date(2026, 1, 15), 2) == date(2025, 11, 15)


def test_same_user_same_plan():
    assert _plan().entries == _plan().entries


def test_different_users_differ():
    a = _plan(MAYA)
    b = _plan("00000000-0000-0000-0000-0000000000ff")
    assert [e.amount for e in a.entries] != [e.amount for e in b.entries]


def test_catch_up_regenerates_earlier_days_identically():
    cutoff = END - timedelta(days=10)
    short = _plan(end=cutoff)
    full = _plan()
    assert [e for e in full.entries if e.posted_at.date() <= cutoff] == short.entries
    assert all(e.posted_at.date() > cutoff for e in full.entries_after(cutoff))


def test_no_account_goes_below_its_floor():
    plan = _plan()
    balance = {a.key: a.opening_balance for a in plan.accounts}
    floor = {a.key: (-(a.credit_limit or Decimal("0")) if a.kind == "credit" else Decimal("0"))
             for a in plan.accounts}
    for entry in plan.entries:
        balance[entry.account_key] += entry.amount
        assert balance[entry.account_key] >= floor[entry.account_key], entry


def test_movements_balance_out():
    legs = defaultdict(Decimal)
    for entry in _plan().entries:
        if entry.transfer_ref:
            legs[entry.transfer_ref] += entry.amount
    assert legs and all(total == 0 for total in legs.values())


def test_subscriber_netflix_price_change_lands_in_the_last_two_months():
    plan = _plan(SAM, "subscriber")
    netflix = [e for e in plan.entries if e.merchant == "Netflix"]
    assert netflix and all(e.recurring for e in netflix)
    recent = [e for e in netflix if (END.year - e.posted_at.year) * 12 + END.month - e.posted_at.month < 2]
    older = [e for e in netflix if e not in recent]
    assert recent and {e.amount for e in recent} == {Decimal("-22.99")}
    assert older and {e.amount for e in older} == {Decimal("-15.99")}


def test_income_every_fourteen_days_and_purchases_look_real():
    plan = _plan()
    paydays = [e.posted_at.date() for e in plan.entries if e.kind == "income"]
    assert all((b - a).days == 14 for a, b in zip(paydays, paydays[1:]))
    purchases = [e for e in plan.entries if e.is_purchase]
    assert 250 <= len(purchases) <= 600
    assert all(e.description.startswith("Spent $") and e.category and e.merchant for e in purchases)
    assert len({e.expense_id for e in purchases}) == len(purchases)


def test_budgets_and_goals_come_from_the_profile():
    plan = _plan()
    assert plan.budgets["Restaurant"] == Decimal("600.00")
    assert [g.name for g in plan.goals] == ["Japan trip"]
    assert any(e.goal_key == plan.goals[0].key for e in plan.entries)
