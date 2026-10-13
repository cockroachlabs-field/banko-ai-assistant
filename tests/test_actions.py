"""Proposed changes run only after Confirm, exactly once, for their owner."""

import os
from datetime import date

import pytest

from banko_ai.services import actions as A
from banko_ai.services import budgets, ledger, transactions
from banko_ai.services.seeding import seed_user
from tests._money import drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture(scope="module")
def member():
    uid = make_user("balanced")
    seed_user(uid, months=1, today=date.today())
    yield uid
    drop_user(uid)


def _an_expense(user):
    return transactions.list_transactions(user, limit=20)["items"][0]


def test_budget_proposal_runs_on_confirm_once(member):
    p = A.propose(member, "budget", {"category": "Groceries", "amount": "432.10"}, "Set Groceries to $432.10")
    assert p["status"] == "pending"
    before = ledger.data_version(member)
    done = A.confirm(member, p["action_id"])
    assert done["status"] == "done"
    rows = {b["category"]: b for b in budgets.budget_progress(member)["items"]}
    assert float(rows["Groceries"]["budget"]) == 432.10
    again = A.confirm(member, p["action_id"])
    assert again["status"] == "done" and ledger.data_version(member) >= before


def test_recategorize_proposal(member):
    item = next(i for i in transactions.list_transactions(member, limit=50)["items"] if i.get("expense_id"))
    p = A.propose(member, "recategorize", {"expense_id": item["expense_id"], "category": "Entertainment"},
                  "Move it to Entertainment")
    A.confirm(member, p["action_id"])
    assert transactions.transaction_detail(member, item["expense_id"])["category"] == "Entertainment"


def test_cancel_then_confirm_does_nothing(member):
    p = A.propose(member, "budget", {"category": "Shopping", "amount": "99"}, "Shopping $99")
    assert A.cancel(member, p["action_id"])["status"] == "cancelled"
    assert A.confirm(member, p["action_id"])["status"] == "cancelled"


def test_someone_elses_action_is_not_found(member):
    other = make_user()
    try:
        p = A.propose(other, "budget", {"category": "Shopping", "amount": "10"}, "x")
        with pytest.raises(ledger.NotFound):
            A.confirm(member, p["action_id"])
    finally:
        drop_user(other)


def test_unknown_kind_is_refused(member):
    with pytest.raises(ValueError):
        A.propose(member, "transfer", {}, "nope")


def test_recategorize_rejects_unknown_category(member):
    item = next(i for i in transactions.list_transactions(member, limit=50)["items"] if i.get("expense_id"))
    with pytest.raises(ValueError):
        transactions.recategorize(member, item["expense_id"], "Not A Category")
