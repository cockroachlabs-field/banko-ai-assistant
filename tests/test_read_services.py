"""Read services answer from real rows, scoped to one user."""

import base64
import os
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import text

from banko_ai.config.settings import reset_config
from banko_ai.db.engine import get_engine
from banko_ai.services import accounts as A
from banko_ai.services import budgets as B
from banko_ai.services import ledger as L
from banko_ai.services import transactions as T
from banko_ai.services.seeding import seed_user
from tests._money import drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")
TODAY = date(2026, 10, 11)


@pytest.fixture(autouse=True)
def _current_reads(monkeypatch):
    monkeypatch.setenv("REPORTING_FOLLOWER_READS", "false")
    reset_config()


@pytest.fixture(scope="module")
def seeded():
    uid = make_user("diner")
    seed_user(uid, months=2, today=TODAY)
    yield uid
    drop_user(uid)


def test_accounts_group_and_net_worth(seeded):
    out = A.list_accounts(seeded)
    assert {a["kind"] for a in out["accounts"]} == {"checking", "savings", "credit"}
    assert out["net_worth"] == pytest.approx(sum(a["balance"] for a in out["accounts"]))
    assert 0 <= out["utilization"] <= 1
    assert set(out["groups"]) == {"cash", "credit"}


def test_other_users_account_is_invisible(seeded):
    other = make_user()
    try:
        acct = L.open_account(other, kind="checking", name="C", institution="B", mask="1")
        assert A.get_account(seeded, acct) is None
        assert A.statement(seeded, acct, "2026-09") is None
    finally:
        drop_user(other)


def test_statement_adds_up_and_exports(seeded):
    checking = next(a for a in A.list_accounts(seeded)["accounts"] if a["kind"] == "checking")
    months = A.statement_months(seeded, checking["account_id"])
    st = A.statement(seeded, checking["account_id"], months[1])
    assert st["closing"] == pytest.approx(st["opening"] + st["charges"] + st["credits"])
    assert st["entries"]
    csv = A.statement_csv(st)
    assert csv.splitlines()[0] == "Date,Description,Kind,Amount"
    assert f"Closing balance,,,{st['closing']:.2f}" in csv


def test_statement_month_boundary_is_utc():
    uid = make_user()
    try:
        acct = L.open_account(uid, kind="checking", name="C", institution="B", mask="1", opening_balance="0")
        late = datetime(2026, 9, 30, 23, 30, tzinfo=timezone.utc)
        L.post_history(uid, [L.HistoryRow(entry_id=str(uuid.uuid4()), account_id=acct, posted_at=late,
                                          amount=Decimal("5.00"), kind="income", description="Late deposit")])
        L.rebalance(uid)
        assert len(A.statement(uid, acct, "2026-09")["entries"]) == 1
        assert A.statement(uid, acct, "2026-10")["entries"] == []
    finally:
        drop_user(uid)


def test_bad_month_is_rejected(seeded):
    checking = A.list_accounts(seeded)["accounts"][0]
    with pytest.raises(ValueError):
        A.statement(seeded, checking["account_id"], "2026-13")


def test_net_worth_history_ends_at_todays_net_worth(seeded):
    history = A.net_worth_history(seeded, months=3, today=TODAY)
    assert [h["month"] for h in history] == ["2026-08", "2026-09", "2026-10"]
    assert history[-1]["net_worth"] == pytest.approx(A.list_accounts(seeded)["net_worth"])


def test_transactions_page_with_a_stable_cursor(seeded):
    first = T.list_transactions(seeded, limit=10)
    second = T.list_transactions(seeded, limit=10, cursor=first["next_cursor"])
    ids = [i["id"] for i in first["items"]] + [i["id"] for i in second["items"]]
    assert len(ids) == len(set(ids)) == 20
    stamps = [i["posted_at"] for i in first["items"] + second["items"]]
    assert stamps == sorted(stamps, reverse=True)
    assert first["items"][0]["category_icon"]


def test_garbage_cursor_starts_over(seeded):
    junk = base64.urlsafe_b64encode(b"not|a|cursor").decode()
    assert T.list_transactions(seeded, limit=5, cursor=junk)["items"]


def test_filters_narrow_the_list(seeded):
    restaurants = T.list_transactions(seeded, limit=200, category="restaurant")["items"]
    assert restaurants and all(i["category"] == "Restaurant" for i in restaurants)
    incomes = T.list_transactions(seeded, limit=200, kind="income")["items"]
    assert incomes and all(i["amount"] > 0 for i in incomes)


def test_semantic_search_uses_the_per_user_vector_index(seeded):
    vec = "'[" + ",".join(["0.01"] * 384) + "]'"
    sql = (T.semantic_sql().replace(":qvec", vec).replace(":u", f"'{seeded}'")
           .replace(":k", "50").replace(":limit", "20"))
    with get_engine().connect() as c:
        plan = "\n".join(r[0] for r in c.execute(text("EXPLAIN " + sql)))
    assert "vector search" in plan and "idx_expenses_user_embedding" in plan
    hits = T.semantic_search(seeded, "pizza dinner", limit=5)["items"]
    assert hits and all(h["distance"] is not None for h in hits)


def test_transaction_detail_has_similar_and_residency(seeded):
    item = T.list_transactions(seeded, limit=50, kind="purchase")["items"][0]
    detail = T.transaction_detail(seeded, item["id"])
    assert detail["merchant"] == item["merchant"]
    assert detail["similar"] and "residency" in detail
    assert T.transaction_detail(seeded, str(uuid.uuid4())) is None
    assert T.transaction_detail(seeded, "not-a-uuid") is None


def test_budget_progress_and_update(seeded):
    out = B.budget_progress(seeded, today=TODAY)
    restaurant = next(i for i in out["items"] if i["category"] == "Restaurant")
    assert restaurant["budget"] == 600.0 and restaurant["status"] in ("ok", "watch", "over")
    try:
        B.set_budget(seeded, "restaurant", "650")
        assert next(i for i in B.budget_progress(seeded, today=TODAY)["items"]
                    if i["category"] == "Restaurant")["budget"] == 650.0
    finally:
        B.set_budget(seeded, "restaurant", "600")
    with pytest.raises(L.LedgerError):
        B.set_budget(seeded, "income", "100")
    with pytest.raises(L.LedgerError):
        B.set_budget(seeded, "not a category", "100")


def test_zero_budget_deletes_it():
    uid = make_user()
    try:
        B.set_budget(uid, "restaurant", "100")
        assert [i["category"] for i in B.budget_progress(uid, today=TODAY)["items"]] == ["Restaurant"]
        B.set_budget(uid, "restaurant", 0)
        assert B.budget_progress(uid, today=TODAY)["items"] == []
    finally:
        drop_user(uid)


def test_other_users_transaction_is_invisible(seeded):
    purchase = T.list_transactions(seeded, limit=50, kind="purchase")["items"][0]
    other = make_user()
    try:
        acct = L.open_account(other, kind="checking", name="C", institution="B", mask="1", opening_balance="0")
        entry_id = str(uuid.uuid4())
        L.post_history(other, [L.HistoryRow(entry_id=entry_id, account_id=acct,
                                            posted_at=datetime(2026, 9, 3, tzinfo=timezone.utc),
                                            amount=Decimal("9.00"), kind="income", description="Gift")])
        assert T.transaction_detail(other, entry_id)["amount"] == 9.0
        assert T.transaction_detail(seeded, entry_id) is None
        assert T.transaction_detail(other, purchase["id"]) is None
        assert T.transaction_detail(other, purchase["expense_id"]) is None
    finally:
        drop_user(other)


def test_cursor_is_stable_when_timestamps_tie():
    uid = make_user()
    try:
        acct = L.open_account(uid, kind="checking", name="C", institution="B", mask="1", opening_balance="0")
        same = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
        rows = [L.HistoryRow(entry_id=str(uuid.uuid4()), account_id=acct, posted_at=same,
                             amount=Decimal("1.00"), kind="income", description=f"Tie {n}") for n in range(5)]
        L.post_history(uid, rows)
        seen, cursor = [], None
        for _ in range(5):
            page = T.list_transactions(uid, limit=2, cursor=cursor, account_id=acct)
            seen += [i["id"] for i in page["items"]]
            cursor = page["next_cursor"]
            if cursor is None:
                break
        assert cursor is None
        assert sorted(seen) == sorted(r.entry_id for r in rows) and len(seen) == 5
    finally:
        drop_user(uid)


def test_statement_csv_defuses_formulas():
    uid = make_user()
    try:
        acct = L.open_account(uid, kind="checking", name="C", institution="B", mask="1", opening_balance="0")
        L.post_history(uid, [L.HistoryRow(entry_id=str(uuid.uuid4()), account_id=acct,
                                          posted_at=datetime(2026, 9, 10, tzinfo=timezone.utc),
                                          amount=Decimal("-12.50"), kind="purchase",
                                          description='=HYPERLINK("http://x","y")')])
        L.rebalance(uid)
        csv = A.statement_csv(A.statement(uid, acct, "2026-09"))
        line = next(row for row in csv.splitlines() if "HYPERLINK" in row)
        assert "'=HYPERLINK" in line and line.endswith(",purchase,-12.50")
    finally:
        drop_user(uid)
