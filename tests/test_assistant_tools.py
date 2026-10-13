"""Assistant tools: validated arguments, user scoping, the per-user vector
index, and errors that never escape."""

import os
from datetime import date, timedelta

import pytest
from sqlalchemy import text

from banko_ai.assistant import tools as T
from banko_ai.db.engine import get_engine
from banko_ai.services import ledger
from banko_ai.services._sql import vector_text
from banko_ai.services.accounts import list_accounts
from banko_ai.services.seeding import seed_user
from banko_ai.utils.request_trace import current_trace, reset_trace, start_trace
from tests._money import drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture(scope="module")
def member():
    uid = make_user("diner")
    seed_user(uid, months=2, today=date.today())
    yield uid
    with get_engine().begin() as c:
        for t in ("search_cache", "answer_cache", "memories", "assistant_actions"):
            c.execute(text(f"DELETE FROM {t} WHERE user_id = CAST(:u AS UUID)"), {"u": uid})
    drop_user(uid)


@pytest.fixture
def ctx(member):
    return T.ToolContext(user_id=member)


@pytest.fixture
def trace():
    token = start_trace()
    yield
    reset_trace(token)


def test_the_tool_set_matches_the_spec():
    assert set(T.TOOLS) == {
        "search_transactions", "spend_summary", "receipt_details", "similar_transactions", "duplicate_check",
        "merchant_history", "budget_status", "recurring_bills", "goals_status", "balances", "cash_flow",
        "recall", "remember", "propose_recategorize", "propose_budget"}
    described = T.describe_tools()
    for name in T.TOOLS:
        assert described.count(f"{name}(") == 1


def test_search_returns_own_rows_and_caches(ctx, trace):
    first = T.run_tool("search_transactions", {"query": "pizza", "limit": 5}, ctx)
    assert first["ok"] and 0 < len(first["result"]["items"]) <= 5
    assert all(i["expense_id"] for i in first["result"]["items"])
    again = T.run_tool("search_transactions", {"query": "pizza", "limit": 5}, ctx)
    assert again["result"] == first["result"]
    assert any(s["kind"] == "cache" and s["status"] == "hit" for s in current_trace())


def test_bad_arguments_and_unknown_tools_never_raise(ctx):
    bad = T.run_tool("search_transactions", {"limit": "lots"}, ctx)
    assert bad == {"tool": "search_transactions", "ok": False, "error": "bad arguments", "ms": bad["ms"]}
    assert T.run_tool("drop_tables", {}, ctx)["ok"] is False
    assert T.run_tool("balances", None, ctx)["ok"] is True


def test_spend_summary_matches_sql(ctx, member):
    start = date.today() - timedelta(days=30)
    out = T.run_tool("spend_summary", {"date_from": str(start)}, ctx)["result"]
    with get_engine().connect() as c:
        total = c.execute(text("SELECT COALESCE(sum(expense_amount), 0) FROM expenses WHERE user_id = CAST(:u AS UUID) "
                               "AND expense_date >= :s AND expense_date < :e"),
                          {"u": member, "s": start, "e": date.today() + timedelta(days=1)}).scalar()
    assert out["total"] == pytest.approx(float(total), abs=0.01)


def test_merchant_history_and_similar_stay_with_the_user(ctx, member):
    with get_engine().connect() as c:
        merchant, expense_id = c.execute(text(
            "SELECT merchant, expense_id FROM expenses WHERE user_id = CAST(:u AS UUID) "
            "AND embedding IS NOT NULL LIMIT 1"), {"u": member}).fetchone()
    hist = T.run_tool("merchant_history", {"merchant": merchant}, ctx)["result"]
    assert hist["merchant"] == merchant and hist["visits"] >= 1
    sim = T.run_tool("similar_transactions", {"expense_id": str(expense_id)}, ctx)["result"]
    assert sim and str(expense_id) not in {s["expense_id"] for s in sim}
    other = T.ToolContext(user_id=make_user())
    try:
        assert T.run_tool("similar_transactions", {"expense_id": str(expense_id)}, other)["result"] == []
    finally:
        drop_user(other.user_id)


def test_duplicate_check_finds_a_planted_twin(ctx, member):
    acct = next(a for a in list_accounts(member)["accounts"] if a["kind"] == "credit")
    vec = [0.01] * 384
    first = ledger.post_purchase(member, acct["account_id"], amount="18.40", merchant="Twin Cafe",
                                 category="Coffee", embedding=vec)
    second = ledger.post_purchase(member, acct["account_id"], amount="18.40", merchant="Twin Cafe",
                                  category="Coffee", embedding=vec)
    out = T.run_tool("duplicate_check", {"expense_id": second}, ctx)["result"]
    assert [d["expense_id"] for d in out["candidates"]] == [first]


def test_proposals_are_pending_until_confirmed(ctx):
    out = T.run_tool("propose_budget", {"category": "Groceries", "amount": "321"}, ctx)
    assert out["ok"] and out["result"]["status"] == "pending"
    assert "Groceries" in out["result"]["summary"]


def test_remember_then_recall(ctx):
    T.run_tool("remember", {"content": "I'm saving for a new bike"}, ctx)
    got = T.run_tool("recall", {"query": "what am I saving for"}, ctx)["result"]
    assert "bike" in got[0]["content"]


def test_receipt_details_without_a_receipt(ctx):
    out = T.run_tool("receipt_details", {}, ctx)
    assert out["ok"] and out["result"] is None


def test_similar_transactions_plan_uses_the_vector_index(member):
    with get_engine().connect() as c:
        plan = "\n".join(r[0] for r in c.execute(text(
            "EXPLAIN SELECT expense_id FROM expenses@idx_expenses_user_embedding "
            f"WHERE user_id = '{member}' ORDER BY embedding <=> '{vector_text([0.01] * 384)}' LIMIT 5")))
    assert "vector search" in plan


def test_explain_semantic_shows_the_vector_index(member):
    from banko_ai.services.transactions import explain_semantic
    plan = explain_semantic(member, "pizza")
    assert "idx_expenses_user_embedding" in plan


def test_duplicate_check_skips_purchases_beyond_the_window(ctx, member):
    acct = next(a for a in list_accounts(member)["accounts"] if a["kind"] == "credit")
    vec = [0.02] * 384
    old = ledger.post_purchase(member, acct["account_id"], amount="19.50", merchant="Old Cafe",
                               category="Coffee", embedding=vec,
                               expense_date=date.today() - timedelta(days=30))
    recent = ledger.post_purchase(member, acct["account_id"], amount="19.50", merchant="Old Cafe",
                                  category="Coffee", embedding=vec)
    out = T.run_tool("duplicate_check", {"expense_id": recent}, ctx)["result"]
    assert old not in [d["expense_id"] for d in out["candidates"]]


def test_unresolved_merchant_raises_tool_error(ctx):
    out = T.run_tool("spend_summary", {"merchant": "NoSuchMerchant"}, ctx)
    assert not out["ok"] and "No merchant called 'NoSuchMerchant'" in out["error"]


def test_unresolved_category_raises_tool_error(ctx):
    out = T.run_tool("spend_summary", {"category": "NoSuchCategory"}, ctx)
    assert not out["ok"] and "No category called 'NoSuchCategory'" in out["error"]


def test_library_value_error_becomes_failed(ctx, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("secret library detail")
    monkeypatch.setattr("banko_ai.services.memories.recall", broken)
    out = T.run_tool("recall", {"query": "anything"}, ctx)
    assert not out["ok"] and out["error"] == "failed"
    assert "secret" not in str(out)
