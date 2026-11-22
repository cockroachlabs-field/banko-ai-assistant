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


def test_search_returns_the_cursor_of_its_next_page(ctx):
    from banko_ai.services import transactions as txns
    first = T.run_tool("search_transactions", {"query": "dinner", "limit": 3}, ctx)["result"]
    assert len(first["items"]) == 3 and first["next_cursor"]
    nxt = txns.semantic_search(ctx.user_id, "dinner", limit=3, cursor=first["next_cursor"])["items"]
    assert nxt and {i["expense_id"] for i in nxt}.isdisjoint({i["expense_id"] for i in first["items"]})


def test_a_search_without_a_limit_takes_the_page_size_setting(ctx, monkeypatch):
    from banko_ai.config.settings import reset_config
    monkeypatch.setenv("SEARCH_PAGE_SIZE", "4")
    reset_config()
    try:
        assert len(T.run_tool("search_transactions", {"query": "groceries"}, ctx)["result"]["items"]) == 4
    finally:
        monkeypatch.delenv("SEARCH_PAGE_SIZE")
        reset_config()


def test_a_broken_search_cache_is_a_miss(ctx, monkeypatch):
    from banko_ai.services import caches

    def down(*args, **kwargs):
        raise RuntimeError("search cache down")
    monkeypatch.setattr(caches, "get_search", down)
    monkeypatch.setattr(caches, "put_search", down)
    out = T.run_tool("search_transactions", {"query": "coffee", "limit": 3}, ctx)
    assert out["ok"] and 0 < len(out["result"]["items"]) <= 3


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


def _statements(monkeypatch, pretend_region=None) -> list:
    """The WHERE and parameters of every totals statement and of every plan
    asked for: one builder writes both. With pretend_region, the statement
    runs without the region it was given (this database has one region and
    no such column) and the list keeps the region that was named."""
    from banko_ai.utils import aggregations
    seen, real = [], aggregations._where_and_params

    def spy(intent, user_id, category, region, merchant=None):
        if pretend_region:
            seen.append(region)
            return real(intent, user_id, category, None, merchant)
        seen.append(real(intent, user_id, category, region, merchant))
        return seen[-1]
    monkeypatch.setattr(aggregations, "_where_and_params", spy)
    return seen


@pytest.mark.parametrize("args", [{}, {"category": "dining"}, {"date_from": "2026-10-01", "date_to": "2026-10-31"}])
def test_the_plan_of_a_spend_summary_is_of_the_statement_it_ran(ctx, monkeypatch, args):
    """What the answer shows under "How CockroachDB ran it" has to be the
    statement the figure came from: the same window, the same category."""
    seen = _statements(monkeypatch)
    out = T.run_tool("spend_summary", args, ctx)
    assert out["ok"] and len(seen) == 1
    plan = T.spend_plan(ctx, out["result"])
    assert len(seen) == 2 and seen[0] == seen[1]
    assert "execution time" in plan and "group (scalar)" in plan and "expenses@" in plan


def test_the_plan_of_a_total_at_a_merchant_names_the_merchant(ctx, member, monkeypatch):
    seen = _statements(monkeypatch)
    with get_engine().connect() as c:
        merchant = c.execute(text("SELECT merchant FROM expenses WHERE user_id = CAST(:u AS UUID) LIMIT 1"),
                             {"u": member}).scalar()
    out = T.run_tool("spend_summary", {"merchant": merchant.lower()}, ctx)
    T.spend_plan(ctx, out["result"])
    assert len(seen) == 2 and seen[0] == seen[1] and seen[0][1]["m"] == merchant


def test_spend_summary_names_the_home_region_where_a_read_may(ctx, monkeypatch):
    """On REGIONAL BY ROW a total that names no region looks in all of them:
    about 750 ms a call on the hosted cluster. The tool names the region
    the SQL route names for the same statement, and so does its plan."""
    monkeypatch.setattr(T, "read_region", lambda user_id, url: "eu-west-1")
    named = _statements(monkeypatch, pretend_region=True)
    out = T.run_tool("spend_summary", {}, ctx)
    assert out["ok"] and named == ["eu-west-1"]
    T.spend_plan(ctx, out["result"])
    assert named == ["eu-west-1", "eu-west-1"]


def test_spend_summary_names_no_region_where_a_read_may_not(ctx, monkeypatch):
    """One region, row-level security, or rows that are not all at home."""
    monkeypatch.setattr(T, "read_region", lambda user_id, url: None)
    named = _statements(monkeypatch, pretend_region=True)
    out = T.run_tool("spend_summary", {}, ctx)
    T.spend_plan(ctx, out["result"])
    assert named == [None, None]


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


def test_spend_summary_says_which_purchases_were_the_largest(ctx, member):
    """Its rows are the most recent purchases. A model asked about the
    biggest one picked the largest of those, and the month's rent, paid on
    the first, was never among them."""
    start = date.today() - timedelta(days=45)
    out = T.run_tool("spend_summary", {"date_from": str(start)}, ctx)["result"]
    with get_engine().connect() as c:
        top = c.execute(text("SELECT expense_amount, merchant FROM expenses WHERE user_id = CAST(:u AS UUID) "
                             "AND expense_date >= :s ORDER BY expense_amount DESC, expense_date DESC, expense_id "
                             "LIMIT 3"), {"u": member, "s": start}).fetchall()
    assert [(row["amount"], row["merchant"]) for row in out["largest"]] == [(float(a), m) for a, m in top]
    assert out["largest"][0]["amount"] >= max(row["amount"] for row in out["rows"])


def test_the_planner_is_told_what_largest_is():
    """A key nobody described is a key the model has to guess at."""
    described = T.TOOLS["spend_summary"].description
    assert "largest" in described and "most recent" in described


def test_a_budget_proposal_names_a_real_spending_category(ctx, member):
    """"set my dining out budget to 300" used to store a proposal no Confirm could run."""
    from banko_ai.services import actions, budgets
    out = T.run_tool("propose_budget", {"category": "dining out", "amount": "300"}, ctx)
    assert out["ok"] and out["result"]["summary"] == "Set your Restaurant budget to $300.00 a month"
    assert actions.confirm(member, out["result"]["action_id"])["status"] == "done"
    assert any(b["category"] == "Restaurant" and b["budget"] == 300
               for b in budgets.budget_progress(member)["items"])
    for bad in ("yachts", "Income"):
        refused = T.run_tool("propose_budget", {"category": bad, "amount": "50"}, ctx)
        assert not refused["ok"] and "Groceries" in refused["error"] and "Restaurant" in refused["error"]


def test_spend_summary_understands_catalog_aliases(ctx):
    out = T.run_tool("spend_summary", {"category": "dining"}, ctx)
    assert out["ok"] and out["result"]["category"] == "Restaurant"


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


def test_the_plan_under_a_search_is_of_the_search_that_ran(member, monkeypatch):
    """The search names the home region where a read may. Its plan never
    did: under an answer that read one region, "How CockroachDB ran it"
    showed a statement that read all of them, and the answer said it was
    served from three regions."""
    from banko_ai.services import transactions as txns
    named, real = [], txns.semantic_sql

    def spy(filters="", *, pinned=False):
        named.append(pinned)
        return real(filters)                      # one region here: no such column
    monkeypatch.setattr(txns, "semantic_sql", spy)
    monkeypatch.setattr(txns, "read_region", lambda user_id, url: "eu-west-1")
    assert "idx_expenses_user_embedding" in txns.explain_semantic(member, "pizza")
    assert named == [True]
    monkeypatch.setattr(txns, "read_region", lambda user_id, url: None)
    txns.explain_semantic(member, "pizza")
    assert named == [True, False]


def _visits_named(monkeypatch) -> list:
    """The region each visits statement was given. It runs without it: this
    database has one region and no such column."""
    from banko_ai.services import transactions as txns
    named, real = [], txns._visits_where

    def spy(user_id, name, date_from, date_to, region=None):
        named.append(region)
        return real(user_id, name, date_from, date_to)
    monkeypatch.setattr(txns, "_visits_where", spy)
    return named


def test_a_merchants_visits_name_the_home_region_where_a_read_may(member, monkeypatch):
    """The visits statement named no region, so "when did I last go to
    Costco" looked in every one, and its plan said so. It names the home
    region as the other reads do, in the answer and in its plan."""
    from banko_ai.services import transactions as txns
    with get_engine().connect() as c:
        merchant = c.execute(text("SELECT merchant FROM expenses WHERE user_id = CAST(:u AS UUID) LIMIT 1"),
                             {"u": member}).scalar()
    named = _visits_named(monkeypatch)
    monkeypatch.setattr(txns, "read_region", lambda user_id, url: "eu-west-1")
    assert txns.merchant_history(member, merchant)["visits"] > 0
    assert "group (scalar)" in txns.explain_merchant_visits(member, merchant)
    assert named == ["eu-west-1", "eu-west-1"]
    monkeypatch.setattr(txns, "read_region", lambda user_id, url: None)
    txns.merchant_history(member, merchant)
    txns.explain_merchant_visits(member, merchant)
    assert named == ["eu-west-1", "eu-west-1", None, None]


def test_the_visits_statement_keeps_to_the_region_it_is_given():
    from banko_ai.services import transactions as txns
    where, params = txns._visits_where("u", "Costco", None, None, "eu-west-1")
    assert where.startswith("crdb_region = CAST(:region AS crdb_internal_region) AND user_id = ")
    assert params == {"u": "u", "m": "Costco", "region": "eu-west-1"}
    where, params = txns._visits_where("u", "Costco", date(2026, 10, 1), date(2026, 10, 31))
    assert "crdb_region" not in where and "region" not in params and {"dfrom", "dto"} <= set(params)


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


def test_spend_summary_refuses_a_backwards_window(ctx):
    out = T.run_tool("spend_summary", {"date_from": "2026-10-10", "date_to": "2026-10-01"}, ctx)
    assert not out["ok"] and "date_from" in out["error"]


def test_describe_tools_marks_required_arguments():
    described = T.describe_tools()
    assert "search_transactions(query: str (required), category: str | None" in described
    assert "propose_budget(category: str (required), amount: Decimal (required))" in described
    assert "limit: int (required)" not in described


def test_unresolved_category_raises_tool_error(ctx):
    out = T.run_tool("spend_summary", {"category": "NoSuchCategory"}, ctx)
    assert not out["ok"] and "No category called 'NoSuchCategory'" in out["error"]


def test_a_made_up_expense_id_is_not_a_lookup_that_broke(ctx):
    """Models pass placeholders such as "<expense_id>". That is the model's
    mistake: no candidates, and nothing for the answer to apologize for."""
    out = T.run_tool("duplicate_check", {"expense_id": "<expense_id>"}, ctx)
    assert out["ok"] is True and out["result"] == {"candidates": []}


def test_remembering_nothing_is_bad_arguments(ctx):
    out = T.run_tool("remember", {"content": "   "}, ctx)
    assert out["ok"] is False and out["error"] == "bad arguments"


def test_a_search_while_the_model_loads_fails_in_one_log_line(ctx, monkeypatch, caplog):
    """The search cannot run until the embedding model has loaded. That is a
    failed lookup for the answer, and a line in the log, not a traceback."""
    import logging

    from banko_ai.services import caches
    from banko_ai.utils.embeddings import ModelWarmingUp

    def loading(*args, **kwargs):
        raise ModelWarmingUp("embedding model all-MiniLM-L6-v2 is still loading")
    monkeypatch.setattr(T.txns, "semantic_search", loading)
    monkeypatch.setattr(caches, "get_search", lambda *args, **kwargs: None)
    caplog.set_level(logging.INFO, logger="banko.assistant")
    out = T.run_tool("search_transactions", {"query": "coffee while loading"}, ctx)
    assert out["ok"] is False and out["error"] == "failed"
    about = [r for r in caplog.records if "search_transactions" in r.getMessage()]
    assert [r.getMessage() for r in about] == ["tool search_transactions skipped: the embedding model is still loading"]
    assert about[0].exc_info is None


def test_library_value_error_becomes_failed(ctx, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("secret library detail")
    monkeypatch.setattr("banko_ai.services.memories.recall", broken)
    out = T.run_tool("recall", {"query": "anything"}, ctx)
    assert not out["ok"] and out["error"] == "failed"
    assert "secret" not in str(out)


def test_rejected_tool_arguments_name_the_fields_in_the_log_but_not_the_values(caplog):
    """A planner's bad call is diagnosable from the server log, and the log
    never repeats what the user typed."""
    import logging
    caplog.set_level(logging.INFO, logger="banko.assistant")
    out = T.run_tool("search_transactions", {"query": "my secret dinner", "category": ["my secret dinner"],
                                             "date_from": "lately"},
                     T.ToolContext(user_id="00000000-0000-0000-0000-000000000000"))
    assert out["ok"] is False and out["error"] == "bad arguments"
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "search_transactions" in logged and "category" in logged and "date_from" in logged
    assert "secret dinner" not in logged and "lately" not in logged


def test_an_oversized_page_is_clamped_not_rejected(ctx):
    """watsonx asked for 100 of this month's restaurant purchases; rejecting
    the call dropped its date and category filters with it."""
    from banko_ai.services.transactions import SEARCH_PAGE_MAX
    out = T.run_tool("search_transactions", {"query": "coffee", "limit": 100}, ctx)
    assert out["ok"] is True
    assert T.SearchArgs.model_validate({"query": "coffee", "limit": 100}).limit == SEARCH_PAGE_MAX
    assert T.SearchArgs.model_validate({"query": "coffee", "limit": 0}).limit == 1
    assert T.MonthsArgs.model_validate({"months": 40}).months == 12
    assert T.ExpenseArgs.model_validate({"expense_id": "x", "limit": 99}).limit == 20


def test_an_infinite_count_is_a_bad_argument_not_a_crash(ctx):
    """json.loads turns 1e999 into inf, and int(inf) raises OverflowError."""
    out = T.run_tool("search_transactions", {"query": "coffee", "limit": float("inf")}, ctx)
    assert out["ok"] is False and out["error"] == "bad arguments"


def test_a_clamped_count_is_reported_back_to_the_model(ctx):
    """The model asked for 100 and got 25; it should know, so it never
    narrates a period or a count it did not get."""
    out = T.run_tool("search_transactions", {"query": "coffee", "limit": 100}, ctx)
    assert out["ok"] is True and out["clamped"] == {"limit": 25}
    assert "clamped" not in T.run_tool("search_transactions", {"query": "coffee", "limit": 5}, ctx)
