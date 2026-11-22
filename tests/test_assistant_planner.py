"""The planner: the model proposes tool steps as JSON, code validates and
runs them, never more than the step limit."""

import json

from banko_ai.assistant import planner as P
from banko_ai.assistant.tools import ToolContext
from tests._chat import ScriptedChat


def test_parse_plan_tolerates_junk():
    assert P.parse_plan('{"steps": [{"tool": "balances", "args": {}}]}').steps[0].tool == "balances"
    fenced = 'Sure!\n```json\n{"steps": [{"tool": "budget_status"}]}\n```'
    assert [s.tool for s in P.parse_plan(fenced).steps] == ["budget_status"]
    assert P.parse_plan("I cannot help with that").steps == []
    assert P.parse_plan('{"steps": "balances"}').steps == []
    assert P.parse_plan('{"steps": [{"args": {}}, {"tool": "balances"}]}').steps[0].tool == "balances"
    assert P.parse_plan("").steps == []


def test_executor_caps_steps_and_skips_unknown_tools(monkeypatch):
    ran = []

    def fake_run(name, args, ctx):
        ran.append(name)
        ok = name != "nope"
        return {"tool": name, "ok": ok, "result": {"n": len(ran)} if ok else None,
                "error": None if ok else "unknown tool", "ms": 1.0}

    monkeypatch.setattr(P, "run_tool", fake_run)
    steps = [P.Step(tool=t) for t in ("balances", "nope", "budget_status", "goals_status", "cash_flow")]
    seen = []
    out = P.execute(steps, ToolContext(user_id="u"), budget=3, on_step=seen.append)
    assert ran == ["balances", "nope", "budget_status"]
    assert [r["ok"] for r in out] == [True, False, True]
    assert seen == out


def test_plan_and_act_loops_once_with_results(monkeypatch):
    monkeypatch.setattr(P, "run_tool", lambda name, args, ctx: {"tool": name, "ok": True, "result": {"x": 1}, "ms": 1})
    chat = ScriptedChat('{"steps": [{"tool": "balances"}]}', '{"steps": [{"tool": "goals_status"}]}',
                        '{"steps": [{"tool": "cash_flow"}]}')
    out = P.plan_and_act(chat, "am I on track?", ToolContext(user_id="u"), history=[], memories=[], max_steps=5)
    assert [r["tool"] for r in out] == ["balances", "goals_status"]
    assert len(chat.calls) == 2
    assert '"tool": "balances"' in chat.prompts()[-1]


def test_empty_plan_falls_back_to_search(monkeypatch):
    calls = []
    monkeypatch.setattr(P, "run_tool", lambda name, args, ctx: calls.append((name, args)) or
                        {"tool": name, "ok": True, "result": {"items": []}, "ms": 1})
    P.plan_and_act(ScriptedChat("no idea", '{"steps": []}'), "late night food", ToolContext(user_id="u"),
                   history=[], memories=[])
    assert calls[0] == ("search_transactions", {"query": "late night food"})


def _failing_search(calls):
    """run_tool as the real one behaves on granite's bad plan: a search
    without its required query fails validation, anything else works."""
    def fake(name, args, ctx):
        calls.append((name, args))
        ok = not (name == "search_transactions" and not (args or {}).get("query"))
        return {"tool": name, "ok": ok, "result": {"items": []} if ok else None,
                "error": None if ok else "bad arguments", "ms": 1}
    return fake


def test_a_plan_with_no_successful_step_falls_back_to_search(monkeypatch):
    calls = []
    monkeypatch.setattr(P, "run_tool", _failing_search(calls))
    bad = '{"steps": [{"tool": "search_transactions", "args": {"category": "dining out"}}]}'
    out = P.plan_and_act(ScriptedChat(bad, '{"steps": []}'), "where could I cut back on dining out?",
                         ToolContext(user_id="u"), history=[], memories=[])
    assert calls == [("search_transactions", {"category": "dining out"}),
                     ("search_transactions", {"query": "where could I cut back on dining out?"})]
    assert [r["ok"] for r in out] == [False, True]


def test_no_fallback_when_a_step_worked(monkeypatch):
    calls = []
    monkeypatch.setattr(P, "run_tool", _failing_search(calls))
    plan = ('{"steps": [{"tool": "search_transactions", "args": {"category": "dining out"}}, '
            '{"tool": "spend_summary", "args": {"category": "dining out"}}]}')
    P.plan_and_act(ScriptedChat(plan, '{"steps": []}'), "q", ToolContext(user_id="u"), history=[], memories=[])
    assert [name for name, _ in calls] == ["search_transactions", "spend_summary"]


def test_a_failed_fallback_is_not_repeated(monkeypatch):
    calls = []

    def all_fail(name, args, ctx):
        calls.append(name)
        return {"tool": name, "ok": False, "result": None, "error": "failed", "ms": 1}
    monkeypatch.setattr(P, "run_tool", all_fail)
    P.plan_and_act(ScriptedChat("no plan", '{"steps": []}'), "q", ToolContext(user_id="u"), history=[], memories=[])
    assert calls == ["search_transactions"]


def test_planner_prompt_is_grounded(monkeypatch):
    from datetime import date

    from banko_ai.assistant.prompts import planner_messages
    from banko_ai.assistant.tools import describe_tools
    from banko_ai.config.settings import reset_config
    monkeypatch.setenv("ASSISTANT_MAX_STEPS", "7")
    reset_config()
    system = planner_messages("q", history=[], memories=[], context_facts="", tools=describe_tools())[0]["content"]
    assert date.today().isoformat() in system
    assert "Groceries" in system and "Restaurant" in system and "Income" not in system
    assert "at most 7 steps" in system and "five" not in system
    assert "search_transactions(query: str (required), category: str | None" in system
    assert "spend_summary(date_from: datetime.date | None" in system


TOOL_LIST = """\
- search_transactions(query: str (required), category: str | None, date_from: datetime.date | None, date_to: datetime.date | None, limit: int | None): find purchases by meaning (e.g. 'late night food', 'pizza')
- spend_summary(date_from: datetime.date | None, date_to: datetime.date | None, category: str | None, merchant: str | None): exact totals from SQL for a window, optionally one category or merchant; rows are the most recent purchases and largest the three biggest, so name the biggest from largest
- receipt_details(document_id: str | None): the active receipt (or one by document_id): merchant, total, items, linked expense
- similar_transactions(expense_id: str (required), limit: int): purchases most like one expense
- duplicate_check(expense_id: str (required), limit: int): possible duplicates of one expense: same amount, nearby date, similar description
- merchant_history(merchant: str (required)): visits, total, average and recent purchases at one merchant
- budget_status(): this month's budgets with spent and remaining per category
- recurring_bills(): detected bills and subscriptions with cadence, next due and price changes
- goals_status(): savings goals with progress and projected completion
- balances(): accounts with balances, credit utilization and net worth
- cash_flow(months: int): income and spending by month
- recall(query: str (required)): the user's saved memories relevant to a question
- remember(content: str (required)): save a durable fact the user stated about themselves
- propose_recategorize(expense_id: str (required), category: str (required)): propose moving an expense to another category (runs only after Confirm)
- propose_budget(category: str (required), amount: Decimal (required)): propose a monthly budget for a category (runs only after Confirm)"""


def test_the_tool_list_the_planner_reads_changes_on_purpose_only():
    """Every character of this list is part of the planner's prompt. Seven
    of them once changed what gpt-4.1 does with "coffee": limit: int became
    limit: int | None when the page size became a setting, and the planner
    asked for spend_summary where it had picked nothing and the search had
    run. No test saw it, because a scripted model plans what it is told to.

    So a change to an argument, a type or a description fails here first.
    Make it on purpose: update this copy, then ask the model the hosted
    instance runs a few short questions again ("coffee", "my coffee
    expenses", "starbucks", "groceries") and look at which tools it picks."""
    from banko_ai.assistant.tools import describe_tools
    assert describe_tools() == TOOL_LIST


def test_planner_prompt_lists_tools_memories_and_context():
    from banko_ai.assistant.prompts import planner_messages
    msgs = planner_messages("q", history=[{"role": "user", "content": "earlier"}],
                            memories=[{"content": "saving for Japan"}], context_facts="Nudge: coffee is up 30%",
                            tools="- balances(): accounts")
    text = json.dumps(msgs)
    assert "balances()" in text and "saving for Japan" in text and "coffee is up 30%" in text
    assert msgs[-1]["role"] == "user"


def test_answer_prompt_sets_language_and_citation_format():
    """The model copies a ready-made cite value from the facts; a placeholder
    pattern in the prompt is what granite copied literally."""
    from banko_ai.assistant.prompts import answer_messages
    msgs = answer_messages("q", language="French", history=[], memories=[], facts="[]", context_facts="")
    system = msgs[0]["content"]
    assert "French" in system and '"cite"' in system and "Banko" in system
    assert "[txn:" not in system and "<expense_id>" not in system
    assert "not done until the user confirms" in system
