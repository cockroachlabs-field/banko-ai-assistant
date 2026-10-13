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


def test_planner_prompt_lists_tools_memories_and_context():
    from banko_ai.assistant.prompts import planner_messages
    msgs = planner_messages("q", history=[{"role": "user", "content": "earlier"}],
                            memories=[{"content": "saving for Japan"}], context_facts="Nudge: coffee is up 30%",
                            tools="- balances(): accounts")
    text = json.dumps(msgs)
    assert "balances()" in text and "saving for Japan" in text and "coffee is up 30%" in text
    assert msgs[-1]["role"] == "user"


def test_answer_prompt_sets_language_and_citation_format():
    from banko_ai.assistant.prompts import answer_messages
    msgs = answer_messages("q", language="French", history=[], memories=[], facts="[]", context_facts="")
    system = msgs[0]["content"]
    assert "French" in system and "[txn:" in system and "Banko" in system
