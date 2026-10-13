"""The planner: the model proposes tool steps as JSON; code validates and
runs them. A plan the model gets wrong costs one search, never an error."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from pydantic import BaseModel, Field, ValidationError

from ..config.settings import get_config
from .prompts import planner_messages
from .tools import ToolContext, describe_tools, run_tool

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


class Step(BaseModel):
    tool: str
    args: dict = Field(default_factory=dict)


class Plan(BaseModel):
    steps: list[Step] = Field(default_factory=list)


def first_object(raw: str) -> dict | None:
    fenced = _FENCE.search(raw)
    candidates = [fenced.group(1)] if fenced else []
    candidates.append(raw)
    decoder = json.JSONDecoder()
    for text in candidates:
        start = text.find("{")
        while start != -1:
            try:
                obj, _ = decoder.raw_decode(text[start:])
                if isinstance(obj, dict):
                    return obj
            except ValueError:
                pass
            start = text.find("{", start + 1)
    return None


def parse_plan(raw: str) -> Plan:
    obj = first_object(raw or "")
    if not obj or not isinstance(obj.get("steps"), list):
        return Plan()
    steps = []
    for item in obj["steps"]:
        if not isinstance(item, dict):
            continue
        try:
            steps.append(Step.model_validate({"tool": item.get("tool"), "args": item.get("args") or {}}))
        except ValidationError:
            continue
    return Plan(steps=steps)


def execute(steps: list[Step], ctx: ToolContext, *, budget: int,
            on_step: Callable[[dict], None] | None = None) -> list[dict]:
    results = []
    for step in steps[:max(0, budget)]:
        result = {**run_tool(step.tool, step.args, ctx), "args": step.args}
        results.append(result)
        if on_step:
            on_step(result)
    return results


def plan_and_act(chat, message: str, ctx: ToolContext, *, history: list[dict], memories: list[dict],
                 context_facts: str = "", max_steps: int | None = None,
                 on_step: Callable[[dict], None] | None = None) -> list[dict]:
    budget = max_steps or get_config().assistant_max_steps
    tools = describe_tools()

    def ask(results: list[dict] | None = None) -> Plan:
        return parse_plan(chat.complete(planner_messages(message, history=history, memories=memories,
                                                         context_facts=context_facts, tools=tools,
                                                         results=results, max_steps=budget), temperature=0))

    # Nothing usable, or nothing that worked: ground the answer in a semantic
    # search of the question instead of letting the model guess.
    grounding = Step(tool="search_transactions", args={"query": message[:300]})
    plan = ask()
    results = execute(plan.steps or [grounding], ctx, budget=budget, on_step=on_step)
    if plan.steps and not any(r["ok"] for r in results):
        results += execute([grounding], ctx, budget=1, on_step=on_step)
    remaining = budget - len(results)
    if remaining > 0:
        more = ask(results)
        results += execute(more.steps, ctx, budget=remaining, on_step=on_step)
    return results
