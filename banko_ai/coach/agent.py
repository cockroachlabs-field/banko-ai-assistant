"""CoachAgent — planner-executor agent with two modes.

Reactive mode (this task): entry receives a Signal; planner emits a JSON
plan; executor calls tools; synthesizer drafts a nudge.

Conversational mode (Task 6): entry receives user message + thread history;
planner decomposes into tool calls; executor iterates with a hard cap;
reply is returned (and persisted to checkpointer in Task 6).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from .signals import Signal
from .tools import COACH_TOOLS

log = logging.getLogger("banko.coach.agent")


_PLANNER_SYSTEM_PROMPT = """You are the PLANNER for Banko's Spending Coach.
Given a streaming spending signal, decide which tools to call to gather
enough EVIDENCE to draft a useful, non-judgmental nudge. A great nudge
cites concrete numbers — totals, percentages, merchants — not just the
raw signal payload.

Available tools:
  - get_user_budget(category)
      -> {monthly_budget, source}
  - get_recent_transactions(category, limit, days)
      -> [{description, amount, merchant, ...}]
  - get_recent_signals(limit)
      -> [{signal_type, payload, ...}]
  - get_monthly_summary(year, month, top_merchants_k)
      -> {total, by_category, top_merchants, transaction_count}
  - get_spending_velocity(category, monthly_budget)
      -> {spent_so_far, projected_eom, pct_of_budget, on_track}
  - get_top_merchants(days, k, category)
      -> [{merchant, total, transaction_count}]
  - detect_subscriptions(lookback_days, min_occurrences)
      -> [{merchant, typical_amount, latest_amount, amount_drift, ...}]

Respond with JSON only. Schema:
  {"steps": [{"tool": "<name>", "args": {<kwargs>}}, ...]}

Rules:
- 2-3 steps is the sweet spot. Never more than 5.
- For budget_threshold signals: ALWAYS call get_user_budget(category)
  AND get_spending_velocity(category, monthly_budget) — they give the
  synthesizer pace + projection. Optionally add get_top_merchants(
  days=14, k=3, category) to identify what is driving the overshoot.
- For anomaly signals: call get_recent_transactions with the merchant's
  category (limit=10, days=30) to compare against the user's pattern;
  optionally get_top_merchants(days=30, k=5) for context.
- For recurring_drift signals: call detect_subscriptions(lookback_days=
  120, min_occurrences=3) — the synthesizer can then quote the price
  change ("Netflix went from $9.99 to $15.99 this cycle").
- Empty plan is allowed only if the payload alone is genuinely enough.
- monthly_budget for get_spending_velocity should match what
  get_user_budget returned; if you're unsure, omit get_spending_velocity
  and let the synthesizer work with just the budget.

Output JSON only. No prose."""


_SYNTH_SYSTEM_PROMPT = """You are the SYNTHESIZER for Banko's Spending Coach.
Draft one nudge: 1-3 sentences, supportive tone, no emojis, no exclamation
marks. ALWAYS cite the concrete numbers in tool_results when present —
percentages, dollar amounts, projected end-of-month totals, merchant
names. Never lecture the user. End with one specific, low-friction
suggestion (not a question). Output the nudge text only, no JSON, no
preamble."""


_CONVERSE_PLANNER_PROMPT = """You are the PLANNER for Banko's Spending Coach
in conversational mode. The user is following up on a nudge or asking a
direct finance question. Decompose into tool calls.

Available tools:
  - get_user_budget(category)
  - set_budget(category, amount)
  - get_recent_transactions(category, limit, days)
  - get_recent_signals(limit)
  - explain_nudge(nudge_id)
  - get_monthly_summary(year, month, top_merchants_k)
  - get_spending_velocity(category, monthly_budget)
  - get_top_merchants(days, k, category)
  - detect_subscriptions(lookback_days, min_occurrences)

Respond with JSON only:
  {"steps": [{"tool": "<name>", "args": {<kwargs>}}, ...]}

Rules:
- 0-3 steps. Empty steps means "answer from the conversation alone."
- "How am I doing on X?" -> get_user_budget + get_spending_velocity.
- "Why am I overspending on X?" -> get_spending_velocity +
  get_top_merchants(category=X).
- "Show me my subscriptions" -> detect_subscriptions.
- "Explain this nudge" + context.nudge_id -> explain_nudge(nudge_id).
- "Set my X budget to $Y" -> set_budget(category=X, amount=Y).
- "Show me my dining last 2 weeks" -> get_recent_transactions.

Output JSON only."""


_CONVERSE_SYNTH_PROMPT = """You are the SYNTHESIZER for Banko's Spending
Coach in conversational mode. Reply naturally in 1-3 sentences using the
tool results. Use concrete numbers from the tool results when relevant.
No emojis, no exclamation marks. Output the reply text only."""


@dataclass
class CoachAgent:
    database_url: str
    llm_invoker: Callable[..., Any]
    provider_name: str = "unknown"
    tool_overrides: dict[str, Callable[..., Any]] = field(default_factory=dict)
    max_steps: int = 5
    checkpointer: Any | None = None

    def react(self, signal: Signal) -> dict[str, Any]:
        """Reactive mode: signal in, nudge out."""
        plan = self._plan_for_signal(signal)
        tool_trace = self._execute_plan(plan, signal.user_id)
        message = self._synthesize_nudge(signal, tool_trace)
        return {
            "message": message,
            "tool_trace": tool_trace,
            "provider_used": self.provider_name,
        }

    def _plan_for_signal(self, signal: Signal) -> list[dict[str, Any]]:
        user_msg = (
            f"Signal type: {signal.signal_type.value}\n"
            f"Severity: {signal.severity}\n"
            f"Payload: {json.dumps(signal.payload)}\n"
            f"User id: {signal.user_id}\n"
        )
        messages = [
            SystemMessage(content=_PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
        raw = self._invoke_llm(messages)
        return self._parse_plan(raw)

    def _parse_plan(self, raw: Any) -> list[dict[str, Any]]:
        text = raw.content if hasattr(raw, "content") else str(raw)
        try:
            parsed = json.loads(text.strip())
            steps = parsed.get("steps", []) or []
            return [s for s in steps if isinstance(s, dict)
                    and "tool" in s]
        except (json.JSONDecodeError, AttributeError) as e:
            log.warning("planner returned invalid JSON, falling back to "
                        "empty plan", extra={"error": str(e)})
            return []

    def _execute_plan(self, steps: list[dict[str, Any]],
                      user_id: str) -> list[dict[str, Any]]:
        trace: list[dict[str, Any]] = []
        for step in steps[: self.max_steps]:
            tool_name = step.get("tool")
            args = step.get("args") or {}
            fn = self.tool_overrides.get(tool_name) or COACH_TOOLS.get(tool_name)
            if fn is None:
                trace.append({"tool": tool_name, "error": "unknown tool"})
                continue
            try:
                if tool_name in self.tool_overrides:
                    result = fn(user_id=user_id, **args)
                else:
                    result = fn(user_id=user_id, database_url=self.database_url,
                                **args)
                trace.append({"tool": tool_name, "args": args, "result": result})
            except Exception as e:
                log.exception("tool failed", extra={"tool": tool_name})
                trace.append({"tool": tool_name, "args": args,
                              "error": str(e)})
        return trace

    def _synthesize_nudge(self, signal: Signal,
                          tool_trace: list[dict[str, Any]]) -> str:
        context = {
            "signal_type": signal.signal_type.value,
            "severity": signal.severity,
            "payload": signal.payload,
            "tool_results": [{"tool": t.get("tool"),
                              "result": t.get("result")}
                             for t in tool_trace if "result" in t],
        }
        messages = [
            SystemMessage(content=_SYNTH_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps(context)),
        ]
        raw = self._invoke_llm(messages)
        text = raw.content if hasattr(raw, "content") else str(raw)
        return text.strip()

    def converse(self, user_id: str, message: str,
                 history: list[dict[str, str]] | None = None,
                 context: dict[str, Any] | None = None,
                 thread_id: str | None = None) -> dict[str, Any]:
        """Conversational mode: user message in, reply text out.

        `history` is a list of {role, content} dicts (oldest first).
        `context` lets the caller inject structured hints (e.g. a
        nudge_id the user is following up on)."""
        history = history or []
        history_text = "\n".join(
            f"{h.get('role', 'user')}: {h.get('content', '')}"
            for h in history
        )
        user_block = (
            (f"Conversation so far:\n{history_text}\n\n" if history_text else "")
            + (f"Context: {json.dumps(context)}\n\n" if context else "")
            + f"User just said: {message}"
        )
        planner_msgs = [
            SystemMessage(content=_CONVERSE_PLANNER_PROMPT),
            HumanMessage(content=user_block),
        ]
        raw = self._invoke_llm(planner_msgs)
        steps = self._parse_plan(raw)
        tool_trace = self._execute_plan(steps, user_id)

        synth_context: dict[str, Any] = {
            "user_message": message,
            "history": history,
            "tool_results": [{"tool": t.get("tool"), "result": t.get("result")}
                             for t in tool_trace if "result" in t],
        }
        if context:
            synth_context["context"] = context

        synth_msgs = [
            SystemMessage(content=_CONVERSE_SYNTH_PROMPT),
            HumanMessage(content=json.dumps(synth_context)),
        ]
        raw = self._invoke_llm(synth_msgs)
        text = raw.content if hasattr(raw, "content") else str(raw)

        if self.checkpointer is not None and thread_id is not None:
            try:
                self.checkpointer.put(
                    config={"configurable": {"thread_id": thread_id}},
                    checkpoint={"v": 1, "ts": text[:200],
                                "channel_values": {"messages": history + [
                                    {"role": "user", "content": message},
                                    {"role": "assistant", "content": text}
                                ]}},
                    metadata={"source": "coach", "user_id": user_id},
                    new_versions={},
                )
            except Exception as e:
                log.warning("checkpoint write failed",
                            extra={"thread_id": thread_id, "error": str(e)})

        return {
            "message": text.strip(),
            "tool_trace": tool_trace,
            "provider_used": self.provider_name,
        }

    def _invoke_llm(self, messages: list) -> Any:
        return self.llm_invoker(messages)


def default_llm_invoker(messages: list, temperature: float = 0.3) -> Any:
    """Default invoker: build a LangChain LLM via the existing factory and
    call it. Kept module-level so tests can import it without instantiating
    the agent."""
    from banko_ai.agents.llm_factory import get_llm_for_agent
    llm = get_llm_for_agent(temperature=temperature)
    return llm.invoke(messages)


def build_checkpointer(database_url: str):
    """Return a CockroachDBSaver bound to a long-lived psycopg connection.
    Returns None when langchain-cockroachdb (or its psycopg dep) isn't
    importable so tests in environments without the extra don't crash.

    The library ships `from_conn_string` as a contextmanager; we want a
    process-lifetime saver, so we open the connection ourselves and pass
    it to the constructor, then call setup() once to create the
    checkpoint tables if they don't exist."""
    try:
        from langchain_cockroachdb import CockroachDBSaver
        from psycopg import Connection
        from psycopg.rows import dict_row
    except ImportError as e:
        log.warning("langchain-cockroachdb not installed; checkpointing "
                    "disabled: %s", e)
        return None

    from ..utils.db_retry import to_libpq_url
    pg_url = to_libpq_url(database_url)

    try:
        conn = Connection.connect(
            pg_url, autocommit=True, prepare_threshold=5, row_factory=dict_row,
        )
        saver = CockroachDBSaver(conn)
        saver.setup()
        return saver
    except Exception as e:
        log.warning("checkpointer setup failed; checkpointing disabled",
                    extra={"error": str(e)})
        return None
