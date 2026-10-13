"""Every prompt the assistant sends, in one place. The assistant's name
comes from BRAND_NAME."""

from __future__ import annotations

import json
from datetime import date

from ..config.catalog import get_catalog
from ..config.settings import get_config


def _memories(memories: list[dict]) -> str:
    return "\n".join(f"- {m['content']}" for m in memories) or "- none"


def planner_messages(message: str, *, history: list[dict], memories: list[dict], context_facts: str,
                     tools: str, results: list[dict] | None = None, max_steps: int | None = None,
                     today: date | None = None) -> list[dict]:
    cfg = get_config()
    labels = ", ".join(get_catalog().spending_labels())
    system = (
        f"You plan tool calls for {cfg.brand_name}, a banking assistant. Today is "
        f"{(today or date.today()).isoformat()}. Reply with JSON only, shaped like "
        '{"steps": [{"tool": "<name>", "args": {...}}]}. Use only these tools, and always pass every '
        "argument marked (required):\n"
        f"{tools}\n"
        f"Spending categories: {labels}. A category argument must be one of these names.\n"
        "Rules: never do arithmetic yourself, use spend_summary for totals; dates are YYYY-MM-DD; "
        f"propose_* tools only when the user asks for a change; at most {max_steps or cfg.assistant_max_steps} "
        'steps; return {"steps": []} when the results already answer the question.\n'
        f"What you know about the user:\n{_memories(memories)}")
    if context_facts:
        system += f"\nThe user opened this conversation from:\n{context_facts}"
    msgs = [{"role": "system", "content": system}, *history]
    if results is None:
        msgs.append({"role": "user", "content": message})
    else:
        msgs.append({"role": "user", "content": (
            f"Question: {message}\nResults so far:\n{json.dumps(results, default=str)[:6000]}\n"
            "Return more steps only if something is still missing.")})
    return msgs


def answer_messages(message: str, *, language: str, history: list[dict], memories: list[dict], facts: str,
                    context_facts: str) -> list[dict]:
    brand = get_config().brand_name
    # No marker pattern here: a placeholder in the prompt is what small models
    # copy literally. Each purchase in the facts carries its own "cite" value.
    system = (
        f"You are {brand}, a friendly banking assistant. Answer in {language}. Use only the facts "
        "below; if they do not answer the question, say so plainly. Never invent numbers and never "
        "do arithmetic beyond what the facts state. When you mention a specific purchase from the "
        'facts, put that purchase\'s "cite" value right after the mention, copied exactly; never '
        "write one for anything that has no cite. A proposal is not done until the user confirms "
        "it: say it is ready to confirm, never that it is done. Keep it short: a sentence or two, "
        "then bullets if they help. Markdown is fine.\n"
        f"What you know about the user:\n{_memories(memories)}")
    if context_facts:
        system += f"\nThe user opened this conversation from:\n{context_facts}"
    return [{"role": "system", "content": system}, *history,
            {"role": "user", "content": f"{message}\n\nFacts:\n{facts}"}]


def memory_extraction_messages(message: str) -> list[dict]:
    return [{"role": "system", "content": (
        "Decide whether the user's message states a durable fact about themselves worth remembering "
        "for future conversations (a savings goal, a preference, a recurring habit). Reply with JSON "
        'only: {"kind": "goal" | "preference" | "fact", "content": "<one short sentence in the third '
        'person is not needed, keep the user\'s words>"} or {} when there is nothing durable.')},
        {"role": "user", "content": message}]
