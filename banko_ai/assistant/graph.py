"""The assistant: one LangGraph StateGraph per turn, one checkpointed thread
per conversation.

    START -> ingest -> route -> aggregate | receipt | memory | offtopic | plan
          -> respond -> learn -> END

Nodes push events (steps, tokens, citations, proposals) through LangGraph's
stream writer; run_turn hands them to the caller, which turns them into
Server-Sent Events. The model is reached only through the Chat protocol."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from ..ai_providers.streaming import Chat, ProviderChat
from ..config.settings import get_config
from ..db.engine import get_engine
from ..db.regions import money_region
from ..services import caches, ledger, memories, reports, threads
from ..services import transactions as txns
from ..utils.request_trace import serving_regions, trace_step
from . import answers, narration
from .planner import first_object, plan_and_act
from .prompts import answer_messages, memory_extraction_messages
from .router import reads_as_follow_up
from .router import route as decide
from .tools import ToolContext, run_tool

log = logging.getLogger("banko.assistant")
_IMPLICIT = re.compile(r"\b(i'?m saving for|i am saving for|my goal is|i prefer|i usually|i always|i never)\b", re.I)


def _window(old: list[dict] | None, new: list[dict] | None) -> list[dict]:
    keep = max(2, get_config().assistant_history_window)
    return ((old or []) + (new or []))[-keep:]


class TurnState(TypedDict, total=False):
    user_id: str
    thread_id: str
    message: str
    language: str
    context: dict | None
    messages: Annotated[list[dict], _window]
    active_receipt: dict | None
    memories: list[dict]
    context_facts: str
    route: dict
    results: list[dict]
    answer: str
    cached: bool
    meta: dict
    proposals: list[dict]
    citations: list[dict]


@dataclass
class TurnDeps:
    chat: Chat
    url: str | None = None


def _emit(event: dict) -> None:
    try:
        get_stream_writer()(event)
    except Exception:  # outside a graph run (unit calls): nothing listens
        pass


def _step(label: str, detail: str = "", ms: float | None = None, status: str = "info") -> None:
    trace_step("assistant", label, detail, ms=ms, status=status)
    _emit({"type": "step", "label": label, "detail": detail, "ms": None if ms is None else round(ms, 1),
           "status": status})


def _ctx(state: TurnState, deps: TurnDeps) -> ToolContext:
    return ToolContext(user_id=state["user_id"], thread_id=state.get("thread_id"), url=deps.url,
                       active_receipt=state.get("active_receipt"))


# ── nodes ────────────────────────────────────────────────────────────────

def ingest(state: TurnState, runtime: Runtime[TurnDeps]) -> dict:
    deps = runtime.context
    user_id, message = state["user_id"], state["message"]
    recalled: list[dict] = []
    t0 = time.perf_counter()
    try:
        recalled = memories.recall(user_id, message, url=deps.url)
    except Exception:
        log.warning("memory recall skipped", exc_info=True)
    if recalled:
        _step("Memory recall", f"{len(recalled)} memories", ms=(time.perf_counter() - t0) * 1000, status="success")
    facts = ""
    ctx = state.get("context") or {}
    if ctx.get("kind") == "nudge":
        nudge = reports.nudge_detail(user_id, str(ctx.get("id")), url=deps.url)
        facts = f"Coach nudge from {nudge['created_at']}: {nudge['message']}" if nudge else ""
    elif ctx.get("kind") == "transaction":
        detail = txns.transaction_detail(user_id, str(ctx.get("id")), url=deps.url)
        facts = f"Transaction: {json.dumps(detail, default=str)}" if detail else ""
    return {"memories": recalled, "context_facts": facts, "results": [], "answer": "", "cached": False,
            "meta": {}, "proposals": [], "citations": [], "route": {}}


def route_node(state: TurnState, runtime: Runtime[TurnDeps]) -> dict:
    t0 = time.perf_counter()
    r = decide(state["message"], active_receipt=state.get("active_receipt"), context=state.get("context"))
    _step("Intent routing", f"{r.kind}: {r.detail}", ms=(time.perf_counter() - t0) * 1000)
    return {"route": r.as_dict()}


def aggregate(state: TurnState, runtime: Runtime[TurnDeps]) -> dict:
    from ..utils.aggregations import explain_aggregation, resolve_category, resolve_merchant, run_aggregation
    from ..utils.intent_classifier import AggregationIntent
    deps, user_id = runtime.context, state["user_id"]
    url = deps.url or get_config().database_url
    i = state["route"]["intent"]
    intent = AggregationIntent(i["operation"], i["subject"], date.fromisoformat(i["window_start"]),
                               date.fromisoformat(i["window_end"]))
    merchant = None
    if intent.subject and resolve_category(intent.subject, url) is None:
        merchant = resolve_merchant(intent.subject, user_id, url)
    with get_engine(url).connect() as conn:
        region = money_region(conn, user_id, url)
    narration.aggregation(intent, region)
    t0 = time.perf_counter()
    agg = run_aggregation(intent, user_id, url, region=region, merchant=merchant)
    db_ms = (time.perf_counter() - t0) * 1000
    narration.aggregation_sql(agg, db_ms)
    _step("Aggregation SQL", f"${agg.total:,.2f} across {agg.count} transactions "
          f"({agg.merchant or agg.category or 'all categories'}), computed by CockroachDB", ms=db_ms, status="success")
    explain = explain_aggregation(intent, user_id, url, region=region, merchant=merchant)
    served = serving_regions(explain) or region
    card = answers.render_aggregation_markdown(agg)
    language = state.get("language") or "English"
    key = caches.search_key(user_id, ledger.data_version(user_id, deps.url),
                            {"agg": [agg.operation, agg.category, agg.merchant, str(agg.window_start),
                                     str(agg.window_end), agg.total, agg.count], "lang": language})
    hit = None
    try:
        hit = caches.get_search(user_id, key, url=deps.url, stat="narrative")
    except Exception:  # a cache outage is a miss, never a broken answer
        log.warning("narrative cache read skipped", exc_info=True)
    if hit:
        text, cached = hit[0]["answer"], True
        narration.narrative_cache(True, caches.estimate_tokens(text))
        _step("Narrative cache", "hit: SQL reran, the prose was reused", status="hit")
    else:
        narration.narrative_cache(False)
        insights = answers.aggregation_insights(agg, state["message"], deps.chat) if agg.count else None
        if insights:
            narration.insights()
        english = card + (f"\n\n{insights}" if insights else "")
        text, cached = answers.translate(english, language, deps.chat), False
        if text != english:
            narration.rendered(language)
        try:
            caches.put_search(user_id, key, [{"answer": text}], url=deps.url, stat="narrative")
        except Exception:
            log.warning("narrative cache write skipped", exc_info=True)
    return {"answer": text, "cached": cached,
            "meta": {"region": served, "db_ms": int(db_ms), "explain": explain}}


def receipt(state: TurnState, runtime: Runtime[TurnDeps]) -> dict:
    result = run_tool("receipt_details", {}, _ctx(state, runtime.context))
    _step("Receipt details", "the active receipt" if result["ok"] and result.get("result") else "not found",
          ms=result["ms"], status="success" if result["ok"] else "warning")
    return {"results": [result]}


def memory(state: TurnState, runtime: Runtime[TurnDeps]) -> dict:
    deps, r = runtime.context, state["route"]
    language = state.get("language") or "English"
    if r["kind"] == "remember":
        saved = memories.remember(state["user_id"], r["payload"], source_thread_id=state.get("thread_id"),
                                  url=deps.url)
        _step("Memory saved", saved["content"], status="success")
        text = f"Got it. I'll remember that {answers.second_person(saved['content'])}."
    else:
        removed = memories.forget_matching(state["user_id"], r["payload"], url=deps.url)
        _step("Memory removed" if removed else "Nothing to forget", ", ".join(removed))
        text = (f"Done, I forgot that {answers.second_person(removed[0])}." if removed
                else "I didn't have anything like that saved.")
    return {"answer": answers.translate(text, language, deps.chat)}


def offtopic(state: TurnState, runtime: Runtime[TurnDeps]) -> dict:
    return {"answer": answers.translate(answers.redirect_message(), state.get("language") or "English",
                                        runtime.context.chat)}


def _cacheable(state: TurnState) -> bool:
    """Whether the answer cache may serve or keep this turn's answer. A
    follow-up ("what about restaurants?") means something only after the
    turn before it, so its text alone must never key an answer."""
    if state.get("active_receipt") or state.get("context"):
        return False
    return not (state.get("messages") and reads_as_follow_up(state["message"]))


def plan(state: TurnState, runtime: Runtime[TurnDeps]) -> dict:
    deps, user_id = runtime.context, state["user_id"]
    language = state.get("language") or "English"
    if _cacheable(state):
        hit = None
        try:
            from ..services._vector import embed
            hit = caches.get_answer(user_id, version=ledger.data_version(user_id, deps.url), language=language,
                                    query=state["message"], qvec=embed(state["message"]), url=deps.url)
        except Exception:  # a cache outage is a miss, never a broken answer
            log.warning("answer cache read skipped", exc_info=True)
        if hit:
            narration.answer_cache(True, hit["tokens"])
            _step("Answer cache", f"hit, about {hit['tokens']} tokens saved", status="hit")
            return {"answer": hit["answer"], "cached": True}
        narration.answer_cache(False)
        _step("Answer cache", "miss", status="miss")

    def on_step(result: dict) -> None:
        _step(f"Tool {result['tool']}", "ok" if result["ok"] else result.get("error", "failed"),
              ms=result["ms"], status="success" if result["ok"] else "warning")

    results = plan_and_act(deps.chat, state["message"], _ctx(state, deps),
                           history=state.get("messages") or [], memories=state.get("memories") or [],
                           context_facts=state.get("context_facts") or "", on_step=on_step)
    # Kept in event shape, so the streamed proposal and the returned one match.
    proposals = [{"type": "proposal", **r["result"]} for r in results
                 if r["ok"] and r["tool"].startswith("propose_")]
    return {"results": results, "proposals": proposals, "meta": _search_meta(state, deps, results)}


def _search_meta(state: TurnState, deps: TurnDeps, results: list[dict]) -> dict:
    """The region badge and EXPLAIN details for the chat page, from the
    turn's semantic search when there was one."""
    db_ms = int(sum(r.get("ms") or 0 for r in results))
    query = next(((r.get("args") or {}).get("query") for r in results
                  if r["tool"] == "search_transactions" and r["ok"]), None)
    url = deps.url or get_config().database_url
    with get_engine(url).connect() as conn:
        home = money_region(conn, state["user_id"], url)
    if not query:
        return {"region": home, "db_ms": db_ms}
    try:
        t0 = time.perf_counter()
        explain = txns.explain_semantic(state["user_id"], query, url=deps.url)
        _step("EXPLAIN ANALYZE", "plan captured", ms=(time.perf_counter() - t0) * 1000)
    except Exception:
        log.warning("semantic EXPLAIN skipped", exc_info=True)
        return {"region": home, "db_ms": db_ms}
    return {"region": serving_regions(explain) or home, "db_ms": db_ms, "explain": explain}


PROPOSAL_STATUS = "waiting for you to confirm"
CONFIRM_LINE = "Waiting for your confirmation: {summary}."


def _with_cites(value: Any) -> Any:
    """A tool result as the model reads it: every purchase carries a ready
    made "cite" marker to copy, so the model never has to build one."""
    if isinstance(value, dict):
        out = {k: _with_cites(v) for k, v in value.items()}
        if value.get("expense_id"):
            out["cite"] = f"[txn:{value['expense_id']}]"
        return out
    if isinstance(value, list):
        return [_with_cites(v) for v in value]
    return value


def _facts(results: list[dict]) -> str:
    useful = []
    for r in results:
        if not r.get("ok"):
            continue
        result = r.get("result")
        if r["tool"].startswith("propose_") and isinstance(result, dict):
            # No action id: the model cited it as a purchase. And not done yet.
            result = {"proposal": result.get("summary"), "status": PROPOSAL_STATUS}
        else:
            result = _with_cites(result)
        useful.append({"tool": r["tool"], "result": result})
    return json.dumps(useful, default=str)[:8000] if useful else "[]"


def _positive(amount: Any) -> Any:
    return abs(amount) if isinstance(amount, (int, float)) else amount


def _cited(answer: str, results: list[dict]) -> list[dict]:
    rows: dict[str, dict] = {}

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("expense_id"):
                rows.setdefault(str(value["expense_id"]).lower(), value)
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk([r.get("result") for r in results])
    return [{"expense_id": rows[i]["expense_id"], "merchant": rows[i].get("merchant"),
             "amount": _positive(rows[i].get("amount")), "date": rows[i].get("date")}
            for i in answers.citation_ids(answer) if i in rows]


def _cited_from_ledger(user_id: str, answer: str, deps: TurnDeps) -> list[dict]:
    """Citations for an answer that arrived without tool results (a cache
    hit): each marked id is looked up again, scoped to this user."""
    out = []
    for i in answers.citation_ids(answer):
        try:
            d = txns.transaction_detail(user_id, i, url=deps.url)
        except Exception:
            log.warning("citation lookup skipped", exc_info=True)
            continue
        if d and d.get("expense_id"):
            out.append({"expense_id": d["expense_id"], "merchant": d.get("merchant"),
                        "amount": _positive(d.get("amount")), "date": d.get("posted_at")})
    return out


def _confirmation(proposals: list[dict], language: str, chat: Chat) -> str:
    """The fixed line a proposal turn ends with, whatever the model said:
    nothing changes until the user clicks Confirm."""
    lines = "\n".join(CONFIRM_LINE.format(summary=str(p.get("summary") or "").rstrip(".")) for p in proposals)
    return answers.translate(lines, language, chat)


def respond(state: TurnState, runtime: Runtime[TurnDeps]) -> dict:
    deps = runtime.context
    language = state.get("language") or "English"
    answer = state.get("answer") or ""
    generated = False
    if answer:
        for i in range(0, len(answer), 48):
            _emit({"type": "token", "text": answer[i:i + 48]})
    else:
        msgs = answer_messages(state["message"], language=language,
                               history=state.get("messages") or [], memories=state.get("memories") or [],
                               facts=_facts(state.get("results") or []),
                               context_facts=state.get("context_facts") or "")
        narration.generating(language)
        t0 = time.perf_counter()
        parts = []
        for piece in deps.chat.stream(msgs, temperature=0.3):
            parts.append(piece)
            _emit({"type": "token", "text": piece})
        answer = "".join(parts).strip()
        generated = True
        _step("LLM answer", "streamed", ms=(time.perf_counter() - t0) * 1000, status="success")
    results = state.get("results") or []
    citations = _cited(answer, results) if results else _cited_from_ledger(state["user_id"], answer, deps)
    # From here on the answer holds canonical markers for real purchases only.
    answer = answers.resolve_citations(answer, {c["expense_id"] for c in citations})
    proposals = state.get("proposals") or []
    if proposals:
        line = _confirmation(proposals, language, deps.chat)
        _emit({"type": "token", "text": f"\n\n{line}" if answer else line})
        answer = f"{answer}\n\n{line}" if answer else line
    # Only answers whose whole value is in the text are cached: a proposal
    # card or an empty reply would replay wrong on a hit.
    elif generated and state["route"].get("kind") == "plan" and _cacheable(state) and answer:
        try:
            from ..services._vector import embed
            caches.put_answer(state["user_id"], version=ledger.data_version(state["user_id"], deps.url),
                              language=language, query=state["message"], qvec=embed(state["message"]),
                              answer=answer, tokens=caches.estimate_tokens(answer), url=deps.url)
        except Exception:
            log.warning("answer cache write skipped", exc_info=True)
    for c in citations:
        _emit({"type": "citation", **c})
    for p in proposals:
        _emit(p)
    return {"answer": answer, "citations": citations,
            "messages": [{"role": "user", "content": state["message"]},
                         {"role": "assistant", "content": answers.strip_citations(answer)}]}


def learn(state: TurnState, runtime: Runtime[TurnDeps]) -> dict:
    deps, user_id = runtime.context, state["user_id"]
    first_turn = len(state.get("messages") or []) <= 2
    try:
        threads.touch_thread(user_id, state["thread_id"], added=2,
                             title=threads.title_from(state["message"]) if first_turn else None, url=deps.url)
    except Exception:
        log.warning("thread bookkeeping skipped", exc_info=True)
    if (get_config().assistant_implicit_memory and state["route"].get("kind") not in ("remember", "forget")
            and _IMPLICIT.search(state["message"])):
        try:
            raw = deps.chat.complete(memory_extraction_messages(state["message"]), temperature=0)
            obj = first_object(raw) or {}
            if obj.get("content") and obj.get("kind") in ("fact", "preference", "goal"):
                saved = memories.remember(user_id, obj["content"], kind=obj["kind"],
                                          source_thread_id=state["thread_id"], url=deps.url)
                if saved["created"]:
                    _step("Memory saved", saved["content"], status="success")
        except Exception:
            log.warning("implicit memory skipped", exc_info=True)
    return {}


def _branch(state: TurnState) -> str:
    kind = state["route"]["kind"]
    return {"remember": "memory", "forget": "memory"}.get(kind, kind)


def build_graph(checkpointer):
    g = StateGraph(TurnState, context_schema=TurnDeps)
    for name, fn in (("ingest", ingest), ("route", route_node), ("aggregate", aggregate), ("receipt", receipt),
                     ("memory", memory), ("offtopic", offtopic), ("plan", plan), ("respond", respond),
                     ("learn", learn)):
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "route")
    g.add_conditional_edges("route", _branch, ["aggregate", "receipt", "memory", "offtopic", "plan"])
    for name in ("aggregate", "receipt", "memory", "offtopic", "plan"):
        g.add_edge(name, "respond")
    g.add_edge("respond", "learn")
    g.add_edge("learn", END)
    return g.compile(checkpointer=checkpointer)


_graph = None
_graph_lock = threading.Lock()


def get_graph():
    global _graph
    with _graph_lock:
        if _graph is None:
            from .checkpoint import get_saver
            _graph = build_graph(get_saver())
        return _graph


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def run_turn(user_id: str, thread_id: str, message: str, *, language_code: str | None = None,
             context: dict | None = None, chat=None, url: str | None = None, on_event=None, graph=None) -> dict:
    graph = graph or get_graph()
    deps = TurnDeps(chat=chat or ProviderChat(), url=url)
    inputs: dict[str, Any] = {"user_id": user_id, "thread_id": thread_id, "message": message[:4000],
                              "language": answers.language_name(language_code)}
    if context is not None:  # otherwise the thread keeps the context it was opened with
        inputs["context"] = context
    final: dict = {}
    for mode, payload in graph.stream(inputs, _config(thread_id), context=deps, stream_mode=["custom", "values"]):
        if mode == "custom":
            if on_event:
                on_event(payload)
        else:
            final = payload
    return {"answer": final.get("answer", ""), "citations": final.get("citations", []),
            "proposals": final.get("proposals", []), "meta": final.get("meta", {}),
            "cached": bool(final.get("cached")), "route": final.get("route", {})}


def set_active_receipt(thread_id: str, receipt_info: dict, *, graph=None) -> None:
    (graph or get_graph()).update_state(_config(thread_id), {"active_receipt": receipt_info}, as_node="learn")


def set_context(thread_id: str, context: dict, *, graph=None) -> None:
    """Open a thread from a nudge or a transaction; every turn of it sees that context."""
    (graph or get_graph()).update_state(_config(thread_id), {"context": context}, as_node="learn")


def thread_state(thread_id: str, *, graph=None) -> dict:
    state = (graph or get_graph()).get_state(_config(thread_id))
    return dict(state.values or {})


def thread_messages(thread_id: str, *, graph=None) -> list[dict]:
    state = (graph or get_graph()).get_state(_config(thread_id))
    return list((state.values or {}).get("messages") or [])
