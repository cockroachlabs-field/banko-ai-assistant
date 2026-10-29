"""The assistant: one LangGraph StateGraph per turn, one checkpointed thread
per conversation.

    START -> ingest -> route -> aggregate | visits | receipt | memory | offtopic | plan | more
          -> respond -> learn -> END

A visit question about a place that is none of the user's merchants goes on
from visits to plan, which searches by meaning instead.

Nodes push events (steps, tokens, citations, proposals) through LangGraph's
stream writer; run_turn hands them to the caller, which turns them into
Server-Sent Events. respond ends with one "final" event carrying the whole
answer, so the caller can send it before learn and the checkpoint write
finish. Cache writes and the implicit memory extraction run in the
background (assistant.background). The model is reached only through the
Chat protocol."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Annotated, Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from ..ai_providers.streaming import Chat, ProviderChat
from ..config.settings import get_config
from ..db.engine import get_engine
from ..db.regions import money_region, read_region
from ..services import caches, ledger, memories, receipts, reports, threads
from ..services import transactions as txns
from ..utils.request_trace import serving_regions, trace_step
from . import answers, background, narration
from .planner import first_object, parse_plan, plan_and_act
from .prompts import answer_messages, memory_extraction_messages
from .router import reads_as_follow_up
from .router import route as decide
from .tools import SearchArgs, ToolContext, run_tool

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
    last_route: str | None  # the previous turn's route; "receipt" right after an upload
    results: list[dict]
    answer: str
    streamed: bool  # the answer already went out as tokens while it was made
    cached: bool
    saved_tokens: int  # the model's work a cache hit stood in for, estimated
    meta: dict
    proposals: list[dict]
    citations: list[dict]
    page: dict | None  # the search this turn showed: q, category, from, to and the next cursor
    last_search: dict | None  # the previous turn's page, which "show more" continues


@dataclass
class TurnDeps:
    chat: Chat
    url: str | None = None


def _emit(event: dict) -> None:
    try:
        get_stream_writer()(event)
    except Exception:  # outside a graph run (unit calls): nothing listens
        pass


def _step(label: str, detail: str = "", ms: float | None = None, status: str = "info",
          key: str | None = None) -> None:
    """A finished piece of work: a trace step, and a step event. key names
    the running phase it closes, so the bubble swaps that chip for this one."""
    trace_step("assistant", label, detail, ms=ms, status=status)
    event = {"type": "step", "label": label, "detail": detail, "ms": None if ms is None else round(ms, 1),
             "status": status}
    if key:
        event["key"] = key
    _emit(event)


def _running(key: str, label: str) -> None:
    """A long phase starting, for the bubble's status line ("Planning the
    steps..."). Only an event: the trace records what finished."""
    _emit({"type": "step", "key": key, "label": label, "detail": "", "ms": None, "status": "running"})


def _since(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000


# The tools that search the user's history get a phase of their own.
_TOOL_PHASES = {"search_transactions": "Searching your transactions",
                "similar_transactions": "Searching your transactions",
                "duplicate_check": "Searching your transactions",
                "merchant_history": "Searching your transactions",
                "spend_summary": "Running the numbers in CockroachDB"}


class _Planning:
    """The planner's model, with each call shown as a phase while it runs
    and a Planner step when it returns."""

    def __init__(self, chat: Chat):
        self.chat, self.calls = chat, 0

    def complete(self, messages: list[dict], *, temperature: float = 0.3) -> str:
        self.calls += 1
        key = f"planner:{self.calls}"
        _running(key, "Planning the steps")
        t0 = time.perf_counter()
        try:
            raw = self.chat.complete(messages, temperature=temperature)
        except Exception:
            _step("Planner", "the model did not answer", ms=_since(t0), status="warning", key=key)
            raise
        tools = [s.tool for s in parse_plan(raw).steps]
        empty = "no tools picked" if self.calls == 1 else "nothing more to look up"
        _step("Planner", ", ".join(tools) or empty, ms=_since(t0), status="success", key=key)
        return raw

    def stream(self, messages: list[dict], *, temperature: float = 0.3):
        return self.chat.stream(messages, temperature=temperature)


def _ctx(state: TurnState, deps: TurnDeps) -> ToolContext:
    return ToolContext(user_id=state["user_id"], thread_id=state.get("thread_id"), url=deps.url,
                       active_receipt=state.get("active_receipt"))


# ── nodes ────────────────────────────────────────────────────────────────

def ingest(state: TurnState, runtime: Runtime[TurnDeps]) -> dict:
    deps = runtime.context
    user_id, message = state["user_id"], state["message"]
    recalled: list[dict] = []
    _running("ingest", "Recalling what you told me")
    t0 = time.perf_counter()
    try:
        recalled = memories.recall(user_id, message, url=deps.url)
    except Exception:
        log.warning("memory recall skipped", exc_info=True)
    if recalled:
        _step("Memory recall", f"{len(recalled)} memories", ms=_since(t0), status="success", key="ingest")
    else:
        _step("Memory recall", "none that fit this question", ms=_since(t0), key="ingest")
    facts = ""
    ctx = state.get("context") or {}
    if ctx.get("kind") == "nudge":
        nudge = reports.nudge_detail(user_id, str(ctx.get("id")), url=deps.url)
        facts = f"Coach nudge from {nudge['created_at']}: {nudge['message']}" if nudge else ""
    elif ctx.get("kind") == "transaction":
        detail = txns.transaction_detail(user_id, str(ctx.get("id")), url=deps.url)
        facts = f"Transaction: {json.dumps(detail, default=str)}" if detail else ""
    return {"memories": recalled, "context_facts": facts, "results": [], "answer": "", "streamed": False,
            "cached": False, "saved_tokens": 0, "meta": {}, "proposals": [], "citations": [], "route": {},
            "page": None}


def route_node(state: TurnState, runtime: Runtime[TurnDeps]) -> dict:
    t0 = time.perf_counter()
    r = decide(state["message"], active_receipt=state.get("active_receipt"), context=state.get("context"),
               last_route=state.get("last_route"), has_history=bool(state.get("messages")),
               last_search=state.get("last_search"))
    _step("Intent routing", f"{r.kind}: {r.detail}", ms=(time.perf_counter() - t0) * 1000)
    return {"route": r.as_dict()}


NOT_FOUND = "I couldn't find a category or merchant called '{subject}'."


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
        t0 = time.perf_counter()
        merchant = resolve_merchant(intent.subject, user_id, url)
        if merchant is None:
            # A total across every category would answer a question nobody asked.
            _step("Category lookup", f"no category or merchant called '{intent.subject}'",
                  ms=(time.perf_counter() - t0) * 1000, status="warning")
            return {"answer": answers.translate(NOT_FOUND.format(subject=intent.subject),
                                                state.get("language") or "English", deps.chat)}
    # Named only when all the user's rows live in their home region: a sum
    # pinned there would miss rows written elsewhere and answer "none".
    region = read_region(user_id, url)
    narration.aggregation(intent, region)
    if region:
        _step("Region pruning", f"Scoped to {region} (REGIONAL BY ROW)")
    _running("sql", "Running the numbers in CockroachDB")
    t0 = time.perf_counter()
    agg = run_aggregation(intent, user_id, url, region=region, merchant=merchant)
    db_ms = _since(t0)
    narration.aggregation_sql(agg, db_ms)
    _step("Aggregation SQL", f"${agg.total:,.2f} across {agg.count} transactions "
          f"({agg.merchant or agg.category or 'all categories'}), computed by CockroachDB", ms=db_ms, status="success",
          key="sql")
    language = state.get("language") or "English"
    key, hit = None, None
    try:
        key = caches.search_key(user_id, ledger.data_version(user_id, deps.url),
                                {"agg": [agg.operation, agg.category, agg.merchant, str(agg.window_start),
                                         str(agg.window_end), agg.total, agg.count], "lang": language})
        hit = caches.get_search(user_id, key, url=deps.url, stat="narrative")
    except Exception:  # a cache outage is a miss, never a broken answer
        log.warning("narrative cache read skipped", exc_info=True)
    saved = 0
    if hit:
        text, cached, streamed = hit[0]["answer"], True, False
        saved = caches.estimate_tokens(text)
        narration.narrative_cache(True, saved)
        _step("Narrative cache", f"hit: SQL reran, the prose was reused, about {saved:,} tokens saved", status="hit")
    else:
        narration.narrative_cache(False)
        _step("Narrative cache", "miss: generating the narrative", status="miss")
        # An English card with no purchases goes out without the model.
        writes = bool(agg.count) or language != "English"
        if writes:
            _running("answer", "Writing the answer")
        t0 = time.perf_counter()
        (text, finished), cached, streamed = _narrate(agg, state["message"], language, deps.chat), False, True
        if writes:
            _step("LLM answer", "narrative streamed" if finished else "the model stopped early; its words are kept",
                  ms=_since(t0), status="success" if finished else "warning", key="answer")
        if key is not None and finished:  # a narrative the model never finished is never replayed
            background.submit("narrative cache write", caches.put_search, user_id, key, [{"answer": text}],
                              url=deps.url, stat="narrative")
    explain, served = _query_plan(lambda: explain_aggregation(intent, user_id, url, region=region, merchant=merchant),
                                  region)
    return {"answer": text, "cached": cached, "streamed": streamed, "saved_tokens": saved,
            "meta": {"region": served, "db_ms": int(db_ms), "explain": explain}}


def visits(state: TurnState, runtime: Runtime[TurnDeps]) -> dict:
    """"Did I go to IKEA?": the visits come from SQL and the card from code,
    so no model lists, drops or invents a date. A place that is none of the
    user's merchants ("the gym") goes on to the planner."""
    from ..utils.intent_classifier import extract_window
    deps, user_id = runtime.context, state["user_id"]
    place = state["route"].get("payload") or ""
    window = extract_window(state["message"])
    days = {"date_from": window[0], "date_to": window[1] - timedelta(days=1)} if window else {}
    url = deps.url or get_config().database_url
    with get_engine(url).connect() as conn:
        region = money_region(conn, user_id, url)
    _running("sql", "Running the numbers in CockroachDB")
    t0 = time.perf_counter()
    found = txns.merchant_history(user_id, place, recent=10, url=deps.url, **days)
    db_ms = _since(t0)
    if found is None:
        _step("Merchant lookup", f"no merchant called '{place}'; searching by meaning instead", ms=db_ms,
              status="warning", key="sql")
        return {"route": {**state["route"], "kind": "plan", "detail": f"'{place}' is not one of your merchants"}}
    last_ever = None
    if window and not found["visits"]:
        last_ever = (txns.merchant_history(user_id, found["merchant"], recent=0, url=deps.url) or {}).get("last")
    narration.visits(found["merchant"], found["visits"], db_ms)
    _step("Visits SQL", f"{found['visits']} visits to {found['merchant']}, ${found['total']:,.2f}, computed by "
          "CockroachDB", ms=db_ms, status="success", key="sql")
    card = answers.render_visits_markdown(found, window=window, last_ever=last_ever)
    explain, served = _query_plan(
        lambda: txns.explain_merchant_visits(user_id, found["merchant"], url=deps.url, **days), region)
    return {"answer": answers.translate(card, state.get("language") or "English", deps.chat),
            "meta": {"region": served, "db_ms": int(db_ms), "explain": explain}}


def _query_plan(explain_fn, region: str | None) -> tuple[str | None, str | None]:
    """EXPLAIN ANALYZE for the trace, and the region that served the query.
    The answer is already written, so a database that can't give the plan
    (a node going away mid-demo) costs the plan, never the answer."""
    _running("explain", "Reading the query plan")
    t0 = time.perf_counter()
    try:
        explain = explain_fn()
    except Exception:
        log.warning("EXPLAIN skipped", exc_info=True)
        _step("EXPLAIN ANALYZE", "plan not captured", ms=_since(t0), status="warning", key="explain")
        return None, region
    served = serving_regions(explain) or region
    _step("EXPLAIN ANALYZE", "plan captured" + (f", served by {served}" if served else ""), ms=_since(t0),
          key="explain")
    return explain, served


def _emit_text(text: str) -> None:
    for i in range(0, len(text), 48):
        _emit({"type": "token", "text": text[i:i + 48]})


def _relay(open_stream, prefix: str = "") -> tuple[str, bool]:
    """Send a model stream out as token events and return exactly what was
    sent, and whether the stream finished. Whitespace at either end is held
    back, and prefix goes out just before the first real text, so nothing
    dangles when the stream is empty. A stream that fails to open or breaks
    keeps what already went out."""
    sent: list[str] = []
    held, started = "", False
    try:
        for piece in open_stream():
            if not started:
                piece = piece.lstrip()
                if not piece:
                    continue
                started, piece = True, prefix + piece
            body = piece.rstrip()
            if not body:
                held += piece
                continue
            chunk, held = held + body, piece[len(body):]
            _emit({"type": "token", "text": chunk})
            sent.append(chunk)
    except Exception:
        log.warning("model stream stopped early", exc_info=True)
        return "".join(sent), False
    return "".join(sent), True


def _narrate(agg, question: str, language: str, chat: Chat) -> tuple[str, bool]:
    """The aggregation answer, streamed as it is produced. In English the SQL
    card goes out at once and the insights stream after it. In another
    language the insights come first (one call), then the translation of
    card and insights streams. Returns exactly the text sent, and whether
    the model finished all of it (only a finished narrative is cached)."""
    card = answers.render_aggregation_markdown(agg)
    if language == "English":
        _emit_text(card)
        if not agg.count:
            return card, True
        insights, finished = _relay(lambda: chat.stream(answers.insights_messages(agg, question), temperature=0.4),
                                    prefix="\n\n")
        if insights:
            narration.insights()
        return card + insights, finished
    notes = answers.aggregation_insights(agg, question, chat) if agg.count else None
    if notes:
        narration.insights()
    english = card + (f"\n\n{notes}" if notes else "")
    translated, finished = _relay(lambda: chat.stream(answers.translation_messages(english, language),
                                                      temperature=0.1))
    if not translated:  # no translation came back: the English, figures and all, never cached as the translation
        _emit_text(english)
        return english, False
    if translated != english:
        narration.rendered(language)
    return translated, finished and (notes is not None or not agg.count)


RECEIPT_KEYS = ("document_id", "merchant", "amount", "date", "category", "items", "expense_id")
NO_RECEIPT = "I don't see a receipt in this conversation yet. Upload one and ask me about it."


def receipt(state: TurnState, runtime: Runtime[TurnDeps]) -> dict:
    deps = runtime.context
    ctx, adopted, which = _ctx(state, deps), None, "the active receipt"
    if not (state.get("active_receipt") or {}).get("document_id"):
        # "how much was this receipt?" in a thread without one: the user's latest.
        t0 = time.perf_counter()
        latest = receipts.latest_receipt(state["user_id"], url=deps.url)
        if latest is None:
            _step("Receipt details", "no receipt yet", ms=(time.perf_counter() - t0) * 1000, status="warning")
            return {"answer": answers.translate(NO_RECEIPT, state.get("language") or "English", deps.chat)}
        adopted, which = {k: latest.get(k) for k in RECEIPT_KEYS}, "your latest receipt"
        ctx = replace(ctx, active_receipt=adopted)
    result = run_tool("receipt_details", {}, ctx)
    _step("Receipt details", which if result["ok"] and result.get("result") else "not found",
          ms=result["ms"], status="success" if result["ok"] else "warning")
    # The latest receipt becomes this thread's receipt, so "was the wine on it?" works next.
    return {"results": [result], **({"active_receipt": adopted} if adopted else {})}


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
    if not state.get("messages"):
        return True
    return not (reads_as_follow_up(state["message"]) or (state.get("route") or {}).get("follow_up"))


def plan(state: TurnState, runtime: Runtime[TurnDeps]) -> dict:
    deps, user_id = runtime.context, state["user_id"]
    language = state.get("language") or "English"
    if _cacheable(state):
        hit = None
        _running("answer_cache", "Checking the answer cache")
        t0 = time.perf_counter()
        try:
            from ..services._vector import embed
            hit = caches.get_answer(user_id, version=ledger.data_version(user_id, deps.url), language=language,
                                    query=state["message"], qvec=embed(state["message"]), url=deps.url)
        except Exception:  # a cache outage is a miss, never a broken answer
            log.warning("answer cache read skipped", exc_info=True)
        if hit:
            narration.answer_cache(True, hit["tokens"])
            _step("Answer cache", f"hit, about {hit['tokens']:,} tokens saved", ms=_since(t0), status="hit",
                  key="answer_cache")
            return {"answer": hit["answer"], "cached": True, "saved_tokens": int(hit["tokens"])}
        narration.answer_cache(False)
        _step("Answer cache", "miss", ms=_since(t0), status="miss", key="answer_cache")

    tools_run = 0

    def on_start(step) -> None:
        nonlocal tools_run
        tools_run += 1
        if step.tool in _TOOL_PHASES:
            _running(f"tool:{tools_run}", _TOOL_PHASES[step.tool])

    def on_step(result: dict) -> None:
        _step(f"Tool {result['tool']}", "ok" if result["ok"] else result.get("error", "failed"),
              ms=result["ms"], status="success" if result["ok"] else "warning", key=f"tool:{tools_run}")

    results = plan_and_act(_Planning(deps.chat), state["message"], _ctx(state, deps),
                           history=state.get("messages") or [], memories=state.get("memories") or [],
                           context_facts=state.get("context_facts") or "", on_step=on_step, on_start=on_start)
    # Kept in event shape, so the streamed proposal and the returned one match.
    proposals = [{"type": "proposal", **r["result"]} for r in results
                 if r["ok"] and r["tool"].startswith("propose_")]
    meta, page = _search_meta(state, deps, results), _page(results)
    if page and page["next"]:
        meta["search"] = page  # the chat's Show more pages it
    return {"results": results, "proposals": proposals, "meta": meta, "page": page}


def _page(results: list[dict]) -> dict | None:
    """The turn's semantic search as "show more" continues it: the same
    search _search_meta explains, with the cursor of its next page."""
    found = next((r for r in results if r["tool"] == "search_transactions" and r["ok"]), None)
    if found is None:
        return None
    a = SearchArgs.model_validate(found.get("args") or {})
    return {"q": a.query, "category": a.category, "from": a.date_from.isoformat() if a.date_from else None,
            "to": a.date_to.isoformat() if a.date_to else None, "next": (found.get("result") or {}).get("next_cursor")}


MORE_LINE = "Here are {n} more matches for “{q}”."
ALL_SHOWN = "That's every match I found for “{q}”."


def more(state: TurnState, runtime: Runtime[TurnDeps]) -> dict:
    """The next page of the previous turn's search, straight from the
    database: no planner and no model, a fixed line plus the purchases."""
    deps, last = runtime.context, state.get("last_search") or {}
    language = state.get("language") or "English"
    if not last.get("next"):
        _step("Next page", f"nothing after the last page of '{last.get('q', '')}'")
        return {"answer": answers.translate(ALL_SHOWN.format(q=last.get("q", "")), language, deps.chat),
                "page": last}
    _running("tool:1", _TOOL_PHASES["search_transactions"])
    t0 = time.perf_counter()
    found = txns.semantic_search(state["user_id"], last["q"], limit=txns.search_page_size(), cursor=last["next"],
                                 category=last.get("category"),
                                 date_from=date.fromisoformat(last["from"]) if last.get("from") else None,
                                 date_to=date.fromisoformat(last["to"]) if last.get("to") else None, url=deps.url)
    db_ms = _since(t0)
    items = [{"expense_id": i["expense_id"], "merchant": i["merchant"], "amount": _positive(i["amount"]),
              "date": i["posted_at"]} for i in found["items"]]
    _step("Next page", f"{len(items)} more for '{last['q']}'", ms=db_ms, status="success", key="tool:1")
    page = {**last, "next": found["next_cursor"]}
    url = deps.url or get_config().database_url
    with get_engine(url).connect() as conn:
        home = money_region(conn, state["user_id"], url)
    line = MORE_LINE.format(n=len(items), q=last["q"]) if items else ALL_SHOWN.format(q=last["q"])
    return {"answer": answers.translate(line, language, deps.chat), "page": page,
            "meta": {"region": home, "db_ms": int(db_ms), "search": {**page, "items": items}}}


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
    _running("explain", "Reading the query plan")
    t0 = time.perf_counter()
    try:
        explain = txns.explain_semantic(state["user_id"], query, url=deps.url)
        _step("EXPLAIN ANALYZE", "plan captured", ms=_since(t0), key="explain")
    except Exception:
        log.warning("semantic EXPLAIN skipped", exc_info=True)
        _step("EXPLAIN ANALYZE", "plan not captured", ms=_since(t0), status="warning", key="explain")
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
            if isinstance(result, dict):  # the page cursor is for the chat, not the model
                result = {k: v for k, v in result.items() if k != "next_cursor"}
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
        if not state.get("streamed"):  # a fixed or cached answer; a streamed one already went out
            _emit_text(answer)
    else:
        msgs = answer_messages(state["message"], language=language,
                               history=state.get("messages") or [], memories=state.get("memories") or [],
                               facts=_facts(state.get("results") or []),
                               context_facts=state.get("context_facts") or "")
        narration.generating(language)
        _running("answer", "Writing the answer")
        t0 = time.perf_counter()
        parts = []
        for piece in deps.chat.stream(msgs, temperature=0.3):
            parts.append(piece)
            _emit({"type": "token", "text": piece})
        answer = "".join(parts).strip()
        generated = True
        _step("LLM answer", "streamed", ms=_since(t0), status="success", key="answer")
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
    # card or an empty reply would replay wrong on a hit, and an answer no
    # tool grounded would keep a guess for every later ask.
    elif (generated and state["route"].get("kind") == "plan" and _cacheable(state) and answer
          and any(r.get("ok") for r in results)):
        background.submit("answer cache write", _keep_answer, state["user_id"], language, state["message"],
                          answer, deps.url)
    for c in citations:
        _emit({"type": "citation", **c})
    for p in proposals:
        _emit(p)
    # Everything the answer event needs, sent now: learn and the checkpoint
    # write still have to run, and nobody should wait for them.
    _emit({"type": "final", "answer": answer, "citations": citations, "proposals": proposals,
           "cached": bool(state.get("cached")), "saved_tokens": int(state.get("saved_tokens") or 0),
           "route": state.get("route") or {}, "meta": dict(state.get("meta") or {})})
    return {"answer": answer, "citations": citations, "last_route": state["route"].get("kind"),
            "last_search": state.get("page"),
            "messages": [{"role": "user", "content": state["message"]},
                         {"role": "assistant", "content": answers.strip_citations(answer)}]}


def _keep_answer(user_id: str, language: str, query: str, answer: str, url: str | None) -> None:
    """The answer cache write, after the answer went out. The data version
    is read here, after the turn, as the inline write always did."""
    from ..services._vector import embed
    caches.put_answer(user_id, version=ledger.data_version(user_id, url), language=language, query=query,
                      qvec=embed(query), answer=answer, tokens=caches.estimate_tokens(answer), url=url)


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
        # One more model call; the answer is already out, so it runs behind it.
        background.submit("implicit memory", _extract_memory, deps.chat, user_id, state["message"],
                          state["thread_id"], deps.url)
    return {}


def _extract_memory(chat: Chat, user_id: str, message: str, thread_id: str, url: str | None) -> None:
    raw = chat.complete(memory_extraction_messages(message), temperature=0)
    obj = first_object(raw) or {}
    if obj.get("content") and obj.get("kind") in ("fact", "preference", "goal"):
        memories.remember(user_id, obj["content"], kind=obj["kind"], source_thread_id=thread_id, url=url)


def _branch(state: TurnState) -> str:
    kind = state["route"]["kind"]
    return {"remember": "memory", "forget": "memory"}.get(kind, kind)


def _after_visits(state: TurnState) -> str:
    return "plan" if state["route"]["kind"] == "plan" else "respond"


def build_graph(checkpointer):
    g = StateGraph(TurnState, context_schema=TurnDeps)
    for name, fn in (("ingest", ingest), ("route", route_node), ("aggregate", aggregate), ("visits", visits),
                     ("receipt", receipt), ("memory", memory), ("offtopic", offtopic), ("plan", plan),
                     ("more", more), ("respond", respond), ("learn", learn)):
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "route")
    g.add_conditional_edges("route", _branch,
                            ["aggregate", "visits", "receipt", "memory", "offtopic", "plan", "more"])
    g.add_conditional_edges("visits", _after_visits, ["plan", "respond"])
    for name in ("aggregate", "receipt", "memory", "offtopic", "plan", "more"):
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
            "cached": bool(final.get("cached")), "saved_tokens": int(final.get("saved_tokens") or 0),
            "route": final.get("route", {})}


def set_active_receipt(thread_id: str, receipt_info: dict, *, graph=None) -> None:
    """Make an upload the thread's receipt. The upload counts as a turn about
    it, so "is it a duplicate?" right after still means the receipt."""
    (graph or get_graph()).update_state(_config(thread_id), {"active_receipt": receipt_info, "last_route": "receipt"},
                                        as_node="learn")


def set_context(thread_id: str, context: dict, *, graph=None) -> None:
    """Open a thread from a nudge or a transaction; every turn of it sees that context."""
    (graph or get_graph()).update_state(_config(thread_id), {"context": context}, as_node="learn")


def thread_state(thread_id: str, *, graph=None) -> dict:
    state = (graph or get_graph()).get_state(_config(thread_id))
    return dict(state.values or {})


def thread_messages(thread_id: str, *, graph=None) -> list[dict]:
    state = (graph or get_graph()).get_state(_config(thread_id))
    return list((state.values or {}).get("messages") or [])
