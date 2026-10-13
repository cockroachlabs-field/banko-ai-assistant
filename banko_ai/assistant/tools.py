"""The assistant's tools. Each takes validated arguments, runs user-scoped
service code (all SQL lives in banko_ai/services), and returns JSON-safe
results. run_tool never raises: a bad plan step becomes a recorded error,
not a failed answer."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ..config.catalog import get_catalog
from ..config.settings import get_config
from ..services import accounts, actions, budgets, caches, goals, ledger, memories, receipts, recurring, reports
from ..services import transactions as txns
from ..services._sql import jsonable
from ..utils.request_trace import trace_step
from . import narration

log = logging.getLogger("banko.assistant")


class ToolError(Exception):
    """User-facing error message from a tool."""
    pass


@dataclass(frozen=True)
class ToolContext:
    user_id: str
    thread_id: str | None = None
    url: str | None = None
    active_receipt: dict | None = None


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    fn: Callable[[ToolContext, Any], Any]


class NoArgs(BaseModel):
    pass


class SearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    category: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    limit: int = Field(10, ge=1, le=25)


class SpendArgs(BaseModel):
    date_from: date | None = None
    date_to: date | None = None
    category: str | None = None
    merchant: str | None = None


class ReceiptArgs(BaseModel):
    document_id: str | None = None


class ExpenseArgs(BaseModel):
    expense_id: str
    limit: int = Field(5, ge=1, le=20)


class MerchantArgs(BaseModel):
    merchant: str = Field(min_length=1, max_length=120)


class MonthsArgs(BaseModel):
    months: int = Field(3, ge=1, le=12)


class QueryArgs(BaseModel):
    query: str = Field(min_length=1, max_length=300)


class RememberArgs(BaseModel):
    content: str = Field(min_length=1, max_length=500)


class RecategorizeArgs(BaseModel):
    expense_id: str
    category: str = Field(min_length=1, max_length=60)


class BudgetArgs(BaseModel):
    category: str = Field(min_length=1, max_length=60)
    amount: Decimal = Field(gt=0, le=1_000_000)


def _search(ctx: ToolContext, a: SearchArgs) -> dict:
    narration.vector_search(a.query, a.limit)
    payload = a.model_dump(mode="json")
    key, cached = None, None
    try:
        key = caches.search_key(ctx.user_id, ledger.data_version(ctx.user_id, ctx.url), payload)
        cached = caches.get_search(ctx.user_id, key, url=ctx.url)
    except Exception:  # a cache outage is a miss, never a failed search
        log.warning("search cache read skipped", exc_info=True)
    if cached is not None:
        narration.vector_cache(len(cached))
        trace_step("cache", "Search cache", f"hit for '{a.query}'", status="hit")
        return {"items": cached}
    narration.vector_cache(None)
    trace_step("cache", "Search cache", f"miss for '{a.query}'", status="miss")
    t0 = time.perf_counter()
    found = txns.semantic_search(ctx.user_id, a.query, limit=a.limit, category=a.category,
                                 date_from=a.date_from, date_to=a.date_to, url=ctx.url)["items"]
    narration.vector_rows(len(found), (time.perf_counter() - t0) * 1000)
    items = [{"expense_id": i["expense_id"], "date": i["posted_at"], "merchant": i["merchant"],
              "amount": abs(i["amount"]), "category": i["category"], "description": i["description"]}
             for i in found]
    if key is not None:
        try:
            caches.put_search(ctx.user_id, key, items, url=ctx.url)
        except Exception:
            log.warning("search cache write skipped", exc_info=True)
    return {"items": items}


def _spend(ctx: ToolContext, a: SpendArgs) -> dict:
    from ..utils.aggregations import resolve_category, resolve_merchant, run_aggregation
    from ..utils.intent_classifier import AggregationIntent
    url = ctx.url or get_config().database_url
    end = (a.date_to or date.today()) + timedelta(days=1)
    start = a.date_from or (end - timedelta(days=31))
    if a.merchant:
        merchant = resolve_merchant(a.merchant, ctx.user_id, url)
        if merchant is None:
            raise ToolError(f"No merchant called '{a.merchant}' in your history")
    else:
        merchant = None
    if a.category and not merchant:
        if resolve_category(a.category, url) is None:
            raise ToolError(f"No category called '{a.category}'")
    intent = AggregationIntent("sum", a.category if not merchant else None, start, end)
    agg = run_aggregation(intent, ctx.user_id, url, merchant=merchant)
    return {"total": agg.total, "count": agg.count, "average": agg.average, "category": agg.category,
            "merchant": agg.merchant, "from": str(start), "to": str(end - timedelta(days=1)),
            "rows": agg.rows[:10]}


def _receipt(ctx: ToolContext, a: ReceiptArgs) -> dict | None:
    if a.document_id:
        return receipts.get_receipt(ctx.user_id, a.document_id, url=ctx.url)
    if ctx.active_receipt and ctx.active_receipt.get("document_id"):
        return receipts.get_receipt(ctx.user_id, ctx.active_receipt["document_id"], url=ctx.url)
    return None


def _duplicates(ctx: ToolContext, a: ExpenseArgs) -> dict:
    cfg = get_config()
    return txns.duplicate_check(ctx.user_id, a.expense_id, window_days=cfg.duplicate_window_days,
                                max_distance=cfg.receipt_duplicate_max_distance, url=ctx.url)


def _propose_recategorize(ctx: ToolContext, a: RecategorizeArgs) -> dict:
    detail = txns.transaction_detail(ctx.user_id, a.expense_id, url=ctx.url)
    if not detail:
        raise ledger.NotFound("That transaction was not found")
    labels = {c.label.lower(): c.label for c in get_catalog().categories}
    label = labels.get(a.category.strip().lower())
    if label is None:
        raise ToolError("Pick one of the catalog categories")
    summary = f"Move {detail.get('merchant') or 'this purchase'} on {str(detail.get('posted_at'))[:10]} to {label}"
    return actions.propose(ctx.user_id, "recategorize", {"expense_id": a.expense_id, "category": label},
                           summary, thread_id=ctx.thread_id, url=ctx.url)


def _propose_budget(ctx: ToolContext, a: BudgetArgs) -> dict:
    amount = a.amount.quantize(Decimal("0.01"))
    return actions.propose(ctx.user_id, "budget", {"category": a.category, "amount": str(amount)},
                           f"Set your {a.category} budget to ${amount:,.2f} a month",
                           thread_id=ctx.thread_id, url=ctx.url)


TOOLS: dict[str, Tool] = {t.name: t for t in (
    Tool("search_transactions", "find purchases by meaning (e.g. 'late night food', 'pizza')", SearchArgs, _search),
    Tool("spend_summary", "exact totals from SQL for a window, optionally one category or merchant", SpendArgs, _spend),
    Tool("receipt_details", "the active receipt (or one by document_id): merchant, total, items, linked expense",
         ReceiptArgs, _receipt),
    Tool("similar_transactions", "purchases most like one expense", ExpenseArgs,
         lambda c, a: txns.similar_transactions(c.user_id, a.expense_id, limit=a.limit, url=c.url)),
    Tool("duplicate_check", "possible duplicates of one expense: same amount, nearby date, similar description",
         ExpenseArgs, _duplicates),
    Tool("merchant_history", "visits, total, average and recent purchases at one merchant", MerchantArgs,
         lambda c, a: txns.merchant_history(c.user_id, a.merchant, url=c.url)),
    Tool("budget_status", "this month's budgets with spent and remaining per category", NoArgs,
         lambda c, a: budgets.budget_progress(c.user_id, url=c.url)),
    Tool("recurring_bills", "detected bills and subscriptions with cadence, next due and price changes", NoArgs,
         lambda c, a: recurring.recurring_json(recurring.detect_recurring(c.user_id, url=c.url))),
    Tool("goals_status", "savings goals with progress and projected completion", NoArgs,
         lambda c, a: goals.list_goals(c.user_id, url=c.url)),
    Tool("balances", "accounts with balances, credit utilization and net worth", NoArgs,
         lambda c, a: accounts.list_accounts(c.user_id, url=c.url)),
    Tool("cash_flow", "income and spending by month", MonthsArgs,
         lambda c, a: reports.cashflow(c.user_id, months=a.months, url=c.url)),
    Tool("recall", "the user's saved memories relevant to a question", QueryArgs,
         lambda c, a: memories.recall(c.user_id, a.query, url=c.url)),
    Tool("remember", "save a durable fact the user stated about themselves", RememberArgs,
         lambda c, a: memories.remember(c.user_id, a.content, source_thread_id=c.thread_id, url=c.url)),
    Tool("propose_recategorize", "propose moving an expense to another category (runs only after Confirm)",
         RecategorizeArgs, _propose_recategorize),
    Tool("propose_budget", "propose a monthly budget for a category (runs only after Confirm)",
         BudgetArgs, _propose_budget),
)}


def describe_tools() -> str:
    lines = []
    for t in TOOLS.values():
        fields = ", ".join(f"{n}: {getattr(f.annotation, '__name__', str(f.annotation))}"
                           for n, f in t.args.model_fields.items())
        lines.append(f"- {t.name}({fields}): {t.description}")
    return "\n".join(lines)


def run_tool(name: str, args: dict | None, ctx: ToolContext) -> dict:
    t0 = time.perf_counter()

    def done(**out) -> dict:
        ms = round((time.perf_counter() - t0) * 1000, 1)
        return {"tool": name, **out, "ms": ms}

    tool = TOOLS.get(name)
    if tool is None:
        trace_step("tool", f"Unknown tool {name}", "skipped", status="warning")
        return done(ok=False, error="unknown tool")
    try:
        parsed = tool.args.model_validate(args or {})
    except ValidationError:
        trace_step("tool", name, "bad arguments, skipped", status="warning")
        return done(ok=False, error="bad arguments")
    try:
        result = jsonable(tool.fn(ctx, parsed))
    except (ledger.NotFound, ToolError) as exc:
        trace_step("tool", name, str(exc), status="warning")
        return done(ok=False, error=str(exc))
    except Exception:
        log.warning("tool %s failed", name, exc_info=True)
        trace_step("tool", name, "failed; details are in the server log", status="error")
        return done(ok=False, error="failed")
    out = done(ok=True, result=result)
    trace_step("tool", name, _summary(result), ms=out["ms"], status="success")
    return out


def _summary(result: Any) -> str:
    if isinstance(result, dict) and "items" in result:
        return f"{len(result['items'])} rows"
    if isinstance(result, list):
        return f"{len(result)} rows"
    return "done" if result is not None else "nothing found"
