"""What the demo console prints while the assistant works, in the wording it
has always used: the aggregation, the vector search, the answer cache, the
model answering and the receipt pipeline. Plain stdout, like the boot lines.

Every line describes work that just ran. Never pass a secret, a connection
URL or a file path in here; filenames and queries come from users, so they
are printed without control characters."""

from __future__ import annotations

from ..config.settings import get_config


def _say(line: str) -> None:
    try:
        print(line, flush=True)
    except Exception:  # a closed, broken or non-UTF-8 stdout never stops the work it describes
        pass


def _plain(value: object) -> str:
    return "".join(ch for ch in str(value) if ch.isprintable())


def _service() -> str:
    return get_config().ai_service


# ── aggregation: SQL does the math ───────────────────────────────────────

def aggregation(intent, region: str | None) -> None:
    _say("\n🧮 AGGREGATION (agentic routing):")
    _say(f"1. Intent: {intent.operation} on '{_plain(intent.subject or 'all spending')}' | window "
         f"{intent.window_start} to {intent.window_end}")
    if region:
        _say(f"   Region-pruned to {region} (REGIONAL BY ROW)")


def aggregation_sql(agg, ms: float) -> None:
    scope = agg.category or getattr(agg, "merchant", None) or "all"
    _say(f"2. SQL answered in {int(ms)}ms: ${agg.total:,.2f} across {agg.count} transactions (category: {scope})")
    _say("   Figures computed by CockroachDB; the model never does the arithmetic")


def narrative_cache(hit: bool, tokens: int = 0) -> None:
    _say(f"3. ✅ Narrative cache HIT (est. {tokens} tokens saved)" if hit
         else "3. ❌ Narrative cache MISS, generating")


def insights() -> None:
    _say(f"   Insights narrated by {_service()} around the pinned figures")


def rendered(language: str) -> None:
    _say(f"   Rendered in {language}, figures preserved")


# ── vector search ────────────────────────────────────────────────────────

def vector_search(query: str, limit: int) -> None:
    _say("\n🔍 VECTOR SEARCH (with caching):")
    _say(f"1. Query: '{_plain(query)}' | Limit: {limit}")


def vector_cache(cached: int | None) -> None:
    _say("2. ❌ Vector search cache MISS, querying database" if cached is None
         else f"2. ✅ Vector search cache HIT! Found {cached} cached results")


def vector_rows(count: int, ms: float) -> None:
    _say(f"3. Database query returned {count} expense records in {ms:.1f}ms")


# ── the answer cache and the model's answer ─────────────────────────────

def answer_cache(hit: bool, tokens: int = 0) -> None:
    _say(f"✅ Answer cache HIT (est. {tokens} tokens saved)" if hit else "❌ Answer cache MISS")


def generating(language: str) -> None:
    _say(f"Using {_service()} for response generation in {language}")


# ── receipts ─────────────────────────────────────────────────────────────

def receipt_uploaded(filename: str) -> None:
    _say(f"📄 Receipt uploaded: {_plain(filename)}")


def receipt_fields(fields: dict) -> None:
    _say(f"📊 Extracted fields: {fields!r}")


def receipt_posted(expense_id: str) -> None:
    _say(f"💰 Expense posted through the ledger: {expense_id}")


def receipt_duplicate(detail: str) -> None:
    _say(f"⚠️  Duplicate detected: {detail}, nothing posted")


def receipt_check(step: str) -> None:
    _say("🕵️  Running fraud check..." if step == "fraud" else "📊 Running budget check...")


def receipt_check_result(result: str) -> None:
    _say(f"   {result}")


def receipt_failed(message: str) -> None:
    _say(f"❌ {message}")
