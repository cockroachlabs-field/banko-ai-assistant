"""
Intent classifier for Banko AI.

Two-stage gate. A cheap lexical/fuzzy pre-pass catches the common case where
the query obviously names a financial concept ("delta expenses", "$42 at
starbucks", "how much did I spend on groceries") — including typos like
"delta exepnses" that MiniLM-anchor cosine similarity scores below threshold.
Only queries that miss the lexical gate fall through to the embedding check.

Why the lexical pre-pass exists: before the embedding gate was added
(2026-03-17, commit f93782d), every query went straight to RAG and the LLM
itself was robust to typos. The embedding gate intercepted before the LLM
could see them; the pre-pass restores that tolerance for queries that name
known financial signals.
"""

import difflib
import re
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from banko_ai.utils.embeddings import load_embedding_model

_model = None

FINANCIAL_ANCHORS = [
    "expense report spending budget",
    "transaction payment purchase receipt",
    "credit card debit charge bill",
    "grocery shopping restaurant dining coffee food",
    "monthly cost salary income savings",
    "financial summary category breakdown",
    "subscription recurring payment",
    "refund reimbursement overdraft",
    "rent mortgage utility electricity",
    "travel hotel flight uber lyft taxi",
    "duplicate fraud suspicious charge",
    "how much did I spend on",
    "amazon walmart target costco starbucks",
]

# Queries that are clearly non-financial get high similarity to these anchors,
# so we check against them and reject if the query is closer to this set.
NON_FINANCIAL_ANCHORS = [
    "weather forecast temperature rain",
    "joke funny humor laugh",
    "recipe cook bake ingredients",
    "sports game score team",
    "politics president election government",
    "movie film actor director",
    "song music lyrics album",
    "translate language dictionary words",
    "science physics chemistry biology",
    "history war ancient civilization",
]

REDIRECT_MESSAGE = (
    "I'm Banko, your personal finance assistant. "
    "I can help with expense tracking, spending analysis, budgets, "
    "receipt processing, and fraud detection. "
    "What would you like to know about your finances?"
)

SIMILARITY_THRESHOLD = 0.20

# Core finance vocabulary the lexical/fuzzy pre-pass checks before the
# embedding gate. Hits here short-circuit to True without loading MiniLM.
# Keep this list small and high-signal — false positives let off-topic queries
# through to the LLM, but that's a softer failure than blocking real ones.
FINANCIAL_KEYWORDS = frozenset({
    # core verbs
    "spend", "spent", "spending", "paid", "pay", "bought", "buy", "charged",
    "owe", "owed",
    # core nouns
    "expense", "expenses", "transaction", "transactions", "receipt", "receipts",
    "budget", "budgets", "balance", "bill", "bills", "payment", "payments",
    "refund", "refunds", "charge", "charges", "cost", "costs", "income",
    "salary", "savings", "subscription", "subscriptions", "purchase",
    "purchases", "invoice", "invoices", "deposit", "withdrawal", "fraud",
    "duplicate",
    # categories
    "grocery", "groceries", "restaurant", "dining", "food", "gas", "fuel",
    "rent", "mortgage", "utility", "utilities", "electricity", "internet",
    "travel", "hotel", "flight", "flights", "uber", "lyft", "taxi",
    # common merchants we know will show up in demo data
    "amazon", "walmart", "target", "costco", "starbucks", "delta", "united",
    "american", "southwest", "nike", "apple",
    # currency words (the symbol is checked separately)
    "dollar", "dollars", "usd",
})

# Words used in fuzzy matching only — they're the most-likely-typo'd core
# financial terms plus distinctive-shape merchant names. Kept small so
# difflib stays fast and unrelated tokens don't accidentally match (avoid
# adding short or generic-shape words like "delta" or "apple" here).
_FUZZY_CORE = frozenset({
    "expense", "expenses", "spend", "spent", "spending", "transaction",
    "receipt", "budget", "payment", "refund", "charge", "subscription",
    "starbucks", "amazon", "walmart", "costco",
})

_FUZZY_CUTOFF = 0.80  # difflib SequenceMatcher ratio; 0.80 = one typo in
                      # ~5 chars, tight enough to avoid runaway false hits

_TOKEN_RE = re.compile(r"[a-zA-Z$]+")


def _lexical_or_fuzzy_financial_hit(query: str) -> bool:
    """Cheap pre-pass: True if the query mentions a financial concept by
    exact substring, currency symbol, or close fuzzy match against the core
    vocabulary. Catches typos the MiniLM gate can't recover from."""
    lower = query.lower()
    if "$" in lower:
        return True
    tokens = _TOKEN_RE.findall(lower)
    for tok in tokens:
        if tok in FINANCIAL_KEYWORDS:
            return True
    # Fuzzy second pass — only for tokens long enough to be meaningful
    for tok in tokens:
        if len(tok) < 4:
            continue
        if difflib.get_close_matches(tok, _FUZZY_CORE, n=1, cutoff=_FUZZY_CUTOFF):
            return True
    return False


def _get_model():
    global _model
    if _model is None:
        _model = load_embedding_model()
    return _model


_financial_embeddings: np.ndarray | None = None
_non_financial_embeddings: np.ndarray | None = None


def _get_financial_embeddings() -> np.ndarray:
    global _financial_embeddings
    if _financial_embeddings is None:
        model = _get_model()
        _financial_embeddings = model.encode(FINANCIAL_ANCHORS, normalize_embeddings=True)
    return _financial_embeddings


def _get_non_financial_embeddings() -> np.ndarray:
    global _non_financial_embeddings
    if _non_financial_embeddings is None:
        model = _get_model()
        _non_financial_embeddings = model.encode(NON_FINANCIAL_ANCHORS, normalize_embeddings=True)
    return _non_financial_embeddings


def is_financial_query(query: str) -> bool:
    """Return True if the query is related to personal finance.

    Stage 1 (cheap): lexical + fuzzy match against FINANCIAL_KEYWORDS. Catches
    obvious cases and typos like 'delta exepnses' that MiniLM scores below
    threshold. Short-circuits before loading the embedding model.

    Stage 2 (embedding fallback): two-sided cosine similarity vs financial
    and non-financial anchor sets. Only runs when the lexical pass misses.
    """
    if _lexical_or_fuzzy_financial_hit(query):
        return True

    model = _get_model()
    query_embedding = model.encode([query], normalize_embeddings=True)[0]

    fin_sims = np.dot(_get_financial_embeddings(), query_embedding)
    non_fin_sims = np.dot(_get_non_financial_embeddings(), query_embedding)

    max_fin = float(np.max(fin_sims))
    max_non_fin = float(np.max(non_fin_sims))

    # Financial if closer to financial anchors or above threshold
    if max_fin >= SIMILARITY_THRESHOLD and max_fin > max_non_fin:
        return True
    return False


# ── Aggregation intent ────────────────────────────────────────────────────
# Questions like "how much did I spend on restaurants in the past 60 days"
# must be answered by SQL, not by the LLM summing retrieval hits. This
# stays lexical on purpose: no extra model calls, identical behavior on
# every provider, works airgapped.

@dataclass(frozen=True)
class AggregationIntent:
    operation: str          # sum | count | average
    subject: str | None     # raw category phrase, None means all categories
    window_start: date      # inclusive
    window_end: date        # exclusive


_OP_PATTERNS = [
    ("average", r"\baverage\b|\bavg\b|\bmean\b"),
    ("count", r"\bhow many\b|\bnumber of\b|\bcount\b"),
    ("sum", r"\bhow much\b|\btotal\b|\bspent\b|\bspend\b|\bsum\b"),
]

_SUBJECT_RE = re.compile(
    r"\b(?:on|for|at)\s+([a-z][a-z '&-]{2,30}?)"
    r"(?:\s+(?:in|during|over|the|this|last|past|since)\b|[?.!,]|$)")

# "average grocery spend", "total restaurant expenses": subject sits between
# the operation word and a spend noun, no preposition involved.
_SUBJECT_INLINE_RE = re.compile(
    r"\b(?:average|avg|mean|total)\s+([a-z][a-z '&-]{2,30}?)\s+"
    r"(?:spend|spending|expenses?|purchases?|costs?)\b")

_WINDOW_PATTERNS = [
    (re.compile(r"(?:past|last)\s+(\d+)\s+days?"), "days"),
    (re.compile(r"(?:past|last)\s+(\d+)\s+weeks?"), "weeks"),
    (re.compile(r"(?:past|last)\s+(\d+)\s+months?"), "months"),
    (re.compile(r"this\s+month"), "this_month"),
    (re.compile(r"last\s+month"), "last_month"),
    (re.compile(r"this\s+year"), "this_year"),
    (re.compile(r"since\s+([a-z]+)"), "since_month"),
]

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}


def _resolve_window(q: str, today: date) -> tuple[date, date] | None:
    for rx, kind in _WINDOW_PATTERNS:
        m = rx.search(q)
        if not m:
            continue
        end = today + timedelta(days=1)
        if kind == "days":
            return end - timedelta(days=int(m.group(1)) + 1), end
        if kind == "weeks":
            return end - timedelta(weeks=int(m.group(1)), days=1), end
        if kind == "months":
            return end - timedelta(days=30 * int(m.group(1)) + 1), end
        if kind == "this_month":
            start = today.replace(day=1)
            nxt = (start.replace(year=start.year + 1, month=1)
                   if start.month == 12
                   else start.replace(month=start.month + 1))
            return start, nxt
        if kind == "last_month":
            first_this = today.replace(day=1)
            last_start = (first_this.replace(year=first_this.year - 1, month=12)
                          if first_this.month == 1
                          else first_this.replace(month=first_this.month - 1))
            return last_start, first_this
        if kind == "this_year":
            return today.replace(month=1, day=1), end
        if kind == "since_month":
            mon = _MONTHS.get(m.group(1))
            if mon:
                yr = today.year if mon <= today.month else today.year - 1
                return date(yr, mon, 1), end
    return None


def extract_window(query: str,
                   today: date | None = None) -> tuple[date, date] | None:
    """Public helper: the date window named in a question, or None. Used by
    the RAG path to push a date predicate even for non-aggregation asks."""
    return _resolve_window(query.lower(), today or date.today())


def classify_aggregation(query: str,
                         today: date | None = None) -> AggregationIntent | None:
    q = query.lower().strip()
    today = today or date.today()
    op = next((name for name, rx in _OP_PATTERNS if re.search(rx, q)), None)
    if op is None:
        return None
    # "show / list / recent / why" questions are retrieval or coach asks,
    # not aggregation, even when a spend word appears.
    if re.search(r"\bshow\b|\blist\b|\brecent\b|\bwhy\b|\bwhich\b", q):
        return None
    window = _resolve_window(q, today)
    if window is None:
        end = today + timedelta(days=1)
        window = (end - timedelta(days=90), end)
    m = _SUBJECT_RE.search(q) or _SUBJECT_INLINE_RE.search(q)
    subject = m.group(1).strip() if m else None
    return AggregationIntent(operation=op, subject=subject,
                             window_start=window[0], window_end=window[1])
