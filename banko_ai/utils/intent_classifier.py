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

from . import clock

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
    operation: str          # sum | count | average | largest | smallest
    subject: str | None     # raw category phrase, None means all categories
    window_start: date      # inclusive
    window_end: date        # exclusive
    limit: int = 1          # largest and smallest: how many purchases were asked for


_OP_PATTERNS = [
    ("average", r"\baverage\b|\bavg\b|\bmean\b"),
    ("count", r"\bhow many\b|\bnumber of\b|\bcount\b"),
    ("sum", r"\bhow much\b|\btotal\b|\bspent\b|\bspend\b|\bsum\b"),
]

# "my biggest purchase last month", "top 5 biggest purchases this year", "the
# most expensive restaurant purchase I made in October": the largest purchase
# is a fact SQL has. The planner's tools return the rows most like the
# wording, or the most recent, never the largest.
#
# The route takes a question only when it accounts for every word of it: an
# optional lead-in ("what was", "show me"), my or the, "top", how many, the
# size word, up to three words naming a category or a merchant, a purchase
# noun, then a place ("at Costco") and a window it can read, the window
# anywhere. One word left over (a second clause, "that month", "on it",
# "under $100", "ever") and it is not this route's question: answering the
# part it understood answers something nobody asked. Such a question stays
# where it was before: with the totals when it asks for one, with the
# planner, and the conversation it can see, otherwise.
_NUMBER_WORDS = {w: n for n, w in enumerate(
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen twenty".split(), start=1)} | {"thirty": 30, "forty": 40, "fifty": 50, "hundred": 100}
# Up to three digits: "top 100" is a count, "biggest 2025 purchases" names a year.
_COUNT = r"(?:\d{1,3}|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + ")"
_SIZE_WORD = (r"(?:biggest|largest|most\s+expensive|priciest|costliest|highest"
              r"|smallest|cheapest|least\s+expensive|lowest)")
_SMALL_WORDS = ("smallest", "cheapest", "least", "lowest")
_NAME_WORD = r"[a-z][a-z'&.-]*"
_PLACE_WORD = r"[a-z0-9][a-z0-9'&.+-]*"
_ASK = r"(?:please\s+)?(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
_LEAD = (rf"{_ASK}(?:show(?:\s+me)?|list|tell\s+me(?:\s+about)?|give\s+me|find(?:\s+me)?|get\s+me|display|name)"
         r"|what\s+(?:was|were|is|are)|what's|whats|which\s+(?:was|were|is|are)"
         # One purchase answers "how much": its amount is the card's headline.
         r"|(?P<amount>how\s+(?:much|big|large)\s+(?:was|were|is|are)"
         r"|how\s+much\s+did\s+i\s+(?:spend|pay)\s+(?:on|for))"
         r"|when\s+(?:was|were|is)|when\s+did\s+i\s+(?:make|have)"
         r"|where\s+(?:was|were|is)|where\s+did\s+i\s+(?:make|have)"
         r"|what\s+can\s+you\s+tell\s+me\s+about|what\s+do\s+you\s+know\s+about"
         r"|i\s+(?:want|need|would\s+like)\s+to\s+(?:see|know)|i'd\s+like\s+to\s+(?:see|know)|let\s+me\s+see")
# "tell me what my biggest purchase was": the verb comes after the purchase.
_LEAD_EMBEDDED = (rf"{_ASK}(?:tell|show|remind)\s+me\s+what|do\s+you\s+know\s+what"
                  r"|i\s+(?:want|need|would\s+like)\s+to\s+know\s+what|i'd\s+like\s+to\s+know\s+what")
# The nouns are the ones a row of expenses answers to. A payment can be a
# card payment, a bill is a recurring one, an expense is as often a category
# ("my biggest expense"), an order is not one yet: those stay the planner's.
_PURCHASES = (rf"(?:(?P<det>my|the|our)\s+)?(?:(?P<top>top)\s+)?(?:(?P<before>{_COUNT})\s+)?"
              rf"(?:(?P<size>{_SIZE_WORD})\s+)?(?:(?P<after>{_COUNT})\s+)?"
              rf"(?P<between>(?:{_NAME_WORD}\s+){{0,3}}?)"
              r"(?:purchase|transaction|charge)(?P<plural>s)?"
              r"(?:\s+(?:that\s+)?i(?:\s+have|'ve|\s+had)?\s+made)?")
_PLACE = (rf"(?:\s+(?:at|from|on|in|for)\s+(?:the\s+)?(?P<place>{_PLACE_WORD}(?:\s+{_PLACE_WORD}){{0,4}}))?"
          r"(?:\s+so\s+far)?(?:\s+please)?")
_SIZE_QUESTIONS = (re.compile(rf"^(?:(?:{_LEAD})\s+)?{_PURCHASES}{_PLACE}$"),
                   re.compile(rf"^(?:{_LEAD_EMBEDDED})\s+{_PURCHASES}\s+(?:was|were|is|are){_PLACE}$"))
# "in the last 30 days", "from last month": what ties a window to the rest.
_BEFORE_A_WINDOW = re.compile(r"(?:\b(?:in|during|over|for|within|from|of|throughout)\s+)?(?:\bthe\s+)?$")
_SIZE_FILLERS = frozenset("single individual overall one-off".split())


def _size_question(q: str, today: date, has_history: bool = False) -> AggregationIntent | None:
    """The intent of a question about the largest or smallest purchases that
    is nothing but that question, else None."""
    q = " ".join(re.sub(r"[?.!\s]+$", "", q.replace("’", "'").replace(",", " ")).split())
    found = _find_window(q, today)
    if found is not None:
        (i, j), window = found[2], found[:2]
        if (i and q[i - 1].isalnum()) or (j < len(q) and q[j].isalnum()):
            return None
        # The window is read: what is left has to be the question about purchases.
        before, after = _BEFORE_A_WINDOW.sub("", q[:i]), re.sub(r"^ so far\b", "", q[j:])
        q = " ".join(f"{before} {after}".split())
    else:
        end = today + timedelta(days=1)
        window = (end - timedelta(days=90), end)
    m = next((m for m in (rx.match(q) for rx in _SIZE_QUESTIONS) if m), None)
    if m is None or not (m.group("top") or m.group("size")):
        return None
    between = [w for w in m.group("between").split() if w not in _SIZE_FILLERS]
    place = (m.group("place") or "").split()
    names = between + place
    if ((between and place) or _NEVER_A_NAME.intersection(names) or _NEVER_BETWEEN.intersection(between)
            or any(w.startswith("non-") for w in names)):
        return None     # a category and a place at once, or a word that names neither
    subject = " ".join(names) or None
    asked = [n for n in (m.group("before"), m.group("after")) if n]
    if len(asked) > 1 or (between and not asked and not m.group("size")):
        return None     # "Top Golf purchases": top is as likely the start of a name
    if has_history and m.group("det") == "the" and not (found or subject or asked):
        return None     # "the biggest purchase", mid-conversation: the biggest of what was just shown
    limit = (int(asked[0]) if asked[0].isdigit() else _NUMBER_WORDS[asked[0]]) if asked else (
        3 if m.group("plural") else 1)
    if limit < 1 or (limit > 1 and m.groupdict().get("amount")):
        return None     # "how much were my top 5 purchases" asks for a sum the card does not give
    operation = "smallest" if (m.group("size") or "").startswith(_SMALL_WORDS) else "largest"
    return AggregationIntent(operation, subject, window[0], window[1], limit)


_SUBJECT_RE = re.compile(
    r"\b(?:on|for|at)\s+([a-z][a-z '&-]{2,30}?)"
    r"(?:\s+(?:in|during|over|the|this|last|past|since|today|yesterday)\b|[?.!,]|$)")

# "average grocery spend", "total restaurant expenses": subject sits between
# the operation word and a spend noun, no preposition involved.
_SUBJECT_INLINE_RE = re.compile(
    r"\b(?:average|avg|mean|total)\s+([a-z][a-z '&-]{2,30}?)\s+"
    r"(?:spend|spending|expenses?|purchases?|costs?)\b")

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], start=1)}
_MONTH_NAMES = {**_MONTHS, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
                "sept": 9, "oct": 10, "nov": 11, "dec": 12}
# Only after in or during: "how much may I spend" and "the march" name no month.
_IN_MONTH = r"\b(?:in|during)\s+(" + "|".join(sorted(_MONTH_NAMES, key=len, reverse=True)) + r")\.?"
_YEAR = r"((?:19|20)\d{2})(?![\w'-])"

_WINDOW_PATTERNS = [
    # "in October 2025", "in October" (the latest one that has begun), "in 2025".
    (re.compile(_IN_MONTH + r"(?:\s+of)?[\s,]+" + _YEAR), "month_year"),
    (re.compile(_IN_MONTH + r"(?![\w'-])"), "month"),
    (re.compile(r"\b(?:in|during)\s+" + _YEAR), "year"),
    (re.compile(r"(?:past|last)\s+(\d+)\s+days?"), "days"),
    (re.compile(r"(?:past|last)\s+(\d+)\s+weeks?"), "weeks"),
    (re.compile(r"(?:past|last)\s+(\d+)\s+months?"), "months"),
    (re.compile(r"this\s+month"), "this_month"),
    (re.compile(r"last\s+month"), "last_month"),
    (re.compile(r"this\s+year"), "this_year"),
    (re.compile(r"\blast\s+year\b"), "last_year"),
    # Calendar weeks start on Monday; "the past week" and "the past month"
    # roll back from today, like "the past 7 days".
    (re.compile(r"\bthis\s+week\b"), "this_week"),
    (re.compile(r"\blast\s+week\b"), "last_week"),
    (re.compile(r"\bpast\s+week\b"), "past_week"),
    (re.compile(r"\bpast\s+month\b"), "past_month"),
    (re.compile(r"\btoday\b"), "today"),
    (re.compile(r"\byesterday\b"), "yesterday"),
    (re.compile(r"since\s+([a-z]+)"), "since_month"),
]

# "for this month", "at all", "on average per month": a window or emphasis
# after the preposition, not a category, so the question covers everything.
_NOT_A_SUBJECT = frozenset(
    "the this that last past next my a an each every per day days week weeks weekend month months quarter "
    "year years time period today yesterday now all everything total average overall so far".split())
# In a size question a word like these between the size and the noun, or in
# the place, names no category and no merchant. The totals drop such a word
# and answer for everything; here that would be an answer to another
# question ("my biggest weekend purchase" is not the biggest of any day).
_NEVER_A_NAME = _NOT_A_SUBJECT | frozenset(
    "i me we you it its they them their these those there here our your mine some any both another other others "
    "same such which what who whose and or but not no nor never none without except excluding besides than "
    "on at in from for of to with by under over above below between within about into before after since until "
    "during was were is are be been did do does have has had made make bought buy paid pay received receive "
    "spent spend recent recently latest newest oldest previous first ever tomorrow currently monthly weekly "
    "weekends weekday weekdays daily yearly annual quarterly recurring regular times sum count number amount "
    "purchase purchases transaction transactions charge charges expense expenses payment payments bill bills "
    "order orders item items thing things list store stores merchant merchants category categories type types "
    "kind kinds receipt receipts account accounts".split())
# Between the size word and the noun only: another size ("biggest and
# smallest"), or how it was paid, which is not something this route filters
# on. After "at" these can be a name (Top Golf, Cash App, Big Lots).
_NEVER_BETWEEN = frozenset(
    "top biggest largest smallest cheapest priciest costliest highest lowest most least expensive big large small "
    "cheap high low more less credit debit cash card cards check online".split())


def _month_window(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)


def _find_window(q: str, today: date) -> tuple[date, date, tuple[int, int]] | None:
    """The window a question names (end exclusive) and where in the text it
    names it, or None."""
    for rx, kind in _WINDOW_PATTERNS:
        m = rx.search(q)
        if not m:
            continue
        window = _window_of(kind, m, today)
        if window is not None:
            return window[0], window[1], m.span()
    return None


def _window_of(kind: str, m: re.Match, today: date) -> tuple[date, date] | None:
    end = today + timedelta(days=1)
    if kind == "month_year":
        return _month_window(int(m.group(2)), _MONTH_NAMES[m.group(1)])
    if kind == "month":
        month = _MONTH_NAMES[m.group(1)]
        return _month_window(today.year if month <= today.month else today.year - 1, month)
    if kind == "year":
        return date(int(m.group(1)), 1, 1), date(int(m.group(1)) + 1, 1, 1)
    if kind == "days":
        return end - timedelta(days=int(m.group(1)) + 1), end
    if kind == "weeks":
        return end - timedelta(weeks=int(m.group(1)), days=1), end
    if kind == "months":
        return end - timedelta(days=30 * int(m.group(1)) + 1), end
    if kind == "this_month":
        return _month_window(today.year, today.month)
    if kind == "last_month":
        first_this = today.replace(day=1)
        last_start = (first_this.replace(year=first_this.year - 1, month=12)
                      if first_this.month == 1
                      else first_this.replace(month=first_this.month - 1))
        return last_start, first_this
    if kind == "this_year":
        return today.replace(month=1, day=1), end
    if kind == "last_year":
        return date(today.year - 1, 1, 1), date(today.year, 1, 1)
    if kind == "this_week":
        return today - timedelta(days=today.weekday()), end
    if kind == "last_week":
        monday = today - timedelta(days=today.weekday())
        return monday - timedelta(weeks=1), monday
    if kind == "past_week":
        return end - timedelta(weeks=1, days=1), end
    if kind == "past_month":
        return end - timedelta(days=31), end
    if kind == "today":
        return today, end
    if kind == "yesterday":
        return today - timedelta(days=1), today
    if kind == "since_month":
        mon = _MONTHS.get(m.group(1))
        if mon:
            yr = today.year if mon <= today.month else today.year - 1
            return date(yr, mon, 1), end
    return None


def _resolve_window(q: str, today: date) -> tuple[date, date] | None:
    found = _find_window(q, today)
    return found[:2] if found else None


def extract_window(query: str,
                   today: date | None = None) -> tuple[date, date] | None:
    """Public helper: the date window named in a question, or None. Used by
    the RAG path to push a date predicate even for non-aggregation asks."""
    return _resolve_window(query.lower(), today or clock.today())


def classify_aggregation(query: str,
                         today: date | None = None,
                         has_history: bool = False) -> AggregationIntent | None:
    """has_history says the thread already has turns: "the biggest purchase"
    with no window or place of its own then leans on them."""
    q = query.lower().strip()
    today = today or clock.today()
    size = _size_question(q, today, has_history)
    if size is not None:
        return size
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
    if subject and all(w in _NOT_A_SUBJECT for w in subject.split()):
        subject = None
    return AggregationIntent(operation=op, subject=subject,
                             window_start=window[0], window_end=window[1])
