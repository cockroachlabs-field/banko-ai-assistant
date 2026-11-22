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

from ..config.settings import get_config
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
    # The subject is a word the total could not place ("coffee total", "how
    # much do I have"), not one named behind on, for or at. A category or a
    # merchant of the user's still gets its total from SQL; anything else is
    # the planner's, never a total over everything.
    loose: bool = False


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
# Only after in, during or for: "how much may I spend" and "the march" name
# no month. "total expenses for october" said it could not find a category
# called october. After for, the month in full: "what do I pay for apr" asks
# about a rate.
_IN_MONTH = (r"\b(?:in|during)\s+(?:the\s+month\s+of\s+)?("
             + "|".join(sorted(_MONTH_NAMES, key=len, reverse=True)) + r")\.?")
_FOR_MONTH = r"\bfor\s+(?:the\s+month\s+of\s+)?(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")"
_YEAR = r"((?:19|20)\d{2})(?![\w'-])"

_WINDOW_PATTERNS = [
    # "in October 2025", "in October" (the latest one that has begun), "in 2025".
    (re.compile(_IN_MONTH + r"(?:\s+of)?[\s,]+" + _YEAR), "month_year"),
    (re.compile(_IN_MONTH + r"(?![\w'-])"), "month"),
    (re.compile(_FOR_MONTH + r"(?:\s+of)?[\s,]+" + _YEAR), "month_year"),
    (re.compile(_FOR_MONTH + r"(?![\w'-])"), "month"),
    (re.compile(r"\b(?:in|during|for|throughout)\s+(?:the\s+year\s+)?" + _YEAR), "year"),
    (re.compile(rf"(?:\bthese\s+)?(?:past|last|previous)\s+({_COUNT})\s+days?"), "days"),
    (re.compile(rf"(?:\bthese\s+)?(?:past|last|previous)\s+({_COUNT})\s+weeks?"), "weeks"),
    (re.compile(rf"(?:\bthese\s+)?(?:past|last|previous)\s+({_COUNT})\s+months?"), "months"),
    # "this month to date" is this month and "this past month" the 31 days
    # behind today: both stand before "this month" here.
    (re.compile(r"\b(?:this\s+)?year[\s-]to[\s-]date\b|\bytd\b"), "this_year"),
    (re.compile(r"\b(?:this\s+)?month[\s-]to[\s-]date\b|\bmtd\b"), "this_month"),
    (re.compile(r"\b(?:this\s+)?week[\s-]to[\s-]date\b|\bwtd\b"), "this_week"),
    # "The past week", "the last month", "over the last year" roll back from
    # today, like "the past 7 days". "Last week", "last month" and "last
    # year" alone are the calendar ones.
    (re.compile(r"\b(?:this\s+)?past\s+week\b|(?<=\bthe\s)last\s+week\b"), "past_week"),
    (re.compile(r"\b(?:this\s+)?past\s+month\b|(?<=\bthe\s)last\s+month\b"), "past_month"),
    (re.compile(r"\b(?:this\s+)?past\s+year\b|(?<=\bthe\s)last\s+year\b"), "past_year"),
    (re.compile(r"(?:this|\bcurrent)\s+(?:whole\s+|entire\s+)?month"), "this_month"),
    (re.compile(r"last\s+month"), "last_month"),
    (re.compile(r"(?:this|\bcurrent)\s+(?:whole\s+|entire\s+|calendar\s+)?year"), "this_year"),
    (re.compile(r"\blast\s+year\b"), "last_year"),
    # Calendar weeks start on Monday.
    (re.compile(r"\b(?:this|current)\s+(?:whole\s+|entire\s+)?week\b"), "this_week"),
    (re.compile(r"\blast\s+week\b"), "last_week"),
    # "my total for the year": the one that is running. Only after for:
    # "during the week" means the weekdays.
    (re.compile(r"\bfor\s+the\s+month\b"), "this_month"),
    (re.compile(r"\bfor\s+the\s+week\b"), "this_week"),
    (re.compile(r"\bfor\s+the\s+year\b"), "this_year"),
    (re.compile(r"\btodays?\b"), "today"),
    (re.compile(r"\byesterdays?\b"), "yesterday"),
    # "since january", "since january 1": from the first of that month.
    (re.compile(r"since\s+([a-z]+)(?:\s+(?:the\s+)?(?:1st|1|first)\b)?"), "since_month"),
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
        try:
            window = _window_of(kind, m, today)
        except (OverflowError, ValueError):  # "the last 1,000,000 days": further back than dates go
            window = None
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
    if kind in ("days", "weeks", "months"):
        n = int(m.group(1)) if m.group(1).isdigit() else _NUMBER_WORDS[m.group(1)]  # "the last two weeks"
        if kind == "days":
            return end - timedelta(days=n + 1), end
        if kind == "weeks":
            return end - timedelta(weeks=n, days=1), end
        return end - timedelta(days=30 * n + 1), end
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
    if kind == "past_year":
        return end - timedelta(days=365), end  # a day more, and the first of a month falls into it twice
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


# ── What a total is made of ───────────────────────────────────────────────
# A total answers a question only when it accounts for every word of it, as
# the size route does. A total's own words name nothing: who is asking and
# the asking ("can you tell me", "please"), the counting ("how much",
# "total", "average"), the spending and its rows ("did I spend",
# "purchases"). The dates are read and cut out. Any word left over is what
# the total is of ("coffee total", "how much is my rent") or something a
# total of spending cannot answer ("how much do I have", "how much can I
# spend", "per month", "last weekend"). Dropped, as it used to be, the sum
# ran over everything.
#
# So this list is kept to words that name nothing and change nothing. A word
# missing here sends its question to the planner, which costs model calls; a
# word too many answers a question for all spending. Left out on purpose:
# what points at something (it, that, those, there, mine), what turns a
# question around (not, only, more, over, than), what asks what is allowed,
# needed or yet to come (can, will, should, need, going to), what names a
# place or a kind (store, merchant, category), every word of time (a window
# that was read is gone by now, so "last", "month" or "weekly" left over is
# a window nobody read, or a rate) and every number. Every word here is
# needed by a row of tests/fixtures/plain_totals.tsv or by a total with a
# subject (tests/test_totals_subject.py checks that): one nothing needs has
# no business in it.
_TOTAL_WORDS = frozenset("""
    i me my myself we our the a
    of in on at for to from by with about and but so
    do does did done is are was were been being am
    how howmuch what whats whatd whatre
    much many total totaled totalled sum amount amt number
    count average avg overall altogether together combined cumulative grand running
    final full complete entire whole exact precise actual real rough approximate ballpark figure tally value
    spend spent spended spending pay paid paying payed buy bought purchased
    charged billed cost costs expense expenses expenditure expenditures expend expended outgoings outlay
    purchase purchases transaction transactions charges
    money dollar dollars bucks quid usd
    went gone manage
    all everything anything something stuff times
    just also actually exactly honestly simply totally even really
    already again roughly approximately around personally alone like
    please pls plz kindly thanks thx ty cheers ok okay alright right well hey hi hello yo hmm um uh oh ah
    wow whoa yikes dang damn omg geez jeez heck lol btw idk anyway though tho sorry sure yes
    yeah great cool perfect nice awesome wait seriously
    im id breakdown
    posted happened occurred recorded processed cleared
""".split())
# "how much have I spent" has spending in it and "how many transactions did
# I have" has the rows; "how much do I have" asks for a balance. "how much
# did I get charged" is spending; "how much did I get" is not. "how many
# purchases did I make" counts purchases; "how much did I make" asks for
# income. A payment is as often a card payment as a purchase, so it is no
# row here.
_HAVE = frozenset("have has had having hav ive youve weve".split())
_GET = frozenset("get gets got getting gotten".split())
_MAKE = frozenset("make makes made making".split())
_SPENT = frozenset(
    "spent spended paid payed bought made cost been gone charged billed purchased done expended totaled totalled".split())
_ROWS = frozenset("expense expenses purchase purchases transaction transactions charges".split())
_A_WORD = re.compile(r"[^\W\d_]+|\d+(?:[.,]\d+)*")
# Time, wherever a window did not take it: a rate ("per month", "weekly"), a
# window nobody read ("last weekend", "3 months ago"), a pointer ("that").
_TIME = frozenset(
    "day days daily week weeks weekly weekend weekends weekday weekdays fortnight month months monthly quarter "
    "quarters quarterly year years yearly annual annually today yesterday tomorrow tonight morning afternoon evening "
    "night nights noon midnight time period date ago last past next previous this that since until per each every "
    "monday tuesday wednesday thursday friday saturday sunday".split()) | frozenset(_MONTH_NAMES)
# Behind on, for or at these are no subject either: "how much did I pay for that".
_POINTERS = frozenset("it its they them their these those there here one ones".split())
# Behind on, for or at these name nothing and ask nothing: "at all", "on
# average", "on everything", "for me". Any other word there is a name by its
# place, a total's own word or not: "at the dollar store".
_FILLER = frozenset(
    "the my our a an all everything anything something stuff things total average overall whole so far me what "
    "spend spending purchase purchases expense expenses transaction transactions charges costs money".split())
# What no name ends with, though it stands where one would ("on coffee
# though", "on rent for the past year"), and what turns a subject into a
# comparison or an exception ("on coffee compared to last month", "on
# everything but amazon").
_NO_NAMES_END = frozenset(
    "though tho overall altogether please pls plz thanks anyway btw again too also lol exactly roughly approximately "
    "really actually honestly alone combined together total "
    "for of on at in to from by with within during over through across back and or but as".split())
_TURNS = frozenset("compared versus vs than against excluding including without except besides not only but".split())
# "total of my coffee purchases", "total number of coffee purchases", "total value of my purchases".
_INLINE_LEAD = re.compile(
    r"^(?:(?:(?:number|value|amount|sum|cost|count|figure)\s+)?of(?:\s+|$))?(?:all(?:\s+|$))?(?:(?:my|the|our|a|an)(?:\s+|$))?")
# No total is longer than this, and nothing below runs over a page of text.
_LONGEST = 300

# Said before the dates are looked for. "So far" and its kin name no window:
# a total runs up to now anyway, and "up to today" is not today alone.
_TO_NOW = re.compile(
    r"\b(?:so|thus)\s+far\b|\bup\s+to\s+date\b|\bas\s+of\s+(?:right\s+now|now|today|yet)\b"
    r"|(?<!\byear)(?<!\bmonth)(?<!\bweek)(?<!\bquarter)[\s-]to[\s-]date\b"
    r"|\bby\s+now\b|\byet\b|\bthese\s+days\b|\bnowadays\b|\blately\b|\brecently\b|\bof\s+late\b"
    r"|\bat\s+the\s+moment\b|\bright\s+now\b|\bat\s+this\s+point\b|\bcurrently\b|\bpresently\b|\bnow\b"
    r"|\bcurrent\b(?!\s+(?:calendar\s+)?(?:month|week|year)\b)")
# "up to today", "until now": the same, unless a "from" before it makes it
# the far end of a span ("from last month to today").
_UP_TO_NOW = re.compile(r"\b(?:up\s+)?(?:to|until|till|til)\s+(?:now|today|this\s+point)\b")
# A greeting, a word before the question, a tag after it. A word that could
# as well begin a name ("next", "first", "good") is an opener only with a
# comma or a colon behind it, and "mate" is a tag only where nobody's mate
# is meant ("how much did I pay my mate").
_MATE = r"(?:mate|man|dude|bro|buddy|pal)"
_HELLO = re.compile(
    r"\bgood\s+(?:morning|afternoon|evening|day|night)\b|\b(?:hey|hi|hello)\s+there\b|\bthank\s+(?:you|u)\b"
    r"|\bby\s+the\s+way\b"
    r"|\bout\s+of\s+curiosity\b|\b(?:a\s+)?(?:quick|last|one\s+more|one\s+last|another|one)\s+question\b"
    r"|^\W*(?:(?:so|well|ok|okay|and|alright|right)[\s,]+)*then\b[\s,]*"
    r"|^\W*(?:(?:and\s+)?(?:got\s+it|i\s+see|one\s+more(?:\s+thing)?|next(?:\s+question)?|no|good|fine|interesting"
    r"|wait|first|second|seriously|listen|look|question|morning|afternoon|evening)\s*[,:;.!-]+\s*)+"
    rf"|^\W*(?:hey|hi|hello|yo|ok|okay|thanks|so)[\s,]+{_MATE}\b[\s,]*"
    rf"|(?<!\bmy)(?<!\bthe)(?<!\ba)(?<!\bour)(?<!\bpay)[\s,]+{_MATE}\W*$"
    r"|,\s*(?:right|huh|eh|no|correct|you\s+know|period|full\s+stop)\W*$|\btbh\b"
    r"|(?<=[a-z])\.(?:com|net|org|co|io)\b")
# The card of a total of everything lists its largest categories already.
_BY_CATEGORY = re.compile(
    r"\b(?:by|per|for\s+each|in\s+each|in\s+every|across\s+all|in\s+all|across|each|every|all)\s+(?:my\s+)?"
    r"(?:category|categories)\b")
# Asking to be told is no part of what is asked: "can you tell me how much I
# spent", "I'd like to know", "can I get my total", "..., do you know?",
# "I'm not sure how much I spent". Anywhere else "can" asks what is allowed
# and "need" what is needed, and neither is a total. Read with the
# apostrophes gone ("id like").
_FIND_OUT = (r"(?:check|see|find\s+out|figure\s+out|work\s+out|look\s+up|calculate|compute|determine|understand|know"
             r"|add\s+up|sum\s+up|tally\s+up|tally|total\s+up|total|sum|count\s+up|count)")
_ASKING = re.compile(
    r"\b(?:can|could|would|will)\s+(?:you|u)\s+(?:(?:please|kindly|just|quickly)\s+)*"
    rf"(?:(?:tell|show|give|remind|let|get)\s+me(?:\s+know)?|help\s+me(?:\s+{_FIND_OUT})?|{_FIND_OUT})\b"
    rf"|\bhelp\s+me\s+{_FIND_OUT}\b|\bwould\s+you\s+mind\s+telling\s+me\b"
    r"|\b(?:do|did|would)\s+(?:you|u)\s+(?:happen\s+to\s+)?know\b|\b(?:you\s+)?know\b(?=\s+(?:how|what)\b)"
    r"|\bdo\s+(?:you|u)\s+have\b(?=\s+(?:my|the|a)\b)"
    r"|\bif\s+you\s+(?:know|dont\s+mind|can|could|would|will|please)\b"
    # "can I get my total": what follows is the total. "how much can I get" asks what is on offer.
    r"|\b(?:can|could|may)\s+i\s+(?:please\s+)?(?:get|see|have|know|ask)\b(?=\s+(?:my|the|a|an|how|what)\b)"
    r"|\b(?:i\s+(?:just\s+)?(?:want|need|wanna)|i\s+would\s+(?:like|love)|id\s+(?:like|love))(?:\s+to)?"
    r"\s+(?:know|see|find\s+out|check|ask)\b"
    r"|\b(?:i\s+(?:just\s+)?(?:want|need)|i\s+would\s+like|id\s+like)\s+"
    r"(?=(?:the|my|a)\s+(?:total|sum|amount|number|count|average)\b)"
    r"|\b(?:i\s+)?(?:am\s+|was\s+)?(?:just\s+)?(?:curious|wondering|wonder)(?:\s+about)?\b"
    r"|\b(?:i\s+am\s+|im\s+)?not\s+sure\b|\b(?:i\s+)?(?:dont|do\s+not)\s+know\b|\b(?:i\s+have\s+)?no\s+idea\b"
    rf"|\bi\s+(?:forgot|forget)\b|\b(?:i\s+am\s+|im\s+)?(?:just\s+)?trying\s+to\s+{_FIND_OUT}\b|\b(?:just\s+)?checking\b"
    r"|\b(?:tell|remind|give|get)\s+me\b|\b(?:let\s+me|lemme)\s+(?:know|see)\b|\bany\s+idea\b"
    # Told to work it out: only where the question begins.
    rf"|^\W*(?:(?:ok|okay|please|pls|hey|hi|so|well|and|now|then)[\s,]+)*"
    rf"(?:{_FIND_OUT}|get|pull\s+up|pull|bring\s+up|fetch|grab|display)\b")
# Turns of phrase that are a total's own though one of their words is not.
_OWN_PHRASES = re.compile(
    r"\bbreak(?:s|ing)?\s+(?:it\s+)?down\b|\bbroken\s+down\b"
    r"|\bin\s+(?:total|all)\b|\ball\s+in\s+all\b|\ball\s+told\b|\bin\s+the\s+end\b|\bat\s+all\b"
    r"|\bof\s+course\b|\bin\s+general\b|\bon\s+the\s+whole\b|\bgive\s+or\s+take\b"
    # Only set off at the end: "did I spend more or less this month" compares, "after all expenses" subtracts.
    r"|,\s*more\s+or\s+less(?=\W*$)|\bafter\s+all(?=\W*$)"
    r"|\bacross\s+(?:everything|the\s+board)\b|\bincluding\s+everything\b|\beverything\s+included\b"
    r"|\ball\s+(?:inclusive|included)\b|\bout\s+of\s+pocket\b"
    r"|\b(?:across|on|in|from)\s+all\s+(?:of\s+)?(?:my\s+)?(?:accounts|cards)\b"
    r"|\b(?:add|adds|added|adding|sum|sums|summed|summing|tally|tallied|tallying|count|counting|total|totals"
    r"|totaled|totalled)\s+up(?:\s+to)?\b"
    r"|\b(?:end|ends|ended|ending|wind|winds|winding|wound)\s+up\b(?!\s+with\b)"
    r"|\b(?:come|comes|came|coming)\s+to\b(?!\s+(?:me|us)\b)"
    r"|\b(?:went|gone|go|goes)\s+through\b(?!\s+\w)|\btook\s+place\b"
    # "how many times did I eat at chipotle": whatever was done there, it counts the purchases.
    r"|\bhow\s+many\s+times\s+(?:did|have|do|does)\s+(?:i|we)\s+[a-z]+(?=\s+at\b)"
    r"|\bshopping\s+(?=(?:for|at)\b)")
# A total's own only beside its rows: "how many transactions in my account",
# "how many purchases are there". "how much is in my account" and "how much
# is there" ask for a balance.
_WHERE_ROWS_ARE = re.compile(
    r"\bin\s+my\s+(?:account|history|records?)\b|\b(?:is|are|was|were)\s+there\b|\b(?:have|has)\s+there\s+been\b")
# Money going out is money spent.
_WENT_OUT = re.compile(
    r"\b(?:went|goes|gone|came|come|comes)\s+out\s+of\s+my\s+accounts?\b|\bleft\s+my\s+accounts?\b"
    r"|\b(?:went|goes|gone)\s+out\b|\b(?:paid|pay|pays|paying)\s+out\b")
# "how much did I have to pay" is what was paid; "how much do I have to pay" is what is owed.
_HAD_TO = re.compile(r"\b(did\s+(?:i|we))\s+have\s+to\s+(?=(?:pay|spend)\b)|\b((?:have|has)\s+(?:i|we))\s+had\s+to\s+(?:pay|spend)\b")
# Money coming in: "how much was I paid", "how much did uber pay me", "my pay".
_PAID_TO_ME = re.compile(
    r"\b(?:was|were|am|are)\s+(?:i|we)\s+(?:paid|payed)\b|\b(?:i|we)\s+(?:was|were|am|are|got|get)\s+(?:paid|payed)\b"
    r"|\bbeen\s+(?:paid|payed)\b|\b(?:get|gets|got|getting)\s+(?:paid|payed)\b"
    r"|\b(?:paid|payed|owed|due|coming)\s+to\s+(?:me|us)\b|\b(?:my|our|take\s+home|total)\s+pay\b"
    r"|\b(?:pays|paid|payed|gave|gives|owes|owed)\s+(?:me|us)\b"
    r"|\b(?:did|does|do|has|have|will|would|can|could|should)\s+(?!(?:you|u|i|we)\b)[a-z]+(?:\s+[a-z]+){0,3}?"
    r"\s+(?:pay|give|owe)\s+(?:me|us)\b")
# A way of paying, not a place.
_WALLET = re.compile(r"\b(?:apple|google|amazon|samsung|shop)\s+pay\b")
# No total at all, whatever its other words: a comparison ("did I spend as
# much as last month"), an exception ("everything but rent"), a judgement
# ("how much is too much", "is it ok to spend"), one purchase ("how much was
# the charge") and a question about a word ("what does total mean").
_NOT_A_TOTAL = re.compile(
    r"\bas\s+(?:much|many)\b|\bwhat\s+(?:does|do|did)\b.*\bmean\b|\b(?:everything|all|anything)\s+but\b"
    r"|\btoo\s+(?:much|many)\b|\b(?:is|was|be)\s+(?:it\s+)?(?:ok|okay|alright|fine|safe|reasonable)\b"
    r"|\b(?:was|is)\s+(?:the|my|that|this)\s+(?:purchase|charge|transaction|expense)\b"
    r"(?!\s+(?:total|count|history|amount|sum|average)\b)")
# "how much have I spend": the grammar slips, the question does not.
_HAVE_SPEND = re.compile(r"\b((?:have|has)\s+(?:i|we)|(?:i|we)\s+have)\s+(?:spend|pay|buy)\b")
# What only leads up to the dates. Over, across and through do so before
# "the" ("over the last 30 days"); "how much did I go over this month" asks
# about a limit. "Back in october" is the dates (the window begins at "in",
# so classify_aggregation takes that "back" off); "pay back last month" is
# not.
_BEFORE_DATES = re.compile(
    r"(?:\b(?:over|in|during)\s+the\s+course\s+of\s+(?:the\s+)?"
    r"|(?:\b(?:in|for|during|over|throughout)\s+)?(?:\bthe\s+)?\b(?:whole|entire|all)\s+of\s+(?:the\s+)?"
    r"|\b(?:in|for|from|during|within|throughout)\s+(?:the\s+)?"
    r"|\b(?:over|across|through)\s+the\s+|\bback\s+in\s+(?:the\s+)?|\b(?:only|just|earlier)\s+|\bthe\s+)?$")
# The dates as one end of a span, or as a bound: "since last month", "as of
# yesterday", "through today", "up to last month". Not the window that was
# asked for, so a word is kept for it.
# "what did my total come to last month" and "how many went through this
# month" are turns of phrase, not bounds.
_A_BOUND = re.compile(
    r"\b(?:since|thru|until|till|til|before|after|by|as\s+of)\s+(?:the\s+)?$"
    r"|(?<!\bwent)(?<!\bgone)(?<!\bgo)(?<!\bgoes)\s+through\s+(?:the\s+)?$"
    r"|(?<!\bcome)(?<!\bcame)(?<!\bcomes)(?<!\bup)(?<!\bamount)(?<!\bamounts)(?<!\bamounted)\s+to\s+(?:the\s+)?$")
_FAR_END = re.compile(r"^\s*(?:to|until|till|til|through|thru|up\s+to)\b")
# The words people mistype most in a total ("how much did i spnd", "how mcuh").
# A slip of one of them is that word, not a subject: one letter added,
# dropped or changed behind the same first two letters, or two neighbours
# swapped behind the same first one. "send" is not "spend", "expensive" not
# "expenses", and the real words below are themselves.
_OFTEN_MISTYPED = ("spend", "spent", "spending", "expense", "expenses", "purchase", "purchases", "transaction",
                   "transactions", "total", "much")
_NOT_A_SLIP = frozenset(
    "sped speed spendy spelt speeding expanse expanses expensed purchaser purchasers tonal must munch mulch".split())
_categories: tuple[int, frozenset[str]] | None = None


def _one_slip(word: str, meant: str, swap_only: bool = False) -> bool:
    """One letter added, dropped, changed or swapped with its neighbour."""
    if word == meant or abs(len(word) - len(meant)) > 1:
        return False
    if len(word) == len(meant):
        apart = [i for i in range(len(word)) if word[i] != meant[i]]
        swapped = (len(apart) == 2 and apart[1] == apart[0] + 1
                   and word[apart[0]] == meant[apart[1]] and word[apart[1]] == meant[apart[0]])
        return swapped or (len(apart) == 1 and not swap_only)
    if swap_only:
        return False
    short, long = sorted((word, meant), key=len)
    return any(long[:i] + long[i + 1:] == short for i in range(len(long)))


def _meant(w: str) -> str | None:
    """The word of a total that w is a slip of, or None."""
    if len(w) < 4 or not w.isascii() or not w.isalpha() or w in _NOT_A_SLIP or w in _TOTAL_WORDS:
        return None
    return next((meant for meant in _OFTEN_MISTYPED
                 if w[0] == meant[0] and _one_slip(w, meant, swap_only=w[1] != meant[1])), None)


def _category_names() -> frozenset[str]:
    """The catalog's one-word names. None of them is ever a total's own
    word: a catalog of one's own (CATEGORIES_FILE) may call a category
    Payments."""
    global _categories
    from ..config.catalog import get_catalog
    catalog = get_catalog()
    if _categories is None or _categories[0] != id(catalog):
        names = {name.lower() for c in catalog.categories for name in (c.key, c.label, *c.aliases)}
        _categories = (id(catalog), frozenset(n for n in names if " " not in n and "_" not in n))
    return _categories[1]


def _a_totals_word(w: str) -> bool:
    return (w in _TOTAL_WORDS and w not in _category_names()) or _meant(w) is not None


def _plain(q: str) -> str:
    """The question with what only surrounds it taken off, before the dates
    and the subject are looked for: the assistant's name as a form of
    address, a greeting, a tag, "so far", and "by category", which the card
    of a total answers already. q comes with its spaces collapsed."""
    brand = r"\s+".join(re.escape(w) for w in get_config().brand_name.lower().split())
    name = rf"(?:{brand}|assistant)" if brand else "assistant"
    q = re.sub(rf"^\W*(?:(?:hey|hi|hello|ok|okay|yo|thanks|so)[\s,]+)?{name}(?!\w)[\s,:;.!-]*", "", q)
    # At the end only where nobody's assistant is meant: "how much do I pay my assistant".
    q = re.sub(rf"(?<!\bmy)(?<!\bthe)(?<!\ban)(?<!\ba)(?<!\bour)(?<!\byour)[\s,]+{name}\W*$", "", q)
    q = _BY_CATEGORY.sub(" ", _HELLO.sub(" ", q.replace("’", "'")))
    if not re.search(r"\bfrom\b", q):
        q = _UP_TO_NOW.sub(" ", q)
    return " ".join(_TO_NOW.sub(" ", q).split())


def _core(phrase: str, inline: bool) -> tuple[str | None, list[str]]:
    """The subject in the words behind on, for or at, or (inline) between
    total and a spend noun, and the words handed back to be accounted for
    with the rest of the question.

    Behind on, for or at the words are a name by their place, so little is
    taken off: what no name ends with ("coffee though", "rent for"), the
    words of time behind it ("coffee per month", handed back: they are no
    part of a name and not a total's own either), and whatever follows a
    word that turns the question ("coffee compared to", handed back too).
    "at best buy" keeps its "buy", and "at the dollar store" is a subject
    though every word of it is one a total calls its own. None when nothing
    is left that could name anything ("for that", "on weekends", "at all")."""
    typed = phrase.split()
    plain = [w.replace("'", "") for w in typed]  # "trader joe's" stays as typed
    lo, hi, handed = 0, len(plain), []
    turn = next((i for i, w in enumerate(plain) if w in _TURNS), None)
    if turn is not None:
        # "but" is a total's own word where it joins two sentences; here it makes an exception.
        handed, hi = ["except" if w == "but" else w for w in plain[turn:]], turn
    if inline:
        lo = len(_INLINE_LEAD.match(" ".join(plain)).group().split())  # type: ignore[union-attr]
        if turn is not None and lo > hi:
            lo = hi
    if inline and _NEVER_BETWEEN.intersection(plain[lo:hi]):
        return None, plain[lo:hi] + handed  # "average of my largest purchases": a size, not a name
    while hi > lo and (plain[hi - 1] in _NO_NAMES_END or plain[hi - 1] in _TIME):
        hi -= 1
        if plain[hi] in _TIME:
            handed.insert(0, plain[hi])
    if lo < hi and all(w in _FILLER or w in _TIME or w in _POINTERS for w in plain[lo:hi]):
        handed, hi = [w for w in plain[lo:hi] if w not in _FILLER] + handed, lo
    return (" ".join(typed[lo:hi]) or None), handed


def _unplaced(text: str) -> tuple[list[str], bool]:
    """The words that are none of a total's own, in order, and whether the
    question has rows or times in it ("how many coffee purchases", "how many
    times did I buy coffee"): a count counts purchases, nothing else."""
    text = re.sub(r"(?<=\w)['’]ve\b", " have", text)            # "how much've I spent"
    text = re.sub(r"(?<=\w)['’]ll\b", " will", text).replace("’", "").replace("'", "")
    text = _WALLET.sub(" wallet ", _PAID_TO_ME.sub(" income ", text))
    text = _NOT_A_TOTAL.sub(" than ", text)
    text = _HAVE_SPEND.sub(lambda m: f"{m.group(1)} spent", text)
    text = _HAD_TO.sub(lambda m: f"{m.group(1)} " if m.group(1) else f"{m.group(2)} spent", text)
    text = _WENT_OUT.sub(" paid ", _ASKING.sub(" ", text))
    rows = any(w in _ROWS or _meant(w) in _ROWS for w in _A_WORD.findall(text))
    if rows:
        text = _WHERE_ROWS_ARE.sub(" ", text)
    words = _A_WORD.findall(_OWN_PHRASES.sub(" ", text))
    spent = any(w in _SPENT or _meant(w) == "spent" for w in words)
    left = []
    for i, w in enumerate(words):
        if w in _HAVE:
            own = (spent or rows) and words[i + 1:i + 2] != ["to"]
        elif w in _GET:
            own = spent
        elif w in _MAKE:
            own = rows
        else:
            own = _a_totals_word(w)
        if not own:
            left.append(w)
    return left, rows or "times" in words


def extract_window(query: str,
                   today: date | None = None) -> tuple[date, date] | None:
    """Public helper: the date window named in a question, or None. Used by
    the RAG path to push a date predicate even for non-aggregation asks."""
    q = " ".join(query.lower().split())
    if not re.search(r"\bfrom\b", q):
        q = _UP_TO_NOW.sub(" ", q)  # "up to today" is not today alone
    return _resolve_window(_TO_NOW.sub(" ", q), today or clock.today())


def find_window(query: str, today: date | None = None) -> tuple[date, date, tuple[int, int]] | None:
    """Public helper: the date window a question names (end exclusive) and
    where in the lowercased text it names it, or None. The answer cache cuts
    the dates out of a question with it and compares what is left."""
    return _find_window(query.lower(), today or clock.today())


def _operation(q: str) -> tuple[str, str] | None:
    """What a question asks to have worked out (sum, count, average) and the
    question without "I mean", which is not the mean. None when it asks for
    nothing of the kind, or to be shown rows."""
    q = re.sub(r"\bi\s+mean\b[\s,]*", "", q)
    op = next((name for name, rx in _OP_PATTERNS if re.search(rx, q)), None)
    if op is None:
        return None
    if op == "count" and re.search(r"\bhow\s+many\s+(?:dollars|bucks|quid|usd)\b", q):
        op = "sum"  # "how many dollars did I spend" counts dollars, not purchases
    # "show / list / recent / why" questions are retrieval or coach asks,
    # not aggregation, even when a spend word appears.
    if re.search(r"\bshow\b|\blist\b|\brecent\b|\bwhy\b|\bwhich\b", q):
        return None
    return op, q


def _total_of(q: str, today: date) -> tuple[tuple[date, date], str | None, list[str], bool]:
    """A total's question taken apart: its window, the subject behind on, for
    or at (or between total and a spend noun), the words that are none of a
    total's own, and whether it has rows or times to count."""
    q = _plain(q)
    named = _find_window(q, today)
    if named is None:
        end = today + timedelta(days=1)
        window, rest = (end - timedelta(days=90), end), q
    else:
        window = named[:2]
        # The dates are no part of the subject: "total expenses for october"
        # is not a total "for october". A comma stands where they were, so a
        # subject before them still ends there, and a possessive behind them
        # goes with them ("last month's total", "this months total").
        head, tail = q[:named[2][0]], re.sub(r"^(?:['’]?s\b)?(?:\s*only\b)?", "", q[named[2][1]:])
        if q[named[2][0]:].startswith(("in ", "during ")):
            head = re.sub(r"\bback\s+$", "", head)  # "back in october"
        if _A_BOUND.search(head) or _FAR_END.match(tail):
            # One end of a span ("from last month to today") or a bound ("as
            # of yesterday"): not the dates that were asked for.
            head = f"{_A_BOUND.sub('', head)} since "
        rest = f"{head[:_BEFORE_DATES.search(head).start()]} , {tail}"  # type: ignore[union-attr]
    found = _SUBJECT_RE.search(rest) or _SUBJECT_INLINE_RE.search(rest)
    subject, outside = None, rest
    if found:
        subject, handed = _core(found.group(1), inline=found.re is _SUBJECT_INLINE_RE)
        outside = f"{rest[:found.start(1)]} {' '.join(handed)} {rest[found.end(1):]}"
    left, counted = _unplaced(outside)
    return window, subject, left, counted


def more_than_a_total(query: str, today: date | None = None) -> bool:
    """A total with its subject behind on, for or at, and more asked than
    the total of it: "how much can I spend on groceries", "how much do I
    spend at starbucks a month". classify_aggregation declines such a
    question, and so must whoever would answer for the subject alone."""
    q = " ".join(query.lower().split())
    found = _operation(q) if len(q) <= _LONGEST else None
    if found is None:
        return False
    _, subject, left, _ = _total_of(found[1], today or clock.today())
    return subject is not None and bool(left)


def classify_aggregation(query: str,
                         today: date | None = None,
                         has_history: bool = False) -> AggregationIntent | None:
    """has_history says the thread already has turns: "the biggest purchase"
    with no window or place of its own then leans on them."""
    q = " ".join(query.lower().split())
    if len(q) > _LONGEST:
        return None
    today = today or clock.today()
    size = _size_question(q, today, has_history)
    if size is not None:
        return size
    found = _operation(q)
    if found is None:
        return None
    op, q = found
    window, subject, left, counted = _total_of(q, today)
    if subject is not None:
        # Behind on, for or at it is the subject by its place. With a word
        # besides that no total places ("how much can I spend on groceries",
        # "on coffee per month"), more is asked than a total of it.
        return None if left else AggregationIntent(op, subject, window[0], window[1])
    if op == "count" and not counted:
        return None  # "how many subscriptions am I paying for", "how many stores": things, not purchases
    if not left:
        return AggregationIntent(op, None, window[0], window[1])
    return AggregationIntent(op, " ".join(left), window[0], window[1], loose=True)
