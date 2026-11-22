"""The assistant's caches. Both are keyed on the user and the user's data
version, so a cached answer or search result never crosses users and never
outlives a write that changed the user's data (a new receipt, a
recategorization, a memory added or deleted). Hits and misses land in
cache_stats, so /cache-stats keeps its shape, written behind the request."""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import re
import threading
import time
import unicodedata
from collections import OrderedDict
from collections.abc import Callable, Iterable
from datetime import date, datetime, timezone
from typing import Any, NamedTuple

from sqlalchemy import text

from ..config.catalog import get_catalog
from ..config.settings import get_config
from ..db.engine import get_engine, routed, run_transaction
from ..db.regions import money_region, read_region, region_columns
from ..db.rls import request_url
from ..utils import clock
from ..utils.intent_classifier import extract_window, find_window
from ._sql import jsonable, region_pin
from ._vector import nearest

log = logging.getLogger("banko.caches")


def estimate_tokens(text_value: str) -> int:
    """About four characters per token; the trace calls it an estimate."""
    return len(text_value or "") // 4


# cache_stats lives in the primary region: written inline, every lookup on a
# multi-region cluster (hits included) would wait on a cross-region commit.
_stats: queue.Queue[tuple[str, str, int]] = queue.Queue()
_writer: threading.Thread | None = None
_writer_lock = threading.Lock()


def record_stat(cache_type: str, operation: str, tokens_saved: int = 0) -> None:
    global _writer
    _stats.put((cache_type, operation, int(tokens_saved)))
    with _writer_lock:
        if _writer is None or not _writer.is_alive():
            _writer = threading.Thread(target=_write_stats, name="banko-cache-stats", daemon=True)
            _writer.start()


def flush_stats() -> None:
    """Wait until every stat recorded so far is written (or given up on)."""
    _stats.join()


def _write_stats() -> None:
    while True:
        rows = [_stats.get()]
        while True:
            try:
                rows.append(_stats.get_nowait())
            except queue.Empty:
                break
        try:
            with get_engine().begin() as conn:
                conn.execute(text("""
                    INSERT INTO cache_stats (cache_type, operation, tokens_saved)
                    SELECT * FROM unnest(CAST(:t AS STRING[]), CAST(:o AS STRING[]), CAST(:n AS INT8[]))
                """), {"t": [r[0] for r in rows], "o": [r[1] for r in rows], "n": [r[2] for r in rows]})
        except Exception:  # stats never break an answer
            log.debug("cache stats not recorded", exc_info=True)
        finally:
            for _ in rows:
                _stats.task_done()


def search_key(user_id: str, version: int, payload: dict) -> str:
    body = json.dumps({"u": str(user_id), "v": int(version), "p": jsonable(payload)}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


def get_search(user_id: str, key: str, *, url: str | None = None, stat: str | None = "search") -> list | dict | None:
    """stat None: the caller looks under more than one key and counts the outcome once itself."""
    url = request_url(url)

    def fn(conn) -> list | dict | None:
        row = conn.execute(text("""
            UPDATE search_cache SET hits = hits + 1
            WHERE cache_key = :k AND user_id = CAST(:u AS UUID) AND expires_at > now()
            RETURNING results
        """), {"k": key, "u": user_id}).fetchone()
        return None if row is None else (row[0] if isinstance(row[0], (list, dict)) else json.loads(row[0]))

    results = run_transaction(fn, url=url, user_id=user_id)
    if stat:
        record_stat(stat, "hit" if results is not None else "miss")
    return results


def put_search(user_id: str, key: str, results: list | dict, *, url: str | None = None,
               stat: str | None = "search", minutes: int | None = None) -> None:
    """minutes defaults to SEARCH_CACHE_TTL_MINUTES, the life of a search
    window someone pages through; a narrative passes the answer cache's.
    stat None: a second entry for one answer, counted once already."""
    url = request_url(url)
    minutes = minutes or get_config().search_cache_ttl_minutes

    def fn(conn) -> None:
        region = money_region(conn, user_id, url)
        rcol, rval = region_columns(region)
        conn.execute(text(f"""
            UPSERT INTO search_cache ({rcol}cache_key, user_id, results, hits, created_at, expires_at)
            VALUES ({rval}:k, CAST(:u AS UUID), CAST(:r AS JSONB), 0, now(), now() + :m * INTERVAL '1 minute')
        """), {"k": key, "u": user_id, "r": json.dumps(jsonable(results)), "m": minutes, "crdb_region": region})

    run_transaction(fn, url=url, user_id=user_id)
    if stat:
        record_stat(stat, "write")


# Full-width forms (１２．９９) read as their plain ones, and the dashes and the minus sign a phone types as a
# hyphen. Not NFKC: that turns ½ into 1/2 and "1½%" into "112%".
_FULL_WIDTH = ({code: code - 0xFEE0 for code in range(0xFF01, 0xFF5F)} | {0x3000: 0x20}
               | {code: 0x2D for code in (*range(0x2010, 0x2016), 0x2212)})
_COMPARES = "<>=≤≥≠"


def normalize_question(question: str) -> str:
    """The question as the answer cache compares it: lowercase, no
    punctuation, single spaces. What tells two numbers apart stays: the
    point of 12.99 and of .99, the slash of 11/2, the hyphen of 10-20, a
    minus sign, a percent sign, and <, > and = ("$12.99" and "$1,299", "> $100"
    and "< $100" were one question each). A thousands comma goes, so $1,000
    is $1000. The marks a script writes its vowels with stay on their
    letters."""
    asked = (question or "").translate(_FULL_WIDTH).lower().replace("!=", "≠")
    kept, size = [], len(asked)

    def digit_next(i: int) -> bool:
        """The next character that is no currency sign is a digit."""
        i += 1
        while i < size and unicodedata.category(asked[i]) == "Sc":
            i += 1
        return i < size and asked[i].isdigit()

    for i, ch in enumerate(asked):
        before, after = asked[i - 1] if i else "", asked[i + 1] if i + 1 < size else ""
        if ch.isalnum() or ch == "_" or ch.isspace():
            kept.append(ch)
        elif unicodedata.category(ch)[0] == "M":
            # A vowel sign belongs to the letter before it. After anything
            # else it is an emoji's selector, and goes with the emoji.
            if before.isalpha() or (before and unicodedata.category(before)[0] == "M"):
                kept.append(ch)
        elif ch in _COMPARES:
            kept.append(f" {ch} ")
        elif ch == "%":
            if before.isdigit():
                kept.append(ch)
        elif ch == ",":
            thousands = asked[i + 1:i + 4].isdigit() and len(asked[i + 1:i + 4]) == 3 and not asked[i + 4:i + 5].isdigit()
            if before.isdigit() and after.isdigit() and not thousands:
                kept.append(ch)
        elif ch in "./:":
            if after.isdigit() and (before.isdigit() or (ch == "." and not before.isalnum())):
                kept.append(ch)  # 12.99, 11/2, 10:30, and the point of $.99
        elif ch == "-":
            if digit_next(i) and (before.isdigit() or not before.isalnum()):
                kept.append(ch)  # 10-20, 2026-11-02 and -50; "pre-tax" loses its hyphen as before
    return " ".join("".join(kept).split())


CANDIDATES = 25  # questions looked at per lookup, nearest first; one user's cache at one data version is small

# The same question in other words is one shape only: a spending category
# or one of the user's own merchants, asked for bare or as its purchases,
# optionally for some dates. "coffee", "coffee expenses", "my coffee
# spending" and "show my coffee purchases" are one question, and so are
# "starbucks" and "purchases at starbucks" for someone who has bought
# there. Like the SQL routes, the shape takes a question only when it
# accounts for every word of it. Dropping "small words" from any question was
# tried first, and no list of them was small enough: "show my savings account"
# is not "show my savings account transactions", "my last purchase" not "my
# last purchases", "did I get a refund" not "can I get a refund". A miss
# costs a model call; a wrong hit is a wrong answer.
_ROWS = frozenset("expenses spending purchases transactions charges".split())  # plural: "my last coffee purchase" is one row
# Asking to be shown. Not find, get, give or tell ("find me flights" asks to
# book), and no question: "what is my coffee spending" asks how much.
_SHOW = frozenset("show list display view".split())
# Category names that are no spending kind when they stand alone or before a
# word for purchases: things with a page of their own, and "internet
# purchases", "phone purchases" (bought online, by phone). A catalog of one's
# own (CATEGORIES_FILE) can add names like these; this knows the packaged ones.
_THINGS = frozenset("subscription subscriptions bills internet phone".split())
# What Banko keeps besides purchases, in the words a person has for it. A
# merchant with one of these in its name is asked for only in words that can
# mean nothing but the shop (_only_the_shop): "my target" may be a savings
# target, "spending on target" may ask whether the spending is on target, and
# "target spending" for the target itself. "target purchases" and "spending
# at target" are the shop.
_OWN = frozenset(
    "target targets goal goals budget budgets balance balances account accounts checking savings saving card cards "
    "credit debit cash money funds transfer transfers deposit deposits income paycheck salary payment payments "
    "refund refunds bill bills subscription subscriptions receipt receipts memory memories statement statements "
    "limit limits fee fees interest loan loans debt debts coach nudge nudges assistant dashboard platform "
    "settings".split())
_SHOP_ROWS = frozenset("purchases transactions charges".split())
_BEFORE_DATES = re.compile(r"(?:\b(?:in|for|during|over)\s+)?(?:\bthe\s+)?$")

# Words that turn a question around or point it somewhere else. Two similar
# questions that differ in one of these are two questions.
_TURNING = frozenset(
    "not no non never without except excluding only but cannot < > = ≤ ≥ ≠ on off "
    "who whom whose where when why how what which "
    "dont doesnt didnt isnt arent wasnt werent havent hasnt hadnt cant couldnt wont wouldnt shouldnt aint "
    "was were did had will would can could should might must shall "
    "under over above below less more fewer least most than between "
    "before after since until during "
    "biggest largest highest smallest lowest cheapest top bottom best worst "
    "big bigger small smaller large larger high higher low lower expensive cheap cheaper pricey costly "
    "first last next previous this past latest newest oldest recent new old early earlier late later "
    "in out to from into incoming outgoing up down sent received "
    "and or both each every per all "
    "total average sum count min max minimum maximum "
    "paid unpaid due overdue pending declined refunded refund refunds income credit debit deposit deposits "
    "withdrawal withdrawals often rarely always sometimes usually online".split())
_DATE_WORDS = frozenset(
    "january february march april may june july august september october november december "
    "jan feb mar apr jun jul aug sep sept oct nov dec "
    "monday tuesday wednesday thursday friday saturday sunday "
    "mondays tuesdays wednesdays thursdays fridays saturdays sundays mon tue tues wed thu thur thurs fri sat sun "
    "today tomorrow yesterday tonight weekend weekends weekday weekdays day days week weeks month months "
    "year years quarter quarters daily weekly monthly yearly annual annually quarterly "
    "morning mornings afternoon afternoons evening evenings night nights noon midnight "
    "spring summer autumn fall winter season ytd mtd qtd wtd am pm".split())
_NUMBER_WORDS = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million "
    "second third fourth fifth once twice thrice couple few several dozen half double single triple".split())
# Words whose place in a question says nothing: the order check leaves them out.
_ANYWHERE = frozenset(
    "i me my mine the a an do does did is are was were have has had there any some what which who please show list "
    "give tell can could would will you".split())
_UNDOING = ("un", "non", "in", "im", "ir", "il", "dis", "mis", "anti")
_TOKEN = re.compile(r"\d+(?:[.,/:-]\d+)*[a-z%]*|[^\W\d_]+|[<>=≤≥≠]")
_THOUSANDS = re.compile(r",(?=\d{3}(?:\D|$))")


def ask_words(question: str, today: date | None = None, is_merchant: Callable[[str], bool] | None = None) -> str:
    """What two wordings of one question have in common: for a spending
    category or one of the user's merchants, asked for bare or as its
    purchases, the name and the dates ("coffee", "show my coffee expenses"
    and "my coffee purchases" all give "coffee"); for every other question,
    the question itself (normalize_question)."""
    found = subject_ask(question, today, is_merchant)
    return found[0] if found else normalize_question(question)


class Asked(NamedTuple):
    name: str          # what is asked for, as typed: a category, a merchant, or neither
    bare: bool         # no word for the purchases
    when: str          # the dates it names, as of the day it was asked
    shown: bool        # asked to be shown ("show me", "can you list")
    mine: bool         # "my" stood before the name
    rows: str | None   # the word for the purchases
    at: str | None     # the word between the purchases and the name ("purchases at starbucks")

    @property
    def shown_only(self) -> bool:
        """ "show me movies": asked to be shown the thing itself, not one's own and not its purchases."""
        return self.shown and self.bare and not self.mine


def asked_subject(question: str, today: date | None = None) -> Asked | None:
    """The shape alone, whatever the name names: [can you] [show [me]] [my]
    <name> [expenses] [for <dates>], or [my] expenses on <name>. None when
    a word is left over. today: the day it was asked."""
    asked, when, shown = normalize_question(question), "", False
    dates = find_window(asked, today)
    if dates:
        start, end, (i, j) = dates
        head = asked[:i]
        asked = f"{head[:_BEFORE_DATES.search(head).start()]} {asked[j:]}"  # type: ignore[union-attr]
        when = f" {start.isoformat()} {end.isoformat()}"
    words = asked.split()
    if words and words[-1] == "please":
        words = words[:-1]
    if words and words[0] == "please":
        words = words[1:]
    if len(words) >= 2 and words[0] in ("can", "could", "would", "will") and words[1] == "you":
        words = words[3:] if words[2:3] == ["please"] else words[2:]
        if not words or words[0] not in _SHOW:
            return None  # "would you rent": asked for advice, not to be shown
    if words and words[0] in _SHOW:
        words, shown = (words[2:] if words[:2] == ["show", "me"] else words[1:]), True
    mine = words[:1] == ["my"]
    if words and words[0] in ("my", "the"):
        words = words[1:]
    bare, rows, at = False, None, None
    if len(words) >= 3 and words[0] in _ROWS and words[1] in ("on", "for", "of", "at", "from"):
        rows, at = words[0], words[1]
        subject = words[3:] if words[2] in ("my", "the") else words[2:]
    elif words and words[-1] in _ROWS:
        rows, subject = words[-1], words[:-1]
    else:
        subject, bare = words, True
    if not subject or len(subject) > 4:
        return None
    return Asked(" ".join(subject), bare, when, shown, mine, rows, at)


def _only_the_shop(found: Asked) -> bool:
    """Whether the words can mean nothing but purchases at a place of this
    name: "target purchases", "charges from target", "spending at target"."""
    return found.at in ("at", "from") or (found.at is None and found.rows in _SHOP_ROWS)


def subject_ask(question: str, today: date | None = None,
                is_merchant: Callable[[str], bool] | None = None) -> tuple[str, bool] | None:
    """For a question that is a request for one thing's purchases and
    nothing else, what its wordings share and whether it is bare (no word
    for the purchases). None for any other question.

    The thing is a spending category ("coffee": the name as typed, with the
    dates), or, when is_merchant says so, one of the user's own merchants
    ("@starbucks"). A bare one ("coffee", "groceries?") is a whole question
    when it opens a conversation. After "how much did I spend on restaurants
    last month?" it is the short form of another question, so whoever
    matches by this must know where a bare one was asked (see get_answer)."""
    found = asked_subject(question, today)
    if found is None:
        return None
    name, bare, when = found.name, found.bare, found.when
    category = get_catalog().find(name)
    if category is not None:
        # At and from are a merchant's: "purchases at the restaurant" names one restaurant.
        spending = (category.grp in ("needs", "wants") and category.key != "other" and name not in _THINGS
                    and found.at not in ("at", "from"))
        return (name + when, bare) if spending else None
    if is_merchant is None or (_OWN.intersection(name.split()) and not _only_the_shop(found)) or not is_merchant(name):
        return None
    return f"@{name}{when}", bare


def category_ask(question: str, today: date | None = None) -> tuple[str, bool] | None:
    """subject_ask for categories only: what the router can know without the user's rows."""
    return subject_ask(question, today)


# A user's merchants as of one data version, kept in this process by
# database and user: reading them reads the user's purchases, and every
# purchase raises the version. Kept for a while only: an account another
# process deleted and made again starts its versions over, and nothing tells
# this process.
_MERCHANTS: OrderedDict[tuple[str, str], tuple[int, frozenset[str], float]] = OrderedDict()
_MERCHANTS_KEPT = 512      # users
_MERCHANTS_SECONDS = 900.0
_MERCHANTS_LOCK = threading.Lock()
_now = time.monotonic


def merchants_of(user_id: str, version: int | Callable[[], int], url: str | None = None,
                 conn: Any = None, region: str | None = None) -> Callable[[str], bool]:
    """Says whether a name is one of this user's merchants, as the shape
    reads names (_as_asked). Nothing is read until a name asks, and then
    once for each state of the user's money: version is the user's data
    version, or what reads it when a name asks.

    conn: a connection on url to read on, when the caller holds one. Then
    region is the caller's read_region, asked for before it took the
    connection: read_region may open one of its own, and a pool that is
    full would wait on itself."""
    names: frozenset[str] | None = None

    def is_merchant(name: str) -> bool:
        nonlocal names
        if names is None:
            names = _merchants(user_id, int(version() if callable(version) else version), url, conn, region)
        return name in names

    return is_merchant


def _merchants(user_id: str, version: int, url: str | None, conn: Any, region: str | None) -> frozenset[str]:
    url = request_url(url)
    key = (url or "", str(user_id))
    with _MERCHANTS_LOCK:
        kept = _MERCHANTS.get(key)
        if kept is not None and kept[0] == version and _now() - kept[2] <= _MERCHANTS_SECONDS:
            _MERCHANTS.move_to_end(key)
            return kept[1]
    if conn is not None:
        names = frozenset(_merchant_names(conn, user_id, region))
    else:
        region = read_region(user_id, url)
        with get_engine(routed(url, user_id)).connect() as own:
            names = frozenset(_merchant_names(own, user_id, region))
    with _MERCHANTS_LOCK:
        _MERCHANTS[key] = (version, names, _now())
        _MERCHANTS.move_to_end(key)
        while len(_MERCHANTS) > _MERCHANTS_KEPT:
            _MERCHANTS.popitem(last=False)
    return names


def forget_merchants(user_ids: Iterable[str] | None = None) -> None:
    """Drop the merchants kept for these users, or for everyone. A deleted
    account and a database set up again start their versions over, so the
    version no longer says which purchases the names came from."""
    with _MERCHANTS_LOCK:
        if user_ids is None:
            _MERCHANTS.clear()
            return
        gone = {str(user_id) for user_id in user_ids}
        for key in [key for key in _MERCHANTS if key[1] in gone]:
            del _MERCHANTS[key]


def _as_asked(merchant: str) -> str:
    """A merchant's name as the shape reads it in a question: normalized,
    and with no "the" or "my" in front, which the shape takes off a name
    ("the home depot purchases" asks for "home depot")."""
    words = normalize_question(merchant).split()
    return " ".join(words[1:] if len(words) > 1 and words[0] in ("the", "my") else words)


def _merchant_names(conn: Any, user_id: str, region: str | None = None) -> set[str]:
    """Inside the user's home region when a read may name it (region, from
    read_region): on REGIONAL BY ROW tables a lookup by user alone looks in
    every region."""
    return {_as_asked(str(r[0])) for r in conn.execute(text(
        f"SELECT DISTINCT merchant FROM expenses WHERE {region_pin('', region)}user_id = CAST(:u AS UUID) "
        "AND merchant IS NOT NULL"), {"u": user_id, "region": region})}


def other_ask(cached: str, asked: str) -> str | None:
    """Why two similar questions are still two questions, or None. Similarity
    cannot say: "coffee this month" and "coffee last month" are 92% similar,
    "purchases under $100" and "purchases over $100" 95%. This catches other
    dates, other numbers, a word that turns a question around and the same
    words in another order ("from checking to savings"). It does not know
    what a question is about: "coffee expenses" and "tea expenses" pass."""
    if extract_window(cached) != extract_window(asked):
        return "it names other dates"
    one, other = _tokens(cached), _tokens(asked)
    words_one = [t for t in one if not t[0].isdigit()]
    words_other = [t for t in other if not t[0].isdigit()]
    apart = set(words_one) ^ set(words_other)
    if apart & _DATE_WORDS:
        return "it names other dates"
    if _numbers(cached) != _numbers(asked) or apart & _NUMBER_WORDS:
        return "it names other numbers"
    turned = sorted(apart & _TURNING)
    if turned:
        return "it differs in " + " and ".join(f"“{w}”" for w in turned)
    for longer in sorted(apart):
        # "unauthorized" beside "authorized": the same word, undone.
        if any(longer == prefix + base for base in apart for prefix in _UNDOING if len(base) >= 4):
            return f"it differs in “{longer}”"
    shared = (set(words_one) & set(words_other)) - _ANYWHERE
    if [w for w in words_one if w in shared] != [w for w in words_other if w in shared]:
        return "it has the same words in another order"
    return None


def _numbers(question: str) -> list[str]:
    """The numbers a question names, in order and as normalize_question reads them: -50 is not 50, .99 not 99."""
    return [t for t in normalize_question(question).split() if any(ch.isdigit() for ch in t)]


# Words that name a day without the window parser making dates of them.
_DAY_WORDS = _DATE_WORDS | frozenset("ago due upcoming soon lately recently now currently".split())


def _dates_as_of(question: str, day: date) -> tuple[tuple[date, date] | None, bool]:
    """The window a question names as of the day it was asked, and whether
    it also names a day in words no window comes of ("due tomorrow", "next
    week"): such a question is only good on the day it was asked."""
    asked = normalize_question(question)
    found = find_window(asked, day)
    rest = f"{asked[:found[2][0]]} {asked[found[2][1]:]}" if found else asked
    return (found[:2] if found else None), any(w in _DAY_WORDS for w in rest.split())


def _tokens(question: str) -> list[str]:
    """A question's words and numbers for other_ask: "non-recurring" is two
    words, "didn't" one, "$5k" and "3rd" are not 5 and 3, 1,000 is 1000."""
    asked = unicodedata.normalize("NFKC", question or "").lower().replace("'", "").replace("’", "")
    return [_THOUSANDS.sub("", t) for t in _TOKEN.findall(asked)]


def get_answer(user_id: str, *, version: int, language: str, query: str, qvec: str, match: str | None = None,
               similarity: float | None = None, shape: str | None = None, standalone: bool = True,
               previous: str | None = None, thread_id: str | None = None, refused: list | None = None,
               unshaped: bool = False, url: str | None = None) -> dict | None:
    """A cached answer this question may replay. The vector index brings the
    user's nearest cached questions; the match says which of them counts:

      exact    the same question, give or take case, punctuation and spacing
      words    also a category or one of the user's merchants in other words (subject_ask)
      similar  also a question at least `similarity` similar, unless other_ask objects

    Similarity alone decides nothing in the first two: "bills due next week"
    and "bills due this week" sit 0.026 apart, "coffee" and "coffee expenses"
    0.371. The same question wins over the same in other words, and that over
    a similar one. match and similarity default to CACHE_MATCH and
    CACHE_SIMILARITY_THRESHOLD.

    shape names the asker's settings that shape an answer (page size,
    memories): only an answer kept under the same shape counts. unshaped
    lets answers kept before shapes were recorded count as well; the caller
    says so when its shape is the deployment's own, which is what shaped
    them.

    standalone says the question opens its conversation (thread_id), and
    previous is the question that conversation asked just before. They are
    about what a conversation can do to a question's meaning:
      - a category or merchant with a word for its purchases ("coffee
        expenses", "starbucks charges") says all it asks, and counts anywhere;
      - a bare one ("coffee", "groceries?") is a whole question only where
        it opened a conversation. Elsewhere it counts only against the
        question asked right before or after it in the same conversation:
        "coffee", then "coffee expenses";
      - a similar question counts only between two questions that each
        opened a conversation: similarity knows nothing of what a
        conversation was about.

    Dates are read as of the day each question was asked: "coffee this
    month" kept on the 31st does not answer for the month that began since,
    in these words or any other, and a day no window comes of ("due
    tomorrow") is only good on the day it was asked.

    A list passed as refused gets the nearest similar question that was
    turned down, with the reason, when nothing hit."""
    cfg = get_config()
    url = request_url(url)
    match = match or cfg.cache_match
    max_distance = 1.0 - float(cfg.cache_similarity_threshold if similarity is None else similarity)
    today = clock.today()
    want = normalize_question(query)
    asked_dates, asked_loose = _dates_as_of(query, today)
    just_before = normalize_question(previous) if previous and thread_id else None
    region = None if match == "exact" else read_region(user_id, url)  # for the merchants, before a connection is held

    def fn(conn) -> tuple[dict | None, dict | None]:
        near = nearest(conn, table="answer_cache", index="answer_cache_user_embedding",
                       exact_index="answer_cache_user_idx", id_col="cache_id", emb_col="query_embedding",
                       user_id=user_id, qvec=qvec, k=CANDIDATES)
        if not near:
            return None, None
        kept = {str(r[0]): r[1:] for r in conn.execute(text("""
            SELECT cache_id, query_text, COALESCE(standalone, false), created_at, thread_id FROM answer_cache
            WHERE user_id = CAST(:u AS UUID) AND cache_id = ANY(CAST(:ids AS UUID[]))
              AND data_version = :v AND language = :l AND expires_at > now()
              AND (shape IS NOT DISTINCT FROM :shape OR (:unshaped AND shape IS NULL))
        """), {"u": user_id, "ids": [c for c, _ in near], "v": int(version), "l": language, "shape": shape,
               "unshaped": bool(unshaped)})}
        def replayed(ranked: list[tuple]) -> dict | None:
            for _, distance, cache_id, question, kind in sorted(ranked):
                row = conn.execute(text("""
                    UPDATE answer_cache SET hits = hits + 1
                    WHERE cache_id = CAST(:c AS UUID) AND user_id = CAST(:u AS UUID)
                      AND data_version = :v AND language = :l AND expires_at > now()
                    RETURNING answer, tokens
                """), {"c": cache_id, "u": user_id, "v": int(version), "l": language}).fetchone()
                if row is not None:
                    return {"cache_id": cache_id, "answer": row[0], "tokens": int(row[1]), "distance": distance,
                            "question": question, "match": kind}
            return None

        # The same words first: they need nothing read about what the question names.
        the_same, others = [], []
        for cache_id, distance in near:
            if cache_id not in kept:  # another version, language or shape, or expired
                continue
            question, opened, created, thread = kept[cache_id]
            day = created.astimezone(timezone.utc).date()
            same = normalize_question(question)
            kept_dates, kept_loose = _dates_as_of(question, day)
            other_dates = kept_dates != asked_dates or (day != today and (kept_loose or asked_loose))
            if same != want:
                others.append((cache_id, distance, question, bool(opened), day, thread, same, other_dates))
            elif not other_dates:  # the same words about another day's "today" or "this month" are another question
                the_same.append((0, distance, cache_id, question, "exact"))
        if (hit := replayed(the_same)) is not None or match == "exact" or not others:
            return hit, None

        # The user's merchants are read only when a name that is no category asks, and looked up for
        # a kept question only when the one asked names a merchant: no category's words are a merchant's.
        is_merchant = merchants_of(user_id, int(version), url, conn, region)
        asked_for = subject_ask(query, today, is_merchant)
        of_kept = is_merchant if asked_for is not None and asked_for[0].startswith("@") else None
        ranked, turned_down = [], None
        for cache_id, distance, question, opened, day, thread, same, other_dates in others:
            # Asked right before this one, in this very conversation: the same words elsewhere are not that.
            neighbours = just_before is not None and thread == thread_id and same == just_before
            if asked_for is not None and same_subject(asked_for, standalone, subject_ask(question, day, of_kept),
                                                      opened, neighbours):
                ranked.append((1, distance, cache_id, question, "words"))
            elif match == "similar" and standalone and opened and distance <= max_distance:
                why = "it names other dates" if other_dates else other_ask(question, query)
                if why:
                    turned_down = turned_down or {"question": question, "similarity": round(1.0 - distance, 4),
                                                  "why": why}
                    continue
                ranked.append((2, distance, cache_id, question, "similar"))
        return replayed(ranked), turned_down

    hit, turned_down = run_transaction(fn, url=url, user_id=user_id)
    if refused is not None and hit is None and turned_down:
        refused.append(turned_down)
    record_stat("answer", "hit" if hit else "miss", hit["tokens"] if hit else 0)
    return hit


def same_subject(asked: tuple[str, bool] | None, asked_opened: bool, kept: tuple[str, bool] | None,
                 kept_opened: bool, neighbours: bool) -> bool:
    """Whether two questions are one question about a category or a
    merchant in two wordings (subject_ask gave each its name, dates and
    whether it is bare). A bare one counts where it opened its conversation,
    or beside the other question in the same conversation (neighbours)."""
    if asked is None or kept is None or asked[0] != kept[0]:
        return False
    return (not asked[1] or asked_opened or neighbours) and (not kept[1] or kept_opened or neighbours)


def put_answer(user_id: str, *, version: int, language: str, query: str, qvec: str, answer: str,
               tokens: int, shape: str | None = None, standalone: bool = False, thread_id: str | None = None,
               asked_at: datetime | None = None, url: str | None = None) -> None:
    """shape: the asker's settings that shaped this answer. standalone: the
    question opened its conversation (thread_id), so the answer leans on
    nothing said before. asked_at: when the question was asked; the write
    runs behind the answer, and the dates a question names are read as of
    the day it was asked (see get_answer)."""
    url = request_url(url)
    hours = get_config().cache_ttl_hours

    def fn(conn) -> None:
        region = money_region(conn, user_id, url)
        rcol, rval = region_columns(region)
        conn.execute(text(
            "DELETE FROM answer_cache WHERE user_id = CAST(:u AS UUID) AND data_version < :v"
        ), {"u": user_id, "v": int(version)})
        conn.execute(text(f"""
            INSERT INTO answer_cache ({rcol}user_id, data_version, language, query_text, query_embedding,
                                      answer, tokens, created_at, expires_at, shape, standalone, thread_id)
            VALUES ({rval}CAST(:u AS UUID), :v, :l, :q, :qvec, :a, :t, COALESCE(CAST(:at AS TIMESTAMPTZ), now()),
                    now() + :h * INTERVAL '1 hour', :shape, :standalone, :thread)
        """), {"u": user_id, "v": int(version), "l": language, "q": query[:2000], "qvec": qvec,
               "a": answer, "t": int(tokens), "h": hours, "shape": shape, "standalone": bool(standalone),
               "thread": thread_id, "at": asked_at, "crdb_region": region})

    run_transaction(fn, url=url, user_id=user_id, idempotent=False)
    record_stat("answer", "write")


def cache_summary(user_id: str, *, url: str | None = None) -> dict[str, Any]:
    """Counts for the Settings page and the trace."""
    url = request_url(url)
    with get_engine(routed(url, user_id)).connect() as conn:
        answers = conn.execute(text("SELECT count(*) FROM answer_cache WHERE user_id = CAST(:u AS UUID)"),
                               {"u": user_id}).scalar()
        searches = conn.execute(text("SELECT count(*) FROM search_cache WHERE user_id = CAST(:u AS UUID)"),
                                {"u": user_id}).scalar()
    return {"answers": int(answers or 0), "searches": int(searches or 0)}
