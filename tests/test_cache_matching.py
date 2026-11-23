"""Which earlier question's answer the assistant may replay, and how a
question is read as a category or a merchant asked for.

"coffee" and then "coffee expenses" used to be two questions to the answer
cache: only the same words replayed an answer. Similarity cannot tell the two
apart from questions that are not the same: "coffee" and "coffee expenses" are
63% similar, and so are "coffee expenses" and "tea expenses", while "purchases
under $100" and "purchases over $100" are 95% similar. Nor can a list of
words to ignore: "show my savings account" is not "show my savings account
transactions". One shape can: a spending category or one of the user's own
merchants, asked for bare or as its purchases, for the same dates.

That shape is what the router sends to SQL (subject_asked): such a question
is answered by the database on every model, and its card is kept under what
it asks for, never under its words (tests/test_named_alone_from_sql.py). The
answer cache was the first to read the shape, as a third way to match
(words), and reads it no more. It matches two ways (CACHE_MATCH, and each
person's own choice in Settings): exact, the same question; similar, also,
for a conversation's first question, any question at least as similar as the
similarity setting, unless it names other dates or numbers or differs in a
word that turns a question around."""

import os

import pytest
from sqlalchemy import text

from banko_ai.db.engine import get_engine
from banko_ai.services import caches as C
from banko_ai.services import ledger
from banko_ai.services._vector import embed
from tests._money import card_purchase, drop_user, make_user, seeded_diner

needs_db = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


def _asked(question: str, today=None, is_merchant=None):
    """What a question in the shape asks for (C.subject_asked), as one line
    to compare: the name with the days it names, "@" before a merchant's,
    and whether it stands bare. None for any other question."""
    found = C.subject_asked(question, today, is_merchant)
    if found is None:
        return None
    days = "".join(f" {day.isoformat()}" for day in found.window or ())
    return ("@" if found.kind == "merchant" else "") + found.name + days, found.bare


def _one(question: str, today=None, is_merchant=None) -> str:
    """What two wordings of one question share: the thing asked for with
    its days, or, outside the shape, the question as the cache reads it."""
    found = _asked(question, today, is_merchant)
    return found[0] if found else C.normalize_question(question)


# The same question, as the cache reads it.

@pytest.mark.parametrize("one, other", [
    ("what was the $12.99 charge?", "what was the $1,299 charge?"),     # without its point, $12.99 read as 1299
    ("purchases on 11/2", "purchases on 1/12"),
    ("purchases > $100", "purchases < $100"),
    ("purchases ≥ $100", "purchases ≤ $100"),
    ("over 5% of my income", "over 5 of my income"),
    ("the １２．９９ charge", "the １，２９９ charge"),                       # full-width digits
    ("a $.99 charge", "a $99 charge"),
    ("charges of -$50", "charges of $50"),
    ("purchases of $10-$20", "purchases of $1020"),
    ("purchases of $10–$20", "purchases of $1020"),                    # an en dash, as a phone types it
    ("charges of −$50", "charges of $50"),                             # a real minus sign
    ("charges != $50", "charges = $50"),
    ("a 1½% fee", "a 112% fee"),
])
def test_a_question_keeps_what_tells_two_numbers_apart(one, other):
    assert C.normalize_question(one) != C.normalize_question(other)


@pytest.mark.parametrize("question, read", [
    ("What was the $12.99 charge?", "what was the 12.99 charge"),
    ("What's  my BIGGEST bill?! ", "whats my biggest bill"),                  # as it always did
    ("over $1,000.50, please.", "over 1000.50 please"),                       # a thousands comma is no difference
    ("the １２．９９ charge", "the 12.99 charge"),
    ("12,50 euros", "12,50 euros"),                                          # a decimal comma stays
    ("coffee ☕️", "coffee"),                                                  # an emoji goes, selector and all
    ("pre-tax income", "pretax income"),                                      # a hyphen between letters, as before
    ("charges of -$50", "charges of -50"),
    ("purchases of $10-$20", "purchases of 10-20"),
    ("purchases >= $100", "purchases > = 100"),
    ("purchases of $10–$20", "purchases of 10-20"),
    ("charges of −$50", "charges of -50"),
    ("charges != $50", "charges ≠ 50"),
])
def test_how_a_question_reads_to_the_cache(question, read):
    assert C.normalize_question(question) == read


def test_a_thousands_comma_makes_no_other_question():
    assert C.normalize_question("purchases over $1,000") == C.normalize_question("purchases over $1000")


def test_a_question_in_a_script_with_vowel_signs_keeps_them():
    """Hindi writes its vowels as marks on the consonant. Stripped like
    punctuation, दिन (day) and दान (donation) were one word."""
    assert C.normalize_question("दिन का खर्च") != C.normalize_question("दान का खर्च")
    assert C.normalize_question("दिन का खर्च?") == "दिन का खर्च"


# One thing asked for in other words: a spending category, bare or as its purchases.

@pytest.mark.parametrize("question, means, bare", [
    ("coffee", "coffee", True),
    ("Coffee?", "coffee", True),
    ("my coffee", "coffee", True),
    ("show my coffee", "coffee", True),
    ("coffee expenses", "coffee", False),
    ("Coffee expenses?", "coffee", False),
    ("my coffee spending", "coffee", False),
    ("coffee purchases", "coffee", False),
    ("the coffee transactions", "coffee", False),
    ("Show my coffee expenses, please", "coffee", False),
    ("show me my coffee charges", "coffee", False),
    ("can you list the coffee transactions", "coffee", False),
    ("Could you please show me my coffee purchases?", "coffee", False),
    ("view my coffee spending", "coffee", False),
    ("expenses on coffee", "coffee", False),
    ("my spending on coffee", "coffee", False),
    ("purchases for groceries", "groceries", False),
    ("dining out expenses", "dining out", False),
    ("rent", "rent", True),
    ("show my rent transactions", "rent", False),
])
def test_a_category_asked_for_in_other_words_is_one_question(question, means, bare):
    """bare: no word for its purchases. Alone that is a whole question;
    inside a conversation it may be the short form of another one."""
    assert _asked(question) == (means, bare)


def test_the_dates_a_category_is_asked_for_are_part_of_the_question():
    """The same category for the same dates, however the dates are said."""
    this_month = _one("coffee this month")
    assert this_month.startswith("coffee ") and this_month != "coffee"
    for same in ("coffee expenses this month", "show my coffee purchases for this month", "this month coffee spending",
                 "my coffee spending during this month"):
        assert _one(same) == this_month, same
    assert _one("coffee last month") != this_month
    assert _one("coffee last month") == _one("show my coffee expenses for last month")
    assert _one("coffee expenses in the last 30 days") == _one("coffee over the past 30 days")
    assert _one("coffee expenses in the last 30 days") != _one("coffee in the last 7 days")
    assert _one("coffee expenses in october") == _one("coffee in october")


def test_a_span_of_days_no_calendar_holds_is_no_window_and_no_crash():
    """ "coffee last 1,000,000 days" overflowed the date arithmetic: the
    question failed, and kept in the cache it would have failed every lookup
    that met it."""
    from banko_ai.utils.intent_classifier import extract_window, find_window
    for question in ("coffee last 1000000 days", "coffee last 1,000,000 days", "coffee in the past 99999999 months",
                     "coffee last 400000 weeks"):
        assert find_window(question) is None and extract_window(question) is None, question
        assert _asked(question) is None


def test_dates_are_read_as_of_the_day_the_question_was_asked():
    """ "coffee this month" kept on the 31st is about October. Read on the
    1st it would be November's question, and October's answer would be
    replayed for it."""
    from datetime import date
    october = _one("coffee this month", today=date(2026, 10, 31))
    november = _one("coffee this month", today=date(2026, 11, 1))
    assert october != november
    assert october == _one("coffee expenses in october", today=date(2026, 11, 1))
    assert _one("coffee", today=date(2026, 10, 31)) == _one("coffee", today=date(2026, 11, 1))


@pytest.mark.parametrize("question", [
    # Not a spending category: a thing with a page of its own, a merchant, money coming in, a word with two meanings.
    "subscriptions", "show my subscriptions", "bills", "my bills", "income", "income transactions", "starbucks",
    "starbucks purchases", "savings account", "show my savings account transactions", "tea expenses",
    "internet purchases", "my internet", "phone expenses",
    # A category, and one word more.
    "all coffee expenses", "recent coffee expenses", "coffee expenses over $5", "coffee and tea expenses",
    "no coffee expenses", "coffee expenses at starbucks", "biggest coffee purchases", "how is my coffee spending",
    "why are my coffee expenses so high", "coffee payments", "coffee refunds", "can I see my coffee expenses",
    "coffee expense", "coffee purchase", "coffee charge", "coffee transaction", "my last coffee purchase",
    "subscription charges",
    # A question about a category asks how much or what it is; a request to do something is not a request to show.
    "what is coffee", "what is my rent", "is my rent paid", "what are my coffee expenses",
    "what is my coffee spending this month", "whats my coffee spending", "tell me my rent", "find me flights",
    "get me coffee", "find hotel", "give me my coffee expenses", "see my coffee expenses", "get coffee",
    "give coffee expenses", "tell coffee expenses", "would you rent", "can you coffee", "could you please groceries",
    # Nothing about a category at all.
    "expenses", "show my transactions", "what are my charges", "Show me", "?",
    "show my transactions this month", "show my spending this month", "show my last purchases",
    "what did I pay for at Costco", "did I get a refund from Amazon?", "which bills are due", "when do I get paid",
    "do I have any subscriptions", "any charges over $100?", "transfers to savings", "quién me pagó",
])
def test_one_word_the_shape_does_not_account_for_and_it_is_no_category_asked_for(question):
    """Such a question goes on to the routes that read all of it. Taken for
    the category's purchases it would be told about another question."""
    assert _asked(question) is None


@pytest.mark.parametrize("one, other", [
    ("coffee expenses", "tea expenses"),
    ("coffee this month", "coffee last month"),
    ("phone expenses", "internet expenses"),                    # two names of one category are two questions
    ("rent expenses", "mortgage expenses"),
    ("grocery expenses", "restaurant expenses"),
    ("show my savings account", "show my savings account transactions"),
    ("show my last purchase", "show my last purchases"),
    ("show my transactions this month", "show my spending this month"),
    ("what did I pay for at Costco", "what did I pay at Costco"),
    ("subscriptions", "subscription charges"),
    ("what is my coffee spending this month", "show my coffee transactions this month"),
    ("tell me my rent", "rent transactions"),
    ("find me flights", "show my flights"),
    ("purchases under $100", "purchases over $100"),
    ("what was the $12.99 charge", "what was the $1,299 charge"),
    ("did I get a refund from Amazon?", "can I get a refund from Amazon?"),
    ("which bills are due", "which bills were due"),
    ("what is the fee", "what is the fee for"),
    ("what's on this receipt?", "what is this receipt for?"),
    ("where do I spend the most on coffee", "where do I spend most on coffee"),
])
def test_two_questions_stay_two(one, other):
    assert _one(one) != _one(other)


# A merchant of one's own, asked for in other words.

_MINE = {"starbucks", "trader joes", "blue bottle coffee", "whole foods market", "netflix", "cafe"}


def _mine(name: str) -> bool:
    return name in _MINE


@pytest.mark.parametrize("question, means, bare", [
    ("starbucks", "@starbucks", True),
    ("Starbucks?", "@starbucks", True),
    ("my starbucks", "@starbucks", True),
    ("starbucks purchases", "@starbucks", False),
    ("show my Starbucks charges", "@starbucks", False),
    ("can you list my starbucks transactions", "@starbucks", False),
    ("purchases at Starbucks", "@starbucks", False),
    ("my charges from Netflix", "@netflix", False),
    ("Trader Joe's purchases", "@trader joes", False),
    ("blue bottle coffee purchases", "@blue bottle coffee", False),
    ("Whole Foods Market spending", "@whole foods market", False),
])
def test_a_merchant_asked_for_in_other_words_is_one_question(question, means, bare):
    """The same shape as a category, for a name that is one of the user's own merchants."""
    assert _asked(question, is_merchant=_mine) == (means, bare)
    assert _asked(question) is None                                                  # nobody's merchant: no shape


def test_a_merchants_dates_are_part_of_the_question_too():
    bare = _asked("starbucks last month", is_merchant=_mine)
    asked_for = _asked("show my starbucks purchases for last month", is_merchant=_mine)
    assert bare[0] == asked_for[0] and bare[0].startswith("@starbucks ") and (bare[1], asked_for[1]) == (True, False)
    assert bare[0] != _asked("starbucks", is_merchant=_mine)[0]


# The shape as a route reads it: what is asked for, how, and for which days.

def test_the_shape_says_what_is_asked_for_how_and_for_which_days():
    from datetime import date
    today = date(2026, 11, 22)
    found = C.subject_asked("my coffee expenses", today)
    assert (found.kind, found.name, found.bare, found.window, found.shown_only) == ("category", "coffee", False, None,
                                                                                    False)
    assert C.subject_asked("Coffee?", today).bare
    assert C.subject_asked("dining out expenses", today).name == "dining out"       # as typed: the catalog knows the label
    october = (date(2026, 10, 1), date(2026, 11, 1))
    assert C.subject_asked("coffee expenses for october", today).window == october
    shop = C.subject_asked("purchases at Starbucks last month", today, _mine)
    assert (shop.kind, shop.name, shop.bare, shop.window) == ("merchant", "starbucks", False, october)
    assert C.subject_asked("show me movies", today).shown_only                      # the thing itself, not one's own
    assert not C.subject_asked("show me my movies", today).shown_only
    for nothing in ("bills", "how much coffee", "starbucks", "coffee and tea", "other", "purchases at the restaurant"):
        assert C.subject_asked(nothing, today) is None, nothing


@pytest.mark.parametrize("question", [
    "target purchases",                       # a merchant, and not one of this user's
    "starbucks and netflix purchases", "starbucks purchases over $5", "recent starbucks purchases",
    "what did I buy at starbucks", "did I go to starbucks", "how much at starbucks", "starbucks refunds",
    "starbucks purchase", "the starbucks on main street purchases", "purchases to starbucks",
])
def test_a_merchant_and_one_word_more_is_no_merchant_question(question):
    assert _asked(question, is_merchant=_mine) is None


@pytest.mark.parametrize("name", ["target", "goal", "budget", "new balance", "savings", "cash app", "card factory",
                                  "coach"])
def test_a_merchant_named_like_one_of_bankos_own_things_is_asked_for_only_as_the_shop(name):
    """ "my target" may be a savings target, "spending on target" may ask
    whether the spending is on target, and "target spending" for the target
    itself. Only the words that can mean nothing but the shop count."""
    everything = lambda _: True  # noqa: E731
    for question in (f"my {name}", name, f"show me {name}", f"{name} this month", f"spending on {name}",
                     f"my spending on {name}", f"{name} spending", f"{name} expenses", f"expenses for {name}"):
        assert _asked(question, is_merchant=everything) is None, question
    for question in (f"{name} purchases", f"{name} transactions", f"my {name} charges", f"purchases at {name}",
                     f"charges from {name}", f"spending at {name}", f"expenses at {name}"):
        assert _asked(question, is_merchant=everything) == (f"@{name}", False), question


@pytest.mark.parametrize("question", ["purchases at the restaurant", "expenses from the pharmacy", "charges at the hotel",
                                      "spending at restaurants"])
def test_a_category_is_no_place_to_buy_at(question):
    """ "purchases at the restaurant" names one restaurant. At and from are a merchant's."""
    assert _asked(question) is None
    assert _asked(question, is_merchant=lambda _: False) is None


def test_a_category_is_still_what_money_is_spent_on():
    assert _asked("spending on restaurants") == ("restaurants", False)
    assert _asked("expenses for groceries") == ("groceries", False)


def test_a_word_that_ends_in_the_keeps_its_ending_before_the_dates():
    """ "soothe this month" was read as "soo" this month."""
    assert C.asked_subject("soothe this month").name == "soothe"
    assert C.asked_subject("bathe today").name == "bathe"
    assert C.asked_subject("coffee over the last 30 days").name == "coffee"


def _read_from_made_up_databases(monkeypatch, names) -> None:
    """The merchants come from `names`, over a connection that is only its database's URL."""
    monkeypatch.setattr(C, "_merchant_names", names)
    monkeypatch.setattr(C, "request_url", lambda url: url)
    monkeypatch.setattr(C, "read_region", lambda user, url: None)
    monkeypatch.setattr(C, "routed", lambda url, user: url)
    monkeypatch.setattr(C, "get_engine", lambda url: type("E", (), {"connect": lambda self: _Held(url)})())


def test_what_is_kept_about_a_users_merchants_is_per_database_and_not_for_ever(monkeypatch):
    """One process, two databases (a test run, a scratch database): the same
    user id is another user there. And another process may have deleted the
    account, so nothing is kept longer than a while."""
    reads, clock = [], [1000.0]
    _read_from_made_up_databases(monkeypatch, lambda conn, user, region=None: reads.append(conn) or {"starbucks"})
    monkeypatch.setattr(C, "_now", lambda: clock[0])
    C.forget_merchants()
    uid, one, two = "3f0c6d0e-7c55-4c1b-9a65-0f6f3b0a9d11", "postgresql://one/db", "postgresql://two/db"
    assert C.merchants_of(uid, 3, one)("starbucks") and reads == [one]
    assert C.merchants_of(uid, 3, one)("starbucks") and reads == [one]                    # kept
    assert C.merchants_of(uid, 3, two)("starbucks") and reads == [one, two]               # another database
    clock[0] += C._MERCHANTS_SECONDS + 1
    assert C.merchants_of(uid, 3, one)("starbucks") and reads == [one, two, one]          # a while later
    C.forget_merchants([uid])
    assert C.merchants_of(uid, 3, two)("starbucks") and reads == [one, two, one, two]


def test_a_users_data_version_is_read_only_when_a_name_asks(monkeypatch):
    """A category, or a bare name with one of Banko's own words, needs no look at the user's purchases."""
    asked = []
    _read_from_made_up_databases(monkeypatch, lambda conn, user, region=None: {"starbucks"})
    C.forget_merchants()
    is_merchant = C.merchants_of("3f0c6d0e-7c55-4c1b-9a65-0f6f3b0a9d12", lambda: asked.append(1) or 7,
                                 "postgresql://one/db")
    assert _asked("coffee expenses", is_merchant=is_merchant) == ("coffee", False) and asked == []
    assert _asked("my target", is_merchant=is_merchant) is None and asked == []
    assert _asked("starbucks", is_merchant=is_merchant) == ("@starbucks", True) and asked == [1]
    assert _asked("starbucks charges", is_merchant=is_merchant) == ("@starbucks", False) and asked == [1]


def test_the_merchants_are_read_inside_the_home_region_when_a_read_may_name_it(monkeypatch):
    """On REGIONAL BY ROW tables a lookup by user alone looks in every region."""
    seen = []

    class Conn:
        def execute(self, statement, params):
            seen.append((str(statement), params))
            return [("The Home Depot",), ("Starbucks",)]

    assert C._merchant_names(Conn(), "u", "us-east-1") == {"home depot", "starbucks"}
    assert "crdb_region = CAST(:region AS crdb_internal_region) AND" in seen[0][0] and seen[0][1]["region"] == "us-east-1"
    C._merchant_names(Conn(), "u", None)
    assert "crdb_region" not in seen[1][0]
    # Without a connection of the caller's, the region is asked for and then named.
    monkeypatch.setattr(C, "read_region", lambda user, url: "eu-west-1")
    monkeypatch.setattr(C, "get_engine", lambda url: type("E", (), {"connect": lambda self: _Held(Conn())})())
    monkeypatch.setattr(C, "routed", lambda url, user: url)
    C.forget_merchants()
    assert C.merchants_of("3f0c6d0e-7c55-4c1b-9a65-0f6f3b0a9d13", 1, "postgresql://one/db")("home depot")
    assert seen[2][1]["region"] == "eu-west-1"


class _Held:
    """A connection as a context manager, for a test that reads through a made-up one."""

    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self.conn

    def __exit__(self, *exc):
        return False


def test_a_name_that_is_a_category_and_a_merchant_is_the_category():
    """ "cafe" names the Coffee category. A merchant called Cafe does not change what the word asks for."""
    assert _asked("cafe purchases", is_merchant=_mine) == ("cafe", False) == _asked("cafe purchases")


# A similar question that is another question.

@pytest.mark.parametrize("cached, asked, why", [
    ("coffee this month", "coffee last month", "dates"),
    ("coffee expenses", "coffee expenses in october", "dates"),
    ("coffee expenses in october", "coffee expenses in november", "dates"),
    ("october purchases", "november purchases", "dates"),
    ("coffee on mondays", "coffee on fridays", "dates"),
    ("trips last summer", "trips last winter", "dates"),
    ("spending ytd", "spending mtd", "dates"),
    ("purchases after 9 pm", "purchases after 9 am", "dates"),
    ("what bills are due next week?", "what bills are due this week?", "dates"),
    ("purchases on 11/2", "purchases on 2/11", "numbers"),
    ("purchases over $100", "purchases over $50", "numbers"),
    ("purchases over $5k", "purchases over $5", "numbers"),
    ("charges of -$50", "charges of $50", "numbers"),
    ("a $.99 charge", "a $99 charge", "numbers"),
    ("my 3 latest coffee purchases", "my 5 latest coffee purchases", "numbers"),
    ("the 3rd largest purchase", "the 3 largest purchases", "numbers"),
    ("my top five merchants", "my top ten merchants", "numbers"),
    ("purchases under $100", "purchases over $100", "under"),
    ("purchases > $100", "purchases < $100", "<"),
    ("coffee expenses", "expenses that are not coffee", "not"),
    ("which bills did I pay", "which bills didn't I pay", "didnt"),
    ("recurring charges", "non-recurring charges", "non"),
    ("authorized charges", "unauthorized charges", "unauthorized"),
    ("did I get a refund", "will I get a refund", "will"),
    ("which bills are due", "which bills were due", "were"),
    ("where did I buy coffee", "when did I buy coffee", "when"),
    ("am I on track with my budget", "am I off track with my budget", "off"),
    ("my biggest coffee purchase", "my smallest coffee purchase", "biggest"),
    ("expensive purchases", "cheap purchases", "cheap"),
    ("transfers to savings", "transfers from savings", "from"),
    ("transfers from checking to savings", "transfers from savings to checking", "order"),
    ("transfers from checking to savings", "transfer from savings to checking", "order"),
])
def test_a_similar_question_about_other_dates_numbers_or_the_opposite_is_another_question(cached, asked, why):
    said = C.other_ask(cached, asked)
    assert said and why in said, said


@pytest.mark.parametrize("cached, asked", [
    ("coffee", "coffee expenses"),
    ("any unusual charges lately", "have there been any strange charges recently"),
    ("coffee in october", "coffee expenses in october"),
    ("purchases over $100", "show purchases over 100"),
    ("purchases over $1,000", "purchases over 1000"),
])
def test_a_similar_question_about_the_same_thing_raises_no_objection(cached, asked):
    assert C.other_ask(cached, asked) is None


def test_some_other_questions_still_pass_and_the_page_says_so():
    """No list of words closes this: who owes whom is not in any word."""
    assert C.other_ask("who do I owe money", "who owes me money") is None
    from banko_ai.services import preferences as P
    similar = next(c for c in next(s for s in P.SETTINGS if s.key == "cache_match").choices if c[0] == "similar")
    assert "cannot tell what a question is about" in similar[2]


# The cache itself.

@pytest.fixture
def user():
    uid = make_user()
    yield uid
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM answer_cache WHERE user_id = CAST(:u AS UUID)"), {"u": uid})
    drop_user(uid)


def _keep(uid: str, question: str, answer: str = "kept", *, standalone: bool = True, shape: str | None = None) -> int:
    """An answer as a conversation's first question leaves it, unless said otherwise."""
    version = ledger.data_version(uid)
    C.put_answer(uid, version=version, language="English", query=question, qvec=embed(question), answer=answer,
                 tokens=120, standalone=standalone, shape=shape)
    return version


def _ask(uid: str, version: int, question: str, **how):
    return C.get_answer(uid, version=version, language="English", query=question, qvec=embed(question), **how)


@needs_db
def test_one_thing_in_other_words_is_another_question_to_the_answer_cache(user):
    """ "coffee", then "coffee expenses", was one question here, by a third
    way to match (words). Both are a card from SQL now, kept under what they
    ask for, and never come this far. Whoever still asks for words gets the
    same question, as everyone."""
    version = _keep(user, "coffee", "Nine coffee purchases.")
    for how in ("exact", "words"):
        # Not even as a similar question, which "coffee expenses" is at half the way: words is no way to ask for that.
        assert _ask(user, version, "coffee expenses", match=how, similarity=0.5) is None, how
        assert _ask(user, version, "show my coffee purchases", match=how, similarity=0.5) is None, how
        hit = _ask(user, version, "Coffee?", match=how)
        assert hit and hit["answer"] == "Nine coffee purchases." and hit["tokens"] == 120 and hit["match"] == "exact"
    assert _ask(user, version, "coffee expenses", match="similar", similarity=0.5)["match"] == "similar"


@needs_db
def test_the_same_question_is_marked_as_the_same(user):
    version = _keep(user, "coffee expenses")
    for how in ("exact", "similar"):
        hit = _ask(user, version, "Coffee  expenses?", match=how)
        assert hit and hit["match"] == "exact", how


@needs_db
def test_exact_matching_takes_the_same_question_only(user):
    version = _keep(user, "coffee")
    assert _ask(user, version, "coffee expenses", match="exact") is None
    assert _ask(user, version, "Coffee?", match="exact")["answer"] == "kept"


@needs_db
def test_the_same_question_is_the_way_to_match_unless_told_otherwise(user, monkeypatch):
    """A category or merchant in other words is read from SQL and never
    reaches the cache, so matching by it is no setting any more. A
    deployment that still says words gets the same question, as everyone."""
    from banko_ai.config.settings import get_config, reset_config
    version = _keep(user, "coffee")
    assert get_config().cache_match == "exact"
    assert _ask(user, version, "coffee expenses") is None
    assert _ask(user, version, "Coffee?")["match"] == "exact"
    monkeypatch.setenv("CACHE_MATCH", "words")
    reset_config()
    try:
        assert get_config().cache_match == "exact"
        assert _ask(user, version, "coffee expenses") is None
    finally:
        monkeypatch.undo()
        reset_config()
    with pytest.raises(ValueError, match="CACHE_MATCH"):
        monkeypatch.setenv("CACHE_MATCH", "fuzzy")
        reset_config()
        get_config()
    monkeypatch.undo()
    reset_config()


@needs_db
def test_a_question_no_one_can_read_does_not_switch_the_cache_off(user):
    """Kept beside others, "coffee last 1,000,000 days" used to raise in every lookup that met it."""
    version = _keep(user, "coffee last 1,000,000 days", "Too long ago.")
    _keep(user, "coffee", "Nine coffee purchases.")
    assert _ask(user, version, "Coffee?")["answer"] == "Nine coffee purchases."
    assert _ask(user, version, "coffee costs", match="similar", similarity=0.5)["answer"] == "Nine coffee purchases."
    assert _ask(user, version, "Coffee last 1,000,000 days", match="similar")["answer"] == "Too long ago."


def _kept_on(uid: str, question: str, when: str) -> None:
    with get_engine().begin() as c:
        c.execute(text("UPDATE answer_cache SET created_at = CAST(:t AS TIMESTAMPTZ) "
                       "WHERE user_id = CAST(:u AS UUID) AND query_text = :q"), {"u": uid, "q": question, "t": when})


@needs_db
def test_an_answer_about_this_month_does_not_answer_for_the_next(user, monkeypatch):
    """Kept on October 31, "any odd charges this month" is October's. On
    November 1, with no purchase in between, it was replayed for the same
    words."""
    from datetime import date

    from banko_ai.utils import clock
    version = _keep(user, "any odd charges this month", "October's oddities.")
    _keep(user, "any odd charges", "All of them.")
    _kept_on(user, "any odd charges this month", "2026-10-31 22:00:00+00")
    _kept_on(user, "any odd charges", "2026-10-31 22:00:00+00")
    monkeypatch.setattr(clock, "today", lambda: date(2026, 11, 1))
    assert _ask(user, version, "any odd charges this month") is None
    assert _ask(user, version, "any odd charges this month", match="similar", similarity=0.5) is None
    assert _ask(user, version, "Any odd charges?")["answer"] == "All of them."                  # no dates, no day
    monkeypatch.setattr(clock, "today", lambda: date(2026, 10, 31))
    assert _ask(user, version, "Any odd charges this month?")["answer"] == "October's oddities."


@needs_db
def test_a_day_the_cache_cannot_work_out_is_only_good_on_the_day_it_was_asked(user):
    """ "which bills are due tomorrow" names a day in words no window comes
    of. Asked again the next day it is about another day, and was replayed.
    "in october" is October whenever it is asked."""
    version = _keep(user, "which bills are due tomorrow", "Rent, tomorrow.")
    _keep(user, "what did I buy in october", "A few things.")
    assert _ask(user, version, "Which bills are due tomorrow?")["answer"] == "Rent, tomorrow."     # the same day
    _kept_on(user, "which bills are due tomorrow", "2026-10-31 22:00:00+00")
    _kept_on(user, "what did I buy in october", "2026-10-31 22:00:00+00")
    assert _ask(user, version, "Which bills are due tomorrow?") is None
    assert _ask(user, version, "What did I buy in October?")["answer"] == "A few things."


@needs_db
def test_a_similar_question_about_today_kept_yesterday_is_about_another_day(user):
    version = _keep(user, "any unusual charges today", "None today.")
    asked, refused = "have there been any strange charges today", []
    assert _ask(user, version, asked, match="similar", similarity=0.75)["answer"] == "None today."
    _kept_on(user, "any unusual charges today", "2026-10-31 22:00:00+00")
    assert _ask(user, version, asked, match="similar", similarity=0.75, refused=refused) is None
    assert refused and "dates" in refused[0]["why"]


@needs_db
def test_an_answer_is_kept_as_of_the_moment_its_question_was_asked(user):
    """The write runs behind the answer. Asked at 23:59 on the 31st and
    written after midnight, "coffee this month" is still October's."""
    from datetime import datetime, timezone
    asked_at = datetime(2026, 10, 31, 23, 59, 59, tzinfo=timezone.utc)
    C.put_answer(user, version=ledger.data_version(user), language="English", query="coffee this month",
                 qvec=embed("coffee this month"), answer="October's.", tokens=1, asked_at=asked_at)
    with get_engine().connect() as c:
        kept = c.execute(text("SELECT created_at FROM answer_cache WHERE user_id = CAST(:u AS UUID)"), {"u": user}).scalar()
    assert kept.astimezone(timezone.utc) == asked_at


# A merchant of one's own.

@pytest.fixture(scope="module")
def shopper():
    """Someone with purchases at Starbucks and Trader Joe's, and none at Target."""
    uid = seeded_diner(1)
    card_purchase(uid, "Starbucks", "5.25")
    card_purchase(uid, "Trader Joe's", "41.80", category="Groceries")
    yield uid
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM answer_cache WHERE user_id = CAST(:u AS UUID)"), {"u": uid})
    drop_user(uid)


@pytest.fixture
def nothing_kept(shopper):
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM answer_cache WHERE user_id = CAST(:u AS UUID)"), {"u": shopper})
    return shopper


@needs_db
def test_a_merchant_whose_name_begins_with_the_is_asked_for_with_or_without_it(nothing_kept):
    """The shape drops "the" in front of a name, and "The Home Depot" was no name it could find."""
    card_purchase(nothing_kept, "The Home Depot", "64.10", category="Shopping")
    mine = C.merchants_of(nothing_kept, ledger.data_version(nothing_kept))
    for asked in ("the home depot purchases", "home depot purchases", "purchases at the home depot",
                  "show my Home Depot charges"):
        assert _asked(asked, is_merchant=mine) == ("@home depot", False), asked


@needs_db
def test_a_users_merchants_are_read_once_for_each_state_of_their_money(nothing_kept, monkeypatch):
    """Reading them means reading the user's purchases. It is done when a
    name asks for it, and kept until a purchase changes what they are."""
    reads, real = [], C._merchant_names

    def counted(conn, user_id, region=None):
        reads.append(user_id)
        return real(conn, user_id, region)

    def mine():
        return C.merchants_of(nothing_kept, lambda: ledger.data_version(nothing_kept))

    monkeypatch.setattr(C, "_merchant_names", counted)
    C.forget_merchants()
    assert _asked("coffee expenses", is_merchant=mine()) == ("coffee", False) and reads == []           # a category
    assert _asked("my target", is_merchant=mine()) is None and reads == []                 # one of Banko's own words
    assert _asked("starbucks purchases", is_merchant=mine()) == ("@starbucks", False) and len(reads) == 1
    assert _asked("show my starbucks charges", is_merchant=mine()) == ("@starbucks", False) and len(reads) == 1
    card_purchase(nothing_kept, "Peet's Coffee", "4.75")               # another state of their money
    assert _asked("peet's coffee purchases", is_merchant=mine()) == ("@peets coffee", False) and len(reads) == 2
    C.forget_merchants()
    assert _asked("my peets coffee charges", is_merchant=mine()) == ("@peets coffee", False) and len(reads) == 3


@needs_db
def test_an_account_made_again_under_its_old_id_has_its_merchants_read_again():
    """A staff account that sat idle for a month comes back under the id it
    had, and its data version starts over: the same number no longer says
    which purchases the names came from."""
    from banko_ai.services.users import delete_users

    def a_life(merchant: str) -> int:
        """An account with one card and one purchase, and the data version that leaves it at."""
        card = ledger.open_account(uid, kind="credit", name="Card", institution="Banko", mask="4242",
                                   credit_limit="2000")
        ledger.post_purchase(uid, card, amount="12.00", merchant=merchant, category="Coffee")
        return ledger.data_version(uid)

    uid = make_user()
    try:
        version = a_life("Starbucks")
        assert version > 0 and C.merchants_of(uid, version)("starbucks")
        assert delete_users([uid]) == 1
        make_user(user_id=uid)
        assert a_life("Peet's Coffee") == version                      # the same number, other purchases
        mine = C.merchants_of(uid, version)
        assert mine("peets coffee") and not mine("starbucks")
    finally:
        drop_user(uid)


# An answer is replayed only for settings that would have shaped it the same.

@needs_db
def test_an_answer_kept_before_settings_were_recorded_counts_under_the_deployments_own(user):
    """An upgrade must not empty the cache. What was kept before carries no
    settings: it was shaped by the deployment's, and counts for whoever has
    those. Someone with settings of their own asks again."""
    version = ledger.data_version(user)
    C.put_answer(user, version=version, language="English", query="coffee expenses", qvec=embed("coffee expenses"),
                 answer="Kept before the upgrade.", tokens=10)
    usual, mine = "page=10", "page=3"
    assert _ask(user, version, "coffee expenses", shape=usual) is None
    assert _ask(user, version, "coffee expenses", shape=usual, unshaped=True)["answer"] == "Kept before the upgrade."
    assert _ask(user, version, "coffee expenses", shape=mine) is None
    _keep(user, "coffee expenses", "Shaped by three a page.", shape=mine)
    assert _ask(user, version, "coffee expenses", shape=usual, unshaped=True)["answer"] == "Kept before the upgrade."


@needs_db
def test_an_answer_shaped_by_other_settings_is_not_replayed(user):
    """Three purchases a page make another answer than ten. Whoever has ten
    must not get the three, for the same question or for a similar one."""
    version = _keep(user, "coffee", "Three of them.", shape="page=3")
    assert _ask(user, version, "coffee", shape="page=10") is None
    assert _ask(user, version, "coffee expenses", match="similar", similarity=0.5, shape="page=10") is None
    assert _ask(user, version, "coffee") is None                                  # no shape named is not this shape
    assert _ask(user, version, "coffee expenses", match="similar", similarity=0.5, shape="page=3")["answer"] == \
        "Three of them."
    _keep(user, "coffee", "Ten of them.", shape="page=10")
    assert _ask(user, version, "coffee", shape="page=10")["answer"] == "Ten of them."
    assert _ask(user, version, "coffee", shape="page=3")["answer"] == "Three of them."


# A similar question.

@needs_db
def test_a_similar_question_gets_the_answer_when_similarity_is_asked_for(user):
    version = _keep(user, "any unusual charges lately")
    asked = "have there been any strange charges recently"
    assert _ask(user, version, asked, match="exact") is None
    hit = _ask(user, version, asked, match="similar", similarity=0.75)
    assert hit and hit["match"] == "similar" and hit["question"] == "any unusual charges lately"
    assert 0.75 <= 1 - hit["distance"] < 1
    assert _ask(user, version, asked, match="similar", similarity=0.99) is None       # not that similar


@needs_db
def test_a_similar_question_counts_only_between_two_conversations_first_questions(user):
    """Similarity knows nothing of what a conversation was about. "any
    strange charges recently" after "show my Costco purchases" means Costco,
    and so may have "any unusual charges lately" where it was kept."""
    asked = "have there been any strange charges recently"
    version = _keep(user, "any unusual charges lately", "Nothing unusual.")
    assert _ask(user, version, asked, match="similar", similarity=0.75, standalone=False) is None   # asked inside one
    assert _ask(user, version, asked, match="similar", similarity=0.75)["answer"] == "Nothing unusual."
    with get_engine().begin() as c:
        c.execute(text("UPDATE answer_cache SET standalone = false WHERE user_id = CAST(:u AS UUID)"), {"u": user})
    assert _ask(user, version, asked, match="similar", similarity=0.75) is None                    # kept inside one
    with get_engine().begin() as c:
        c.execute(text("UPDATE answer_cache SET standalone = NULL WHERE user_id = CAST(:u AS UUID)"), {"u": user})
    assert _ask(user, version, asked, match="similar", similarity=0.75) is None       # from before this was recorded


@needs_db
def test_a_question_that_is_not_similar_enough_is_a_plain_miss(user):
    version = _keep(user, "coffee expenses")
    refused: list = []
    assert _ask(user, version, "what is my rent", match="similar", similarity=0.75, refused=refused) is None
    assert refused == []


@needs_db
@pytest.mark.parametrize("cached, asked, why", [
    ("coffee this month", "coffee last month", "dates"),
    ("purchases over $100", "purchases over $50", "numbers"),
    ("purchases under $100", "purchases over $100", "under"),
    ("coffee expenses", "expenses that are not coffee", "not"),
])
def test_a_similar_question_that_is_another_question_is_refused_and_the_reason_is_kept(user, cached, asked, why):
    """These pairs are 83% to 95% similar. The reason goes to the trace
    panel, so a miss right beside a near question explains itself."""
    version = _keep(user, cached)
    refused: list = []
    assert _ask(user, version, asked, match="similar", similarity=0.75, refused=refused) is None
    assert len(refused) == 1 and refused[0]["question"] == cached and why in refused[0]["why"], refused
    assert 0.75 <= refused[0]["similarity"] < 1


def _apart(uid: str, asked: str) -> dict:
    with get_engine().connect() as c:
        return dict(c.execute(text("SELECT query_text, query_embedding <=> :q FROM answer_cache "
                                   "WHERE user_id = CAST(:u AS UUID)"), {"u": uid, "q": embed(asked)}).fetchall())


@needs_db
def test_a_refused_neighbour_does_not_hide_a_similar_question_further_away(user):
    """Two similar questions kept, neither in the asked one's words: the
    nearer is about today only and is refused, the farther one answers."""
    asked = "any unusual charges lately"
    near, far = "any unusual charges today", "have there been any strange charges recently"
    assert C.normalize_question(asked) not in (C.normalize_question(near), C.normalize_question(far))
    version = _keep(user, near, "today's")
    _keep(user, far, "the recent ones")
    refused: list = []
    hit = _ask(user, version, asked, match="similar", similarity=0.75, refused=refused)
    assert hit and hit["answer"] == "the recent ones" and hit["match"] == "similar"
    assert refused == []                                               # something answered: nothing to explain
    apart = _apart(user, asked)
    assert apart[near] < apart[far] <= 0.25, apart                     # the refused one really was the nearer


@needs_db
def test_between_two_wordings_of_one_thing_only_similarity_counts(user):
    """ "coffee costs" is similar to "coffee expenses" and near it. The long
    wording asks for the same purchases and is farther: the answer cache
    read both as the category's purchases and took the long one, as the same
    question in other words. It reads no question that way any more. The
    long one differs in "could" and is turned down; the near one answers."""
    asked, similar, same = "coffee expenses", "coffee costs", "could you please show me my coffee purchases"
    assert C.other_ask(similar, asked) is None and "could" in C.other_ask(same, asked)
    version = _keep(user, similar, "what it costs")
    _keep(user, same, "the purchases")
    hit = _ask(user, version, asked, match="similar", similarity=0.5)
    assert hit and hit["answer"] == "what it costs" and hit["match"] == "similar"
    apart = _apart(user, asked)
    assert apart[similar] < apart[same] <= 0.5, apart                  # both were near enough to count
