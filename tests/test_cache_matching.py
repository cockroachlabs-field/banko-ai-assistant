"""Which earlier question's answer the assistant may replay.

"coffee" and then "coffee expenses" used to be two questions to the answer
cache: only the same words replayed an answer. Similarity cannot tell the two
apart from questions that are not the same: "coffee" and "coffee expenses" are
63% similar, and so are "coffee expenses" and "tea expenses", while "purchases
under $100" and "purchases over $100" are 95% similar. Nor can a list of
words to ignore: "show my savings account" is not "show my savings account
transactions". One shape can: a spending category or one of the user's own
merchants, asked for bare or as its purchases, for the same dates. Every
other question is matched word for word.

So there are three ways to match (CACHE_MATCH, and each person's own choice
in Settings): exact, the same question; words, also a category or merchant
in other words; similar, also, for a conversation's first question, any question at
least as similar as the similarity setting, unless it names other dates or
numbers or differs in a word that turns a question around."""

import os

import pytest
from sqlalchemy import text

from banko_ai.db.engine import get_engine
from banko_ai.services import caches as C
from banko_ai.services import ledger
from banko_ai.services._vector import embed
from tests._money import card_purchase, drop_user, make_user, seeded_diner

needs_db = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


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


# The same question in other words: a spending category, asked for bare or as its purchases.

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
    assert C.category_ask(question) == (means, bare)
    assert C.ask_words(question) == means


def test_the_dates_a_category_is_asked_for_are_part_of_the_question():
    """The same category for the same dates, however the dates are said."""
    this_month = C.ask_words("coffee this month")
    assert this_month.startswith("coffee ") and this_month != "coffee"
    for same in ("coffee expenses this month", "show my coffee purchases for this month", "this month coffee spending",
                 "my coffee spending during this month"):
        assert C.ask_words(same) == this_month, same
    assert C.ask_words("coffee last month") != this_month
    assert C.ask_words("coffee last month") == C.ask_words("show my coffee expenses for last month")
    assert C.ask_words("coffee expenses in the last 30 days") == C.ask_words("coffee over the past 30 days")
    assert C.ask_words("coffee expenses in the last 30 days") != C.ask_words("coffee in the last 7 days")
    assert C.ask_words("coffee expenses in october") == C.ask_words("coffee in october")


def test_a_span_of_days_no_calendar_holds_is_no_window_and_no_crash():
    """ "coffee last 1,000,000 days" overflowed the date arithmetic: the
    question failed, and kept in the cache it would have failed every lookup
    that met it."""
    from banko_ai.utils.intent_classifier import extract_window, find_window
    for question in ("coffee last 1000000 days", "coffee last 1,000,000 days", "coffee in the past 99999999 months",
                     "coffee last 400000 weeks"):
        assert find_window(question) is None and extract_window(question) is None, question
        assert C.category_ask(question) is None


def test_dates_are_read_as_of_the_day_the_question_was_asked():
    """ "coffee this month" kept on the 31st is about October. Read on the
    1st it would be November's question, and October's answer would be
    replayed for it."""
    from datetime import date
    october = C.ask_words("coffee this month", today=date(2026, 10, 31))
    november = C.ask_words("coffee this month", today=date(2026, 11, 1))
    assert october != november
    assert october == C.ask_words("coffee expenses in october", today=date(2026, 11, 1))
    assert C.ask_words("coffee", today=date(2026, 10, 31)) == C.ask_words("coffee", today=date(2026, 11, 1))


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
def test_every_other_question_is_compared_word_for_word(question):
    """One word that the shape does not account for, and it is no category
    question: a miss costs a model call, a wrong hit is a wrong answer."""
    assert C.category_ask(question) is None
    assert C.ask_words(question) == C.normalize_question(question)


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
    assert C.ask_words(one) != C.ask_words(other)


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
    assert C.subject_ask(question, is_merchant=_mine) == (means, bare)
    assert C.category_ask(question) is None and C.subject_ask(question) is None      # nobody's merchant: no shape


def test_a_merchants_dates_are_part_of_the_question_too():
    bare = C.subject_ask("starbucks last month", is_merchant=_mine)
    asked_for = C.subject_ask("show my starbucks purchases for last month", is_merchant=_mine)
    assert bare[0] == asked_for[0] and bare[0].startswith("@starbucks ") and (bare[1], asked_for[1]) == (True, False)
    assert bare[0] != C.subject_ask("starbucks", is_merchant=_mine)[0]


@pytest.mark.parametrize("question", [
    "target purchases",                       # a merchant, and not one of this user's
    "starbucks and netflix purchases", "starbucks purchases over $5", "recent starbucks purchases",
    "what did I buy at starbucks", "did I go to starbucks", "how much at starbucks", "starbucks refunds",
    "starbucks purchase", "the starbucks on main street purchases", "purchases to starbucks",
])
def test_a_merchant_and_one_word_more_is_no_merchant_question(question):
    assert C.subject_ask(question, is_merchant=_mine) is None


@pytest.mark.parametrize("name", ["target", "goal", "budget", "new balance", "savings", "cash app", "card factory",
                                  "coach"])
def test_a_merchant_named_like_one_of_bankos_own_things_is_asked_for_only_as_the_shop(name):
    """ "my target" may be a savings target, "spending on target" may ask
    whether the spending is on target, and "target spending" for the target
    itself. Only the words that can mean nothing but the shop count."""
    everything = lambda _: True  # noqa: E731
    for question in (f"my {name}", name, f"show me {name}", f"{name} this month", f"spending on {name}",
                     f"my spending on {name}", f"{name} spending", f"{name} expenses", f"expenses for {name}"):
        assert C.subject_ask(question, is_merchant=everything) is None, question
    for question in (f"{name} purchases", f"{name} transactions", f"my {name} charges", f"purchases at {name}",
                     f"charges from {name}", f"spending at {name}", f"expenses at {name}"):
        assert C.subject_ask(question, is_merchant=everything) == (f"@{name}", False), question


@pytest.mark.parametrize("question", ["purchases at the restaurant", "expenses from the pharmacy", "charges at the hotel",
                                      "spending at restaurants"])
def test_a_category_is_no_place_to_buy_at(question):
    """ "purchases at the restaurant" names one restaurant. At and from are a merchant's."""
    assert C.category_ask(question) is None
    assert C.subject_ask(question, is_merchant=lambda _: False) is None


def test_a_category_is_still_what_money_is_spent_on():
    assert C.category_ask("spending on restaurants") == ("restaurants", False)
    assert C.category_ask("expenses for groceries") == ("groceries", False)


def test_a_word_that_ends_in_the_keeps_its_ending_before_the_dates():
    """ "soothe this month" was read as "soo" this month."""
    assert C.asked_subject("soothe this month").name == "soothe"
    assert C.asked_subject("bathe today").name == "bathe"
    assert C.asked_subject("coffee over the last 30 days").name == "coffee"


def test_what_is_kept_about_a_users_merchants_is_per_database_and_not_for_ever(monkeypatch):
    """One process, two databases (a test run, a scratch database): the same
    user id is another user there. And another process may have deleted the
    account, so nothing is kept longer than a while."""
    reads, clock = [], [1000.0]
    monkeypatch.setattr(C, "_merchant_names", lambda conn, user, region=None: reads.append(conn) or {"starbucks"})
    monkeypatch.setattr(C, "_now", lambda: clock[0])
    C.forget_merchants()
    uid, one, two = "3f0c6d0e-7c55-4c1b-9a65-0f6f3b0a9d11", "postgresql://one/db", "postgresql://two/db"
    assert C.merchants_of(uid, 3, one, conn="one")("starbucks") and reads == ["one"]
    assert C.merchants_of(uid, 3, one, conn="one")("starbucks") and reads == ["one"]                  # kept
    assert C.merchants_of(uid, 3, two, conn="two")("starbucks") and reads == ["one", "two"]           # another database
    clock[0] += C._MERCHANTS_SECONDS + 1
    assert C.merchants_of(uid, 3, one, conn="one")("starbucks") and reads == ["one", "two", "one"]    # a while later
    C.forget_merchants([uid])
    assert C.merchants_of(uid, 3, two, conn="two")("starbucks") and reads == ["one", "two", "one", "two"]


def test_a_users_data_version_is_read_only_when_a_name_asks(monkeypatch):
    """A category, or a bare name with one of Banko's own words, needs no look at the user's purchases."""
    asked = []
    monkeypatch.setattr(C, "_merchant_names", lambda conn, user, region=None: {"starbucks"})
    C.forget_merchants()
    is_merchant = C.merchants_of("3f0c6d0e-7c55-4c1b-9a65-0f6f3b0a9d12", lambda: asked.append(1) or 7,
                                 "postgresql://one/db", conn="one")
    assert C.subject_ask("coffee expenses", is_merchant=is_merchant) == ("coffee", False) and asked == []
    assert C.subject_ask("my target", is_merchant=is_merchant) is None and asked == []
    assert C.subject_ask("starbucks", is_merchant=is_merchant) == ("@starbucks", True) and asked == [1]
    assert C.subject_ask("starbucks charges", is_merchant=is_merchant) == ("@starbucks", False) and asked == [1]


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
    assert C.subject_ask("cafe purchases", is_merchant=_mine) == ("cafe", False) == C.category_ask("cafe purchases")


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


def _keep(uid: str, question: str, answer: str = "kept", *, standalone: bool = True, shape: str | None = None,
          thread: str | None = None) -> int:
    """An answer as a conversation's first question leaves it, unless said otherwise."""
    version = ledger.data_version(uid)
    C.put_answer(uid, version=version, language="English", query=question, qvec=embed(question), answer=answer,
                 tokens=120, standalone=standalone, shape=shape, thread_id=thread)
    return version


def _ask(uid: str, version: int, question: str, **how):
    return C.get_answer(uid, version=version, language="English", query=question, qvec=embed(question), **how)


@needs_db
def test_a_category_in_other_words_gets_the_answer_and_says_whose_it_was(user):
    version = _keep(user, "coffee", "Nine coffee purchases.")
    hit = _ask(user, version, "coffee expenses", match="words")
    assert hit and hit["answer"] == "Nine coffee purchases." and hit["tokens"] == 120
    assert hit["question"] == "coffee" and hit["match"] == "words"       # written for other words: the chat says so
    assert _ask(user, version, "Show my coffee expenses, please", match="words")["question"] == "coffee"
    assert _ask(user, version, "tea expenses", match="words") is None
    assert _ask(user, version, "coffee last month", match="words") is None
    assert _ask(user, version, "all coffee expenses", match="words") is None


@needs_db
def test_the_longer_wording_asked_first_serves_the_shorter_one(user):
    """The two sit far apart as vectors (a third of the way to unrelated),
    so the words decide, never the distance."""
    version = _keep(user, "can you show me my coffee expenses please")
    assert _ask(user, version, "coffee", match="words")["question"] == "can you show me my coffee expenses please"


@needs_db
def test_the_same_question_is_marked_as_the_same(user):
    version = _keep(user, "coffee expenses")
    for how in ("exact", "words", "similar"):
        hit = _ask(user, version, "Coffee  expenses?", match=how)
        assert hit and hit["match"] == "exact", how


@needs_db
def test_exact_matching_takes_the_same_question_only(user):
    version = _keep(user, "coffee")
    assert _ask(user, version, "coffee expenses", match="exact") is None
    assert _ask(user, version, "Coffee?", match="exact")["answer"] == "kept"


@needs_db
def test_words_are_the_way_to_match_unless_told_otherwise(user, monkeypatch):
    version = _keep(user, "coffee")
    assert _ask(user, version, "coffee expenses")["match"] == "words"
    monkeypatch.setenv("CACHE_MATCH", "exact")
    from banko_ai.config.settings import reset_config
    reset_config()
    try:
        assert _ask(user, version, "coffee expenses") is None
    finally:
        monkeypatch.undo()
        reset_config()


@needs_db
def test_a_category_with_a_word_for_its_purchases_counts_anywhere(user):
    """ "coffee expenses" says all it asks: kept inside a conversation or
    asked inside one, it is the same question. Main already replayed the
    same words there."""
    version = _keep(user, "coffee expenses", "Nine coffee purchases.", standalone=False)
    assert _ask(user, version, "my coffee spending", match="words", standalone=False)["answer"] == "Nine coffee purchases."
    assert _ask(user, version, "show my coffee purchases", match="words")["answer"] == "Nine coffee purchases."


@needs_db
def test_a_bare_category_inside_a_conversation_is_no_whole_question(user):
    """ "groceries?" right after "how much did I spend on restaurants last
    month?" asks that about groceries. It is not "show my groceries
    purchases", and what it is told is not that question's answer either."""
    version = _keep(user, "show my groceries purchases", "All of them.")
    assert _ask(user, version, "groceries?", match="words", standalone=False,
                previous="How much did I spend on restaurants last month?") is None
    assert _ask(user, version, "groceries?", match="words")["answer"] == "All of them."       # opening a conversation
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM answer_cache WHERE user_id = CAST(:u AS UUID)"), {"u": user})
    _keep(user, "groceries?", "Last month's groceries.", standalone=False)
    assert _ask(user, version, "show my groceries purchases", match="words") is None
    assert _ask(user, version, "groceries expenses", match="words", standalone=False,
                previous="something else") is None
    assert _ask(user, version, "Groceries", match="words")["match"] == "exact"                # the same words, as ever


@needs_db
def test_a_bare_category_and_the_question_right_after_it_are_one(user):
    """ "coffee" anywhere in a conversation, then "coffee expenses": the
    question just asked, in other words. And the other way round."""
    version = _keep(user, "coffee", "Nine coffee purchases.", standalone=False, thread="here")
    hit = _ask(user, version, "coffee expenses", match="words", standalone=False, previous="Coffee?", thread_id="here")
    assert hit and hit["answer"] == "Nine coffee purchases." and hit["match"] == "words"
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM answer_cache WHERE user_id = CAST(:u AS UUID)"), {"u": user})
    _keep(user, "show my coffee expenses", "Nine coffee purchases.", standalone=False, thread="here")
    assert _ask(user, version, "coffee", match="words", standalone=False, previous="Show my coffee expenses",
                thread_id="here")["answer"] == "Nine coffee purchases."
    assert _ask(user, version, "coffee", match="words", standalone=False, previous="my rent", thread_id="here") is None


@needs_db
def test_the_question_right_before_is_this_conversations_not_anothers_in_the_same_words(user):
    """This conversation asked "coffee" and then "coffee expenses". The
    "coffee" that is kept was asked inside another conversation, after a
    question about last month: the same words, and not the question this
    conversation asked just before."""
    version = _keep(user, "coffee", "Coffee last month.", standalone=False, thread="another")
    assert _ask(user, version, "coffee expenses", match="words", standalone=False, previous="coffee",
                thread_id="this one") is None
    assert _ask(user, version, "coffee expenses", match="words", standalone=False, previous="coffee") is None
    assert _ask(user, version, "coffee expenses", match="words", standalone=False, previous="coffee",
                thread_id="another")["answer"] == "Coffee last month."


@needs_db
def test_a_question_no_one_can_read_does_not_switch_the_cache_off(user):
    """Kept beside others, "coffee last 1,000,000 days" used to raise in every lookup that met it."""
    version = _keep(user, "coffee last 1,000,000 days", "Too long ago.")
    _keep(user, "coffee", "Nine coffee purchases.")
    assert _ask(user, version, "coffee expenses", match="words")["answer"] == "Nine coffee purchases."
    assert _ask(user, version, "Coffee last 1,000,000 days", match="similar")["answer"] == "Too long ago."


@needs_db
def test_a_question_that_is_no_category_is_matched_word_for_word(user):
    """ "any charges over $100?" after "show my Costco purchases" is about
    Costco. It never answers "show my purchases over $100" asked elsewhere,
    in any wording but its own."""
    version = _keep(user, "any charges over $100?", "Two at Costco.", standalone=False)
    assert _ask(user, version, "show my purchases over $100", match="words") is None
    assert _ask(user, version, "charges over $100?", match="words", standalone=False,
                previous="any charges over $100?") is None
    assert _ask(user, version, "Any charges over $100?", match="words")["answer"] == "Two at Costco."


def _kept_on(uid: str, question: str, when: str) -> None:
    with get_engine().begin() as c:
        c.execute(text("UPDATE answer_cache SET created_at = CAST(:t AS TIMESTAMPTZ) "
                       "WHERE user_id = CAST(:u AS UUID) AND query_text = :q"), {"u": uid, "q": question, "t": when})


@needs_db
def test_an_answer_about_this_month_does_not_answer_for_the_next(user, monkeypatch):
    """Kept on October 31, "coffee this month" is October's. On November 1,
    with no purchase in between, it was replayed for the same words, and
    would now be for "coffee in november" as well."""
    from datetime import date

    from banko_ai.utils import clock
    version = _keep(user, "coffee this month", "October's coffee.")
    _keep(user, "coffee", "All coffee.")
    _kept_on(user, "coffee this month", "2026-10-31 22:00:00+00")
    _kept_on(user, "coffee", "2026-10-31 22:00:00+00")
    monkeypatch.setattr(clock, "today", lambda: date(2026, 11, 1))
    assert _ask(user, version, "coffee this month", match="words") is None
    assert _ask(user, version, "coffee expenses in november", match="words") is None
    assert _ask(user, version, "coffee expenses in october", match="words")["answer"] == "October's coffee."
    assert _ask(user, version, "coffee expenses", match="words")["answer"] == "All coffee."     # no dates, no day
    monkeypatch.setattr(clock, "today", lambda: date(2026, 10, 31))
    assert _ask(user, version, "Coffee this month?", match="exact")["answer"] == "October's coffee."


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
def test_a_merchant_of_ones_own_in_other_words_gets_the_answer(nothing_kept):
    """ "starbucks", then "starbucks purchases": one question, like "coffee" and "coffee expenses"."""
    version = _keep(nothing_kept, "starbucks", "Three at Starbucks.")
    for asked in ("starbucks purchases", "show my Starbucks charges", "purchases at starbucks"):
        hit = _ask(nothing_kept, version, asked, match="words")
        assert hit and hit["answer"] == "Three at Starbucks." and (hit["question"], hit["match"]) == ("starbucks", "words")
    assert _ask(nothing_kept, version, "starbucks purchases", match="exact") is None
    assert _ask(nothing_kept, version, "starbucks purchases last month", match="words") is None
    _keep(nothing_kept, "Trader Joe's purchases", "One at Trader Joe's.")
    assert _ask(nothing_kept, version, "my trader joes charges", match="words")["answer"] == "One at Trader Joe's."


@needs_db
def test_a_name_that_is_nobodys_merchant_here_is_matched_word_for_word(nothing_kept):
    """This user never bought at Ikea: what "ikea" asks is anybody's guess."""
    version = _keep(nothing_kept, "ikea", "Nothing at Ikea.")
    assert _ask(nothing_kept, version, "ikea purchases", match="words") is None
    assert _ask(nothing_kept, version, "Ikea?", match="words")["match"] == "exact"
    card_purchase(nothing_kept, "IKEA", "89.00", category="Shopping")       # now they have
    version = _keep(nothing_kept, "ikea", "One at Ikea.")
    assert _ask(nothing_kept, version, "ikea purchases", match="words")["answer"] == "One at Ikea."


@needs_db
def test_a_merchant_whose_name_begins_with_the_is_asked_for_with_or_without_it(nothing_kept):
    """The shape drops "the" in front of a name, and "The Home Depot" was no name it could find."""
    card_purchase(nothing_kept, "The Home Depot", "64.10", category="Shopping")
    version = _keep(nothing_kept, "home depot", "One at The Home Depot.")
    for asked in ("the home depot purchases", "home depot purchases", "purchases at the home depot",
                  "show my Home Depot charges"):
        assert _ask(nothing_kept, version, asked, match="words")["match"] == "words", asked


@needs_db
def test_a_bare_merchant_counts_where_a_bare_category_does(nothing_kept):
    version = _keep(nothing_kept, "starbucks", "Starbucks, last month.", standalone=False, thread="another")
    assert _ask(nothing_kept, version, "starbucks purchases", match="words") is None
    assert _ask(nothing_kept, version, "starbucks purchases", match="words", standalone=False, previous="starbucks",
                thread_id="another")["answer"] == "Starbucks, last month."


@needs_db
def test_a_users_merchants_are_read_once_for_each_state_of_their_money(nothing_kept, monkeypatch):
    """Reading them means reading the user's purchases. It is done when a
    name asks for it, and kept until a purchase changes what they are."""
    reads, real = [], C._merchant_names

    def counted(conn, user_id, region=None):
        reads.append(user_id)
        return real(conn, user_id, region)

    monkeypatch.setattr(C, "_merchant_names", counted)
    C.forget_merchants()
    version = _keep(nothing_kept, "starbucks", "Three at Starbucks.")
    assert _ask(nothing_kept, version, "Starbucks?", match="words")["match"] == "exact" and reads == []   # the same words
    assert _ask(nothing_kept, version, "coffee expenses", match="words") is None and reads == []           # a category
    assert _ask(nothing_kept, version, "starbucks purchases", match="exact") is None and reads == []       # not asked for
    assert _ask(nothing_kept, version, "starbucks purchases", match="words")["match"] == "words" and len(reads) == 1
    assert _ask(nothing_kept, version, "show my starbucks charges", match="words")["match"] == "words" and len(reads) == 1
    card_purchase(nothing_kept, "Peet's Coffee", "4.75")               # another state of their money
    version = _keep(nothing_kept, "peets coffee", "One at Peet's.")
    assert _ask(nothing_kept, version, "peet's coffee purchases", match="words")["match"] == "words" and len(reads) == 2
    C.forget_merchants()
    assert _ask(nothing_kept, version, "my peets coffee charges", match="words")["match"] == "words" and len(reads) == 3


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


@needs_db
def test_the_region_is_asked_for_before_a_connection_is_held(nothing_kept, monkeypatch):
    """read_region may open a connection of its own. Asked for inside the
    answer cache's transaction, a pool that is full would wait on itself."""
    inside, asked, real = [False], [], C.run_transaction

    def tracked(fn, **how):
        def run(conn):
            inside[0] = True
            try:
                return fn(conn)
            finally:
                inside[0] = False
        return real(run, **how)

    monkeypatch.setattr(C, "run_transaction", tracked)
    monkeypatch.setattr(C, "read_region", lambda user, url: asked.append(inside[0]) or None)
    C.forget_merchants()
    version = _keep(nothing_kept, "starbucks", "Three at Starbucks.")
    assert _ask(nothing_kept, version, "starbucks purchases", match="words")["match"] == "words"
    assert asked == [False]


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
    assert _ask(user, version, "show my coffee purchases", shape=usual, unshaped=True)["match"] == "words"
    assert _ask(user, version, "coffee expenses", shape=mine) is None
    _keep(user, "coffee expenses", "Shaped by three a page.", shape=mine)
    assert _ask(user, version, "coffee expenses", shape=usual, unshaped=True)["answer"] == "Kept before the upgrade."



@needs_db
def test_an_answer_shaped_by_other_settings_is_not_replayed(user):
    """Three purchases a page make another answer than ten. Whoever has ten
    must not get the three, in the same words or in other ones."""
    version = _keep(user, "coffee", "Three of them.", shape="page=3")
    assert _ask(user, version, "coffee", shape="page=10") is None
    assert _ask(user, version, "coffee expenses", shape="page=10") is None
    assert _ask(user, version, "coffee") is None                                  # no shape named is not this shape
    assert _ask(user, version, "coffee expenses", shape="page=3")["answer"] == "Three of them."
    _keep(user, "coffee", "Ten of them.", shape="page=10")
    assert _ask(user, version, "coffee", shape="page=10")["answer"] == "Ten of them."
    assert _ask(user, version, "coffee", shape="page=3")["answer"] == "Three of them."


# A similar question.

@needs_db
def test_a_similar_question_gets_the_answer_when_similarity_is_asked_for(user):
    version = _keep(user, "any unusual charges lately")
    asked = "have there been any strange charges recently"
    assert _ask(user, version, asked, match="words") is None
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
    assert C.ask_words(asked) not in (C.ask_words(near), C.ask_words(far))
    version = _keep(user, near, "today's")
    _keep(user, far, "the recent ones")
    refused: list = []
    hit = _ask(user, version, asked, match="similar", similarity=0.75, refused=refused)
    assert hit and hit["answer"] == "the recent ones" and hit["match"] == "similar"
    assert refused == []                                               # something answered: nothing to explain
    apart = _apart(user, asked)
    assert apart[near] < apart[far] <= 0.25, apart                     # the refused one really was the nearer


@needs_db
def test_the_same_question_in_other_words_wins_over_a_nearer_similar_one(user):
    """Both would answer: "coffee costs" is similar and nearer, the long
    wording is the same question and farther. The same question wins."""
    asked, similar, same = "coffee expenses", "coffee costs", "could you please show me my coffee purchases"
    assert C.other_ask(similar, asked) is None and C.ask_words(similar) != C.ask_words(asked) == C.ask_words(same)
    version = _keep(user, similar, "what it costs")
    _keep(user, same, "the purchases")
    hit = _ask(user, version, asked, match="similar", similarity=0.5)
    assert hit and hit["answer"] == "the purchases" and hit["match"] == "words"
    apart = _apart(user, asked)
    assert apart[similar] < apart[same] <= 0.5, apart
    assert _ask(user, version, "coffee outlay", match="similar", similarity=0.5)["match"] == "similar"
