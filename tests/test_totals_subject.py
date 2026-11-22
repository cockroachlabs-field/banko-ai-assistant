"""A total keeps what it is a total of.

"coffee total", "how much coffee" and "how much is my rent" were answered
for all spending: $7,631.76 "on all categories" on an account whose coffee
came to $151.97. The classifier looked for the subject only behind on, for or
at, or between total and a word like spending. Anywhere else the subject was
dropped, and the sum ran over everything. "How much do I have" got the same
answer, and so did "how much do I spend a month".

Now a total has to account for every word of its question, as the size route
does. A total's own words name nothing (who asks, the asking, the counting,
the spending, its rows); the dates are read and cut out; whatever is left is
what the total is of. A spending category or one of the user's merchants,
whole and exactly, gets its total from SQL. Anything else goes to the
planner, never to a total over everything.

The other half is as much a rule: a plain total stays on SQL however it is
dressed. tests/fixtures/plain_totals.tsv holds the everyday phrasings."""

import os
from datetime import date
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import text

from banko_ai.assistant import background
from banko_ai.assistant import graph as G
from banko_ai.assistant.router import route
from banko_ai.db.engine import get_engine
from banko_ai.services import threads
from banko_ai.utils import aggregations as A
from banko_ai.utils.intent_classifier import classify_aggregation, extract_window
from tests._chat import ScriptedChat
from tests._money import card_purchase, drop_user, seeded_diner

TODAY = date(2026, 11, 20)
FIXTURES = Path(__file__).parent / "fixtures"
needs_db = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


def _read(question: str, today: date = TODAY):
    """How the totals route reads a question: None (not its question),
    "everything", or (placed | loose, subject)."""
    intent = classify_aggregation(question, today=today)
    if intent is None:
        return None
    if intent.subject is None:
        return "everything"
    return ("loose" if intent.loose else "placed", intent.subject)


def _lines(name: str) -> list[str]:
    return [line for line in (FIXTURES / name).read_text().splitlines() if line.strip() and not line.startswith("#")]


# The subject, wherever it stands.

@pytest.mark.parametrize("question, subject", [
    ("coffee total", "coffee"),
    ("total coffee", "coffee"),
    ("how much coffee", "coffee"),
    ("how much coffee this month?", "coffee"),
    ("sum of my coffee purchases", "coffee"),
    ("how many coffee purchases this month", "coffee"),
    ("how many times did I buy coffee", "coffee"),
    ("total groceries last month", "groceries"),
    ("how much is my rent", "rent"),
    ("how much went to rent", "rent"),
    ("how much did groceries cost me last month", "groceries"),
    ("how much do I spend eating out", "eating out"),
    ("starbucks total", "starbucks"),
    ("how much does netflix cost me", "netflix"),
    ("how much blue bottle coffee last month", "blue bottle coffee"),
    ("thanks, and my coffee total?", "coffee"),
    ("Banko, can you tell me how much coffee this month", "coffee"),
])
def test_a_total_names_its_subject_wherever_the_subject_stands(question, subject):
    assert _read(question) == ("loose", subject), question


@pytest.mark.parametrize("question, subject", [
    ("how much did I spend on coffee last month", "coffee"),
    ("how much did I spend at trader joe's this year", "trader joe's"),
    ("how much did I spend at whole foods market", "whole foods market"),   # "whole" is its name here
    ("how much did I spend at best buy", "best buy"),                       # and so is "buy"
    ("how much did I spend on Top Golf purchases last month?", "top golf purchases"),
    ("what's my total for groceries", "groceries"),
    ("how many purchases at starbucks this month", "starbucks"),
    ("how many times did I shop at target", "target"),
    ("how many times did I eat at chipotle", "chipotle"),
    ("average grocery spend", "grocery"),
    ("total restaurant expenses", "restaurant"),
    ("thanks, how much did I spend on coffee last month?", "coffee"),
    ("Banko, can you tell me how much I spent on coffee", "coffee"),
    ("how much did I actually end up spending on coffee, roughly", "coffee"),
    ("how much did I spend on moon bases", "moon bases"),                   # not found is said, as before
])
def test_a_subject_behind_on_for_or_at_is_read_as_it_always_was(question, subject):
    assert _read(question) == ("placed", subject), question


@pytest.mark.parametrize("question, subject", [
    ("how much did I spend at the dollar store", "the dollar store"),
    ("how much did I spend at the store", "the store"),
    ("how much did I spend on running", "running"),
    ("how much did I spend at pay pal", "pay pal"),
    ("how much did I spend on myself", "myself"),
    ("total dollar store spending", "dollar store"),
])
def test_a_name_made_of_a_totals_own_words_is_a_name_behind_on_for_or_at(question, subject):
    """ "at the dollar store" is nothing but words a total calls its own. By
    its place it is a subject all the same: dropped, the answer was the
    total of everything."""
    assert _read(question) == ("placed", subject), question


@pytest.mark.parametrize("question", ["how much did I spend at all last month", "how much did I spend on average",
                                      "how much did I spend on everything", "how much did I spend on stuff"])
def test_emphasis_behind_on_for_or_at_is_still_no_subject(question):
    assert _read(question) == "everything", question


@pytest.mark.parametrize("question, subject", [
    ("how much did I spend on coffee though", "coffee"),
    ("how much did I spend on coffee overall", "coffee"),
    ("how much did I spend on coffee so far this month", "coffee"),
    ("how much have I spent on coffee to date", "coffee"),
    ("how much did I spend on rent for the past year", "rent"),
    ("how much did I spend on coffee back in october", "coffee"),
    ("how much did I spend on coffee within the last 30 days", "coffee"),
    ("total of my coffee purchases", "coffee"),
    ("total number of coffee purchases", "coffee"),
    ("how much did I spend at amazon.com", "amazon"),
    ("how much did I spend shopping for groceries", "groceries"),
])
def test_what_no_name_ends_with_is_no_part_of_the_subject(question, subject):
    """Each of these said it could not find a category called "coffee
    though", "coffee so far", "rent for" or "of my coffee"."""
    assert _read(question) == ("placed", subject), question


def test_a_month_after_for_is_the_window_not_the_subject():
    """ "give me my total expenses for october" said it could not find a
    category called october. In January "for december" is the December just
    gone, as "in december" is."""
    october = (date(2026, 10, 1), date(2026, 11, 1))
    assert extract_window("give me my total expenses for october", today=TODAY) == october
    assert _read("give me my total expenses for october") == "everything"
    groceries = classify_aggregation("total groceries for october", today=TODAY)
    assert groceries.subject == "groceries" and (groceries.window_start, groceries.window_end) == october
    january = date(2027, 1, 15)
    for question in ("give me my total expenses for december", "how much did I spend in december"):
        assert extract_window(question, today=january) == (date(2026, 12, 1), date(2027, 1, 1)), question
    assert _read("total groceries for december", january) == ("loose", "groceries")
    assert extract_window("how much did I spend for the month of october", today=TODAY) == october
    assert extract_window("total for june 2025", today=TODAY) == (date(2025, 6, 1), date(2025, 7, 1))
    # Asked in November, "for december" is last December; the word that makes it a plan stays with the question.
    assert _read("how much can I spend for december") == ("loose", "can")


# A plain total stays a plain total.

@pytest.mark.parametrize("line", _lines("plain_totals.tsv"))
def test_an_everyday_total_of_everything_stays_on_sql(line):
    """Greetings, tags, asking to be told, emphasis, other words for total
    and spend, slips of the keyboard: none of them is a subject. A word of a
    total's own that the list lacks would send its question to the planner."""
    question, operation, start, end = line.split("\t")
    intent = classify_aggregation(question, today=TODAY)
    assert intent is not None and intent.subject is None, (question, intent)
    assert (intent.operation, str(intent.window_start), str(intent.window_end)) == (operation, start, end), question


@pytest.mark.parametrize("question", _lines("not_plain_totals.txt"))
def test_a_question_a_total_cannot_answer_is_never_answered_with_one(question):
    """ "how much do I have", "how much was I paid", "how much was my last
    purchase", "how much do I spend a month", "how much did that cost": each
    used to come back as the sum of the last 90 days."""
    assert _read(question) != "everything", question


@pytest.mark.parametrize("question, left", [
    ("how much do I have", "have"),                 # a balance
    ("how much do I owe", "owe"),
    ("how much did I make", "make"),                # income
    ("how much was I paid", "income"),
    ("how much did I save", "save"),
    ("how much did I spend online", "online"),      # a way of paying no category holds
    ("how much did I spend in cash", "cash"),
    ("how much can I spend this month", "can"),     # what is allowed, not what was spent
    ("how much will I spend", "will"),
    ("how much do I need", "need"),
    ("how much did it cost", "it"),
    ("how much did that cost", "that"),
    ("how much is there", "there"),
    ("how much was my last purchase", "last"),      # one purchase
    ("how much more did I spend this month", "more"),
    ("how much did I go over", "go over"),
    ("how much did I go over this month", "go over"),  # with the dates behind it too
    ("how much did I pay back last month", "back"),
    ("how much did my spending go up", "go up"),
    ("how much did I spend on my top 5 purchases", "top 5"),
    ("how many purchases over $100", "over 100"),
    ("where did I spend the most last week?", "where most"),
    ("how much did I spend between october and november", "between october november"),
    ("how much did I spend excluding amazon", "excluding amazon"),
    ("how much did I spend in total, including Amazon?", "including amazon"),
    ("how many expensive purchases did I make", "expensive"),   # not a mistyped "expenses"
    ("how much did I send", "send"),                            # not a mistyped "spend"
    ("how much is tonal", "tonal"),                             # not a mistyped "total"
    ("how much кофе", "кофе"),                                  # a word in any script is a word
])
def test_a_word_a_total_cannot_place_is_kept(question, left):
    """Kept, the graph looks it up and hands the question to the planner.
    Dropped, the answer was a total over everything."""
    assert _read(question) == ("loose", left), question
    assert route(question, today=TODAY).as_dict()["intent"]["loose"] is True


@pytest.mark.parametrize("question", [
    # A balance, money coming in, a payment method.
    "how much is in my account", "how much money is in my account", "what's the total in my account",
    "how much is my pay", "total pay", "how much did I get payed", "how much was paid to me",
    "how much do I have after all expenses", "how much did I end up with",
    "how much did uber pay me", "how much did amazon give me", "how much did I spend with amazon pay",
    # Everything but.
    "total of everything but rent", "everything but rent total", "how much did I spend all but groceries",
    "how much did I spend on everything but amazon",
    # What is allowed, planned or advised.
    "how much can I get", "how much could I get", "how much can I ask for",
    "how much am I going to spend this month", "how much I'll spend this month",
    "how much is okay to spend this month", "how much is okay to spend on groceries",
    "how much is too much for rent", "how much streaming is too much",
    "did I spend more or less this month",
    # Where, whom, which one.
    "how much did I spend in store", "how much did I spend in the US", "how much did I pay my buddy",
    "how much do I pay my assistant", "how many purchases did I make there", "how much was the purchase",
    "how much was mine", "how much did I spend per store", "total by merchant",
    # Things, not purchases.
    "how many stores did I shop at", "how many merchants did I pay this month", "how many categories do I spend on",
    "number of stores", "how many items did I buy", "how many times did I order in",
    "how many times did I deposit in my account",
    # A span of time with two ends, or one that is not the window's.
    "how much did I spend from last month to today", "how much did I spend on coffee from last month to today",
    "how much have I spent on groceries through today", "how much have I spent as of yesterday",
    "how much did I spend on coffee as of last month", "how much do I spend during the week",
])
def test_sql_gives_no_total_to_a_question_that_asks_for_something_else(question):
    """Each of these got a total from SQL: of everything, or of the one
    category or merchant the question happened to name. Now the classifier
    declines, or keeps words that are no category's name, and the graph
    hands the question to the planner."""
    read = _read(question)
    assert read is None or (read[0] == "loose" and A.spending_category(read[1]) is None), (question, read)


@pytest.mark.parametrize("question, left", [
    ("how much did uber pay me", "income"),
    ("how much does my employer pay me", "income"),
    ("how much uber paid me last month", "uber income"),
    ("how much starbucks pays me", "starbucks income"),
    ("how much amazon gave me", "amazon income"),
    ("how much did I spend with amazon pay", "wallet"),
    ("how much did I spend with apple pay this month", "wallet"),
    ("how much did I spend on everything but amazon", "except amazon"),
    ("everything but best buy total", "than best"),
])
def test_a_merchant_that_pays_or_a_wallet_is_not_a_place_money_was_spent(question, left):
    """ "how much did uber pay me" came back as what was spent at Uber, and
    "with amazon pay" as what was spent at Amazon. A word is kept in their
    place, so the merchant's name alone is never what is left."""
    assert _read(question) == ("loose", left), question


def test_i_mean_is_not_the_mean():
    """ "I mean, how much did I spend" got the average purchase."""
    for question in ("I mean how much did I spend", "I mean, how much did I spend on coffee"):
        assert classify_aggregation(question, today=TODAY).operation == "sum", question
    assert _read("I mean how much did I spend") == "everything"


@pytest.mark.parametrize("question", [
    "I'm not sure how much I spent this month", "I don't know how much I've spent this month",
    "I have no idea how much I spent last month", "I forgot how much I spent last month",
    "I'm trying to figure out how much I spent last month", "just checking how much I spent this month",
    "how much did I spend across all my accounts", "how much did I spend across all my cards",
    "how much money went out of my account", "how much money left my account",
    "how much did I spend for each category", "how much did I spend only this month",
    "how much did I spend this month only", "how much did I spend earlier today",
])
def test_more_plain_totals_that_stay_on_sql(question):
    assert _read(question) == "everything", question


@pytest.mark.parametrize("question", [
    "how much can I spend on groceries this month",      # what is allowed
    "how much should I spend on groceries",
    "how much did I save on groceries",
    "how much do I have left for groceries",
    "how much more did I spend on coffee this month",
    "how much did I spend on coffee compared to last month",
    "how much did I spend on coffee per month",          # a rate
    "how much did I spend on coffee last weekend",       # a time no window reads
    "How much did I spend on groceries, and what was my biggest purchase?",
])
def test_more_asked_than_a_total_of_the_subject_is_the_planners(question):
    """Behind on, for or at the subject is found, and the total of it used
    to be the whole answer, whatever else the question asked."""
    assert _read(question) is None, question
    assert route(question, today=TODAY).kind == "plan"


# Time.

@pytest.mark.parametrize("question, left", [
    ("how much do I spend a month", "month"),
    ("how much do I spend per month", "per month"),
    ("how much do I spend each week", "each week"),
    ("how much do I spend monthly", "monthly"),
    ("what's my average monthly spending", "monthly"),
    ("average spend per day", "per day"),
    ("how much did I spend last weekend", "last weekend"),
    ("how much did I spend on weekends", "weekends"),
    ("how much did I spend at night", "night"),
    ("how much did I spend this month and last month", "last month"),
    ("how much have I ever spent", "ever"),
    ("how much did I spend all time", "time"),
    ("how much have I spent since last month", "since"),
    ("how much did I pay for that", "that"),
])
def test_time_no_window_took_is_never_dropped(question, left):
    """A rate ("a month" got the sum of ninety days), a window nobody reads
    ("last weekend" got ninety days too) or a second one. A window that was
    read is cut out, so a word of time left over is always one of these."""
    assert _read(question) == ("loose", left), question


@pytest.mark.parametrize("question, window", [
    ("how much did I spend during the past year", (date(2025, 11, 21), date(2026, 11, 21))),
    ("how much did I spend over the last year", (date(2025, 11, 21), date(2026, 11, 21))),
    ("how much did I spend this month to date", (date(2026, 11, 1), date(2026, 12, 1))),
    ("how much did I spend this year to date", (date(2026, 1, 1), date(2026, 11, 21))),
    ("how much did I spend this past month", (date(2026, 10, 21), date(2026, 11, 21))),
    ("how much did I spend in the last month", (date(2026, 10, 21), date(2026, 11, 21))),
    ("how much did I spend in the last week", (date(2026, 11, 13), date(2026, 11, 21))),
    ("how much did I spend these past 30 days", (date(2026, 10, 21), date(2026, 11, 21))),
    ("how much did I spend over the course of the last 30 days", (date(2026, 10, 21), date(2026, 11, 21))),
    ("how much have I spent since january 1", (date(2026, 1, 1), date(2026, 11, 21))),
    ("what's this months total", (date(2026, 11, 1), date(2026, 12, 1))),
    ("what's last months total", (date(2026, 10, 1), date(2026, 11, 1))),
    ("how much did I spend in the current calendar year", (date(2026, 1, 1), date(2026, 11, 21))),
    ("how much did I spend year to date", (date(2026, 1, 1), date(2026, 11, 21))),
    ("how much did I spend year-to-date", (date(2026, 1, 1), date(2026, 11, 21))),
    ("total spending ytd", (date(2026, 1, 1), date(2026, 11, 21))),
    ("how much did I spend month to date", (date(2026, 11, 1), date(2026, 12, 1))),
    ("what's my total for the year", (date(2026, 1, 1), date(2026, 11, 21))),
    ("how much did I spend this whole month", (date(2026, 11, 1), date(2026, 12, 1))),
    ("how much did I spend in the last two weeks", (date(2026, 11, 6), date(2026, 11, 21))),
    ("how much did I spend in the previous 30 days", (date(2026, 10, 21), date(2026, 11, 21))),
    ("how much did I spend throughout 2025", (date(2025, 1, 1), date(2026, 1, 1))),
])
def test_windows_the_totals_read(question, window):
    """Most of these were answered for the last 90 days, which only the
    small print said. "The past year" is the 365 days up to today: a day
    more, and the first of the month falls into it twice."""
    assert extract_window(question, today=TODAY) == window, question
    assert _read(question) == "everything"


def test_last_month_and_last_year_are_still_the_calendar_ones():
    assert extract_window("how much did I spend last month", today=TODAY) == (date(2026, 10, 1), date(2026, 11, 1))
    assert extract_window("how much did I spend last week", today=TODAY) == (date(2026, 11, 9), date(2026, 11, 16))
    assert extract_window("how much did I spend last year", today=TODAY) == (date(2025, 1, 1), date(2026, 1, 1))


def test_during_the_week_names_no_window():
    """It means the weekdays. Read as this week, it also cut "how often do I go to costco during the week" down to it."""
    assert extract_window("how often do I go to costco during the week", today=TODAY) is None
    assert extract_window("what's my total for the week", today=TODAY) == (date(2026, 11, 16), date(2026, 11, 21))


@pytest.mark.parametrize("question", ["how much did I spend up to today", "how much did I spend as of today",
                                      "how much have I spent until today", "how much have I spent so far"])
def test_up_to_today_is_not_today_alone(question):
    """ "as of today" was answered with today's purchases, on the totals and on the visits route."""
    intent = classify_aggregation(question, today=TODAY)
    assert intent.subject is None and (intent.window_end - intent.window_start).days == 90, question
    assert extract_window(question, today=TODAY) is None


def test_what_was_last_months_total_is_last_months():
    intent = classify_aggregation("what was last month's total", today=TODAY)
    assert intent.subject is None and intent.window_start == date(2026, 10, 1)
    assert _read("this month's coffee total") == ("loose", "coffee")


def test_a_long_run_of_spaces_or_hyphens_costs_nothing():
    """Forty thousand spaces took seventeen seconds."""
    import time
    t0 = time.perf_counter()
    assert _read("how much did I spend" + " " * 40000 + "today") == "everything"
    assert _read("how much did I spend " + "-" * 40000 + " today") is None           # too long to be a total
    assert _read("how much " + "really " * 5000 + "did I spend") is None
    assert time.perf_counter() - t0 < 2


# Counting.

@pytest.mark.parametrize("question", ["how many subscriptions am I paying for", "how many concerts did I go to",
                                      "how many flights did I buy", "how many subscriptions do I have",
                                      "how many payments did I make", "how many did I buy", "how many this month"])
def test_a_count_counts_purchases_and_nothing_else(question):
    """ "how many subscriptions" is not how many times one was charged, and a payment may be a card payment."""
    assert _read(question) is None, question


def test_how_many_dollars_is_a_sum():
    intent = classify_aggregation("how many dollars did I spend this month", today=TODAY)
    assert intent.operation == "sum" and intent.subject is None


# The list itself.

def test_no_category_is_one_of_a_totals_own_words():
    """A category in that list would be a subject dropped again."""
    from banko_ai.config.catalog import get_catalog
    from banko_ai.utils.intent_classifier import _GET, _HAVE, _MAKE, _TOTAL_WORDS, _meant
    names = set()
    for c in get_catalog().categories:
        for name in (c.key.replace("_", " "), c.label, *c.aliases):
            names.update(name.lower().split())
    own = _TOTAL_WORDS | _HAVE | _GET | _MAKE
    assert not names & own, names & own
    assert not [n for n in names if _meant(n)], [n for n in names if _meant(n)]


# A word a total calls its own for no reason is a subject waiting to be
# dropped. Each one is needed: by an everyday total in the fixture, or by a
# total with a subject, which only comes out when the word beside it is not
# taken for one.
_NEEDED_BESIDE_A_SUBJECT = {
    "by": ("how much was I charged by netflix", "netflix"),
    "from": ("how many charges from netflix", "netflix"),
    "with": ("how much did I spend with amazon", "amazon"),
    "me": ("how much does netflix cost me", "netflix"),
    "went": ("how much went to rent", "rent"),
}


def test_every_word_a_total_calls_its_own_is_needed(monkeypatch):
    from banko_ai.utils import intent_classifier as I
    rows = [line.split("\t") for line in _lines("plain_totals.tsv")]
    words_of = {row[0]: set(I._A_WORD.findall(row[0].lower().replace("'", "").replace("’", ""))) for row in rows}
    own, unneeded = I._TOTAL_WORDS, []

    def still_plain(question, operation, start, end):
        intent = classify_aggregation(question, today=TODAY)
        return (intent is not None and intent.subject is None
                and (intent.operation, str(intent.window_start), str(intent.window_end)) == (operation, start, end))

    for word in sorted(own):
        monkeypatch.setattr(I, "_TOTAL_WORDS", own - {word})
        with_it = [row for row in rows if word in words_of[row[0]]]
        if I._meant(word) is not None and with_it:
            continue                      # "spent" without itself is still read, as a slip of "spend"
        if any(not still_plain(*row) for row in with_it):
            continue
        question, subject = _NEEDED_BESIDE_A_SUBJECT.get(word, ("", ""))
        if not question or _read(question) == ("loose", subject):
            unneeded.append(word)
    monkeypatch.setattr(I, "_TOTAL_WORDS", own)
    assert not unneeded, unneeded
    for question, subject in _NEEDED_BESIDE_A_SUBJECT.values():
        assert _read(question) == ("loose", subject), question


def test_a_catalog_of_ones_own_keeps_its_names(monkeypatch):
    """CATEGORIES_FILE may call a category Money or Charges: its name is never a total's own word."""
    from banko_ai.config import catalog
    from banko_ai.utils import intent_classifier as I
    packaged = catalog.get_catalog()

    class Own:
        categories = [*packaged.categories, type("C", (), {"key": "charges", "label": "Charges", "aliases": ("fees",)})()]

    monkeypatch.setattr(catalog, "get_catalog", lambda: Own())
    monkeypatch.setattr(I, "_categories", None)
    assert _read("total charges") == ("loose", "charges")
    monkeypatch.undo()
    I._categories = None
    assert _read("total charges") == "everything"


@pytest.mark.parametrize("typed", ["how much did i spnd", "how much did i sepnd", "how mcuh did i spend",
                                   "how much have i spnt", "total expences", "how many purchses did I make",
                                   "how much did I spend in ttoal"])
def test_a_slip_of_the_keyboard_is_no_subject(typed):
    assert _read(typed) == "everything", typed


def test_the_assistants_name_is_a_form_of_address_only(monkeypatch):
    """Every word of BRAND_NAME used to be ignored wherever it stood: with
    "Summit Savings" for a name, "how much is in my savings" was a total."""
    from banko_ai.config.settings import get_config
    monkeypatch.setattr(get_config(), "brand_name", "Summit Savings")
    assert _read("Summit Savings, how much did I spend last month?") == "everything"
    assert _read("how much is in my savings") == ("loose", "savings")


# Through the assistant.

@pytest.fixture(scope="module")
def member():
    uid = seeded_diner(2)
    for n in range(4):
        card_purchase(uid, "Blue Bottle Coffee", f"{4 + n}.50")
    card_purchase(uid, "Best Buy", "129.99", category="Shopping")
    card_purchase(uid, "Hallmark Cards", "12.00", category="Shopping")
    card_purchase(uid, "Target", "41.20", category="Shopping")
    card_purchase(uid, "The Home Depot", "64.10", category="Shopping")
    yield uid
    with get_engine().begin() as c:
        for t in ("search_cache", "answer_cache", "memories", "assistant_actions", "receipts"):
            c.execute(text(f"DELETE FROM {t} WHERE user_id = CAST(:u AS UUID)"), {"u": uid})
    drop_user(uid)


def _ask(member, message, chat):
    g, thread = G.build_graph(InMemorySaver()), threads.create_thread(member)["thread_id"]
    try:
        out = G.run_turn(member, thread, message, chat=chat, graph=g)
        background.drain()
        return out
    finally:
        threads.delete_thread(member, thread)


def _coffee(member) -> tuple[float, int]:
    with get_engine().connect() as c:
        total, n = c.execute(text(
            "SELECT sum(expense_amount), count(*) FROM expenses WHERE user_id = CAST(:u AS UUID) "
            "AND shopping_type = 'Coffee' AND expense_date > current_date - 90"), {"u": member}).one()
    return float(total), int(n)


def _planner():
    return ScriptedChat('{"steps": []}', '{"steps": []}', "Let me look at that another way.")


@needs_db
@pytest.mark.parametrize("question", ["coffee total", "how much coffee", "total coffee", "sum of my coffee purchases",
                                      "thanks, how much coffee?"])
def test_a_coffee_total_is_coffee(member, question):
    total, n = _coffee(member)
    out = _ask(member, question, ScriptedChat("- Steady."))
    first = out["answer"].splitlines()[0]
    assert out["route"]["kind"] == "aggregate"
    assert f"**${total:,.2f}** on Coffee" in first and f"({n} transactions)" in first, first


@needs_db
def test_a_total_at_a_merchant_is_that_merchants(member):
    out = _ask(member, "blue bottle coffee total", ScriptedChat("- Steady."))
    first = out["answer"].splitlines()[0]
    assert out["route"]["kind"] == "aggregate" and "Blue Bottle Coffee" in first and "all categories" not in first, first


@needs_db
def test_a_merchant_whose_name_holds_a_totals_own_word_is_still_found(member):
    """ "best buy total": "buy" is a word of any total, and Best Buy is where the money went."""
    out = _ask(member, "best buy total", ScriptedChat("- Steady."))
    first = out["answer"].splitlines()[0]
    assert out["route"]["kind"] == "aggregate" and "Best Buy" in first and "$129.99" in first, first


@needs_db
@pytest.mark.parametrize("question", ["how much do I have", "how much did I save", "how much did I spend online",
                                      "how much do I spend a month", "how much did I spend last weekend",
                                      "how much was my last purchase"])
def test_a_total_of_something_banko_cannot_place_goes_to_the_planner(member, question):
    """It used to come back as all spending, said with a straight face."""
    asked = _planner()
    out = _ask(member, question, asked)
    assert out["route"]["kind"] == "plan" and "all categories" not in out["answer"]
    assert out["answer"] == "Let me look at that another way." and asked.calls


@needs_db
@pytest.mark.parametrize("question", ["other total", "what's my total income", "how much did I transfer",
                                      "how much shipping"])
def test_a_word_that_names_no_place_money_is_spent_goes_to_the_planner(member, question):
    """Other, Income and Transfer are categories, and none is a place money
    is spent: a total of one would be a line of "No matching transactions".
    "shipping" is not Shopping: behind on, for or at a near miss is a fair
    guess, for a word nothing placed it is another question's answer."""
    out = _ask(member, question, _planner())
    assert out["route"]["kind"] == "plan" and out["answer"] == "Let me look at that another way.", out["answer"]


@needs_db
@pytest.mark.parametrize("question", ["how much are my bills", "total bills", "how much is my phone",
                                      "target total", "how much is my target"])
def test_a_name_that_may_be_something_else_alone_goes_to_the_planner(member, question):
    """ "my bills" are more than the Utilities category, "my phone" may be
    the device, and "my target" a savings target, though this member shops
    at Target. Alone, such a name is the planner's, as it is no question of
    its own to the answer cache."""
    out = _ask(member, question, _planner())
    assert out["route"]["kind"] == "plan" and out["answer"] == "Let me look at that another way.", out["answer"]


@needs_db
def test_said_outright_such_a_name_is_read_as_it_always_was(member):
    out = _ask(member, "how much did I spend at target", ScriptedChat("- Steady."))
    first = out["answer"].splitlines()[0]
    assert out["route"]["kind"] == "aggregate" and "$41.20" in first and "Target" in first, first


@needs_db
def test_a_merchant_stored_with_the_is_found_without_it(member):
    out = _ask(member, "home depot total", ScriptedChat("- Steady."))
    first = out["answer"].splitlines()[0]
    assert out["route"]["kind"] == "aggregate" and "$64.10" in first and "The Home Depot" in first, first


@needs_db
def test_a_total_that_names_nothing_of_the_users_reads_their_purchases_once(member, monkeypatch):
    """ "what's the sum of 2 and 2" is a total by its words, so the user's
    merchants are asked about before it is turned away. They are kept: the
    next such message reads nothing."""
    from banko_ai.services import caches
    reads, read_names = [], caches._merchant_names
    monkeypatch.setattr(caches, "_merchant_names",
                        lambda conn, user, region=None: reads.append(user) or read_names(conn, user, region))
    monkeypatch.setattr(A, "_merchants_of", lambda user, url: reads.append("names") or [])
    caches.forget_merchants()
    unused = ScriptedChat("unused")
    for question in ("what's the sum of 2 and 2", "total eclipse of the heart", "average rainfall in seattle"):
        assert _ask(member, question, unused)["route"]["kind"] == "offtopic", question
    assert reads == [member] and unused.calls == []


@needs_db
def test_a_category_nothing_was_bought_in_goes_to_the_planner():
    """ "how much electronics" from someone with no such purchase: the word may not have been the subject."""
    uid = seeded_diner(1)
    try:
        with get_engine().begin() as c:
            c.execute(text("DELETE FROM expenses WHERE user_id = CAST(:u AS UUID) AND shopping_type = 'Electronics'"),
                      {"u": uid})
        out = _ask(uid, "how much electronics", _planner())
        assert out["route"]["kind"] == "plan" and out["answer"] == "Let me look at that another way."
    finally:
        drop_user(uid)


@needs_db
@pytest.mark.parametrize("question", [
    "how much did I spend in total, including Best Buy?",      # everything, said twice
    "how much did I spend excluding best buy",                 # everything but
    "how much best buy and blue bottle coffee",                # two of them
    "how much do I owe best buy",                              # another question about it
    "how much did best buy go up",
    "how much is my best buy card",
])
def test_a_merchant_beside_other_words_does_not_take_the_total(member, question):
    """A merchant named anywhere in the message used to be enough, and
    "excluding best buy" came back as the total at Best Buy."""
    out = _ask(member, question, _planner())
    assert out["route"]["kind"] == "plan" and "Best Buy" not in out["answer"], out["answer"]


@needs_db
def test_the_words_left_over_name_something_only_whole_and_exactly(member):
    """Not a near miss, and not a name beside other words."""
    assert A.loose_subject("hallmark cards total", "hallmark cards", member, None) == (None, "Hallmark Cards")
    assert A.loose_subject("card total", "card", member, None) == (None, None)             # not Hallmark Cards
    assert A.loose_subject("how much hallmark", "hallmark", member, None) == (None, None)
    assert A.loose_subject("best buy total", "best", member, None) == (None, "Best Buy")    # "buy" went to the total
    assert A.loose_subject("how much is best", "best", member, None) == (None, None)        # not named whole
    assert A.loose_subject("how much coffee", "coffee", member, None) == ("Coffee", None)
    assert A.loose_subject("how much shipping", "shipping", member, None) == (None, None)   # not Shopping
    assert A.loose_subject("the other day", "other", member, None) == (None, None)
    assert A.loose_subject("total income", "income", member, None) == (None, None)
    assert A.loose_subject("excluding best buy", "excluding best", member, None) == (None, None)
    # With the names the caller has, the purchases are read only for a merchant's stored name.
    known = {"best buy", "hallmark cards", "home depot"}
    assert A.loose_subject("best buy total", "best", member, None, known) == (None, "Best Buy")
    assert A.loose_subject("home depot total", "home depot", member, None, known) == (None, "The Home Depot")
    assert A.loose_subject("how much coffee", "coffee", "nobody", None, known) == ("Coffee", None)
    assert A.loose_subject("how much sushi", "sushi", "nobody", None, known) == (None, None)


@needs_db
def test_a_total_at_a_merchant_the_money_check_does_not_know_comes_from_sql(member):
    """ "total zebra crossing" says nothing about money to the check that
    decides, and was turned away. For someone who has bought at a place of
    that name it is a total."""
    unused = ScriptedChat("unused")
    before = _ask(member, "total zebra crossing", unused)
    assert before["route"]["kind"] == "offtopic" and unused.calls == []
    card_purchase(member, "Zebra Crossing", "18.40", category="Restaurant")
    out = _ask(member, "total zebra crossing", ScriptedChat("- Steady."))
    first = out["answer"].splitlines()[0]
    assert out["route"]["kind"] == "aggregate" and "$18.40" in first and "Zebra Crossing" in first, first
    assert _ask(member, "total eclipse of the heart", unused)["route"]["kind"] == "offtopic" and unused.calls == []


@needs_db
def test_a_total_of_everything_still_comes_from_sql(member):
    for question in ("how much did I spend", "thanks, how much did I spend in total?", "what's my grand total"):
        out = _ask(member, question, ScriptedChat("- Steady."))
        assert out["route"]["kind"] == "aggregate" and "on all categories" in out["answer"].splitlines()[0], question


@needs_db
def test_a_count_of_a_categorys_purchases_comes_from_sql(member):
    _, n = _coffee(member)
    out = _ask(member, "how many coffee purchases", ScriptedChat("- Steady."))
    first = out["answer"].splitlines()[0]
    assert out["route"]["kind"] == "aggregate" and f"**{n} transactions** on Coffee" in first, first
