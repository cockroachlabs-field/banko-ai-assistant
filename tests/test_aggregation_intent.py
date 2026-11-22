"""Window and operation extraction for aggregation questions.

These are pure functions, no DB or LLM. The dates below pin behavior to a
fixed "today" so the suite does not rot."""

from datetime import date

import pytest

from banko_ai.utils.intent_classifier import classify_aggregation, extract_window

TODAY = date(2026, 7, 24)


def test_jims_question_parses():
    intent = classify_aggregation(
        "How much money did I spend on restaurants in the past 60 days?",
        today=TODAY)
    assert intent is not None
    assert intent.operation == "sum"
    assert intent.subject == "restaurants"
    assert intent.window_start == date(2026, 5, 25)
    assert intent.window_end == date(2026, 7, 25)


def test_count_this_month():
    intent = classify_aggregation(
        "how many transactions did I make this month?", today=TODAY)
    assert intent.operation == "count"
    assert intent.window_start == date(2026, 7, 1)
    assert intent.window_end == date(2026, 8, 1)


def test_average_last_month_with_category():
    intent = classify_aggregation(
        "what was my average grocery spend last month", today=TODAY)
    assert intent.operation == "average"
    assert intent.subject == "grocery"
    assert intent.window_start == date(2026, 6, 1)
    assert intent.window_end == date(2026, 7, 1)


def test_no_window_defaults_to_90_days():
    intent = classify_aggregation("total spent on coffee", today=TODAY)
    assert intent.operation == "sum"
    assert (intent.window_end - intent.window_start).days == 90


def test_a_window_or_filler_after_for_or_at_is_not_a_subject():
    """"for this month" is the window and "at all" is emphasis: both mean
    every category, never a category called "this month"."""
    for q in ("how much did I spend for this month?", "how much did I spend at all last month",
              "how much did I spend on average?", "how much did I spend for the year"):
        assert classify_aggregation(q, today=TODAY).subject is None, q
    assert classify_aggregation("how much did I spend on dining last month", today=TODAY).subject == "dining"


def test_a_rate_is_not_a_total_of_everything():
    """ "on average per month" got the average purchase, and "how much do I
    spend a month" the sum of ninety days: neither is a month's spending.
    The words of the rate are kept, and the planner gets the question
    (tests/test_totals_subject.py)."""
    for q in ("how much did I spend on average per month?", "how much do I spend a month", "average monthly spending"):
        intent = classify_aggregation(q, today=TODAY)
        assert intent is None or (intent.loose and intent.subject), q


def test_non_aggregation_returns_none():
    assert classify_aggregation("show me my recent coffee purchases",
                                today=TODAY) is None
    assert classify_aggregation("why did I get a nudge?", today=TODAY) is None


def test_day_and_week_windows():
    wednesday = date(2026, 10, 14)

    def window(q):
        intent = classify_aggregation(q, today=wednesday)
        return intent.window_start, intent.window_end

    assert window("how much did I spend today") == (date(2026, 10, 14), date(2026, 10, 15))
    assert window("how much did I spend yesterday") == (date(2026, 10, 13), date(2026, 10, 14))
    # Calendar weeks start on Monday; "the past week" is the last 7 days.
    assert window("how much did I spend this week") == (date(2026, 10, 12), date(2026, 10, 15))
    assert window("how much did I spend on groceries last week") == (date(2026, 10, 5), date(2026, 10, 12))
    assert window("how much did I spend in the past week") == (date(2026, 10, 7), date(2026, 10, 15))
    assert window("how much did I spend in the past month") == (date(2026, 9, 14), date(2026, 10, 15))
    assert window("how much did I spend last year") == (date(2025, 1, 1), date(2026, 1, 1))


def test_a_weekend_is_not_a_week():
    intent = classify_aggregation("how much did I spend last weekend", today=date(2026, 10, 14))
    assert (intent.window_end - intent.window_start).days == 90


def test_a_day_word_after_the_category_is_not_part_of_it():
    for q in ("how much did I spend on groceries yesterday", "how much did I spend on coffee today?"):
        intent = classify_aggregation(q, today=date(2026, 10, 14))
        assert intent.subject in ("groceries", "coffee") and (intent.window_end - intent.window_start).days == 1, q


def test_a_calendar_month_or_year_is_a_window():
    """"in October" used to be no window at all, so the answer covered the
    last 90 days and said so only in its small print."""
    def window(q, today=date(2026, 11, 20)):
        intent = classify_aggregation(q, today=today)
        return intent.window_start, intent.window_end

    assert window("how much did I spend in October?") == (date(2026, 10, 1), date(2026, 11, 1))
    assert window("how much did I spend on groceries in october") == (date(2026, 10, 1), date(2026, 11, 1))
    assert window("how many purchases did I make during November") == (date(2026, 11, 1), date(2026, 12, 1))
    # The most recent such month that has begun: December is last year's.
    assert window("how much did I spend in December") == (date(2025, 12, 1), date(2026, 1, 1))
    assert window("total spent on coffee in March 2025") == (date(2025, 3, 1), date(2025, 4, 1))
    assert window("total spent on coffee in december of 2024") == (date(2024, 12, 1), date(2025, 1, 1))
    assert window("how much did I spend in sept") == (date(2026, 9, 1), date(2026, 10, 1))
    assert window("how much did I spend in Oct. 2025") == (date(2025, 10, 1), date(2025, 11, 1))
    assert window("how much did I spend in 2025") == (date(2025, 1, 1), date(2026, 1, 1))
    assert window("how much did I spend during 2026?") == (date(2026, 1, 1), date(2027, 1, 1))
    assert classify_aggregation("how much did I spend on groceries in october", today=TODAY).subject == "groceries"


def test_may_and_march_are_months_only_after_in_or_during():
    def days(q):
        intent = classify_aggregation(q, today=date(2026, 11, 20))
        return (intent.window_end - intent.window_start).days

    assert extract_window("how much may I spend on coffee", today=date(2026, 11, 20)) is None   # a verb
    assert days("how much did I spend on the march to the stadium") == 90
    assert days("how much did I spend in may") == 31
    assert days("how much did I spend during march") == 31
    assert days("how much did I spend in 2500 words") == 90        # not a year anyone banked in
    assert days("total expenses for may") == 31                    # after for, the month in full
    assert extract_window("what do I pay for apr", today=date(2026, 11, 20)) is None   # a rate, not April


# The largest or smallest purchase is a fact SQL has. It used to go to the
# planner, whose tools return the rows most like the wording, or the most
# recent ones, never the largest: a month's rent was left out, and a June
# purchase was once named as last month's. The route takes a question only
# when it understands every word of it. Anything it would have to ignore (a
# second clause, an amount, a window it cannot read, a word that leans on
# the turn before) leaves the question where it was before: with the totals
# when it asks for one, with the planner otherwise.
SIZES = ("largest", "smallest")
ASKED = date(2026, 11, 20)                       # a Friday
_DEFAULT = (date(2026, 8, 23), date(2026, 11, 21))   # no window named: the last 90 days
_OCTOBER = (date(2026, 10, 1), date(2026, 11, 1))
_THIS_YEAR = (date(2026, 1, 1), date(2026, 11, 21))
_THIS_MONTH = (date(2026, 11, 1), date(2026, 12, 1))


def _size(question: str, today: date = ASKED):
    intent = classify_aggregation(question, today=today)
    return intent if intent is not None and intent.operation in SIZES else None


@pytest.mark.parametrize("row, question, operation, limit, subject, window", [
    (1, "What was my biggest purchase last month?", "largest", 1, None, _OCTOBER),
    (1, "What can you tell me about my biggest purchase last month?", "largest", 1, None, _OCTOBER),
    (2, "top 5 purchases this year", "largest", 5, None, _THIS_YEAR),
    (3, "cheapest charge this week", "smallest", 1, None, (date(2026, 11, 16), date(2026, 11, 21))),
    (4, "how much was my biggest purchase", "largest", 1, None, _DEFAULT),
    (5, "my biggest purchase at Costco this year", "largest", 1, "costco", _THIS_YEAR),
    (6, "What was my priciest dining purchase this year?", "largest", 1, "dining", _THIS_YEAR),
    (7, "when did I make my biggest purchase at Costco?", "largest", 1, "costco", _DEFAULT),
    (8, "What are my top 5 biggest purchases?", "largest", 5, None, _DEFAULT),              # two size words
    (9, "top 3 most expensive purchases this year", "largest", 3, None, _THIS_YEAR),
    (11, "my top twelve purchases", "largest", 12, None, _DEFAULT),                         # the card lists ten
    (16, "What was my biggest purchase in October?", "largest", 1, None, _OCTOBER),
    (17, "biggest purchase in 2025", "largest", 1, None, (date(2025, 1, 1), date(2026, 1, 1))),
    (17, "biggest purchase in the last six months", "largest", 1, None, (date(2026, 5, 24), date(2026, 11, 21))),
    (36, "most expensive subscription charge", "largest", 1, "subscription", _DEFAULT),
    # Taken here with a subject the graph looks up. One that is no category
    # and none of the user's merchants, or has no purchases, goes on to the
    # planner from there (tests/test_assistant_graph.py).
    (39, "my largest income transaction", "largest", 1, "income", _DEFAULT),
    (0, "biggest purchase on black friday", "largest", 1, "black friday", _DEFAULT),
    (0, "Which was my biggest purchase last month?", "largest", 1, None, _OCTOBER),
    (0, "How much was my most expensive purchase this year?", "largest", 1, None, _THIS_YEAR),
    (0, "show me my top 5 purchases this year", "largest", 5, None, _THIS_YEAR),
    (0, "my 3 biggest transactions in the past 30 days", "largest", 3, None, (date(2026, 10, 21), date(2026, 11, 21))),
    (0, "what are my largest purchases this month", "largest", 3, None, _THIS_MONTH),      # several, no number: three
    (0, "What was my smallest purchase last month?", "smallest", 1, None, _OCTOBER),
    (0, "top 5 cheapest purchases this year", "smallest", 5, None, _THIS_YEAR),
    (0, "the five largest charges this month", "largest", 5, None, _THIS_MONTH),
    (0, "what's the most expensive purchase I made this year?", "largest", 1, None, _THIS_YEAR),
    (0, "Can you tell me what my biggest purchase was last month?", "largest", 1, None, _OCTOBER),
    (0, "Last month, what was my biggest purchase?", "largest", 1, None, _OCTOBER),
    (0, "please show me my top 3 restaurant purchases in October 2025", "largest", 3, "restaurant",
     (date(2025, 10, 1), date(2025, 11, 1))),
    (0, "my biggest purchase from Whole Foods Market last month", "largest", 1, "whole foods market", _OCTOBER),
    (0, "my biggest single purchase last month", "largest", 1, None, _OCTOBER),            # "single" names nothing
    (0, "top 5 restaurant purchases this year", "largest", 5, "restaurant", _THIS_YEAR),
    (0, "How much did I spend on my biggest purchase last month?", "largest", 1, None, _OCTOBER),
    (0, "top 50 purchases this year", "largest", 50, None, _THIS_YEAR),                    # the card lists ten
    (0, "top 100 purchases this year", "largest", 100, None, _THIS_YEAR),                  # and says so, as for 50
])
def test_a_size_question_it_fully_understands_is_an_sql_question(row, question, operation, limit, subject, window):
    intent = _size(question)
    assert intent is not None, question
    assert (intent.operation, intent.limit, intent.subject) == (operation, limit, subject), question
    assert (intent.window_start, intent.window_end) == window, question


@pytest.mark.parametrize("row, question", [
    (10, "what were my biggest and smallest purchases last month?"),     # two questions in one
    (12, "what was my biggest recent purchase?"),                        # "recent" is no category
    (13, "biggest credit card purchase last month"),                     # a payment method, which is not a filter here
    (14, "my highest monthly bill"),
    (15, "my biggest weekend purchase"),                                 # a window it cannot read
    (17, "what's my largest purchase ever?"),
    (17, "biggest purchase last quarter"),
    (17, "biggest purchase two months ago"),
    (18, "largest purchase each month this year"),
    (19, "total of my top 5 purchases"),
    (20, "average of my largest purchases"),
    (21, "how many large purchases did I make"),
    (22, "what's my total spending and top 3 purchases this month"),
    (23, "How much did I spend on groceries, and what was my biggest purchase?"),
    (24, "which one was the biggest?"),                                  # leans on the turn before
    (25, "of those, which purchase was the biggest?"),
    (26, "and the biggest purchase?"),
    (26, "what about my biggest purchase that month?"),
    (27, "what was the biggest purchase in that list?"),
    (28, "what is my biggest expense category?"),                        # a category, not a purchase
    (28, "which category is my biggest expense?"),
    (28, "which merchant got my biggest purchase?"),
    (28, "top purchases by category"),
    (28, "who are my top 5 merchants?"),
    (28, "what's my biggest month for spending?"),
    (29, "biggest purchase at each store"),
    (30, "What's my biggest expense?"),                                  # one purchase, or a category?
    (30, "what are my top 5 expenses"),
    (31, "what's the biggest item on this receipt?"),
    (32, "what was the biggest charge on that receipt?"),
    (33, "what was the biggest charge on it?"),
    (34, "is this my biggest purchase this month?"),
    (35, "what's my biggest bill?"),
    (36, "what's my highest subscription?"),
    (37, "what is the lowest payment I can make on my credit card?"),
    (38, "what was my largest deposit?"),
    (38, "biggest transfer last month"),
    (40, "what was the largest payment I received last month?"),
    (40, "my largest payment to my credit card"),
    (41, "what was not my biggest purchase?"),
    (41, "apart from my biggest purchase, what else did I buy last month?"),
    (42, "excluding rent, what was my biggest purchase last month?"),
    (43, "what was my biggest purchase under $100?"),
    (43, "largest transactions over $500"),
    (44, "¿Cuál fue mi compra más grande el mes pasado?"),
    (45, "what was my most recent purchase?"),
    (0, "why was my biggest purchase so high last month?"),
    (0, "what's the highest transaction fee I paid?"),
    (0, "what's my top payment method?"),
    (0, "largest coffee order this month"),                              # an order, a bill, a payment: the planner's
    (0, "what was my largest expense last month"),
    (0, "my biggest restaurant purchase at Nobu"),                       # a category and a place: one filter at a time
    (0, "my biggest purchases at Costco and Target"),
    (0, "my biggest non-rent purchase last month"),                      # everything but: not a filter here
    (0, "my biggest purchase besides rent"),
    (0, "what was my second biggest purchase last month"),
    (0, "top restaurant purchases this year"),                           # Top Golf purchases: a name, or the top ones?
    (0, "how much did I spend on Top Golf purchases last month?"),
    (0, "how much were my top 3 purchases this month?"),                 # a sum the card does not give
    (0, "how much did I spend on my top 5 purchases"),
    (0, "top 1000 purchases"),
    (0, "biggest 2025 purchases"),                                       # a year is not how many
    (0, "top 0 purchases"),
    (0, "What was my biggest purchase last month? And the smallest?"),
])
def test_a_question_it_would_have_to_guess_at_is_not_a_size_question(row, question):
    assert _size(question) is None, question


def test_the_biggest_purchase_leans_on_the_conversation_when_there_is_one():
    """"What was the biggest purchase?" after a list of groceries means the
    biggest of those, and only the planner sees the list. As a first
    question, or with a window or a place of its own, it means what it says."""
    for question in ("what was the biggest purchase?", "what were the biggest purchases?",
                     "what was the biggest purchase I made?"):
        assert _size(question) is not None, question
        assert classify_aggregation(question, today=ASKED, has_history=True) is None, question
    # A window, a place or a number of its own: a new question wherever it is asked.
    for question in ("what was the biggest purchase last month?", "the biggest purchase at Costco",
                     "show me the 10 largest transactions", "what was my biggest purchase?", "top 5 purchases"):
        intent = classify_aggregation(question, today=ASKED, has_history=True)
        assert intent is not None and intent.operation == "largest", question


def test_what_the_size_route_leaves_reads_as_a_total_or_goes_to_the_planner():
    """Not taken by the size route, a question that asks for a total is still
    read as one, and the rest still goes to the planner. A total keeps the
    words it cannot place ("top 5", "large") instead of dropping them:
    dropped, "total of my top 5 purchases" was answered with all spending.
    Kept, the graph finds no category or merchant of that name and hands the
    question to the planner (tests/test_totals_subject.py). A second
    question behind a total's subject is the planner's from the start."""
    def read(question):
        intent = classify_aggregation(question, today=ASKED)
        return (intent.operation, intent.subject, intent.loose) if intent else None

    assert read("what's my total spending and top 3 purchases this month") == ("sum", "top 3", True)
    assert read("How much did I spend on groceries, and what was my biggest purchase?") is None
    assert read("total of my top 5 purchases") == ("sum", "top 5", True)
    assert read("average of my largest purchases") == ("average", "largest", True)
    assert read("how many large purchases did I make") == ("count", "large", True)
    assert read("how much did I spend on Top Golf purchases last month?") == ("sum", "top golf purchases", False)
    for planner in ("What's my biggest expense?", "and the biggest purchase?", "what was my biggest recent purchase?",
                    "what were my biggest and smallest purchases last month?", "what was my largest deposit?",
                    "show me my recent coffee purchases", "which one was the biggest?"):
        assert read(planner) is None, planner


def test_totals_averages_and_counts_read_as_before():
    assert classify_aggregation("how much did I spend on restaurants last month", today=TODAY).operation == "sum"
    assert classify_aggregation("average grocery spend this year", today=TODAY).operation == "average"
    assert classify_aggregation("how many coffee purchases this month", today=TODAY).operation == "count"
    assert classify_aggregation("how much did I spend on restaurants last month", today=TODAY).limit == 1
