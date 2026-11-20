"""Window and operation extraction for aggregation questions.

These are pure functions, no DB or LLM. The dates below pin behavior to a
fixed "today" so the suite does not rot."""

from datetime import date

import pytest

from banko_ai.utils.intent_classifier import classify_aggregation

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
              "how much did I spend on average per month?", "how much did I spend for the year"):
        assert classify_aggregation(q, today=TODAY).subject is None, q
    assert classify_aggregation("how much did I spend on dining last month", today=TODAY).subject == "dining"


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


# The largest or smallest purchase is a fact SQL has. It used to go to the
# planner, whose tools return the rows most like the wording, or the most
# recent ones, never the largest: a month's rent was left out, and a June
# purchase was once named as last month's.
@pytest.mark.parametrize("question, operation, limit", [
    ("What can you tell me about my biggest purchase last month?", "largest", 1),
    ("what was my largest expense last month", "largest", 1),
    ("How much was my most expensive purchase this year?", "largest", 1),
    ("Which was my biggest purchase last month?", "largest", 1),
    ("show me my top 5 purchases this year", "largest", 5),
    ("my 3 biggest transactions in the past 30 days", "largest", 3),
    ("what are my largest purchases this month", "largest", 3),
    ("What was my smallest purchase last month?", "smallest", 1),
    ("cheapest charge this week", "smallest", 1),
])
def test_the_largest_or_smallest_purchase_is_an_sql_question(question, operation, limit):
    intent = classify_aggregation(question, today=TODAY)
    assert intent is not None, question
    assert (intent.operation, intent.limit) == (operation, limit), question


def test_a_largest_purchase_question_keeps_its_window_and_subject():
    intent = classify_aggregation("What was my biggest purchase last month?", today=TODAY)
    assert (intent.window_start, intent.window_end) == (date(2026, 6, 1), date(2026, 7, 1)) and intent.subject is None
    assert classify_aggregation("my biggest purchase at Costco this year", today=TODAY).subject == "costco"
    assert classify_aggregation("most expensive restaurant purchase last month", today=TODAY).subject == "restaurant"
    assert classify_aggregation("largest coffee order this month", today=TODAY).subject == "coffee"
    assert classify_aggregation("my biggest single purchase last month", today=TODAY).subject is None


@pytest.mark.parametrize("question", [
    "which one was the biggest?",                      # leans on the turn before: the planner's
    "of those, which purchase was the biggest?",
    "what is my biggest expense category?",            # a category, not a purchase
    "which category is my biggest expense?",
    "what's my biggest month for spending?",
    "who are my top 5 merchants?",
    "why was my biggest purchase so high last month?",  # a why question
    "what was my most recent purchase?",               # recent is not a size
])
def test_what_is_not_about_one_purchase_stays_where_it_was(question):
    intent = classify_aggregation(question, today=TODAY)
    assert intent is None or intent.operation not in ("largest", "smallest"), question


def test_totals_averages_and_counts_read_as_before():
    assert classify_aggregation("how much did I spend on restaurants last month", today=TODAY).operation == "sum"
    assert classify_aggregation("average grocery spend this year", today=TODAY).operation == "average"
    assert classify_aggregation("how many coffee purchases this month", today=TODAY).operation == "count"
    assert classify_aggregation("how much did I spend on restaurants last month", today=TODAY).limit == 1
