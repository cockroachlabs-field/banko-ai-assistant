"""Window and operation extraction for aggregation questions.

These are pure functions, no DB or LLM. The dates below pin behavior to a
fixed "today" so the suite does not rot."""

from datetime import date

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
