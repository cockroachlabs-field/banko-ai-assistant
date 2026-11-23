"""Every figure a model writes has to be one it was given.

SQL computes the figures and code writes the cards, but a model still writes
the sentences around an open question's facts, the observations under a
total and the translation of a card. It is told never to invent a number and
never to do arithmetic, and now and then it rounds one, adds two up or miscounts
a list all the same. So what it writes is read before it is shown: an
amount, a count, a percentage or a date that is in nothing it was given is
not shown."""

from datetime import date

import pytest

from banko_ai.assistant import figures as F

DINING = [{"tool": "spend_summary", "result": {
    "total": 947.33, "count": 17, "average": 55.73, "category": "Restaurant", "merchant": None,
    "from": "2026-10-01", "to": "2026-10-31",
    "rows": [{"date": "2026-10-30", "merchant": "Pizza Hut", "amount": 67.37},
             {"date": "2026-10-28", "merchant": "Domino's", "amount": 41.2},
             {"date": "2026-10-25", "merchant": "7-Eleven", "amount": 1850.0}],
    "largest": [{"date": "2026-10-25", "merchant": "7-Eleven", "amount": 1850.0}],
    "top_merchants": [{"merchant": "Domino's", "total": 310.5, "count": 6},
                      {"merchant": "Pizza Hut", "total": 201.11, "count": 4}]}}]
ASKED = "how much did I spend eating out in october?"


@pytest.fixture(scope="module")
def given():
    return F.given(DINING, ASKED)


@pytest.mark.parametrize("line", [
    "You spent $947.33 on restaurants across 17 transactions.",
    "Your average was **$55.73** a visit.",
    "The largest was $67.37 at Pizza Hut on 2026-10-30.",
    "Domino's took $310.50 of it, in 6 purchases.",
    "Domino's: $310.5 (6 purchases)",                      # one decimal, as JSON wrote it
    "| 2026-10-28 | Domino's | $41.20 |",
    "That covers October 1 to October 31, 2026.",
    "From Oct 1st through 31 October you ate out 17 times.",
    "In October 2026 you ate out 17 times.",
    "It was a busy 2026.",
    "The biggest single purchase was $1,850.00 at 7-Eleven.",
    "The biggest single purchase was 1850 dollars.",
    "Here are the three latest purchases:",               # a count in words is no figure
    "Here are the 3 latest purchases:",                   # counted, and three rows were given
    "Your top two merchants were Domino's and Pizza Hut.",
    "It has been the same for the past four months.",      # the months were counted, in words
    "Here are the eleven latest purchases:",
    # Worked out from two figures that were given, and the working is right.
    "Pizza Hut and Domino's together came to $511.61.",                       # 310.50 + 201.11
    "Domino's took $109.39 more than Pizza Hut.",                             # 310.50 - 201.11
    "That is $51.75 a purchase at Domino's.",                                 # 310.50 over 6
    "Everything but Domino's came to $636.83.",                               # 947.33 - 310.50
    # A share, with what it is a share of written beside it.
    "Domino's took $310.50 of the $947.33, which is 32.78%.",
    "Domino's took $310.50 of the $947.33: about 33%.", "Of $947.33, Domino's took $310.50 (32.8%).",
    "1. Pizza Hut\n2. Domino's",                           # a list's own numbering
    "Nothing unusual here.",
    "It is the 2nd largest of the month.",                 # a rank, not a figure
])
def test_a_figure_that_was_given_stands(given, line):
    assert F.unfounded(line, given) == [], line


@pytest.mark.parametrize("line, bad", [
    ("You spent about $950 on restaurants.", ["$950"]),                       # rounded
    ("That is roughly $55 a visit.", ["$55"]),
    ("Pizza Hut and Domino's together came to $511.71.", ["$511.71"]),        # added up, wrongly
    ("Domino's took $109.17 more than Pizza Hut.", ["$109.17"]),              # subtracted, wrongly
    ("Domino's took $310.50 of the $947.33, which is 35%.", ["35%"]),        # divided, wrongly
    ("Domino's was 32.78% of it.", ["32.78%"]),              # right, but of what? Nothing beside it says
    ("That is $52.75 a purchase at Domino's.", ["$52.75"]),
    ("You ate out 18 times.", ["18"]),                                        # miscounted
    ("Domino's and Pizza Hut were 10 purchases.", ["10"]),                    # a count is given as it is, or not shown
    ("The last one was on 2026-10-29.", ["2026-10-29"]),
    ("The last one was on October 29.", ["October 29"]),
    ("That is up 12% on September.", ["12%"]),
    ("In November 2026 you ate out 17 times.", ["November 2026"]),
    ("Back in 2025 it was less.", ["2025"]),
    ("You spent $947.33, which is $36.44 more than last month.", ["$36.44"]),
    ("| 2026-10-28 | Domino's | $41.25 |", ["$41.25"]),
])
def test_a_figure_nobody_gave_is_named(given, line, bad):
    assert F.unfounded(line, given) == bad


def test_a_range_of_days_is_two_days():
    """gpt-4.1: "This month (Nov 1–23), you spent $498.90 on groceries,
    compared to $474.12 last month (Oct 1–31)." The 23 and the 31 are days."""
    seen = F.given({"from": "2026-11-01", "to": "2026-11-23"}, {"from": "2026-10-01", "to": "2026-10-31"})
    for line in ("This month (Nov 1–23) you spent more than last month (Oct 1–31).", "From November 1-23 it was quiet.",
                 "Between 1–23 November it was quiet.", "From Nov 1st – 23rd, 2026 it was quiet."):
        assert F.unfounded(line, seen) == [], line
    assert F.unfounded("This month (Nov 1–24) was busy.", seen) == ["Nov 1–24"]
    assert F.unfounded("From Nov 2–23 it was quiet.", seen) == ["Nov 2–23"]


def test_a_footnote_mark_is_no_figure(given):
    """granite marks its sources its own way now and then: "(cite: [1])", "[Cite: 2]"."""
    for line in ("- Pizza Hut: $67.37 on 2026-10-30 (cite: [1])", "Domino's: $41.20 due on 2026-10-28 [Cite: 2]",
                 "Pizza Hut took $201.11 [3]."):
        assert F.unfounded(line, given) == [], line
    assert F.unfounded("You ate out [18] times.", given) == []      # a mark, whatever it stands beside
    assert F.unfounded("You ate out 18 times [2].", given) == ["18"]


def test_a_percentage_is_not_vouched_for_by_an_amount():
    """granite, under a grocery total: "Whole Foods accounts for 57% of your
    recent grocery spending", beside a $57.00 purchase on the card. Whole
    Foods took $410.05 of $758.38: 54%. A percentage is given when it was
    given as one: under a name that says so, as a share between 0 and 1, or
    with its sign in a text."""
    card = F.given({"total": 758.38, "count": 11, "top_merchants": [{"merchant": "Whole Foods", "total": 410.05}],
                    "rows": [{"merchant": "Trader Joe's", "amount": 57.0}]})
    assert F.unfounded("- Whole Foods accounts for 57% of your recent grocery spending", card) == ["57%"]
    assert F.unfounded("- Whole Foods took $410.05 of the $758.38, about 54%.", card) == []
    accounts = F.given({"accounts": [{"apr": 22.99, "balance": 57.0}], "utilization": 0.4217,
                        "budgets": [{"category": "Groceries", "pct": 0.8731}], "goal": {"progress": 0.5}},
                       "My savings account pays 4.5% and I keep $57 in cash.")
    assert F.unfounded("22.99% APR, 42% of your limit used (42.17%), 87% of the budget (87.3%), halfway there (50%), "
                       "and 4.5% on savings.", accounts) == []
    assert F.unfounded("57% of your limit is used.", accounts) == ["57%"]
    assert F.unfounded("88% of the budget is used.", accounts) == ["88%"]


def test_a_month_given_by_its_number_is_given_by_its_name():
    """The cash flow comes month by month, each named "2026-09". A model
    writes "September 2026", and that is the month it was given."""
    flow = F.given([{"tool": "cashflow", "result": {"months": [
        {"month": "2026-09", "income": 5000.0, "spending": 3723.41, "net": 1276.59},
        {"month": "2026-10", "income": 5000.0, "spending": 852.52, "net": 4147.48}]}}])
    assert F.unfounded("- September 2026: net positive, you saved $1,276.59.\n- Oct 2026: $4,147.48.", flow) == []
    assert F.unfounded("In September you saved $1,276.59.", flow) == []        # a month alone is no figure
    assert F.unfounded("- August 2026: you saved $1,276.59.", flow) == ["August 2026"]
    assert F.unfounded("- September 2025: you saved $1,276.59.", flow) == ["September 2025"]
    # A day gives its month too.
    days = F.given(["Paid on 2026-11-02."])
    assert F.unfounded("That was in November 2026.", days) == []


def test_a_figure_worked_out_from_two_that_were_given_is_checked_not_refused():
    """ "What share of my spending goes to restaurants?" asks for a figure
    no lookup returns. The model divides, and code checks the division: a
    sum, a difference, a share or an average of two figures it was given
    stands when it is right, to the cent or to the decimals it is written
    with, and is a figure from nowhere when it is not."""
    spent = F.given([{"tool": "spend_summary", "result": {"total": 946.15, "count": 17}},
                     {"tool": "spend_summary", "result": {"total": 4114.88, "count": 131}}])
    assert F.unfounded("22.99% of your spending went to restaurants: $946.15 of $4,114.88.", spent) == []
    assert F.unfounded("About 23% went to restaurants ($946.15 of $4,114.88).", spent) == []
    assert F.unfounded("23.99% of your spending went to restaurants: $946.15 of $4,114.88.", spent) == ["23.99%"]
    assert F.unfounded("The rest came to $3,168.73.", spent) == []                       # 4,114.88 - 946.15
    assert F.unfounded("That is $55.66 a visit.", spent) == []                           # 946.15 over 17
    assert F.unfounded("That is $31.41 a purchase overall.", spent) == []                # 4,114.88 over 131
    assert F.unfounded("That is $31.14 a purchase overall.", spent) == ["$31.14"]
    assert F.unfounded("Together that is 148 purchases.", spent) == ["148"]              # a count: given, or not shown
    # How much more, as a share: 3,168.73 over 946.15.
    assert F.unfounded("Everything else ($3,168.73) is 334.91% of restaurants ($946.15).", spent) == []
    # And how much more or less one is than the other: (4,114.88 - 946.15) over 4,114.88.
    assert F.unfounded("Restaurants ($946.15) came to 77.01% less than everything ($4,114.88).", spent) == []
    assert F.unfounded("Restaurants ($946.15) came to 67.01% less than everything ($4,114.88).", spent) == ["67.01%"]
    # Three figures deep is not worked out for anybody: it is not shown.
    assert F.unfounded("At this pace that is $11,353.80 a year.", spent) == ["$11,353.80"]


def test_a_share_waits_for_the_figures_it_is_a_share_of():
    """gpt-4.1, asked what share of spending goes to restaurants: "22.99% of
    your total spending went to restaurants. This is calculated from $946.15
    spent on restaurants out of a total of $4,114.88 spent overall." The
    share comes first and what it is a share of after it. The sentence
    waits, and what follows waits behind it, so the answer keeps its order."""
    spent = F.given([{"tool": "spend_summary", "result": {"total": 946.15, "count": 17}},
                     {"tool": "spend_summary", "result": {"total": 4114.88, "count": 131}}])
    reader = F.Reader(spent)
    out = [reader.feed(piece) for piece in (
        "Based on your spending, 22.99% of your total", " spending went to restaurants. This is calculated",
        " from $946.15 spent on restaurants out of a total of $4,114.88 spent overall. Anything", " else?")]
    out.append(reader.finish())
    assert out[:2] == ["", ""]                                  # the share alone: not yet
    assert "".join(out) == ("Based on your spending, 22.99% of your total spending went to restaurants. This is "
                            "calculated from $946.15 spent on restaurants out of a total of $4,114.88 spent overall. "
                            "Anything else?")
    assert reader.held == []
    # A share nothing in the answer accounts for is held back when the answer ends, and the rest goes out.
    reader = F.Reader(spent)
    said = reader.feed("About 40% of your spending went to restaurants. You spent $946.15 there. ") + reader.finish()
    assert said == "You spent $946.15 there. " and reader.held == [
        ("About 40% of your spending went to restaurants. ", ["40%"])]
    # A share that was given needs nothing beside it.
    cards = F.given([{"tool": "balances", "result": {"accounts": [{"utilization": 0.42, "apr": 22.99}]}}])
    assert F.unfounded("Your card is 42% used at 22.99% APR.", cards) == []


def test_what_the_user_said_is_given_too(given):
    """A question's own numbers, the conversation so far and what the user
    asked to be remembered are things the model was given."""
    asked = F.given(DINING, "did I stay under my $900 limit across 20 visits?")
    assert F.unfounded("You asked about a $900 limit over 20 visits.", asked) == []
    assert F.unfounded("You asked about a $900 limit over 20 visits.", given) == ["$900", "20"]
    remembered = F.given(DINING, ASKED, [{"role": "assistant", "content": "Groceries came to $502.72 on 2026-09-30."}],
                         ["I'm saving $300 a month for a trip to Japan"])
    assert F.unfounded("Next to groceries ($502.72 as of September 30) and the $300 you put aside.", remembered) == []


@pytest.mark.parametrize("line", [
    "Gastaste 947,33 $ en restaurantes.",                  # a decimal comma
    "Die größte Ausgabe war 1.850,00 $.",                  # a point for thousands
    "Vous avez dépensé 1 850,00 $ en une fois.",           # a space for thousands
    "Du hast 17 Mal auswärts gegessen.",
])
def test_figures_are_read_as_other_languages_write_them(given, line):
    assert F.unfounded(line, given) == [], line


def test_a_comma_before_three_digits_groups_thousands():
    """ "$8,600" is eight thousand six hundred. Read as 8.60 as well, it
    stood whenever some purchase had cost $8.60: a rounded yearly figure
    went out on the strength of one coffee."""
    coffee = F.given({"rows": [{"amount": 8.6}, {"amount": 1.85}], "total": 954.96})
    assert F.unfounded("That is about $8,600 a year.", coffee) == ["$8,600"]
    assert F.unfounded("That is about $1,850 a year.", coffee) == ["$1,850"]
    assert F.unfounded("One was $8.60 and one $1.85.", coffee) == []
    # A point before three digits may be either: a thousand in German, a decimal in English.
    assert F.unfounded("Das waren 1.850 Dollar.", F.given({"total": 1850.0})) == []
    assert F.unfounded("It was 1.850 away.", F.given({"distance": 1.85})) == []


def test_a_share_is_given_as_a_share_or_as_a_percentage():
    cards = F.given([{"tool": "balances", "result": {"accounts": [{"name": "Rewards Card", "utilization": 0.42,
                                                                 "apr": 22.99, "balance": -3360.0}]}}])
    assert F.unfounded("Your card is 42% used at 22.99% APR, with $3,360.00 owed.", cards) == []
    assert F.unfounded("Your card is 45% used.", cards) == ["45%"]


def test_a_number_in_a_name_is_no_figure(given):
    assert F.unfounded("You stopped at 7-Eleven on 2026-10-25.", given) == []
    studio = F.given([{"tool": "merchant_history", "result": {"merchant": "Studio 54", "visits": 3}}])
    assert F.unfounded("You went to Studio 54 three times.", studio) == []
    assert F.unfounded("You went to Studio 55 three times.", studio) == ["55"]


def test_a_number_inside_a_date_or_an_id_gives_no_count():
    """ "2026-10-18" holds an 18. It is a day, and must not make "18
    purchases" a figure anybody gave. Nor does an id's digits."""
    only_dates = F.given([{"tool": "search_transactions", "result": {"items": [
        {"expense_id": "7c698448-28e5-56c6-beb1-aa803d80a888", "date": "2026-10-18", "merchant": "Peet's Coffee",
         "amount": 6.69, "description": "Spent $6.69 on coffee at Peet's Coffee using Credit Card.",
         "posted_at": "2026-10-18T12:00:00+00:00", "distance": 0.3712}]}}])
    assert F.unfounded("You made 18 purchases.", only_dates) == ["18"]
    assert F.unfounded("You made 28 purchases in 2026.", only_dates) == ["28"]
    assert F.unfounded("One purchase: $6.69 at Peet's Coffee on October 18 [txn:7c698448-28e5-56c6-beb1-aa803d80a888].",
                       only_dates) == []
    assert F.unfounded("It was 0.37 away from what you asked.", only_dates) == []


def test_what_is_no_figure_is_not_read_as_one(given):
    for line in ("See you at 9:30 tomorrow.", "- Pizza Hut\n- Domino's",
                 "Compare with [txn:7c698448-28e5-56c6-beb1-aa803d80a888].", "It is your one big purchase.",
                 "A couple of things stand out.", "### Top merchants"):
        assert F.unfounded(line, given) == [], line


def test_a_text_is_cut_where_a_sentence_or_a_line_ends():
    text = ("You spent $947.33 on restaurants. That is about $950. Here is where it went:\n"
            "- Domino's: $310.50 (6 purchases)\n- Pizza Hut: $201.11\n\n| a | b |\n|---|---|\n| 1.5 | 2 |")
    assert F.units(text) == ["You spent $947.33 on restaurants. ", "That is about $950. ", "Here is where it went:\n",
                             "- Domino's: $310.50 (6 purchases)\n", "- Pizza Hut: $201.11\n", "\n", "| a | b |\n",
                             "|---|---|\n", "| 1.5 | 2 |"]
    assert "".join(F.units(text)) == text
    # An abbreviation, a decimal point and a domain end no sentence.
    assert F.units("It cost $12.50 at St. Mark's Deli, e.g. on Fridays. Next.") == [
        "It cost $12.50 at St. Mark's Deli, e.g. on Fridays. ", "Next."]


def test_only_what_stands_is_let_through_as_it_arrives(given):
    """Pieces arrive as the model writes them. A sentence goes out when it
    is whole and every figure in it was given, and never before."""
    reader = F.Reader(given)
    out = []
    for piece in ("You spent $94", "7.33 on restaurants. That", " is about $9", "50 a month. Domino's took $310.50", "."):
        out.append(reader.feed(piece))
    out.append(reader.finish())
    assert out == ["", "You spent $947.33 on restaurants. ", "", "", "", "Domino's took $310.50."]
    assert reader.shown == "You spent $947.33 on restaurants. Domino's took $310.50."
    assert reader.held == [("That is about $950 a month. ", ["$950"])]
    assert (reader.checked, reader.figures) == (3, 3)


def test_a_sentence_held_back_does_not_hold_up_what_follows(given):
    """Only a share waits, for the figures it is a share of. A sentence with
    an amount from nowhere is settled at once, and the next one goes out as
    soon as it is whole."""
    reader = F.Reader(given)
    assert reader.feed("That is about $950 a month. Domino's took $310.50. Pizza") == "Domino's took $310.50. "
    assert reader.held == [("That is about $950 a month. ", ["$950"])]


def test_a_reader_with_nothing_held_shows_the_text_as_it_was(given):
    text = "You spent $947.33 on restaurants across 17 transactions.\n\n- Domino's: $310.50\n- Pizza Hut: $201.11"
    reader = F.Reader(given)
    assert reader.feed(text) + reader.finish() == text and reader.held == [] and reader.shown == text


CARD = ("You spent **$947.33** on Restaurant from 2026-10-01 to 2026-10-31 (17 transactions).\n\n"
        "| 2026-10-30 | Pizza Hut | $67.37 |")


@pytest.mark.parametrize("translation", [
    "Gastaste **$947.33** en Restaurant del 2026-10-01 al 2026-10-31 (17 transacciones).\n\n| 2026-10-30 | Pizza Hut | $67.37 |",
    "Gastaste **947,33 $** en Restaurant del 2026-10-01 al 2026-10-31 (17 transacciones).\n\n| 2026-10-30 | Pizza Hut | 67,37 $ |",
    # A date in the language's own words: its numbers are the date's.
    "Gastaste **$947.33** en Restaurant del 1 de octubre de 2026 al 31 de octubre de 2026 (17 transacciones).\n\n"
    "| 30 de octubre de 2026 | Pizza Hut | $67.37 |",
    "Sie haben vom 01.10.2026 bis 31.10.2026 **947,33 $** für Restaurant ausgegeben (17 Transaktionen).\n\n"
    "| 30.10.2026 | Pizza Hut | 67,37 $ |",
    # A count spelled out is no figure lost: the amounts are what must be there.
    "Gastaste **$947.33** en Restaurant del 2026-10-01 al 2026-10-31 (diecisiete transacciones).\n\n"
    "| 2026-10-30 | Pizza Hut | $67.37 |",
])
def test_a_translation_that_keeps_the_figures_stands(translation):
    assert F.translation_keeps(CARD, translation), translation


@pytest.mark.parametrize("translation, why", [
    (CARD.replace("947.33", "947.83"), "an amount changed"),
    (CARD.replace("(17 transactions)", "(18 transactions)"), "a count changed"),
    (CARD.replace("(17 transactions)", "(17 transactions, 3 of them large)"), "a figure added"),
    ("Gastaste **$947.33** en Restaurant del 2026-10-01 al 2026-10-31 (17 transacciones).", "a purchase and its amount lost"),
    (CARD.replace("2026-10-31", "2026-11-30"), "a day changed"),
    (CARD.replace("$67.37", "$67"), "an amount rounded"),
    ("", "nothing came back"),
])
def test_a_translation_that_changes_adds_or_loses_a_figure_does_not(translation, why):
    assert not F.translation_keeps(CARD, translation), why


@pytest.mark.parametrize("translation", [
    "レストランで**$947.33**を使いました（2026-10-01から2026-10-31まで、17件）。\n\n| 2026-10-30 | Pizza Hut | $67.37 |",
    "您在餐厅消费了**$947.33**（2026-10-01至2026-10-31，17笔交易）。\n\n| 2026-10-30 | Pizza Hut | $67.37 |",
    "आपने रेस्टोरेंट पर 2026-10-01 से 2026-10-31 तक **$947.33** खर्च किए (17 लेनदेन)।\n\n| 2026-10-30 | Pizza Hut | $67.37 |",
    "2026년 10월 1일부터 2026년 10월 31일까지 레스토랑에서 **$947.33**를 지출했습니다(17건).\n\n| 2026년 10월 30일 | Pizza Hut | $67.37 |",
    "Vous avez dépensé **947,33 $** au restaurant du 1er octobre 2026 au 31 octobre 2026 (17 transactions).\n\n"
    "| 30 octobre 2026 | Pizza Hut | 67,37 $ |",
])
def test_a_translation_into_a_script_written_without_spaces_is_read_too(translation):
    assert F.translation_keeps(CARD, translation), translation


@pytest.mark.parametrize("translation", [
    "レストランで**$947.33**を使いました（2026-10-01から2026-10-31まで、18件）。\n\n| 2026-10-30 | Pizza Hut | $67.37 |",
    "您在餐厅消费了**$974.33**（2026-10-01至2026-10-31，17笔交易）。\n\n| 2026-10-30 | Pizza Hut | $67.37 |",
    "आपने रेस्टोरेंट पर 2026-10-01 से 2026-10-31 तक **$947.33** खर्च किए (17 लेनदेन)।",
])
def test_and_a_figure_changed_or_lost_in_one_is_seen(translation):
    assert not F.translation_keeps(CARD, translation), translation


def test_a_sentence_ends_where_its_script_ends_one():
    assert F.units("合計は$947.33です。内訳は次のとおりです。") == ["合計は$947.33です。", "内訳は次のとおりです。"]
    assert F.units("कुल $947.33 है। विवरण नीचे है।") == ["कुल $947.33 है। ", "विवरण नीचे है।"]
    assert F.units("Gastaste $947.33. ¿Quieres ver más?") == ["Gastaste $947.33. ", "¿Quieres ver más?"]
    assert F.units("Vous avez dépensé 947,33 $. À voir aussi.") == ["Vous avez dépensé 947,33 $. ", "À voir aussi."]


def test_a_number_inside_a_word_is_still_no_figure(given):
    assert F.unfounded("She flew on an A380 with a v2 ticket.", given) == []
    assert F.unfounded("合計は18件です。", given) == ["18"]                # but one after a letter of such a script is


def test_a_text_with_no_figures_translates_to_one_with_none():
    assert F.translation_keeps("I couldn't look that up just now. Please ask again in a moment.",
                               "No pude consultarlo ahora mismo. Vuelve a preguntar en un momento.")
    assert not F.translation_keeps("I couldn't look that up just now.", "No pude consultarlo: faltan 3 datos.")


def test_the_days_given_are_read_from_every_way_a_day_is_written():
    seen = F.given(["Paid on October 18, 2026 and again on 2026-11-02.", {"due": "2026-12-01T00:00:00+00:00"}])
    assert {date(2026, 10, 18), date(2026, 11, 2), date(2026, 12, 1)} <= seen.dates
    for line in ("On Oct 18 and Nov 2.", "Due December 1, 2026.", "Due on 12/01/2026.", "Due 1 December."):
        assert F.unfounded(line, seen) == [], line
    assert F.unfounded("Due December 2.", seen) == ["December 2"]
