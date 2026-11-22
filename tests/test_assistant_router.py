"""Routing decides without a model: memory, receipt follow-ups, off-topic,
SQL aggregations, then the planner."""

from datetime import date

import pytest

from banko_ai.assistant.router import route

RECEIPT = {"document_id": "d1", "merchant": "Trader Joe's", "items": ["Pinot Noir wine", "Sourdough bread"],
           "amount": 23.47}
TODAY = date(2026, 10, 12)


def test_memory_commands_win():
    r = route("Remember that my rent is due on the 1st")
    assert r.kind == "remember" and r.payload == "my rent is due on the 1st"
    assert route("please forget about the Japan trip").payload == "the Japan trip"
    assert route("remember that", active_receipt=RECEIPT).kind != "remember"


def test_receipt_reference_beats_aggregation():
    r = route("how much was this receipt?", active_receipt=RECEIPT, today=TODAY)
    assert r.kind == "receipt"
    assert route("was the wine expensive?", active_receipt=RECEIPT).kind == "receipt"
    assert route("is it a duplicate", active_receipt=RECEIPT, last_route="receipt").kind == "receipt"
    assert route("what did I buy at trader joe's", active_receipt=RECEIPT).kind == "receipt"


def test_without_a_receipt_the_same_words_aggregate_or_go_to_the_latest_receipt():
    assert route("how much did I spend on groceries last month", today=TODAY).kind == "aggregate"
    assert route("how much was this receipt?", today=TODAY).kind == "receipt"
    assert route("what did I buy on that receipt", today=TODAY).kind == "receipt"


def test_weak_references_give_way_to_a_real_aggregation():
    """"the total" with a subject or a window is a new question, not the receipt."""
    r = route("what's the total I spent on restaurants this month?", active_receipt=RECEIPT, today=TODAY)
    assert r.kind == "aggregate" and r.intent.subject == "restaurants"
    assert route("how much was the total?", active_receipt=RECEIPT, today=TODAY).kind == "receipt"
    assert route("how much was this receipt in total this month?", active_receipt=RECEIPT,
                 today=TODAY).kind == "receipt"


def test_a_bare_pronoun_needs_the_receipt_to_be_the_topic():
    unrelated = route("is it a good idea to save more?", active_receipt=RECEIPT, last_route="aggregate", today=TODAY)
    assert unrelated.kind != "receipt"
    assert route("is it a good idea to save more?", active_receipt=RECEIPT, last_route="receipt",
                 today=TODAY).kind == "receipt"
    assert route("is that expensive?", active_receipt=RECEIPT, last_route="receipt").kind == "receipt"
    assert route("is that expensive?", active_receipt=RECEIPT, last_route="plan").kind != "receipt"


def test_short_or_follow_up_messages_mid_conversation_reach_the_planner():
    for message in ("which one was the biggest?", "tell me more", "thanks!", "what's the capital of France?"):
        assert route(message, has_history=True, today=TODAY).kind == "plan", message
    long_off_topic = "what is the boiling point of water at sea level in degrees celsius"
    assert route(long_off_topic, has_history=True, today=TODAY).kind == "offtopic"
    assert route("what's the capital of France?", today=TODAY).kind == "offtopic"


def test_unrelated_question_with_a_receipt_is_not_hijacked():
    r = route("how much did I spend on restaurants in the last 60 days", active_receipt=RECEIPT, today=TODAY)
    assert r.kind == "aggregate" and r.intent.operation == "sum"


def test_off_topic():
    assert route("what's the capital of France?").kind == "offtopic"


def test_context_goes_to_the_planner():
    r = route("why did you flag this?", context={"kind": "nudge", "id": "n1"})
    assert r.kind == "plan"


def test_open_questions_plan():
    assert route("where could I cut back on dining out?").kind == "plan"


def test_as_dict_is_json_safe():
    import json
    d = route("how much did I spend on groceries last month", today=TODAY).as_dict()
    json.dumps(d)
    assert d["kind"] == "aggregate" and d["intent"]["operation"] == "sum"


def test_this_year_not_hijacked_by_receipt():
    r = route("how much have I spent this year?", active_receipt=RECEIPT, today=TODAY)
    assert r.kind == "aggregate"


def test_this_month_not_hijacked_by_receipt():
    r = route("is this month worse than last?", active_receipt=RECEIPT, today=TODAY)
    assert r.kind == "plan"


def test_item_words_in_idioms_not_hijacked():
    r = route("what's the bread and butter of my spending?", active_receipt=RECEIPT, today=TODAY)
    assert r.kind != "receipt"


def test_remember_to_is_not_a_memory_command():
    r = route("remember to pay rent on Friday")
    assert r.kind != "remember"


SEARCHED = {"q": "coffee", "category": None, "from": None, "to": None, "next": "bzEw"}


def test_asking_for_more_pages_the_last_search():
    for text in ("show more", "more", "More please", "next", "next page", "show me 10 more", "next 10",
                 "keep going", "load more results", "any more?", "show more of those", "ok, show more."):
        r = route(text, last_search=SEARCHED, has_history=True, today=TODAY)
        assert r.kind == "more" and r.follow_up, text


def test_more_inside_a_real_question_is_not_paging():
    for text in ("how can I save more?", "next month", "did I spend more on coffee than tea?",
                 "more coffee purchases", "tell me more about my budget", "what's next on my bills?"):
        assert route(text, last_search=SEARCHED, has_history=True, today=TODAY).kind != "more", text


def test_more_with_no_search_before_it_goes_to_the_planner():
    assert route("show more", has_history=True, today=TODAY).kind == "plan"


def test_visit_questions_are_answered_by_sql():
    for text, phrase in (("did i go to ikea", "ikea"), ("Did I go to IKEA last month?", "IKEA last month"),
                         ("have I been to Costco", "Costco"), ("when did I last go to Starbucks?", "Starbucks"),
                         ("how often do I go to Whole Foods", "Whole Foods"),
                         ("how many times did I go to Target", "Target"),
                         ("did I shop at Walmart this year", "Walmart this year"),
                         ("did I eat at Chipotle recently", "Chipotle recently"),
                         ("when was my last visit to Trader Joe's", "Trader Joe's"),
                         ("have I ever ordered from DoorDash?", "DoorDash"), ("did I visit REI", "REI"),
                         ("did i go to the gym", "the gym")):
        r = route(text, today=TODAY)
        assert (r.kind, r.payload) == ("visits", phrase), text


def test_visit_words_inside_other_questions_route_as_before():
    assert route("how much did I spend at Target last month", today=TODAY).kind == "aggregate"
    for text in ("did I go over my dining budget?", "where did I go for dinner last week?",
                 "have I been saving enough?", "did I spend more on coffee than tea?"):
        assert route(text, today=TODAY).kind != "visits", text


def test_a_visit_question_about_the_active_receipt_stays_with_the_receipt():
    assert route("did I go to Trader Joe's?", active_receipt=RECEIPT, today=TODAY).kind == "receipt"
    # A window makes it a new question, as for any receipt follow-up.
    assert route("did I go to Trader Joe's this week?", active_receipt=RECEIPT, today=TODAY).kind == "visits"


def test_a_largest_purchase_question_routes_to_sql_with_how_many():
    from banko_ai.assistant.router import route
    found = route("show me my top 5 purchases this year")
    assert found.kind == "aggregate" and found.as_dict()["intent"]["operation"] == "largest"
    assert found.as_dict()["intent"]["limit"] == 5
    assert route("how much did I spend on coffee last month").as_dict()["intent"]["limit"] == 1
    # Leaning on the list the turn before showed: the conversation's, not a new SQL question.
    assert route("which one was the biggest?", has_history=True).kind == "plan"
    leaning = route("what was the biggest purchase?", has_history=True)
    assert leaning.kind == "plan" and leaning.follow_up       # its words alone must never key a cached answer
    assert route("and the biggest purchase?", has_history=True).kind == "plan"
    assert route("what was the biggest purchase?").kind == "aggregate"          # a first question means what it says
    assert route("what was my biggest purchase in October?", has_history=True).kind == "aggregate"
    # "When did I make" opens a visit question; this one asks which purchase.
    assert route("when did I make my biggest purchase at Costco?").kind == "aggregate"


def test_a_size_question_and_the_active_receipt():
    """A receipt on the table does not turn a question about every purchase
    into one about that receipt, and a question about the receipt is not
    answered from every purchase."""
    from banko_ai.assistant.router import route
    asked = route("What was my biggest purchase last month?", active_receipt=RECEIPT, last_route="receipt",
                  has_history=True, today=TODAY)
    assert asked.kind == "aggregate" and asked.intent.operation == "largest"
    for about_it in ("what was the biggest charge on it?", "what was the biggest charge on that receipt?",
                     "is this my biggest purchase this month?"):
        assert route(about_it, active_receipt=RECEIPT, last_route="receipt", has_history=True,
                     today=TODAY).kind == "receipt", about_it
    # "The biggest charge", with a receipt just uploaded and nothing else said yet: of that receipt, most likely.
    assert route("what was the biggest charge?", active_receipt=RECEIPT, last_route="receipt",
                 today=TODAY).kind == "plan"
    # Some turns later "it" no longer means the receipt: the conversation's, not SQL's.
    later = route("what was the biggest charge on it?", active_receipt=RECEIPT, last_route="plan", has_history=True,
                  today=TODAY)
    assert later.kind == "plan"


COFFEE_RECEIPT = {"document_id": "d-coffee", "merchant": "BLUE BOTTLE COFFEE", "amount": 9.75, "category": "Coffee",
                  "items": ["Latte", "Croissant"]}


def test_a_category_word_in_the_merchants_name_is_not_a_question_about_the_receipt():
    """With a Blue Bottle Coffee receipt on the table, "coffee expenses" was
    answered from that one receipt: the merchant's name holds the word
    coffee. The question is about every coffee purchase there is."""
    assert route("coffee expenses", active_receipt=COFFEE_RECEIPT, today=TODAY).kind == "plan"
    assert route("my coffee spending", active_receipt=COFFEE_RECEIPT, today=TODAY).kind != "receipt"
    assert route("anything from a cafe lately", active_receipt={**COFFEE_RECEIPT, "merchant": "Local Cafe"},
                 today=TODAY).kind != "receipt"
    # What only this merchant is called still means its receipt.
    assert route("anything odd about blue bottle", active_receipt=COFFEE_RECEIPT, today=TODAY).kind == "receipt"
    assert route("what did I get at Blue Bottle Coffee", active_receipt=COFFEE_RECEIPT, today=TODAY).kind == "receipt"
    assert route("is the local one cheaper", active_receipt={**COFFEE_RECEIPT, "merchant": "Local Cafe"},
                 today=TODAY).kind == "receipt"
    # And so does everything that pointed at the receipt before.
    assert route("how much was this receipt?", active_receipt=COFFEE_RECEIPT, today=TODAY).kind == "receipt"
    assert route("was the latte on it?", active_receipt=COFFEE_RECEIPT, last_route="receipt").kind == "receipt"


# A category of the catalog is a money question, and its purchases are not a line on a receipt.

@pytest.mark.parametrize("name", ["phone", "rideshare", "transit", "health", "medical", "hardware", "takeout",
                                  "streaming", "electronics", "entertainment", "movies", "concerts"])
def test_a_category_named_alone_is_a_money_question(name):
    """Twelve of the catalog's names were "not a money question" when they
    opened a conversation: no word in "electronics" says money."""
    for question in (name, f"my {name}", f"{name} this month", f"show my {name} purchases"):
        opened = route(question, today=TODAY)
        assert opened.kind == "plan" and not opened.follow_up, question


@pytest.mark.parametrize("name", ["electronics", "streaming", "movies"])
def test_inside_a_conversation_such_a_name_still_leans_on_it(name):
    """As before: short and in the middle of a conversation, it is a
    follow-up, and its answer is never kept under that one word."""
    for question in (name, f"my {name}"):
        later = route(question, has_history=True, today=TODAY)
        assert later.kind == "plan" and later.follow_up, question


@pytest.mark.parametrize("question", ["show me movies", "list movies", "can you show me movies today?",
                                      "list concerts this week", "please show streaming"])
def test_asking_to_be_shown_a_category_word_alone_is_still_no_money_question(question):
    """ "show me movies" asks for films. With a word for the purchases it was a money question already."""
    assert route(question, today=TODAY).kind == "offtopic"


@pytest.mark.parametrize("question", ["show my electronics", "show me my electronics", "list my streaming",
                                      "can you show me my movies", "display my electronics please",
                                      "view my concerts this month"])
def test_asking_to_be_shown_ones_own_is_a_money_question(question):
    """ "show me movies" asks for films, "show me my movies" for what I spent on them."""
    assert route(question, today=TODAY).kind == "plan"


def test_words_that_only_happen_to_be_a_category_stay_where_they_were():
    assert route("other", today=TODAY).kind == "offtopic"                       # no question in it
    assert route("what movies should I watch tonight?", today=TODAY).kind == "offtopic"
    assert route("is my phone waterproof", today=TODAY).kind == "offtopic"


COFFEE_LINE = {"document_id": "d2", "merchant": "Corner Bakery", "items": ["Drip Coffee", "Bagel"], "amount": 7.5}


def test_a_categorys_purchases_are_not_a_line_on_the_receipt():
    """ "my coffee expenses" went to the receipt because one of its lines is
    a coffee, while "coffee expenses" went to the planner. A category asked
    for as its purchases is about the category either way."""
    for question in ("coffee expenses", "my coffee expenses", "show my coffee purchases", "the coffee transactions"):
        assert route(question, active_receipt=COFFEE_LINE, last_route="receipt", today=TODAY).kind == "plan", question
    # The line itself is still the receipt's.
    for question in ("was the coffee on it?", "how much was the coffee?", "the coffee"):
        assert route(question, active_receipt=COFFEE_LINE, last_route="receipt", today=TODAY).kind == "receipt", question


def test_purchases_at_the_restaurant_on_the_receipt_are_the_receipts():
    """ "at the restaurant" is one place, the one on the table, not the category."""
    dinner = {"document_id": "d3", "merchant": "Luigi's", "items": ["Restaurant dinner", "Tiramisu"], "amount": 62.0}
    assert route("purchases at the restaurant", active_receipt=dinner, last_route="receipt", today=TODAY).kind == "receipt"
    assert route("restaurant purchases", active_receipt=dinner, last_route="receipt", today=TODAY).kind == "plan"
