"""Routing decides without a model: memory, receipt follow-ups, off-topic,
SQL aggregations, then the planner."""

from datetime import date

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
