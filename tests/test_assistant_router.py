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
    assert route("is it a duplicate", active_receipt=RECEIPT).kind == "receipt"
    assert route("what did I buy at trader joe's", active_receipt=RECEIPT).kind == "receipt"


def test_without_a_receipt_the_same_words_aggregate_or_plan():
    assert route("how much did I spend on groceries last month", today=TODAY).kind == "aggregate"
    assert route("how much was this receipt?", today=TODAY).kind in ("plan", "aggregate")


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
