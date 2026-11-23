"""A follow-up to an answer from SQL is answered from SQL.

"How much did I spend on restaurants last month?" is read from SQL. "What
about coffee?" right after it was the planner's: a model read the
conversation, picked a tool and its window, and wrote the answer, which no
cache kept. But it asks the question before about another name, and SQL has
that question: the same operation, the same days, the other name. The same
goes for other days ("and this month?"), and for the cards of a category or
a merchant.

Only words that point back and then name a category, one of the user's
merchants, some days, or a name with its days. Anything else that leans on
the conversation is still the planner's."""

import os
from datetime import date, timedelta

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import text

from banko_ai.assistant import background
from banko_ai.assistant import graph as G
from banko_ai.assistant.router import route
from banko_ai.db.engine import get_engine
from banko_ai.services import threads
from banko_ai.utils.intent_classifier import EVER
from tests._chat import ScriptedChat
from tests._money import card_purchase, drop_user, seeded_diner

TODAY = date(2026, 10, 12)
SEPTEMBER = (date(2026, 9, 1), date(2026, 10, 1))
OCTOBER = (date(2026, 10, 1), date(2026, 11, 1))
SHOPS = {"starbucks", "netflix", "blue bottle coffee"}

# What the turn before asked SQL, as the graph keeps it for the next turn.
TOTAL = {"op": "sum", "subject": "Restaurant", "from": "2026-09-01", "to": "2026-10-01", "limit": 1}
EVERYTHING = {"op": "sum", "subject": None, "from": "2026-09-01", "to": "2026-10-01", "limit": 1}
TOP_THREE = {"op": "largest", "subject": None, "from": "2026-09-01", "to": "2026-10-01", "limit": 3}
CARD = {"op": "purchases", "subject": "Coffee", "from": None, "to": None, "limit": 1}
CARD_FOR_DAYS = {"op": "purchases", "subject": "Coffee", "from": "2026-09-01", "to": "2026-10-01", "limit": 1}
VISITS = {"op": "visits", "subject": "Starbucks", "from": None, "to": None, "limit": 1}
VISITS_FOR_DAYS = {"op": "visits", "subject": "Starbucks", "from": "2026-09-01", "to": "2026-10-01", "limit": 1}


def _after(last, message, **kw):
    return route(message, has_history=True, today=TODAY, last_asked=last, is_merchant=SHOPS.__contains__, **kw)


def _total(found):
    return (found.kind, found.intent.operation, found.intent.subject, found.intent.window_start,
            found.intent.window_end, found.follow_up)


@pytest.mark.parametrize("follow", ["what about coffee?", "What about coffee", "and coffee?", "how about coffee",
                                    "and for coffee", "and on coffee?", "same for coffee", "coffee too", "coffee as well?",
                                    "also coffee", "and what about coffee?", "ok, and coffee?", "what about my coffee?"])
def test_after_a_total_another_name_asks_that_total_of_it(follow):
    assert _total(_after(TOTAL, follow)) == ("aggregate", "sum", "coffee", *SEPTEMBER, True), follow


@pytest.mark.parametrize("follow, days", [
    ("what about this month?", OCTOBER), ("and this month?", OCTOBER), ("how about in august", (date(2026, 8, 1), date(2026, 9, 1))),
    ("and for august?", (date(2026, 8, 1), date(2026, 9, 1))), ("what about the last 7 days?", (date(2026, 10, 5), date(2026, 10, 13))),
    ("and in 2025?", (date(2025, 1, 1), date(2026, 1, 1))), ("this month too", OCTOBER),
])
def test_after_a_total_other_days_ask_that_total_for_them(follow, days):
    assert _total(_after(TOTAL, follow)) == ("aggregate", "sum", "Restaurant", *days, True), follow
    assert _total(_after(EVERYTHING, follow)) == ("aggregate", "sum", None, *days, True), follow


def test_a_name_with_its_days_changes_both():
    assert _total(_after(TOTAL, "what about coffee this month?")) == ("aggregate", "sum", "coffee", *OCTOBER, True)
    assert _total(_after(TOTAL, "and groceries in august?")) == ("aggregate", "sum", "groceries", date(2026, 8, 1),
                                                                 date(2026, 9, 1), True)


@pytest.mark.parametrize("op", ["count", "average", "smallest"])
def test_what_was_asked_to_be_worked_out_is_carried(op):
    found = _after({**TOTAL, "op": op}, "what about coffee?")
    assert _total(found) == ("aggregate", op, "coffee", *SEPTEMBER, True)


def test_how_many_were_asked_for_is_carried():
    found = _after(TOP_THREE, "what about coffee?")
    assert _total(found) == ("aggregate", "largest", "coffee", *SEPTEMBER, True) and found.intent.limit == 3
    assert _after(TOP_THREE, "and this month?").intent.limit == 3


def test_one_of_the_users_merchants_is_another_name_too():
    assert _total(_after(TOTAL, "what about starbucks?")) == ("aggregate", "sum", "starbucks", *SEPTEMBER, True)
    assert _total(_after(TOTAL, "and at netflix?")) == ("aggregate", "sum", "netflix", *SEPTEMBER, True)
    # For someone who never bought there it is no name: the planner's, as it was.
    stranger = route("what about starbucks?", has_history=True, today=TODAY, last_asked=TOTAL,
                     is_merchant=lambda name: False)
    assert stranger.kind == "plan"


def test_after_a_categorys_card_another_name_gets_its_card_for_the_same_days():
    groceries = _after(CARD, "what about groceries?")
    assert _total(groceries) == ("aggregate", "purchases", "groceries", EVER, TODAY + timedelta(days=1), True)
    assert _total(_after(CARD_FOR_DAYS, "and groceries?")) == ("aggregate", "purchases", "groceries", *SEPTEMBER, True)
    assert _total(_after(CARD, "and this month?")) == ("aggregate", "purchases", "Coffee", *OCTOBER, True)
    shop = _after(CARD_FOR_DAYS, "what about starbucks?")
    assert (shop.kind, shop.payload, shop.window, shop.follow_up) == ("visits", "starbucks", SEPTEMBER, True)
    assert _after(CARD, "what about starbucks?").window is None


def test_after_a_merchants_card_the_same():
    other = _after(VISITS_FOR_DAYS, "what about netflix?")
    assert (other.kind, other.payload, other.window, other.follow_up) == ("visits", "netflix", SEPTEMBER, True)
    assert (_after(VISITS, "and netflix?").payload, _after(VISITS, "and netflix?").window) == ("netflix", None)
    days = _after(VISITS, "and this month?")
    assert (days.kind, days.payload, days.window) == ("visits", "Starbucks", OCTOBER)
    assert _total(_after(VISITS_FOR_DAYS, "what about coffee?")) == ("aggregate", "purchases", "coffee", *SEPTEMBER, True)


def test_a_card_for_all_time_is_for_all_time_as_of_today():
    """No days were named, so none are carried: everything there is, up to
    the day the follow-up is asked."""
    later = route("what about groceries?", has_history=True, today=TODAY + timedelta(days=3), last_asked=CARD)
    assert (later.intent.window_start, later.intent.window_end) == (EVER, TODAY + timedelta(days=4))


@pytest.mark.parametrize("message", ["what about it?", "what about my budget?", "what about bills?", "and the biggest one?",
                                     "what about saving more?", "how about that", "and why is that?", "same for the rest",
                                     "what about coffee and tea?", "and coffee versus groceries?", "what about next month?",
                                     "and show me movies", "and the month before?"])
def test_anything_else_that_leans_on_the_conversation_is_still_the_planners(message):
    assert _after(TOTAL, message).kind == "plan", message
    assert _after(CARD, message).kind == "plan", message


def test_with_no_answer_from_sql_before_it_nothing_changes():
    """After a planner's answer, a receipt's, or none, "what about coffee?" is what it was."""
    for last in (None, {}):
        found = route("what about coffee?", has_history=True, today=TODAY, last_asked=last)
        assert found.kind == "plan", last
    assert route("what about coffee?", today=TODAY, last_asked=TOTAL, context={"kind": "nudge", "id": "n1"}).kind == "plan"


def test_a_question_before_that_cannot_be_asked_again_leaves_the_follow_up_to_the_planner():
    """What the turn before left is read as it is: a total without its days,
    a card of nothing, or a kind of answer this build does not know (a
    conversation another build wrote) are no question to ask again."""
    for last in ({"op": "sum", "subject": "Coffee", "from": None, "to": None, "limit": 1},
                 {"op": "purchases", "subject": None, "from": None, "to": None, "limit": 1},
                 {"op": "forecast", "subject": "Coffee", "from": "2026-09-01", "to": "2026-10-01", "limit": 1}):
        assert _after(last, "what about groceries?").kind == "plan" or last["op"] == "purchases", last
        assert _after(last, "and this month?").kind == "plan" or last["op"] == "sum", last
    # A total without its days still takes days of the follow-up's own.
    assert _total(_after({"op": "sum", "subject": "Coffee", "from": None, "to": None, "limit": 1},
                         "and this month?")) == ("aggregate", "sum", "Coffee", *OCTOBER, True)


def test_beside_a_receipt_nothing_changes():
    receipt = {"document_id": "d1", "merchant": "Trader Joe's", "items": ["Pinot Noir wine"], "amount": 23.47}
    with_it = route("what about coffee?", has_history=True, today=TODAY, last_asked=TOTAL, active_receipt=receipt,
                    last_route="aggregate")
    assert with_it == route("what about coffee?", has_history=True, today=TODAY, active_receipt=receipt,
                            last_route="aggregate")
    assert route("what about the wine?", has_history=True, today=TODAY, last_asked=TOTAL, active_receipt=receipt,
                 last_route="receipt").kind == "receipt"


def test_a_whole_question_that_starts_with_and_is_its_own_question():
    whole = _after(CARD, "and how much did I spend on coffee last month?")
    assert _total(whole) == ("aggregate", "sum", "coffee", *SEPTEMBER, False)
    assert _after(TOTAL, "show more").kind == "plan"                # no search to go on from


def test_a_name_alone_is_still_its_own_card():
    """No word points back, so it asks what it asks anywhere: the card, for all time."""
    alone = _after(TOTAL, "coffee")
    assert _total(alone) == ("aggregate", "purchases", "coffee", EVER, TODAY + timedelta(days=1), False)


def test_the_route_says_what_it_carried():
    assert _after(TOTAL, "what about coffee?").detail == "the question before, about coffee"
    assert _after(TOTAL, "and this month?").detail == "the question before, for other days"
    assert _after(TOTAL, "what about coffee this month?").detail == "the question before, about coffee, for other days"
    as_state = _after(VISITS, "and this month?").as_dict()
    assert as_state["window"] == ["2026-10-01", "2026-11-01"] and as_state["follow_up"] is True


# ── end to end ───────────────────────────────────────────────────────────

needs_db = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")
SILENT = "no model is asked"


@pytest.fixture(scope="module")
def member():
    uid = seeded_diner(3)
    card_purchase(uid, "Blue Bottle Coffee", "6.25")
    card_purchase(uid, "Corner Kiosk", "4.20")
    yield uid
    with get_engine().begin() as c:
        for t in ("search_cache", "answer_cache", "memories", "assistant_actions", "receipts"):
            c.execute(text(f"DELETE FROM {t} WHERE user_id = CAST(:u AS UUID)"), {"u": uid})
    drop_user(uid)


@pytest.fixture
def chat_thread(member):
    """One conversation: ask(message, chat) returns (answer, events, chat)."""
    g = G.build_graph(InMemorySaver())
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM search_cache WHERE user_id = CAST(:u AS UUID)"), {"u": member})
    thread = threads.create_thread(member)["thread_id"]

    def ask(message, chat=None, **kw):
        chat = chat or ScriptedChat(SILENT)
        events = []
        out = G.run_turn(member, thread, message, chat=chat, graph=g, on_event=events.append, **kw)
        background.drain()
        return out, events, chat
    yield ask
    threads.delete_thread(member, thread)


def _sum(member, where, params=None):
    with get_engine().connect() as c:
        total, count = c.execute(text(
            f"SELECT COALESCE(sum(expense_amount), 0), count(*) FROM expenses WHERE user_id = CAST(:u AS UUID) AND {where}"),
            {"u": member, **(params or {})}).one()
    return float(total), int(count)


def _last_30() -> tuple[date, date]:
    """ "the last 30 days" as the totals read it: thirty days back, through today."""
    return date.today() - timedelta(days=30), date.today() + timedelta(days=1)


@needs_db
def test_what_about_another_category_is_the_same_total_from_sql(chat_thread, member):
    start, end = _last_30()
    first, _, _ = chat_thread("how much did I spend on groceries in the last 30 days", ScriptedChat("- Steady."))
    assert first["route"]["kind"] == "aggregate"
    total, count = _sum(member, "shopping_type = 'Restaurant' AND expense_date >= :s AND expense_date < :e",
                        {"s": start, "e": end})
    narrator = ScriptedChat("- Mostly dinners.")
    out, events, _ = chat_thread("what about restaurants?", narrator)
    assert out["route"]["kind"] == "aggregate" and out["route"]["follow_up"]
    assert out["answer"].startswith(f"You spent **${total:,.2f}** on Restaurant from {start} to "
                                    f"{end - timedelta(days=1)} ({count} transactions).")
    steps = [e["label"] for e in events if e["type"] == "step" and e["status"] != "running"]
    assert "Planner" not in steps and "Aggregation SQL" in steps
    assert len(narrator.calls) == 1                         # the observations under a total, as for any total
    assert out["meta"]["explain"] and "expenses@idx_expenses_user_date" in out["meta"]["explain"]


@needs_db
def test_other_days_and_then_another_name_again(chat_thread, member):
    """Each answer from SQL is what the next follow-up leans on."""
    chat_thread("how much did I spend on groceries in the last 30 days", ScriptedChat("- Steady."))
    month_start = date.today().replace(day=1)
    next_month = (month_start + timedelta(days=32)).replace(day=1)
    out, _, _ = chat_thread("and this month?", ScriptedChat("- Steady."))
    total, count = _sum(member, "shopping_type = 'Groceries' AND expense_date >= :s AND expense_date < :e",
                        {"s": month_start, "e": next_month})
    assert out["route"]["kind"] == "aggregate"
    assert out["answer"].splitlines()[0].startswith(f"You spent **${total:,.2f}** on Groceries from {month_start} to ")
    again, _, _ = chat_thread("what about coffee?", ScriptedChat("- Steady."))
    total, count = _sum(member, "shopping_type = 'Coffee' AND expense_date >= :s AND expense_date < :e",
                        {"s": month_start, "e": next_month})
    assert again["answer"].splitlines()[0].startswith(f"You spent **${total:,.2f}** on Coffee from {month_start} to ")


@needs_db
def test_after_a_card_a_follow_up_is_a_card_and_no_model_is_asked(chat_thread, member):
    card, _, chat = chat_thread("restaurants")
    assert card["route"]["kind"] == "aggregate" and chat.calls == []
    total, count = _sum(member, "shopping_type = 'Groceries'")
    out, events, chat = chat_thread("what about groceries?")
    assert out["route"]["kind"] == "aggregate" and chat.calls == []
    assert out["answer"].startswith(f"You spent **${total:,.2f}** on Groceries since ")
    assert f"({count} transactions)" in out["answer"].splitlines()[0]
    shop, _, chat = chat_thread("and blue bottle coffee?")
    assert shop["route"]["kind"] == "visits" and chat.calls == []
    assert shop["answer"].startswith("You went to **Blue Bottle Coffee** ")
    other, _, chat = chat_thread("what about corner kiosk?")
    assert other["route"]["kind"] == "visits" and chat.calls == []
    assert other["answer"].startswith("You went to **Corner Kiosk** 1 time ")


@needs_db
def test_a_merchants_days_are_carried_to_the_next_merchant(chat_thread, member):
    month_start = date.today().replace(day=1)
    first, _, chat = chat_thread("blue bottle coffee purchases this month")
    assert first["route"]["kind"] == "visits" and chat.calls == []
    assert f"between {month_start} and " in first["answer"]
    other, _, chat = chat_thread("what about corner kiosk?")
    assert other["route"]["kind"] == "visits" and chat.calls == []
    assert f"**Corner Kiosk** 1 time between {month_start} and " in other["answer"]
    earlier, _, chat = chat_thread("and in 2019?")
    assert earlier["route"]["kind"] == "visits" and chat.calls == []
    assert earlier["answer"].startswith("I don't see a visit to **Corner Kiosk** between 2019-01-01 and 2019-12-31.")


@needs_db
def test_what_a_turn_asked_sql_is_kept_for_the_next_one(member):
    """The record a follow-up leans on, as the conversation keeps it. A card
    for all time keeps no days: "everything up to today" asked tomorrow is
    everything up to tomorrow."""
    g = G.build_graph(InMemorySaver())
    thread = threads.create_thread(member)["thread_id"]
    try:
        def kept_after(message, chat=None):
            G.run_turn(member, thread, message, chat=chat or ScriptedChat(SILENT), graph=g)
            background.drain()
            return G.thread_state(thread, graph=g).get("last_asked")
        start, end = _last_30()
        assert kept_after("how much did I spend on groceries in the last 30 days", ScriptedChat("- Steady.")) == {
            "op": "sum", "subject": "Groceries", "from": str(start), "to": str(end), "limit": 1}
        assert kept_after("restaurants") == {"op": "purchases", "subject": "Restaurant", "from": None, "to": None,
                                             "limit": 1}
        assert kept_after("restaurants") == {"op": "purchases", "subject": "Restaurant", "from": None, "to": None,
                                             "limit": 1}                      # from the kept card as well
        month = date.today().replace(day=1)
        visits = kept_after("blue bottle coffee purchases this month")
        # The merchant as its purchases name it, or as it was typed when the card came from the cache.
        assert (visits["op"], visits["subject"].lower(), visits["from"]) == ("visits", "blue bottle coffee", str(month))
        top = kept_after("what were my 3 biggest purchases this year?")
        assert (top["op"], top["subject"], top["limit"]) == ("largest", None, 3)
        assert kept_after("hello there") is None                              # not an answer from SQL
    finally:
        threads.delete_thread(member, thread)


@needs_db
def test_a_follow_up_to_the_largest_purchases_and_to_a_merchants_kept_card(chat_thread, member):
    first, _, chat = chat_thread("what was my biggest purchase this year?")
    assert first["route"]["kind"] == "aggregate" and chat.calls == []
    coffee, _, chat = chat_thread("what about coffee?")
    assert coffee["route"]["kind"] == "aggregate" and chat.calls == []
    assert coffee["answer"].startswith("Your biggest Coffee purchase ")
    chat_thread("blue bottle coffee")
    again, _, _ = chat_thread("blue bottle coffee")
    assert again["cached"]
    other, _, chat = chat_thread("what about corner kiosk?")                  # the card before came from the cache
    assert other["route"]["kind"] == "visits" and chat.calls == []
    assert other["answer"].startswith("You went to **Corner Kiosk** 1 time ")


@needs_db
def test_paging_through_a_list_leaves_the_question_before_where_it_was(chat_thread, member):
    """ "show more" is not another question: what comes after it still leans on the card it paged."""
    card, _, _ = chat_thread("restaurants")
    assert card["meta"]["search"]["next"]
    more, _, chat = chat_thread("show more")
    assert more["route"]["kind"] == "more" and chat.calls == []
    out, _, chat = chat_thread("what about groceries?")
    assert out["route"]["kind"] == "aggregate" and chat.calls == []
    assert " on Groceries since " in out["answer"].splitlines()[0]


@needs_db
def test_a_follow_up_comes_from_the_kept_card_when_there_is_one(chat_thread, member):
    """The question before, about another name, is a question SQL has been
    asked: its card is the one kept for it, whoever asked and however."""
    chat_thread("restaurants")
    first, _, _ = chat_thread("what about groceries?")
    assert not first["cached"]
    chat_thread("restaurants")
    again, _, chat = chat_thread("and groceries?")
    assert again["cached"] and again["answer"] == first["answer"] and chat.calls == []


@needs_db
def test_after_the_planner_answered_a_follow_up_is_the_planners(chat_thread, member):
    planned = ScriptedChat('{"steps": []}', '{"steps": []}', "Nowhere in particular.")
    out, _, _ = chat_thread("where could I cut back on eating out lately?", planned)
    assert out["route"]["kind"] == "plan"
    again = ScriptedChat('{"steps": []}', '{"steps": []}', "Coffee looks fine.")
    out, _, _ = chat_thread("what about coffee?", again)
    assert out["route"]["kind"] == "plan" and out["answer"] == "Coffee looks fine."
    # And a total that went on to the planner is no answer from SQL either.
    chat_thread("how much did I spend on groceries in the last 30 days", ScriptedChat("- Steady."))
    out, _, _ = chat_thread("what about my budget?", ScriptedChat('{"steps": []}', '{"steps": []}', "No budget set."))
    assert out["route"]["kind"] == "plan"
    out, _, _ = chat_thread("what about coffee?", ScriptedChat('{"steps": []}', '{"steps": []}', "Still fine."))
    assert out["route"]["kind"] == "plan"                    # the turn before was the planner's
