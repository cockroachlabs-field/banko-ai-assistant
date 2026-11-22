"""A person's settings shape their own turn of the assistant.

"coffee", then "coffee expenses": the second used to cost a planner call and
an answer all over again. It is the same question in other words, and by
default the cache now takes it for that; the answer says which question it
was written for. How questions are matched, how long a search page is and
what is remembered are each person's to set (services/preferences.py); a
turn started without any uses the deployment's."""

import json
import os
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import text

from banko_ai.assistant import background
from banko_ai.assistant import graph as G
from banko_ai.db.engine import get_engine
from banko_ai.services import memories, threads
from banko_ai.services.preferences import Preferences
from tests._chat import ScriptedChat
from tests._money import card_purchase, drop_user, seeded_diner

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")

SEARCH = '{"steps": [{"tool": "search_transactions", "args": {"query": "%s"}}]}'
DONE = '{"steps": []}'


@pytest.fixture(scope="module")
def member():
    uid = seeded_diner(2)
    for n in range(8):                                   # enough coffee for more than one short page
        card_purchase(uid, "Blue Bottle Coffee", f"{4 + n}.50")
    yield uid
    with get_engine().begin() as c:
        for t in ("search_cache", "answer_cache", "memories", "assistant_actions", "receipts"):
            c.execute(text(f"DELETE FROM {t} WHERE user_id = CAST(:u AS UUID)"), {"u": uid})
    drop_user(uid)


@pytest.fixture(autouse=True)
def cold(member):
    with get_engine().begin() as c:
        for t in ("search_cache", "answer_cache"):
            c.execute(text(f"DELETE FROM {t} WHERE user_id = CAST(:u AS UUID)"), {"u": member})


@pytest.fixture
def g():
    return G.build_graph(InMemorySaver())


def _ask(g, member, message, chat, **how):
    """One question in a conversation of its own, with whatever the turn left behind landed."""
    thread, events = threads.create_thread(member)["thread_id"], []
    try:
        out = G.run_turn(member, thread, message, chat=chat, graph=g, on_event=events.append, **how)
        background.drain()
        return out, events
    finally:
        threads.delete_thread(member, thread)


def _answered(g, member, question, search, answer="Here they are.", **how):
    out, _ = _ask(g, member, question, ScriptedChat(SEARCH % search, DONE, answer), **how)
    assert out["route"]["kind"] == "plan" and not out["cached"] and out["answer"] == answer
    return out


def _cache_step(events) -> dict:
    return next(e for e in events if e["type"] == "step" and e["label"] == "Answer cache")


class _Conversation:
    """One conversation that goes on: every question after the first has the ones before it behind it."""

    def __init__(self, g, member):
        self.g, self.member, self.thread = g, member, threads.create_thread(member)["thread_id"]

    def ask(self, message, chat, **how):
        events = []
        out = G.run_turn(self.member, self.thread, message, chat=chat, graph=self.g, on_event=events.append, **how)
        background.drain()
        return out, events

    def close(self):
        threads.delete_thread(self.member, self.thread)


@pytest.fixture
def conversation(g, member):
    opened = []

    def open_one() -> _Conversation:
        opened.append(_Conversation(g, member))
        return opened[-1]

    yield open_one
    for c in opened:
        c.close()


# How the answer cache matches.

def test_the_same_question_in_other_words_is_answered_from_the_cache(g, member):
    first = _answered(g, member, "coffee", "coffee", "Eight coffee purchases at Blue Bottle.")
    unused = ScriptedChat("unused")
    again, events = _ask(g, member, "coffee expenses", unused)
    assert again["cached"] and again["answer"] == first["answer"] and again["saved_tokens"] > 0
    assert unused.calls == []                                          # no planner, no answer: nothing was asked
    assert again["meta"]["cache"] == {"question": "coffee", "match": "words"}   # the chat says whose answer this is
    step = _cache_step(events)
    assert step["status"] == "hit" and "“coffee”" in step["detail"] and "in other words" in step["detail"]
    third, _ = _ask(g, member, "Show my coffee expenses, please", ScriptedChat("unused"))
    assert third["cached"] and third["meta"]["cache"]["question"] == "coffee"


def test_in_one_conversation_the_same_question_in_other_words_is_answered_from_the_cache(conversation):
    """ "coffee", then "coffee expenses" right below it: the way it was first
    asked for. The second is this conversation's own question again."""
    here = conversation()
    first, _ = here.ask("coffee", ScriptedChat(SEARCH % "coffee", DONE, "Eight coffee purchases at Blue Bottle."))
    unused = ScriptedChat("unused")
    again, events = here.ask("coffee expenses", unused)
    assert again["cached"] and again["answer"] == first["answer"] and unused.calls == []
    assert again["meta"]["cache"] == {"question": "coffee", "match": "words"}
    assert _cache_step(events)["status"] == "hit"


def test_a_category_in_other_words_is_answered_from_the_cache_in_any_conversation(g, member, conversation):
    """A category question says all it asks, so it is the same question
    after anything else was talked about."""
    _answered(g, member, "coffee", "coffee", "Eight coffee purchases at Blue Bottle.")
    here = conversation()
    here.ask("do I have any subscriptions", ScriptedChat(DONE, DONE, "Two subscriptions."))
    unused = ScriptedChat("unused")
    again, _ = here.ask("coffee purchases", unused)
    assert again["cached"] and again["answer"] == "Eight coffee purchases at Blue Bottle." and unused.calls == []
    assert again["meta"]["cache"] == {"question": "coffee", "match": "words"}


def test_a_bare_category_then_its_purchases_right_after_is_answered_from_the_cache(conversation):
    """The way it was asked for: somewhere in a conversation "coffee", then
    "coffee expenses". The question just asked, in other words."""
    here = conversation()
    here.ask("do I have any subscriptions", ScriptedChat(DONE, DONE, "Two subscriptions."))
    first, _ = here.ask("coffee", ScriptedChat(SEARCH % "coffee", DONE, "Eight coffee purchases at Blue Bottle."))
    assert not first["cached"]
    unused = ScriptedChat("unused")
    again, _ = here.ask("coffee expenses", unused)
    assert again["cached"] and again["answer"] == first["answer"] and unused.calls == []
    assert again["meta"]["cache"] == {"question": "coffee", "match": "words"}


def test_a_bare_category_after_another_question_is_asked_not_replayed(g, member, conversation):
    """ "groceries?" right after a question about restaurants in the last 30
    days asks that about groceries. It is not "show my groceries purchases",
    which another conversation asked, and what it is told never answers that
    question either."""
    everything = _answered(g, member, "show my groceries purchases", "groceries", "All the groceries.")
    here = conversation()
    spent, _ = here.ask("How much did I spend on restaurants in the last 30 days?", ScriptedChat("- Steady."))
    assert spent["route"]["kind"] == "aggregate"
    asked = ScriptedChat(SEARCH % "groceries last 30 days", DONE, "Groceries in the last 30 days.")
    short, events = here.ask("groceries?", asked)
    assert short["route"]["kind"] == "plan" and not short["cached"] and asked.calls
    assert short["answer"] == "Groceries in the last 30 days." and _cache_step(events)["status"] == "miss"
    elsewhere, _ = _ask(g, member, "groceries expenses", ScriptedChat("unused"))
    assert elsewhere["cached"] and elsewhere["answer"] == everything["answer"]


def test_the_question_right_before_has_to_be_this_conversations_own(member, conversation):
    """This conversation opened with "coffee". A purchase later, another
    conversation asked "coffee" after a question about the last 30 days, and
    that is the "coffee" now kept: the same words, about something else.
    "coffee expenses" here is asked, not answered with it."""
    here, other = conversation(), conversation()
    here.ask("coffee", ScriptedChat(SEARCH % "coffee", DONE, "All coffee."))
    card_purchase(member, "Blue Bottle Coffee", "5.25")                # a new purchase retires what was kept
    other.ask("How much did I spend on restaurants in the last 30 days?", ScriptedChat("- Steady."))
    other.ask("coffee", ScriptedChat(SEARCH % "coffee last 30 days", DONE, "Coffee in the last 30 days."))
    asked = ScriptedChat(SEARCH % "coffee", DONE, "All coffee, with the new one.")
    mine, _ = here.ask("coffee expenses", asked)
    assert not mine["cached"] and mine["answer"] == "All coffee, with the new one." and asked.calls


def test_an_answer_is_kept_as_of_the_day_its_question_was_asked(g, member, monkeypatch):
    """The write runs behind the answer: the day is the turn's, not the write's."""
    from datetime import date, timezone

    from banko_ai.utils import clock
    monkeypatch.setattr(clock, "today", lambda: date(2026, 10, 31))
    _answered(g, member, "coffee this month", "coffee", "October's coffee.")
    with get_engine().connect() as c:
        kept = c.execute(text("SELECT created_at FROM answer_cache WHERE user_id = CAST(:u AS UUID) "
                              "AND query_text = 'coffee this month'"), {"u": member}).scalar()
    assert kept.astimezone(timezone.utc).date() == date(2026, 10, 31)
    monkeypatch.setattr(clock, "today", lambda: date(2026, 11, 1))
    asked = ScriptedChat(SEARCH % "coffee", DONE, "November's coffee.")
    next_day, _ = _ask(g, member, "coffee this month", asked)
    assert not next_day["cached"] and next_day["answer"] == "November's coffee."


def test_a_question_that_is_no_category_is_asked_in_any_other_wording(g, member, conversation):
    """ "any charges over $100?" right after "show my Costco purchases" is
    about Costco. Its words are near "show my purchases over $100", which
    another conversation asked about everything. Neither is a category, so
    each is matched word for word: the one is asked, and what it is told
    never answers the other."""
    everywhere = _answered(g, member, "show my purchases over $100", "purchases over 100", "Nine, everywhere.")
    costco = conversation()
    costco.ask("show my Costco purchases", ScriptedChat(SEARCH % "costco", DONE, "Four at Costco."))
    asked = ScriptedChat(SEARCH % "costco over 100", DONE, "Two at Costco.")
    leaning, events = costco.ask("any charges over $100?", asked)
    assert not leaning["cached"] and leaning["answer"] == "Two at Costco." and asked.calls
    assert _cache_step(events)["detail"] == "miss"
    same, _ = _ask(g, member, "Show my purchases over $100", ScriptedChat("unused"))
    assert same["cached"] and same["answer"] == everywhere["answer"]                   # never "Two at Costco."
    reworded = ScriptedChat(SEARCH % "over 100", DONE, "Asked.")
    other, _ = _ask(g, member, "purchases over $100", reworded)
    assert not other["cached"] and reworded.calls


def test_the_same_question_again_says_nothing_about_other_words(g, member):
    _answered(g, member, "coffee expenses", "coffee")
    again, events = _ask(g, member, "Coffee expenses?", ScriptedChat("unused"))
    assert again["cached"] and "cache" not in again["meta"]
    assert _cache_step(events)["detail"].startswith("hit, about ")


def test_another_subject_is_another_question(g, member):
    _answered(g, member, "coffee expenses", "coffee")
    other, events = _ask(g, member, "tea expenses", ScriptedChat(SEARCH % "tea", DONE, "No tea."))
    assert not other["cached"] and other["answer"] == "No tea."
    assert _cache_step(events)["status"] == "miss"


def test_someone_who_wants_the_same_question_only_gets_exactly_that(g, member):
    _answered(g, member, "coffee", "coffee")
    exact = Preferences({"cache_match": "exact"})
    asked = ScriptedChat(SEARCH % "coffee", DONE, "Asked again.")
    again, _ = _ask(g, member, "coffee expenses", asked, prefs=exact)
    assert not again["cached"] and again["answer"] == "Asked again." and asked.calls
    same, _ = _ask(g, member, "Coffee?", ScriptedChat("unused"), prefs=exact)
    assert same["cached"]


def test_a_similar_question_is_answered_from_the_cache_and_says_how_similar(g, member):
    first = _answered(g, member, "any unusual charges lately", "unusual charges", "Nothing unusual.")
    asked = "have there been any strange charges recently"
    plain, _ = _ask(g, member, asked, ScriptedChat(SEARCH % "strange charges", DONE, "Nothing strange."))
    assert not plain["cached"]                                        # other words: not without asking for similar
    with get_engine().begin() as c:                                   # ... and that answer is not what is tested next
        c.execute(text("DELETE FROM answer_cache WHERE user_id = CAST(:u AS UUID) AND query_text = :q"),
                  {"u": member, "q": asked})
    similar = Preferences({"cache_match": "similar", "cache_similarity_threshold": 0.75})
    again, events = _ask(g, member, asked, ScriptedChat("unused"), prefs=similar)
    assert again["cached"] and again["answer"] == first["answer"]
    found = again["meta"]["cache"]
    assert found["question"] == "any unusual charges lately" and found["match"] == "similar"
    assert 0.75 <= found["similarity"] < 1
    assert f"{round(found['similarity'] * 100)}% similar" in _cache_step(events)["detail"]
    strict = Preferences({"cache_match": "similar", "cache_similarity_threshold": 0.99})
    none, _ = _ask(g, member, asked, ScriptedChat(SEARCH % "strange charges", DONE, "Nothing strange."), prefs=strict)
    assert not none["cached"]


def test_inside_a_conversation_a_similar_question_is_asked(g, member, conversation):
    """Similarity knows nothing of what the conversation was about: after
    "show my Costco purchases" the question means Costco."""
    _answered(g, member, "any unusual charges lately", "unusual charges", "Nothing unusual.")
    similar = Preferences({"cache_match": "similar", "cache_similarity_threshold": 0.75})
    here = conversation()
    here.ask("show my Costco purchases", ScriptedChat(SEARCH % "costco", DONE, "Four at Costco."), prefs=similar)
    asked = ScriptedChat(SEARCH % "costco strange", DONE, "Nothing strange at Costco.")
    leaning, _ = here.ask("have there been any strange charges recently", asked, prefs=similar)
    assert not leaning["cached"] and leaning["answer"] == "Nothing strange at Costco."


def test_a_similar_question_about_other_dates_is_asked_and_the_trace_says_why(g, member):
    """92% similar, and about another month: similarity alone would have
    replayed this month's answer for last month."""
    _answered(g, member, "coffee purchases this month", "coffee", "This month's coffee.")
    similar = Preferences({"cache_match": "similar", "cache_similarity_threshold": 0.75})
    other, events = _ask(g, member, "coffee purchases last month",
                         ScriptedChat(SEARCH % "coffee", DONE, "Last month's coffee."), prefs=similar)
    assert not other["cached"] and other["answer"] == "Last month's coffee."
    step = _cache_step(events)
    assert step["status"] == "miss" and "“coffee purchases this month”" in step["detail"]
    assert "other dates" in step["detail"] and "% similar" in step["detail"]


def test_a_merchant_asked_for_in_other_words_is_answered_from_the_cache(member, conversation):
    """ "blue bottle coffee", then "blue bottle coffee purchases": one of the user's own merchants."""
    here = conversation()
    first, _ = here.ask("blue bottle coffee", ScriptedChat(SEARCH % "blue bottle", DONE, "Eight at Blue Bottle Coffee."))
    assert first["route"]["kind"] == "plan" and not first["cached"]
    unused = ScriptedChat("unused")
    again, _ = here.ask("blue bottle coffee purchases", unused)
    assert again["cached"] and again["answer"] == "Eight at Blue Bottle Coffee." and unused.calls == []
    assert again["meta"]["cache"] == {"question": "blue bottle coffee", "match": "words"}


def test_a_name_that_is_none_of_ones_merchants_is_asked_in_any_other_wording(conversation):
    """Nobody here bought at Red Bottle: "red bottle coffee", then its purchases, is two questions."""
    here = conversation()
    first, _ = here.ask("red bottle coffee", ScriptedChat(SEARCH % "red bottle", DONE, "Nothing at Red Bottle Coffee."))
    assert first["route"]["kind"] == "plan" and not first["cached"]
    other = ScriptedChat(SEARCH % "red bottle", DONE, "Still nothing at Red Bottle Coffee.")
    stranger, _ = here.ask("red bottle coffee purchases", other)
    assert not stranger["cached"] and other.calls and stranger["answer"] == "Still nothing at Red Bottle Coffee."


def test_a_merchant_of_ones_own_named_alone_is_a_money_question(g, member):
    """ "zorblat" says nothing about money, and whoever never bought there
    is told so. For someone who has, it asks for those purchases, in any of
    the wordings a category has."""
    unused = ScriptedChat("unused")
    before, _ = _ask(g, member, "zorblat", unused)
    assert before["route"]["kind"] == "offtopic" and unused.calls == []
    card_purchase(member, "Zorblat", "18.40", category="Restaurant")
    mine, events = _ask(g, member, "zorblat", ScriptedChat(SEARCH % "zorblat", DONE, "One evening at Zorblat."))
    assert mine["route"]["kind"] == "plan" and mine["answer"] == "One evening at Zorblat."
    assert "one of your merchants" in next(e for e in events if e["type"] == "step"
                                           and e["label"] == "Intent routing")["detail"]
    for wording in ("my zorblat", "Zorblat this month", "show my zorblat"):
        out, _ = _ask(g, member, wording, ScriptedChat(SEARCH % "zorblat", DONE, "Here."))
        assert out["route"]["kind"] == "plan", wording
    # Asked to be shown the thing itself, as "show me movies" is: no more a money question than before.
    for wording in ("show me zorblat", "zebra crossing"):
        other, _ = _ask(g, member, wording, unused)
        assert other["route"]["kind"] == "offtopic" and unused.calls == [], wording


def test_most_of_what_is_off_topic_reads_nothing_about_the_users_purchases(g, member, monkeypatch):
    """Only a few words that could be a name cost a look at the user's
    merchants: not a longer message, not one that asks to be shown a thing."""
    from banko_ai.services import caches
    reads, read_names = [], caches._merchant_names
    monkeypatch.setattr(caches, "_merchant_names",
                        lambda conn, user, region=None: reads.append(user) or read_names(conn, user, region))
    caches.forget_merchants()
    unused = ScriptedChat("unused")
    for message in ("what is the weather like in Lisbon right now", "show me movies", "tell me a joke about cats"):
        out, _ = _ask(g, member, message, unused)
        assert out["route"]["kind"] == "offtopic" and reads == [], message
    out, _ = _ask(g, member, "hello", unused)
    assert out["route"]["kind"] == "offtopic" and len(reads) == 1 and unused.calls == []
    out, _ = _ask(g, member, "thanks", unused)                    # kept since "hello"
    assert out["route"]["kind"] == "offtopic" and len(reads) == 1


def test_an_answer_kept_before_the_upgrade_is_still_replayed(g, member):
    """It carries no settings. Whoever has the deployment's gets it; someone with a page of three asks again."""
    from banko_ai.services import caches, ledger
    from banko_ai.services._vector import embed
    caches.put_answer(member, version=ledger.data_version(member), language="English", query="coffee expenses",
                      qvec=embed("coffee expenses"), answer="Kept before the upgrade.", tokens=10)
    usual, _ = _ask(g, member, "coffee expenses", ScriptedChat("unused"))
    assert usual["cached"] and usual["answer"] == "Kept before the upgrade."
    asked = ScriptedChat(SEARCH % "coffee", DONE, "Three of them.")
    short, _ = _ask(g, member, "coffee expenses", asked, prefs=Preferences({"search_page_size": 3}))
    assert not short["cached"] and short["answer"] == "Three of them."


# Settings that shape an answer are kept with it.

def test_an_answer_is_not_replayed_for_someone_whose_settings_would_shape_it_differently(g, member):
    """Two people showing the same persona: one has three purchases a page.
    Their three-row answer is theirs; the other gets ten rows, asked for
    properly, and then each has their own."""
    short = Preferences({"search_page_size": 3})
    _answered(g, member, "coffee expenses", "coffee", "Three of them.", prefs=short)
    usual, _ = _ask(g, member, "coffee expenses", ScriptedChat(SEARCH % "coffee", DONE, "Ten of them."))
    assert not usual["cached"] and usual["answer"] == "Ten of them."
    mine, _ = _ask(g, member, "coffee", ScriptedChat("unused"), prefs=short)
    assert mine["cached"] and mine["answer"] == "Three of them."
    theirs, _ = _ask(g, member, "coffee", ScriptedChat("unused"))
    assert theirs["cached"] and theirs["answer"] == "Ten of them."


def test_after_a_setting_that_shapes_answers_changes_the_same_question_is_asked_again(g, member):
    """The Settings page says the next question uses what was saved."""
    _answered(g, member, "coffee expenses", "coffee", "Ten of them.")
    one_memory = Preferences({"assistant_memory_recall_k": 1})
    again, _ = _ask(g, member, "coffee expenses", ScriptedChat(SEARCH % "coffee", DONE, "With one memory."),
                    prefs=one_memory)
    assert not again["cached"] and again["answer"] == "With one memory."
    # How questions are matched shapes no answer: changing that keeps what was cached.
    exact, _ = _ask(g, member, "coffee expenses", ScriptedChat("unused"), prefs=Preferences({"cache_match": "exact"}))
    assert exact["cached"] and exact["answer"] == "Ten of them."


# With a receipt on the table.

def _receipt_on_the_table(g, member, thread, doc=None) -> str:
    doc = doc or str(uuid.uuid4())
    with get_engine().begin() as c:
        c.execute(text("UPSERT INTO receipts (document_id, user_id, status, extracted) VALUES "
                       "(CAST(:d AS UUID), CAST(:u AS UUID), 'posted', CAST(:x AS JSONB))"),
                  {"d": doc, "u": member, "x": json.dumps({"merchant": "Blue Bottle Coffee", "amount": 9.75,
                                                           "category": "Coffee", "items": ["Latte", "Croissant"],
                                                           "date": "2026-11-21"})})
    G.set_active_receipt(thread, {"document_id": doc, "merchant": "Blue Bottle Coffee", "items": ["Latte", "Croissant"],
                                  "amount": 9.75}, graph=g)
    return doc


def test_beside_a_receipt_the_conversations_own_question_in_other_words_is_answered_from_the_cache(g, conversation):
    """The demo as it is given: a receipt goes up, then "coffee", then
    "coffee expenses" in the same conversation."""
    here = conversation()
    _receipt_on_the_table(g, here.member, here.thread)
    first, events = here.ask("coffee", ScriptedChat(SEARCH % "coffee", DONE, "Eight coffee purchases."))
    assert first["route"]["kind"] == "plan" and not first["cached"] and _cache_step(events)["status"] == "miss"
    unused = ScriptedChat("unused")
    again, events = here.ask("coffee expenses", unused)
    assert again["cached"] and again["answer"] == "Eight coffee purchases." and unused.calls == []
    assert again["meta"]["cache"] == {"question": "coffee", "match": "words"}
    assert "“coffee”" in _cache_step(events)["detail"]
    same, events = here.ask("Coffee?", ScriptedChat("unused"))
    assert same["cached"] and "cache" not in same["meta"]                     # the same question: nothing to add
    exact = Preferences({"cache_match": "exact"})
    asked = ScriptedChat(SEARCH % "coffee", DONE, "Asked again.")
    strict, _ = here.ask("my coffee spending", asked, prefs=exact)
    assert not strict["cached"] and asked.calls


def test_beside_a_receipt_a_bare_category_counts_only_beside_the_question_next_to_it(g, conversation):
    here = conversation()
    _receipt_on_the_table(g, here.member, here.thread)
    here.ask("any unusual charges lately", ScriptedChat(SEARCH % "unusual", DONE, "Nothing unusual."))
    here.ask("coffee", ScriptedChat(SEARCH % "coffee", DONE, "Eight coffee purchases."))      # bare, not the opener
    next_to_it, _ = here.ask("my coffee spending", ScriptedChat("unused"))
    assert next_to_it["cached"] and next_to_it["meta"]["cache"] == {"question": "coffee", "match": "words"}
    here.ask("Any unusual charges lately?", ScriptedChat("unused"))
    asked = ScriptedChat(SEARCH % "coffee", DONE, "Asked again.")
    later, _ = here.ask("coffee purchases", asked)                     # "coffee" is not the question before any more
    assert not later["cached"] and asked.calls
    here.ask("show my groceries purchases", ScriptedChat(SEARCH % "groceries", DONE, "All the groceries."))
    here.ask("Any unusual charges lately?", ScriptedChat("unused"))
    short = ScriptedChat(SEARCH % "groceries", DONE, "Asked, it may mean something else here.")
    bare, _ = here.ask("groceries", short)                             # bare, and not next to its own question
    assert not bare["cached"] and short.calls


def test_beside_a_receipt_this_month_kept_last_month_is_asked_again(g, conversation, monkeypatch):
    """The same words on the first of the month are about another month."""
    from datetime import date

    from banko_ai.utils import clock
    here = conversation()
    _receipt_on_the_table(g, here.member, here.thread)
    monkeypatch.setattr(clock, "today", lambda: date(2026, 10, 31))
    here.ask("coffee this month", ScriptedChat(SEARCH % "coffee", DONE, "October's coffee."))
    same_day, _ = here.ask("coffee this month", ScriptedChat("unused"))
    assert same_day["cached"]
    monkeypatch.setattr(clock, "today", lambda: date(2026, 11, 1))
    asked = ScriptedChat(SEARCH % "coffee", DONE, "November's coffee.")
    next_day, _ = here.ask("coffee this month", asked)
    assert not next_day["cached"] and next_day["answer"] == "November's coffee."


def test_beside_a_receipt_a_question_that_is_no_category_is_matched_word_for_word(g, conversation):
    here = conversation()
    _receipt_on_the_table(g, here.member, here.thread)
    here.ask("any unusual charges lately", ScriptedChat(SEARCH % "unusual", DONE, "Nothing unusual."))
    asked = ScriptedChat(SEARCH % "unusual", DONE, "Asked again.")
    other, _ = here.ask("unusual charges lately", asked)
    assert not other["cached"] and asked.calls
    same, _ = here.ask("Any unusual charges lately?", ScriptedChat("unused"))
    assert same["cached"] and same["answer"] == "Nothing unusual."


def test_a_question_about_the_receipt_itself_is_matched_word_for_word(g, conversation):
    """About one document every verb counts: what is on it, what it came to
    and the receipt itself are three questions. These are matched word for
    word, whatever else the cache takes for the same question."""
    here = conversation()
    _receipt_on_the_table(g, here.member, here.thread)
    first, _ = here.ask("What's on this receipt?", ScriptedChat("A latte and a croissant."))
    assert first["route"]["kind"] == "receipt" and not first["cached"]
    asked = ScriptedChat("Here it is: $9.75 at Blue Bottle Coffee.")
    other, _ = here.ask("show me this receipt", asked)
    assert other["route"]["kind"] == "receipt" and not other["cached"] and asked.calls
    again, _ = here.ask("what's on this receipt", ScriptedChat("unused"))
    assert again["cached"] and again["answer"] == "A latte and a croissant."


# How many, and what is remembered.

def _found(out) -> list[dict]:
    return next(r for r in out["results"] if r["tool"] == "search_transactions")["result"]["items"]


def _results(g, member, question, chat, **how):
    """The turn's tool results, which the answer event does not carry."""
    thread = threads.create_thread(member)["thread_id"]
    try:
        G.run_turn(member, thread, question, chat=chat, graph=g, **how)
        background.drain()
        return G.thread_state(thread, graph=g)
    finally:
        threads.delete_thread(member, thread)


def test_a_search_page_is_as_long_as_the_person_set(g, member):
    short = Preferences({"search_page_size": 3})
    state = _results(g, member, "coffee expenses", ScriptedChat(SEARCH % "coffee", DONE, "Three of them."), prefs=short)
    assert len(_found(state)) == 3
    assert state["meta"]["search"]["size"] == 3 and state["meta"]["search"]["next"]     # Show more goes on by threes
    # The same question in other words, from someone with the usual page: asked, not served the short answer.
    usual = _results(g, member, "coffee purchases", ScriptedChat(SEARCH % "coffee", DONE, "Ten of them."))
    assert len(_found(usual)) > 3


def test_a_page_size_the_model_asks_for_itself_still_counts(g, member):
    chat = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"query": "coffee", "limit": 2}}]}',
                        DONE, "Two of them.")
    state = _results(g, member, "two coffee purchases", chat, prefs=Preferences({"search_page_size": 5}))
    assert len(_found(state)) == 2


def test_show_more_pages_by_the_persons_size(g, member):
    short = Preferences({"search_page_size": 3})
    thread = threads.create_thread(member)["thread_id"]
    try:
        G.run_turn(member, thread, "coffee expenses", chat=ScriptedChat(SEARCH % "coffee", DONE, "Three."), graph=g,
                   prefs=short)
        more = G.run_turn(member, thread, "show more", chat=ScriptedChat("unused"), graph=g, prefs=short)
        assert more["route"]["kind"] == "more" and len(more["meta"]["search"]["items"]) == 3
        assert more["meta"]["search"]["size"] == 3
    finally:
        background.drain()
        threads.delete_thread(member, thread)


def test_someone_who_turned_remembering_off_is_not_remembered(g, member):
    """ "I usually ..." is the kind of sentence the assistant keeps as a
    memory, with one more model call after the answer. Off: no call, no memory."""
    before = len(memories.list_memories(member))
    chat = ScriptedChat(DONE, DONE, "Noted.", '{"content": "I usually buy coffee at Blue Bottle", "kind": "preference"}')
    out, _ = _ask(g, member, "i usually buy coffee at blue bottle, is that a lot?", chat,
                  prefs=Preferences({"assistant_implicit_memory": False}))
    assert out["answer"] == "Noted."
    assert not any("worth remembering" in prompt for prompt in chat.prompts())     # the model was never asked
    assert len(memories.list_memories(member)) == before


def test_memories_come_to_a_question_as_the_person_set(g, member):
    kept = [memories.remember(member, text_)["memory_id"]
            for text_ in ("I'm saving for a trip to Japan", "I'm saving for a new bicycle", "I'm saving for a kayak")]
    try:
        def recalled(**own) -> list[dict]:
            state = _results(g, member, "what am I saving for?", ScriptedChat(DONE, DONE, "A few things."),
                             prefs=Preferences(own))
            return state["memories"]
        assert len(recalled()) == 3
        assert len(recalled(assistant_memory_recall_k=1)) == 1
        assert recalled(assistant_memory_max_distance=0.1) == []          # none of them is nearly the question itself
    finally:
        for memory_id in kept:
            memories.forget(member, memory_id)


def test_a_turn_started_without_settings_uses_the_deployments(g, member, monkeypatch):
    from banko_ai.config.settings import reset_config
    _answered(g, member, "coffee", "coffee")
    monkeypatch.setenv("CACHE_MATCH", "exact")
    reset_config()
    try:
        again, _ = _ask(g, member, "coffee expenses", ScriptedChat(SEARCH % "coffee", DONE, "Asked again."))
        assert not again["cached"]
    finally:
        monkeypatch.undo()
        reset_config()
