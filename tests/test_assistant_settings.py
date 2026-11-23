"""A person's settings shape their own turn of the assistant.

How questions are matched to cached answers, how long a page of purchases is
and what is remembered are each person's to set (services/preferences.py); a
turn started without any uses the deployment's.

"coffee", then "coffee expenses", was this module's first test: the second
question came from the answer cache, as the same question in other words.
Those questions are read from SQL now and never reach a model or its cache
(tests/test_named_alone_from_sql.py), so the cache is tested here with
questions the planner still answers."""

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

def test_a_question_that_points_back_is_asked_and_its_answer_kept_nowhere(g, member, conversation):
    """ "what about groceries?" right after a question about restaurants in
    the last 30 days asks that about groceries. It takes the conversation to
    read, so the planner is asked, and what it is told answers no other
    question: nothing is kept under these words, and the cache is not even
    looked in."""
    here = conversation()
    spent, _ = here.ask("How much did I spend on restaurants in the last 30 days?", ScriptedChat("- Steady."))
    assert spent["route"]["kind"] == "aggregate"
    asked = ScriptedChat(SEARCH % "groceries last 30 days", DONE, "Groceries in the last 30 days.")
    short, events = here.ask("what about groceries?", asked)
    assert short["route"]["kind"] == "plan" and not short["cached"] and asked.calls
    assert short["answer"] == "Groceries in the last 30 days."
    assert "Answer cache" not in [e.get("label") for e in events]
    with get_engine().connect() as c:
        assert c.execute(text("SELECT count(*) FROM answer_cache WHERE user_id = CAST(:u AS UUID)"),
                         {"u": member}).scalar() == 0
    again = ScriptedChat(SEARCH % "groceries last 30 days", DONE, "Asked again.")
    repeat, _ = here.ask("what about groceries?", again)
    assert not repeat["cached"] and repeat["answer"] == "Asked again."


def test_a_category_named_alone_after_another_question_is_its_card_and_no_model_is_asked(g, member, conversation):
    """ "groceries?" with nothing that points back is a question of its own
    there too: the card SQL writes, the same one wherever it is asked, kept
    for the next time. It used to be the planner's from the second turn of
    a conversation on, and its answer was kept nowhere."""
    here = conversation()
    here.ask("How much did I spend on restaurants in the last 30 days?", ScriptedChat("- Steady."))
    silent = ScriptedChat("no model is asked")
    short, events = here.ask("groceries?", silent)
    assert short["route"]["kind"] == "aggregate" and silent.calls == []
    assert short["answer"].startswith("You spent **$") and " on Groceries since " in short["answer"]
    repeat, _ = here.ask("groceries?", silent)
    assert repeat["cached"] and repeat["answer"] == short["answer"] and silent.calls == []
    opened, _ = conversation().ask("groceries", silent)
    assert opened["cached"] and opened["answer"] == short["answer"] and silent.calls == []


def test_an_answer_is_kept_as_of_the_day_its_question_was_asked(g, member, monkeypatch):
    """The write runs behind the answer: the day is the turn's, not the write's."""
    from datetime import date, timezone

    from banko_ai.utils import clock
    monkeypatch.setattr(clock, "today", lambda: date(2026, 10, 31))
    _answered(g, member, "any unusual charges this month", "unusual charges", "October's oddities.")
    with get_engine().connect() as c:
        kept = c.execute(text("SELECT created_at FROM answer_cache WHERE user_id = CAST(:u AS UUID) "
                              "AND query_text = 'any unusual charges this month'"), {"u": member}).scalar()
    assert kept.astimezone(timezone.utc).date() == date(2026, 10, 31)
    monkeypatch.setattr(clock, "today", lambda: date(2026, 11, 1))
    asked = ScriptedChat(SEARCH % "unusual charges", DONE, "November's oddities.")
    next_day, _ = _ask(g, member, "any unusual charges this month", asked)
    assert not next_day["cached"] and next_day["answer"] == "November's oddities."


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


def test_the_same_question_again_is_answered_from_the_cache(g, member):
    first = _answered(g, member, "any unusual charges lately", "unusual charges", "Nothing unusual.")
    unused = ScriptedChat("unused")
    again, events = _ask(g, member, "Any unusual charges lately?", unused)
    assert again["cached"] and again["answer"] == first["answer"] and again["saved_tokens"] > 0
    assert unused.calls == []                                          # no planner, no answer: nothing was asked
    assert "cache" not in again["meta"]                                # its own words: nothing to say about whose
    assert _cache_step(events)["detail"].startswith("hit, about ")
    reworded = ScriptedChat(SEARCH % "unusual charges", DONE, "Asked.")
    other, _ = _ask(g, member, "unusual charges lately", reworded)     # other words are another question
    assert not other["cached"] and reworded.calls


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
    """Nearly the same words, and about another month: similarity alone
    would have replayed this month's answer for last month."""
    _answered(g, member, "any unusual charges this month", "unusual charges", "This month's oddities.")
    similar = Preferences({"cache_match": "similar", "cache_similarity_threshold": 0.75})
    other, events = _ask(g, member, "any unusual charges last month",
                         ScriptedChat(SEARCH % "unusual charges", DONE, "Last month's oddities."), prefs=similar)
    assert not other["cached"] and other["answer"] == "Last month's oddities."
    step = _cache_step(events)
    assert step["status"] == "miss" and "“any unusual charges this month”" in step["detail"]
    assert "other dates" in step["detail"] and "% similar" in step["detail"]


def test_a_name_that_is_none_of_ones_merchants_is_asked_in_any_other_wording(conversation):
    """Nobody here bought at Red Bottle: "red bottle coffee", then its purchases, is two questions."""
    here = conversation()
    first, _ = here.ask("red bottle coffee", ScriptedChat(SEARCH % "red bottle", DONE, "Nothing at Red Bottle Coffee."))
    assert first["route"]["kind"] == "plan" and not first["cached"]
    other = ScriptedChat(SEARCH % "red bottle", DONE, "Still nothing at Red Bottle Coffee.")
    stranger, _ = here.ask("red bottle coffee purchases", other)
    assert not stranger["cached"] and other.calls and stranger["answer"] == "Still nothing at Red Bottle Coffee."


def test_a_merchant_of_ones_own_named_alone_is_read_from_sql(g, member):
    """ "zorblat" says nothing about money, and whoever never bought there
    is told so. For someone who has, it asks for those purchases, in any of
    the wordings a category has, and SQL answers: no model is asked."""
    unused = ScriptedChat("unused")
    before, _ = _ask(g, member, "zorblat", unused)
    assert before["route"]["kind"] == "offtopic" and unused.calls == []
    card_purchase(member, "Zorblat", "18.40", category="Restaurant")
    for wording in ("zorblat", "my zorblat", "Zorblat this month", "show my zorblat", "zorblat purchases"):
        mine, events = _ask(g, member, wording, unused)
        assert mine["route"]["kind"] == "visits" and unused.calls == [], wording
        assert mine["answer"].startswith("You went to **Zorblat** 1 time "), wording
        assert "one of your merchants" in next(e for e in events if e["type"] == "step"
                                               and e["label"] == "Intent routing")["detail"]
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
    question = "any unusual charges lately"
    caches.put_answer(member, version=ledger.data_version(member), language="English", query=question,
                      qvec=embed(question), answer="Kept before the upgrade.", tokens=10)
    usual, _ = _ask(g, member, question, ScriptedChat("unused"))
    assert usual["cached"] and usual["answer"] == "Kept before the upgrade."
    asked = ScriptedChat(SEARCH % "unusual charges", DONE, "Three of them.")
    short, _ = _ask(g, member, question, asked, prefs=Preferences({"search_page_size": 3}))
    assert not short["cached"] and short["answer"] == "Three of them."


# Settings that shape an answer are kept with it.

def test_an_answer_is_not_replayed_for_someone_whose_settings_would_shape_it_differently(g, member):
    """Two people showing the same persona: one has three purchases a page.
    Their three-row answer is theirs; the other gets ten rows, asked for
    properly, and then each has their own."""
    question, short = "any unusual charges lately", Preferences({"search_page_size": 3})
    _answered(g, member, question, "unusual charges", "Three of them.", prefs=short)
    usual, _ = _ask(g, member, question, ScriptedChat(SEARCH % "unusual charges", DONE, "Ten of them."))
    assert not usual["cached"] and usual["answer"] == "Ten of them."
    mine, _ = _ask(g, member, question, ScriptedChat("unused"), prefs=short)
    assert mine["cached"] and mine["answer"] == "Three of them."
    theirs, _ = _ask(g, member, question, ScriptedChat("unused"))
    assert theirs["cached"] and theirs["answer"] == "Ten of them."


def test_after_a_setting_that_shapes_answers_changes_the_same_question_is_asked_again(g, member):
    """The Settings page says the next question uses what was saved."""
    question = "any unusual charges lately"
    _answered(g, member, question, "unusual charges", "Ten of them.")
    one_memory = Preferences({"assistant_memory_recall_k": 1})
    again, _ = _ask(g, member, question, ScriptedChat(SEARCH % "unusual charges", DONE, "With one memory."),
                    prefs=one_memory)
    assert not again["cached"] and again["answer"] == "With one memory."
    # How questions are matched shapes no answer: changing that keeps what was cached.
    similar, _ = _ask(g, member, question, ScriptedChat("unused"), prefs=Preferences({"cache_match": "similar"}))
    assert similar["cached"] and similar["answer"] == "Ten of them."


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


def test_beside_a_receipt_a_categorys_purchases_come_from_sql_and_the_bare_word_is_asked(g, conversation):
    """The demo as it is given: a receipt goes up, then "coffee", then
    "coffee expenses" in the same conversation. Beside a receipt the bare
    word may be a line on it, so the planner is asked and nothing is kept
    under it. With a word for its purchases it is the category, from SQL."""
    here = conversation()
    _receipt_on_the_table(g, here.member, here.thread)
    first, events = here.ask("coffee", ScriptedChat(SEARCH % "coffee", DONE, "Eight coffee purchases."))
    assert first["route"]["kind"] == "plan" and not first["cached"]
    again, _ = here.ask("coffee", ScriptedChat(SEARCH % "coffee", DONE, "Asked again."))
    assert not again["cached"] and again["answer"] == "Asked again."
    unused = ScriptedChat("unused")
    for whole in ("coffee expenses", "my coffee spending", "show my coffee purchases"):
        listed, _ = here.ask(whole, unused)
        assert listed["route"]["kind"] == "aggregate" and unused.calls == [], whole
        assert listed["answer"].startswith("You spent **$") and " on Coffee since " in listed["answer"], whole


def test_beside_a_receipt_this_month_kept_last_month_is_asked_again(g, conversation, monkeypatch):
    """The same words on the first of the month are about another month."""
    from datetime import date

    from banko_ai.utils import clock
    here = conversation()
    _receipt_on_the_table(g, here.member, here.thread)
    monkeypatch.setattr(clock, "today", lambda: date(2026, 10, 31))
    here.ask("any unusual charges this month", ScriptedChat(SEARCH % "unusual", DONE, "October's oddities."))
    same_day, _ = here.ask("any unusual charges this month", ScriptedChat("unused"))
    assert same_day["cached"]
    monkeypatch.setattr(clock, "today", lambda: date(2026, 11, 1))
    asked = ScriptedChat(SEARCH % "unusual", DONE, "November's oddities.")
    next_day, _ = here.ask("any unusual charges this month", asked)
    assert not next_day["cached"] and next_day["answer"] == "November's oddities."


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
    question, short = "where do I get my coffee before work", Preferences({"search_page_size": 3})
    state = _results(g, member, question, ScriptedChat(SEARCH % "coffee", DONE, "Three of them."), prefs=short)
    assert len(_found(state)) == 3
    assert state["meta"]["search"]["size"] == 3 and state["meta"]["search"]["next"]     # Show more goes on by threes
    # The same question from someone with the usual page: asked, not served the short answer.
    usual = _results(g, member, question, ScriptedChat(SEARCH % "coffee", DONE, "Ten of them."))
    assert len(_found(usual)) > 3


def test_a_page_size_the_model_asks_for_itself_still_counts(g, member):
    chat = ScriptedChat('{"steps": [{"tool": "search_transactions", "args": {"query": "coffee", "limit": 2}}]}',
                        DONE, "Two of them.")
    state = _results(g, member, "two coffee purchases", chat, prefs=Preferences({"search_page_size": 5}))
    assert len(_found(state)) == 2


def test_show_more_pages_by_the_persons_size(g, member):
    short = Preferences({"search_page_size": 3})
    for question, chat in (("where do I get my coffee before work", ScriptedChat(SEARCH % "coffee", DONE, "Three.")),
                           ("coffee expenses", ScriptedChat("unused"))):          # a search, and a list from SQL
        thread = threads.create_thread(member)["thread_id"]
        try:
            first = G.run_turn(member, thread, question, chat=chat, graph=g, prefs=short)
            assert first["meta"]["search"]["size"] == 3 and first["meta"]["search"]["next"], question
            more = G.run_turn(member, thread, "show more", chat=ScriptedChat("unused"), graph=g, prefs=short)
            assert more["route"]["kind"] == "more" and len(more["meta"]["search"]["items"]) == 3, question
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
    first = _answered(g, member, "any unusual charges lately", "unusual charges", "Nothing unusual.")
    asked = "have there been any strange charges recently"
    plain, _ = _ask(g, member, asked, ScriptedChat(SEARCH % "strange charges", DONE, "Nothing strange."))
    assert not plain["cached"]                                        # the deployment matches the same question only
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM answer_cache WHERE user_id = CAST(:u AS UUID) AND query_text = :q"),
                  {"u": member, "q": asked})
    monkeypatch.setenv("CACHE_MATCH", "similar")
    reset_config()
    try:
        again, _ = _ask(g, member, asked, ScriptedChat("unused"))
        assert again["cached"] and again["answer"] == first["answer"]
    finally:
        monkeypatch.undo()
        reset_config()
