"""A category or one of the user's merchants, asked for alone or as its
purchases, is read from SQL and written by code.

Left to the planner, what "coffee" did was the model's choice. On gpt-4.1 it
was a total of the last 31 days; on gpt-4o-mini and granite a search of the
nearest purchases; and seven characters of the planner's tool list had moved
gpt-4.1 from one to the other. The answer is the database's, so it has to be
the same answer on every model and every day: SQL computes it, code writes
the card, and no model is called in English.

Asked again, the card comes from the cache: it is kept under what was asked
for (the category or merchant, the days, the language, the page), with the
state of the user's money, so a repeat or the same thing in other words
reads no purchase at all. And it is the same question wherever in a
conversation it is asked."""

import os
from datetime import date, timedelta

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import event, text
from sqlalchemy.engine import Engine

from banko_ai.assistant import background
from banko_ai.assistant import graph as G
from banko_ai.db.engine import get_engine
from banko_ai.services import threads
from banko_ai.services import transactions as txns
from banko_ai.services.preferences import Preferences
from tests._chat import ScriptedChat
from tests._money import card_purchase, drop_user, seeded_diner

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")

SILENT = "no model is asked"


@pytest.fixture(scope="module")
def member():
    uid = seeded_diner(3)
    card_purchase(uid, "Blue Bottle Coffee", "6.25")
    card_purchase(uid, "Blue Bottle Coffee", "4.75")
    yield uid
    with get_engine().begin() as c:
        for t in ("search_cache", "answer_cache", "memories", "assistant_actions", "receipts"):
            c.execute(text(f"DELETE FROM {t} WHERE user_id = CAST(:u AS UUID)"), {"u": uid})
    drop_user(uid)


@pytest.fixture(autouse=True)
def nothing_kept(member):
    """Each test starts with no card kept for the member, so what one test asked never answers another."""
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM search_cache WHERE user_id = CAST(:u AS UUID)"), {"u": member})


@pytest.fixture
def g():
    return G.build_graph(InMemorySaver())


@pytest.fixture
def statements():
    """Every statement the app sends while the test runs, as text."""
    sent: list[str] = []

    def note(conn, cursor, statement, parameters, context, executemany):
        sent.append(" ".join(statement.split()))
    event.listen(Engine, "before_cursor_execute", note)
    yield sent
    event.remove(Engine, "before_cursor_execute", note)


def _on_purchases(sent: list[str]) -> list[str]:
    return [s for s in sent if " expenses" in s and not s.startswith(("INSERT", "UPSERT"))]


@pytest.fixture
def ask(g, member):
    """Ask in a conversation of its own, or in the one given. Returns (answer, events, chat)."""
    made = []

    def asking(message, *, thread=None, chat=None, **kw):
        if thread is None:
            thread = threads.create_thread(member)["thread_id"]
            made.append(thread)
        chat = chat or ScriptedChat(SILENT)
        events = []
        out = G.run_turn(member, thread, message, chat=chat, graph=g, on_event=events.append, **kw)
        background.drain()
        return out, events, chat
    yield asking
    for thread in made:
        threads.delete_thread(member, thread)


def _sum(member, where="shopping_type = 'Restaurant'", params=None) -> tuple[float, int, date]:
    with get_engine().connect() as c:
        total, count, first = c.execute(text(
            f"SELECT sum(expense_amount), count(*), min(expense_date) FROM expenses "
            f"WHERE user_id = CAST(:u AS UUID) AND {where}"), {"u": member, **(params or {})}).one()
    return float(total or 0), int(count), first


def _steps(events) -> list[str]:
    return [e["label"] for e in events if e["type"] == "step" and e["status"] != "running"]


RESTAURANTS = ("restaurants", "my restaurant expenses", "show me my restaurant purchases", "dining out",
               "Can you list my dining out transactions?", "spending on restaurants", "takeout")


def test_a_category_in_any_of_its_wordings_is_one_card_from_sql_and_no_model(ask, member):
    total, count, first = _sum(member)
    cards = set()
    for question in RESTAURANTS:
        out, events, chat = ask(question)
        assert out["route"]["kind"] == "aggregate" and chat.calls == [], question
        assert out["answer"].startswith(f"You spent **${total:,.2f}** on Restaurant since {first} "
                                        f"({count} transactions)."), question
        steps = _steps(events)
        assert not {"Planner", "LLM answer", "Narrative cache", "Memory recall"} & set(steps), question
        if question == RESTAURANTS[0]:
            # Nobody asked before: SQL reads it, and the plan under it is of that statement.
            assert not out["cached"]
            assert [s for s in steps if s != "Region pruning"] == ["Intent routing", "Answer cache", "Aggregation SQL",
                                                                   "EXPLAIN ANALYZE"], steps
            assert out["meta"]["explain"] and "expenses@idx_expenses_user_date" in out["meta"]["explain"]
        else:
            # The same thing in other words: the card that was kept, and it says whose answer it is.
            assert out["cached"] and steps == ["Intent routing", "Answer cache"], (question, steps)
            assert out["meta"]["cache"] == {"question": RESTAURANTS[0], "match": "words"}, question
            assert "explain" not in out["meta"], question         # no statement ran, so there is no plan to show
        cards.add(out["answer"])
    assert len(cards) == 1                      # the same words back, whatever the wording of the question


def test_asked_again_the_card_comes_from_the_cache_and_reads_no_purchase(ask, member, statements):
    """ "coffee" asked twice ran every statement twice and never said
    cached. The card is kept under the state of the user's money: asked
    again, nothing is read from the purchases."""
    first, _, _ = ask("restaurants")
    assert not first["cached"] and _on_purchases(statements)
    del statements[:]
    again, events, chat = ask("restaurants")
    assert again["cached"] and chat.calls == [] and again["answer"] == first["answer"]
    assert _on_purchases(statements) == [], _on_purchases(statements)
    assert again["saved_tokens"] == 0                           # no model wrote it, so none was saved
    assert "cache" not in again["meta"]                         # the same words: nothing to add under the answer
    hit = next(e for e in events if e["type"] == "step" and e["label"] == "Answer cache" and e["status"] != "running")
    assert hit["status"] == "hit" and "Restaurant" in hit["detail"]
    # Show more under a cached card pages on from the database as it did under the first.
    assert again["meta"]["search"] == first["meta"]["search"] and again["meta"]["search"]["next"]


def test_asked_again_in_the_same_conversation_it_is_the_same_card(ask, member, statements):
    """The report: "coffee" twice in one chat. The second went to the
    planner as a follow-up, three model calls, and no cache keeps those."""
    thread = threads.create_thread(member)["thread_id"]
    try:
        first, _, _ = ask("restaurants", thread=thread)
        del statements[:]
        again, events, chat = ask("restaurants", thread=thread)
        assert again["route"]["kind"] == "aggregate" and chat.calls == []
        assert again["cached"] and again["answer"] == first["answer"] and _on_purchases(statements) == []
        assert "cache" not in again["meta"]
        # And a typed "show more" goes on from the card the cache gave.
        more, _, chat = ask("show more", thread=thread)
        assert more["route"]["kind"] == "more" and chat.calls == []
        assert more["answer"] == "Here are 10 more Restaurant purchases."
    finally:
        threads.delete_thread(member, thread)


def test_a_new_purchase_retires_the_card_that_was_kept(ask, member):
    before, _, _ = ask("my coffee expenses")
    assert ask("my coffee expenses")[0]["cached"]
    card_purchase(member, "Blue Bottle Coffee", "3.10")
    total, count, first = _sum(member, "shopping_type = 'Coffee'")
    after, _, chat = ask("my coffee expenses")
    assert not after["cached"] and chat.calls == [] and after["answer"] != before["answer"]
    assert after["answer"].startswith(f"You spent **${total:,.2f}** on Coffee since {first} ({count} transactions).")


def test_a_card_is_kept_for_what_was_asked_for_and_nothing_else(ask, member):
    """Other days, another category, another page length or another
    language is another card: none of them is answered from this one."""
    ask("restaurants")
    for other, kw in (("restaurants this month", {}), ("groceries", {}),
                      ("restaurants", {"prefs": Preferences({"search_page_size": 5})})):
        out, _, _ = ask(other, **kw)
        assert not out["cached"], other
    short, _, _ = ask("restaurants", prefs=Preferences({"search_page_size": 5}))
    assert short["cached"] and "The latest 5 of " in short["answer"]
    assert ask("restaurants")[0]["cached"] and "The latest 10 of " in ask("restaurants")[0]["answer"]


def test_a_merchants_card_is_kept_the_same_way(ask, member, statements):
    card_purchase(member, "Corner Kiosk", "4.20")
    first, _, _ = ask("blue bottle coffee")
    other, _, _ = ask("corner kiosk")                               # another merchant is another card
    assert not other["cached"] and other["answer"].startswith("You went to **Corner Kiosk** 1 time ")
    del statements[:]
    cards = set()
    for question in ("blue bottle coffee", "Blue Bottle Coffee purchases", "purchases at Blue Bottle Coffee"):
        again, events, chat = ask(question)
        assert again["route"]["kind"] == "visits" and again["cached"] and chat.calls == [], question
        assert _steps(events) == ["Intent routing", "Answer cache"], question
        cards.add(again["answer"])
    assert cards == {first["answer"]} and _on_purchases(statements) == []
    assert again["meta"]["cache"] == {"question": "blue bottle coffee", "match": "words"}
    assert not ask("blue bottle coffee purchases this month")[0]["cached"]        # its own days: another card
    card_purchase(member, "Blue Bottle Coffee", "5.55")
    assert not ask("blue bottle coffee")[0]["cached"]


def test_in_another_language_the_translation_is_kept_and_the_model_is_not_asked_again(ask, member):
    spanish = lambda: ScriptedChat(lambda messages: "Gastaste ... en Restaurant")      # noqa: E731
    first, _, chat = ask("my restaurant expenses", chat=spanish(), language_code="es-ES")
    assert len(chat.calls) == 1 and not first["cached"]
    again, _, chat = ask("restaurants", chat=spanish(), language_code="es-ES")
    assert again["cached"] and chat.calls == [] and again["answer"] == first["answer"]
    assert again["saved_tokens"] > 0                            # the translation the model did not write again
    english, _, chat = ask("restaurants")
    assert not english["cached"] and english["answer"].startswith("You spent")


def test_a_card_the_model_could_not_translate_is_not_kept_as_the_translation(ask, member):
    """With the provider out of reach the card goes out in English. Kept as
    the Spanish answer, it would stay English after the provider is back."""
    def down(messages):
        raise ConnectionError("the provider is out of reach")
    out, _, _ = ask("my restaurant expenses", chat=ScriptedChat(down), language_code="es-ES")
    assert out["answer"].startswith("You spent")
    again, _, chat = ask("my restaurant expenses", chat=ScriptedChat(lambda m: "Gastaste ..."), language_code="es-ES")
    assert not again["cached"] and again["answer"].startswith("Gastaste") and len(chat.calls) == 1


def test_the_largest_purchases_are_a_card_code_writes_and_kept_the_same_way(ask, member, statements):
    """ "my biggest purchase this year" asked twice ran its statements twice
    as well, and in another language asked the model twice."""
    first, events, chat = ask("What was my biggest purchase this year?")
    assert first["route"]["kind"] == "aggregate" and not first["cached"] and chat.calls == []
    assert first["meta"]["explain"] and "expenses@idx_expenses_user_date" in first["meta"]["explain"]
    del statements[:]
    again, events, chat = ask("what was my biggest purchase this year")
    assert again["cached"] and again["answer"] == first["answer"] and chat.calls == []
    assert _on_purchases(statements) == [] and _steps(events) == ["Intent routing", "Answer cache"]
    # How many were asked for, and of what, is another card; and a category's purchases are not its largest.
    three, _, _ = ask("my 3 biggest purchases this year")
    assert not three["cached"] and three["answer"] != first["answer"]
    smallest, _, _ = ask("what was my smallest purchase this year?")
    assert not smallest["cached"] and smallest["answer"] != first["answer"]
    listed, _, _ = ask("restaurant expenses this year")
    largest, _, _ = ask("what was my biggest restaurant purchase this year")
    assert not listed["cached"] and not largest["cached"] and largest["answer"] != listed["answer"]
    card_purchase(member, "Blue Bottle Coffee", "2.95")
    assert not ask("What was my biggest purchase this year?")[0]["cached"]


def test_a_total_at_one_of_the_users_merchants_is_the_same_total_inside_a_conversation(ask, member):
    """ "total zebra crossing" says nothing about money to the check that
    decides. Where it opened a conversation the user's merchants were asked
    before it was turned away; inside one it was taken for a short
    follow-up, and never were."""
    card_purchase(member, "Zebra Crossing", "18.40", category="Restaurant")
    thread = threads.create_thread(member)["thread_id"]
    try:
        ask("how much did I spend on groceries in the last 30 days", thread=thread, chat=ScriptedChat("- Steady."))
        out, _, _ = ask("total zebra crossing", thread=thread, chat=ScriptedChat("- Steady."))
        first = out["answer"].splitlines()[0]
        assert out["route"]["kind"] == "aggregate" and "$18.40" in first and "Zebra Crossing" in first, first
        # Words that are nobody's merchant are still a short follow-up for the planner.
        chat = ScriptedChat('{"steps": []}', '{"steps": []}', "Not sure what you mean.")
        out, _, _ = ask("total eclipse", thread=thread, chat=chat)
        assert out["route"]["kind"] == "plan"
    finally:
        threads.delete_thread(member, thread)


def test_a_cache_that_cannot_be_read_or_written_costs_nothing_but_the_cache(ask, member, monkeypatch):
    def down(*args, **kwargs):
        raise ConnectionError("the gateway node went away")
    monkeypatch.setattr(G.caches, "get_search", down)
    monkeypatch.setattr(G.caches, "put_search", down)
    total, count, first = _sum(member)
    out, events, chat = ask("restaurants")
    assert not out["cached"] and chat.calls == []
    assert out["answer"].startswith(f"You spent **${total:,.2f}** on Restaurant since {first} ({count} transactions).")


def test_an_answer_from_sql_recalls_no_memory_and_needs_no_embedding_model(ask, member, monkeypatch):
    """Every turn began by embedding the question to recall memories, the
    ones SQL answers too, which use none. On a process that had just started
    that waited for the embedding model: up to WARMUP_WAIT_SECONDS before a
    card that needs no model at all."""
    from banko_ai.services import _vector
    from banko_ai.utils.embeddings import ModelWarmingUp

    def loading(text_value):
        raise ModelWarmingUp("embedding model is still loading")
    monkeypatch.setattr(_vector, "embed", loading)
    monkeypatch.setattr(G.memories, "embed", loading, raising=False)
    for question, kind in (("restaurants", "aggregate"), ("blue bottle coffee", "visits"),
                           ("restaurants", "aggregate"), ("hello there", "offtopic")):
        out, events, chat = ask(question)
        assert out["route"]["kind"] == kind and chat.calls == [], question
        assert "Memory recall" not in _steps(events), question
    out, events, _ = ask("how much did I spend on groceries in the last 30 days", chat=ScriptedChat("- Steady."))
    assert out["route"]["kind"] == "aggregate" and "Memory recall" not in _steps(events)


def test_a_question_for_the_planner_still_recalls_what_the_user_told_it(ask, member, monkeypatch):
    recalled = []
    real = G.memories.recall
    monkeypatch.setattr(G.memories, "recall", lambda *a, **k: recalled.append(a[1]) or real(*a, **k))
    chat = ScriptedChat('{"steps": []}', '{"steps": []}', "Nowhere.")
    out, events, chat = ask("where could I cut back on dining out lately?", chat=chat)
    assert out["route"]["kind"] == "plan" and recalled == ["where could I cut back on dining out lately?"]
    steps = _steps(events)
    assert steps.index("Memory recall") > steps.index("Intent routing")
    assert steps.index("Memory recall") < steps.index("Planner")
    ask("restaurants")
    assert len(recalled) == 1


def test_the_card_breaks_the_category_down_and_lists_its_latest_purchases(ask, member):
    out, _, chat = ask("my restaurant expenses")
    assert chat.calls == []
    total, count, _ = _sum(member)
    card = out["answer"]
    with get_engine().connect() as c:
        merchants = c.execute(text(
            "SELECT merchant, sum(expense_amount), count(*) FROM expenses WHERE user_id = CAST(:u AS UUID) "
            "AND shopping_type = 'Restaurant' GROUP BY 1 ORDER BY 2 DESC LIMIT 1"), {"u": member}).one()
    assert f"| {merchants[0]} | ${float(merchants[1]):,.2f} | {merchants[2]} |" in card      # summed over every purchase
    assert f"The latest 10 of {count}:" in card
    listed = txns.list_transactions(member, limit=10, category="Restaurant")["items"]
    rows = [line for line in card.splitlines() if line.startswith("| 20")]
    assert rows == [f"| {str(i['posted_at'])[:10]} | {i['merchant']} | ${abs(float(i['amount'])):,.2f} |" for i in listed]


def test_the_totals_statement_leaves_the_list_to_the_statement_that_pages_it(member):
    """A total reads its latest ten purchases itself. Asked for a category's
    purchases it reads the figure and the breakdown only: the card lists
    them the way Show more goes on, so they are read once, there."""
    from banko_ai.config.settings import get_config
    from banko_ai.utils.aggregations import run_aggregation
    from banko_ai.utils.intent_classifier import EVER, AggregationIntent
    tomorrow, url = date.today() + timedelta(days=1), get_config().database_url
    listed = run_aggregation(AggregationIntent("purchases", "Restaurant", EVER, tomorrow), member, url)
    total, count, first = _sum(member)
    assert (listed.total, listed.count, listed.first_date) == (round(total, 2), count, first)
    assert listed.rows == [] and len(listed.top_merchants) > 1
    assert len(run_aggregation(AggregationIntent("sum", "Restaurant", EVER, tomorrow), member, url).rows) == 10


def test_show_more_under_the_card_goes_on_from_sql(ask, member):
    """ "show me my coffee purchases" was a search, with Show more under it.
    The list from SQL pages on the same way: the button asks the
    transactions API for the next purchases of the category, and a typed
    "show more" brings them without a model."""
    _, count, _ = _sum(member)
    out, _, _ = ask("show me my restaurant purchases")
    page = out["meta"]["search"]
    assert (page["q"], page["category"], page["from"], page["to"], page["size"]) == ("", "Restaurant", None, None, 10)
    first = txns.list_transactions(member, limit=10, category="Restaurant")
    assert page["next"] == first["next_cursor"] and page["next"]
    # What the Show more button asks for.
    second = txns.list_transactions(member, limit=10, category="Restaurant", cursor=page["next"])["items"]
    assert second and not {i["expense_id"] for i in second} & {i["expense_id"] for i in first["items"]}

    thread = threads.create_thread(member)["thread_id"]
    try:
        ask("show me my restaurant purchases", thread=thread)
        more, events, chat = ask("show more", thread=thread)
        assert more["route"]["kind"] == "more" and chat.calls == []
        assert more["answer"] == "Here are 10 more Restaurant purchases."
        items = more["meta"]["search"]["items"]
        assert [i["expense_id"] for i in items] == [i["expense_id"] for i in second]
        assert all(set(i) == {"expense_id", "merchant", "amount", "date"} and i["amount"] > 0 for i in items)
        seen = 20
        while more["meta"]["search"].get("next"):
            more, _, chat = ask("show more", thread=thread)
            assert chat.calls == []
            seen += len(more["meta"]["search"]["items"])
        assert seen == count                                        # every purchase the figure counted, once
        done, _, chat = ask("show more", thread=thread)
        assert done["answer"] == "That's every Restaurant purchase." and chat.calls == []
    finally:
        threads.delete_thread(member, thread)


def test_a_category_for_the_days_it_names(ask, member):
    start = date.today().replace(day=1)
    end = (start + timedelta(days=32)).replace(day=1)
    total, count, _ = _sum(member, "shopping_type = 'Restaurant' AND expense_date >= :s AND expense_date < :e",
                           {"s": start, "e": end})
    out, _, chat = ask("my restaurant expenses this month")
    assert chat.calls == []
    assert out["answer"].startswith(f"You spent **${total:,.2f}** on Restaurant from {start} to "
                                    f"{end - timedelta(days=1)} ({count} transactions).")
    listed = [line.split(" | ")[0].lstrip("| ") for line in out["answer"].splitlines() if line.startswith("| 20")]
    assert len(listed) == min(10, count)
    assert all(str(start) <= day < str(end) for day in listed), listed      # the purchases the figure counted
    if count > 10:
        page = out["meta"]["search"]
        assert (page["from"], page["to"]) == (str(start), str(end - timedelta(days=1)))


def test_show_more_stays_within_the_days_the_list_was_asked_for(ask, member):
    """The next pages are the same question's: the same category and the same days."""
    start = date.today() - timedelta(days=60)
    _, count, _ = _sum(member, "shopping_type = 'Restaurant' AND expense_date >= :s", {"s": start})
    assert count > 10
    thread = threads.create_thread(member)["thread_id"]
    try:
        out, _, _ = ask("restaurant expenses in the last 60 days", thread=thread)
        assert f"({count} transactions)" in out["answer"].splitlines()[0]
        seen = out["answer"].count("\n| 20")
        while out["meta"].get("search", {}).get("next"):
            out, _, chat = ask("show more", thread=thread)
            items = out["meta"]["search"]["items"]
            assert chat.calls == [] and all(str(i["date"])[:10] >= str(start) for i in items), items
            seen += len(items)
        assert seen == count
    finally:
        threads.delete_thread(member, thread)


def test_a_short_category_has_no_next_page_to_offer(ask, member):
    total, count, first = _sum(member, "shopping_type = 'Entertainment'")
    if not 1 <= count <= 10:
        pytest.skip("this seed's entertainment does not fit one page")
    out, _, chat = ask("movies")
    assert chat.calls == []
    assert out["answer"].startswith(f"You spent **${total:,.2f}** on Entertainment since {first} "
                                    f"({count} transactions).")
    assert "search" not in out["meta"] and "The latest" not in out["answer"]
    assert out["answer"].count("\n| 20") == count


def test_a_category_with_nothing_bought_says_so_and_asks_nobody(ask, member):
    """Not the planner's "searching by meaning instead": the category was
    named outright, and the database says there is nothing under it."""
    with get_engine().connect() as c:
        bought = {str(r[0]) for r in c.execute(text(
            "SELECT DISTINCT shopping_type FROM expenses WHERE user_id = CAST(:u AS UUID)"), {"u": member})}
    name = next(n for n in ("Travel", "Electronics", "Home Improvement", "Entertainment") if n not in bought)
    out, events, chat = ask(f"my {name.lower()} expenses")
    assert out["route"]["kind"] == "aggregate" and chat.calls == []
    assert out["answer"] == f"You have no purchases on {name}."
    out, _, chat = ask(f"{name.lower()} expenses last month")
    assert chat.calls == [] and out["answer"].startswith(f"No matching transactions on {name} from ")


def test_one_of_the_users_merchants_in_any_wording_is_one_card_from_sql(ask, member):
    total, count, first = _sum(member, "merchant = 'Blue Bottle Coffee'")
    cards = set()
    for question in ("blue bottle coffee", "Blue Bottle Coffee purchases", "purchases at Blue Bottle Coffee",
                     "show my blue bottle coffee charges"):
        out, events, chat = ask(question)
        assert out["route"]["kind"] == "visits" and chat.calls == [], question
        assert out["answer"].startswith(f"You went to **Blue Bottle Coffee** {count} times since {first} and spent "
                                        f"**${total:,.2f}** there"), question
        if not cards:
            assert out["meta"]["explain"] and "expenses@idx_expenses_user_date" in out["meta"]["explain"]
        assert not {"Planner", "LLM answer", "Memory recall"} & set(_steps(events)), question
        cards.add(out["answer"])
    assert len(cards) == 1


def test_a_whole_question_inside_a_conversation_still_comes_from_sql(ask, member):
    thread = threads.create_thread(member)["thread_id"]
    try:
        ask("how much did I spend on groceries in the last 30 days", thread=thread, chat=ScriptedChat("- Steady."))
        out, _, chat = ask("my restaurant expenses", thread=thread)
        assert out["route"]["kind"] == "aggregate" and chat.calls == []
        out, _, chat = ask("Blue Bottle Coffee purchases", thread=thread)
        assert out["route"]["kind"] == "visits" and chat.calls == []
    finally:
        threads.delete_thread(member, thread)


def test_a_bare_name_inside_a_conversation_is_the_same_card(ask, member):
    """ "restaurants" after a question about groceries was the planner's: a
    model picked what it meant, and its answer was kept nowhere. A name
    asked alone is one question, wherever it is asked. With a word that
    points back ("what about restaurants?") it still takes the conversation."""
    total, count, first = _sum(member)
    thread = threads.create_thread(member)["thread_id"]
    try:
        ask("how much did I spend on groceries in the last 30 days", thread=thread, chat=ScriptedChat("- Steady."))
        out, _, chat = ask("restaurants", thread=thread)
        assert out["route"]["kind"] == "aggregate" and chat.calls == []
        assert out["answer"].startswith(f"You spent **${total:,.2f}** on Restaurant since {first} "
                                        f"({count} transactions).")
        out, _, chat = ask("blue bottle coffee", thread=thread)
        assert out["route"]["kind"] == "visits" and chat.calls == []
        chat = ScriptedChat('{"steps": [{"tool": "spend_summary", "args": {"category": "Restaurant"}}]}',
                            '{"steps": []}', "About the same as groceries.")
        out, _, chat = ask("what about restaurants?", thread=thread, chat=chat)
        assert out["route"]["kind"] == "plan" and out["answer"] == "About the same as groceries."
    finally:
        threads.delete_thread(member, thread)


def test_in_another_language_the_card_is_translated_and_nothing_else_is_asked(ask, member):
    total, _, _ = _sum(member)
    chat = ScriptedChat(lambda messages: "Gastaste ... en Restaurant")
    out, _, chat = ask("my restaurant expenses", chat=chat, language_code="es-ES")
    assert out["answer"].startswith("Gastaste") and len(chat.calls) == 1
    assert f"You spent **${total:,.2f}** on Restaurant since " in chat.prompts()[-1]      # the card it was given


def test_a_merchant_lookup_that_fails_leaves_the_reply_it_had_before(ask, member, monkeypatch):
    def down(*args, **kwargs):
        raise ConnectionError("the gateway node went away")
    monkeypatch.setattr(G.caches, "merchants_of", lambda *a, **k: down)
    out, _, _ = ask("netflix")
    assert out["route"]["kind"] == "offtopic"
    monkeypatch.undo()
    out, _, chat = ask("my restaurant expenses")                     # a category needs no lookup
    assert out["route"]["kind"] == "aggregate"


def test_the_merchants_are_read_once_for_a_state_of_the_users_money(ask, member, monkeypatch):
    read, real = [], G.caches._merchant_names
    monkeypatch.setattr(G.caches, "_merchant_names", lambda *a, **k: read.append(1) or real(*a, **k))
    G.caches.forget_merchants()
    for question in ("blue bottle coffee", "Blue Bottle Coffee purchases", "netflix", "hello"):
        ask(question)
    assert len(read) == 1
    ask("how much did I spend on groceries in the last 30 days", chat=ScriptedChat("- Steady."))   # a total: not asked
    ask("where could I cut back on dining out lately?",
        chat=ScriptedChat('{"steps": []}', '{"steps": []}', "Nowhere."))                              # too long for a name
    assert len(read) == 1
