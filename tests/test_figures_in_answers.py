"""What a model writes about the user's money is read before it is shown.

SQL computes the figures and code writes the cards. A model still writes
the planner's answer, the answer about a receipt, the observations under a
total and the translation of a card. Each is read as it arrives
(assistant/figures.py): a sentence goes out when it is whole and every
figure in it is one the model was given, and a sentence with a figure from
nowhere is never shown."""

import json
import os
import re
import uuid

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import text

from banko_ai.assistant import background
from banko_ai.assistant import graph as G
from banko_ai.db.engine import get_engine
from banko_ai.services import memories, threads
from tests._chat import ScriptedChat
from tests._money import card_purchase, drop_user, seeded_diner

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")

TOTAL_OF_DINING = '{"steps": [{"tool": "spend_summary", "args": {"category": "Restaurant"}}]}'
DONE = '{"steps": []}'
NOWHERE = "$123,456.78"            # an amount in nobody's purchases


@pytest.fixture(scope="module")
def member():
    uid = seeded_diner(3)
    card_purchase(uid, "Blue Bottle Coffee", "6.25")
    yield uid
    with get_engine().begin() as c:
        for t in ("search_cache", "answer_cache", "memories", "assistant_actions", "receipts"):
            c.execute(text(f"DELETE FROM {t} WHERE user_id = CAST(:u AS UUID)"), {"u": uid})
    drop_user(uid)


@pytest.fixture
def ask(member):
    """Ask in a conversation of its own. Returns (answer, events, chat)."""
    g = G.build_graph(InMemorySaver())
    made = []
    with get_engine().begin() as c:
        for t in ("search_cache", "answer_cache"):
            c.execute(text(f"DELETE FROM {t} WHERE user_id = CAST(:u AS UUID)"), {"u": member})

    def asking(message, chat, *, thread=None, **kw):
        if thread is None:
            thread = threads.create_thread(member)["thread_id"]
            made.append(thread)
        events = []
        out = G.run_turn(member, thread, message, chat=chat, graph=g, on_event=events.append, **kw)
        background.drain()
        return out, events, chat
    asking.graph = g
    yield asking
    for thread in made:
        threads.delete_thread(member, thread)


def _facts(messages: list[dict]) -> list[dict]:
    """The facts the answer's model was sent, as it was sent them."""
    return json.loads(messages[-1]["content"].split("Facts:\n", 1)[1])


def _summary(messages: list[dict]) -> dict:
    return next(f["result"] for f in _facts(messages) if f["tool"] == "spend_summary")


def _shown(events: list[dict]) -> str:
    return "".join(e["text"] for e in events if e["type"] == "token")


def _step(events: list[dict], label: str) -> dict | None:
    return next((e for e in events if e["type"] == "step" and e["label"] == label and e["status"] != "running"), None)


def _answers_kept(member: str) -> int:
    with get_engine().connect() as c:
        return int(c.execute(text("SELECT count(*) FROM answer_cache WHERE user_id = CAST(:u AS UUID)"),
                             {"u": member}).scalar())


def test_an_answer_whose_figures_were_all_given_goes_out_whole(ask, member):
    def written(messages):
        found = _summary(messages)
        return (f"You spent ${found['total']:,.2f} on restaurants across {found['count']} purchases. "
                f"That is ${found['average']:,.2f} a visit.\n\n- Nothing else stands out.")
    out, events, _ = ask("where does my dining money tend to go?", ScriptedChat(TOTAL_OF_DINING, DONE, written))
    assert out["route"]["kind"] == "plan"
    assert "a visit." in out["answer"] and out["answer"].endswith("- Nothing else stands out.")
    assert _shown(events).strip() == out["answer"]
    checked = _step(events, "Figures check")
    assert checked["status"] == "success" and "every one from the data" in checked["detail"]
    assert out["meta"]["figures"] == {"read": 3, "left_out": []}
    assert _answers_kept(member) == 1                           # and it is kept for the next ask


def test_a_sentence_with_a_figure_from_nowhere_is_never_shown(ask, member):
    def written(messages):
        found = _summary(messages)
        return (f"You spent ${found['total']:,.2f} on restaurants. That is {NOWHERE} more than you planned. "
                "Nothing else stands out.")
    out, events, _ = ask("where does my dining money tend to go?", ScriptedChat(TOTAL_OF_DINING, DONE, written))
    assert out["answer"].startswith("You spent $") and "Nothing else stands out." in out["answer"]
    assert NOWHERE not in out["answer"] and NOWHERE not in _shown(events)       # not even as it streamed
    assert "123,456" not in _shown(events)
    assert out["answer"].endswith(G.FIGURE_LEFT_OUT)
    assert _shown(events).strip() == out["answer"]
    checked = _step(events, "Figures check")
    assert checked["status"] == "warning" and NOWHERE in checked["detail"]
    assert out["meta"]["figures"]["left_out"] == [
        {"text": f"That is {NOWHERE} more than you planned.", "figures": [NOWHERE]}]
    assert _answers_kept(member) == 0           # not kept: the next ask can get the whole answer


def test_figures_are_judged_to_the_cent_and_counts_by_what_was_counted(ask, member):
    """A rounded amount, two amounts added up wrongly and a count nobody
    counted are figures from nowhere too. Two amounts added up rightly are
    not: the sum is checked, and it stands."""
    def written(messages):
        found = _summary(messages)
        # Far above any two of the figures given, so no sum or difference of them comes to it by chance.
        rounded = f"${round(found['total'] * 9, -2):,.0f}"
        return (f"You spent about {rounded} eating out in a year. "
                f"With the average that makes ${found['total'] * 7 + found['average'] + 1.11:,.2f}. "
                f"That was over {found['count'] + 97} visits. "
                f"Total and average together are ${found['total'] + found['average']:,.2f}. "
                f"The exact total is ${found['total']:,.2f}.")
    out, events, _ = ask("where does my dining money tend to go?", ScriptedChat(TOTAL_OF_DINING, DONE, written))
    assert out["answer"].startswith("Total and average together are $") and "The exact total is $" in out["answer"]
    assert out["answer"].endswith(G.FIGURES_LEFT_OUT.format(n=3))
    assert len(out["meta"]["figures"]["left_out"]) == 3


def test_when_nothing_the_model_wrote_stands_the_answer_says_so(ask, member):
    out, events, _ = ask("where does my dining money tend to go?",
                         ScriptedChat(TOTAL_OF_DINING, DONE, f"You spent {NOWHERE} on restaurants."))
    assert out["answer"] == G.NOTHING_STOOD and _shown(events) == G.NOTHING_STOOD
    assert _answers_kept(member) == 0


def test_what_the_user_said_and_what_was_remembered_are_given_too(ask, member):
    """A figure in the question, earlier in the conversation or in a memory
    is one the model was given: repeating it invents nothing."""
    mid = memories.remember(member, "I'm saving $375.50 a month for a trip")["memory_id"]
    try:
        chat = ScriptedChat(DONE, DONE, "You asked about a $2,500 limit, and you put $375.50 aside each month.")
        out, events, _ = ask("am I close to my $2,500 limit given what I save for the trip each month?", chat)
        assert out["route"]["kind"] == "plan"
        assert out["answer"] == "You asked about a $2,500 limit, and you put $375.50 aside each month."
        assert _step(events, "Figures check")["status"] == "success"
    finally:
        memories.forget(member, mid)


def test_a_figure_from_earlier_in_the_conversation_is_given_too(ask, member):
    thread = threads.create_thread(member)["thread_id"]
    try:
        def first(messages):
            return f"You spent ${_summary(messages)['total']:,.2f} on restaurants."
        one, _, _ = ask("where does my dining money tend to go?", ScriptedChat(TOTAL_OF_DINING, DONE, first),
                        thread=thread)
        amount = re.search(r"\$[\d,]+\.\d{2}", one["answer"]).group(0)
        # No lookup this time: the figure is only in what was said before.
        two, events, _ = ask("and is that a lot, do you think?", ScriptedChat(DONE, DONE, f"{amount} is about usual for you."),
                             thread=thread)
        assert two["route"]["kind"] == "plan" and two["answer"] == f"{amount} is about usual for you."
        assert _step(events, "Figures check")["status"] == "success"
    finally:
        threads.delete_thread(member, thread)


def test_the_note_is_said_in_the_askers_language(ask, member):
    def model(messages):
        said = messages[-1]["content"]
        if "Translate the following" in said:
            return "Omití una frase porque una cifra en ella no está en tus datos."
        return f"Gastaste mucho en restaurantes. Eso son {NOWHERE} más de lo previsto."
    out, _, chat = ask("where does my dining money tend to go?", ScriptedChat(TOTAL_OF_DINING, DONE, model),
                       language_code="es-ES")
    assert out["answer"] == ("Gastaste mucho en restaurantes.\n\n"
                             "Omití una frase porque una cifra en ella no está en tus datos.")
    assert G.FIGURE_LEFT_OUT in chat.prompts()[-1]


def test_an_answer_about_a_receipt_is_read_the_same_way(ask, member):
    doc = str(uuid.uuid4())
    with get_engine().begin() as c:
        c.execute(text("INSERT INTO receipts (document_id, user_id, status, extracted) VALUES "
                       "(CAST(:d AS UUID), CAST(:u AS UUID), 'posted', CAST(:x AS JSONB))"),
                  {"d": doc, "u": member, "x": json.dumps({"merchant": "Harbor Deli", "amount": 23.47,
                                                           "items": ["Turkey club"], "date": "2026-10-10"})})
    thread = threads.create_thread(member)["thread_id"]
    try:
        G.set_active_receipt(thread, {"document_id": doc, "merchant": "Harbor Deli", "items": ["Turkey club"],
                                      "amount": 23.47}, graph=ask.graph)
        chat = ScriptedChat("It came to $23.47 at Harbor Deli on 2026-10-10. With tip that was $28.16.")
        out, events, _ = ask("how much was this receipt?", chat, thread=thread)
        assert out["route"]["kind"] == "receipt"
        assert out["answer"] == f"It came to $23.47 at Harbor Deli on 2026-10-10.\n\n{G.FIGURE_LEFT_OUT}"
        assert "$28.16" not in _shown(events)
    finally:
        threads.delete_thread(member, thread)


# ── the observations under a total ───────────────────────────────────────

TOTAL = "how much did I spend on restaurants in the last 60 days"


def _prompt_figures(messages: list[dict]) -> tuple[str, str]:
    """(the total, the count) as the insights prompt states them."""
    said = messages[-1]["content"]
    return re.search(r"total (\$[\d,]+\.\d{2})", said).group(1), re.search(r"across (\d+) transactions", said).group(1)


def test_an_observation_with_a_figure_from_nowhere_is_dropped(ask, member):
    def insights(messages):
        total, count = _prompt_figures(messages)
        return (f"- All {count} purchases came to {total}.\n- That is {NOWHERE} above your usual.\n"
                "- Dinner is where most of it goes.")
    out, events, chat = ask(TOTAL, ScriptedChat(insights))
    assert out["route"]["kind"] == "aggregate" and len(chat.calls) == 1
    card, _, notes = out["answer"].partition("\n\n- ")
    assert card.startswith("You spent **$") and NOWHERE not in out["answer"] and NOWHERE not in _shown(events)
    assert out["answer"].rstrip().endswith("- Dinner is where most of it goes.") and "purchases came to" in notes
    checked = _step(events, "Figures check")
    assert checked["status"] == "warning" and "1 of 3 observations left out" in checked["detail"]
    assert out["meta"]["figures"]["left_out"][0]["figures"] == [NOWHERE]
    # Not kept: asked again, the model is asked again.
    again, _, chat = ask(TOTAL, ScriptedChat("- Dinner is where most of it goes."))
    assert not again["cached"] and len(chat.calls) == 1


def test_observations_from_the_card_and_its_breakdown_stand_and_are_kept(ask, member):
    def insights(messages):
        total, count = _prompt_figures(messages)
        return f"- All {count} purchases came to {total}.\n- Dinner is where most of it goes."
    out, events, _ = ask(TOTAL, ScriptedChat(insights))
    assert out["answer"].rstrip().endswith("- Dinner is where most of it goes.")
    assert _step(events, "Figures check")["status"] == "success"
    again, _, chat = ask(TOTAL, ScriptedChat("unused"))
    assert again["cached"] and chat.calls == [] and again["answer"] == out["answer"]


def test_what_stands_on_the_card_is_given_to_the_observations():
    """The card says "to 2026-10-31": the last day of the window, which no
    purchase may have fallen on and which the totals themselves keep as the
    day after. An observation that quotes the card quotes what it was given."""
    from datetime import date

    from banko_ai.assistant import answers, figures
    from banko_ai.utils.aggregations import AggregationResult
    agg = AggregationResult(operation="sum", category="Restaurant", window_start=date(2026, 10, 1),
                            window_end=date(2026, 11, 1), total=947.33, count=17, average=55.73,
                            first_date=date(2026, 10, 2), last_date=date(2026, 10, 28),
                            top_merchants=[{"merchant": "Domino's", "total": 310.5, "count": 6, "category": "Restaurant"}])
    known = G._total_given(agg, answers.render_aggregation_markdown(agg), "how much did I spend eating out in october?")
    assert figures.unfounded("- From 2026-10-01 to 2026-10-31 it came to $947.33 over 17 purchases.", known) == []
    assert figures.unfounded("- Domino's took $310.50 of it in 6 purchases, the last on 2026-10-28.", known) == []
    assert figures.unfounded("- Through October 30 it came to $947.33.", known) == ["October 30"]


# ── translations ─────────────────────────────────────────────────────────

def test_a_cards_translation_that_changes_a_figure_is_not_used(ask, member):
    """The card is SQL's. In English with the right figures it beats Spanish
    with a wrong one, and what was shown is not kept as the translation."""
    def wrong(messages):
        english = messages[-1]["content"].split("\n\n", 1)[1]
        amount = re.search(r"\$[\d,]+\.\d{2}", english).group(0)
        return english.replace("You spent", "Gastaste").replace(amount, "$99,999.99", 1)
    out, events, _ = ask("my restaurant expenses", ScriptedChat(wrong), language_code="es-ES")
    assert out["route"]["kind"] == "aggregate" and out["answer"].startswith("You spent **$")
    assert "99,999.99" not in out["answer"]
    assert _step(events, "Figures check")["status"] == "warning"

    def right(messages):
        return messages[-1]["content"].split("\n\n", 1)[1].replace("You spent", "Gastaste")
    again, _, chat = ask("my restaurant expenses", ScriptedChat(right), language_code="es-ES")
    assert not again["cached"] and again["answer"].startswith("Gastaste **$") and len(chat.calls) == 1


def test_a_translated_total_holds_back_a_sentence_with_a_figure_the_english_does_not_have(ask, member):
    def model(messages):
        said = messages[-1]["content"]
        if "Translate the following" not in said:
            return "- Dinner is where most of it goes."
        english = said.split("\n\n", 1)[1]
        first, _, rest = english.partition("\n")
        return (first.replace("You spent", "Gastaste") + f" Eso es {NOWHERE} de más.\n" + rest).replace(
            "Dinner is where most of it goes.", "La cena es donde más se va.")
    out, events, _ = ask(TOTAL, ScriptedChat(model), language_code="es-ES")
    assert out["answer"].startswith("Gastaste **$") and NOWHERE not in out["answer"]
    assert NOWHERE not in _shown(events) and "La cena es donde más se va." in out["answer"]
    assert out["answer"].rstrip().endswith(G.FIGURE_LEFT_OUT)
    again, _, chat = ask(TOTAL, ScriptedChat(model), language_code="es-ES")
    assert not again["cached"] and chat.calls                    # not kept
