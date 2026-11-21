"""One clock: every date Banko stores is a UTC date, so "today" is UTC's.

The ledger, seeding and the reports took today from UTC. The assistant, its
planner prompt and the coach took it from the machine's own clock. West of
UTC the two disagree every evening, from 7 pm in New York and from 4 pm in
Las Vegas: a purchase made then is dated tomorrow by the machine's calendar,
and "today", "this week" and "the last 30 days" all ended before it. A
receipt uploaded in an evening demo was missing from the very next answer.
In a container the machine's clock is UTC, so it never showed there."""

import os
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "banko_ai"
# The machine's own date or time of day: date.today(), datetime.today(),
# and datetime.now() with no zone given.
LOCAL_CLOCK = re.compile(r"\b(?:date|datetime)\.today\(\)|\bdatetime\.now\(\s*\)")


def test_nothing_in_banko_reads_the_machines_own_date():
    offenders = []
    for path in sorted(ROOT.rglob("*.py")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if LOCAL_CLOCK.search(line.split("#", 1)[0]):
                offenders.append(f"{path.relative_to(ROOT.parent)}:{lineno}: {line.strip()}")
    assert not offenders, "Take the day from banko_ai.utils.clock.today():\n" + "\n".join(offenders)


@pytest.fixture
def another_day(monkeypatch):
    """Put this process where the calendar already shows another day than
    UTC's, whatever the hour the suite runs at. Yields UTC's today."""
    hour = datetime.now(timezone.utc).hour
    # POSIX signs are reversed: Etc/GMT+12 is twelve hours behind UTC.
    monkeypatch.setenv("TZ", "Etc/GMT+12" if hour < 12 else "Etc/GMT-14")
    time.tzset()
    utc_today = datetime.now(timezone.utc).date()
    assert date.today() != utc_today, "the fixture did not move the machine's calendar"
    yield utc_today
    monkeypatch.undo()
    time.tzset()


def test_the_clock_is_utc_wherever_the_machine_is(another_day):
    from banko_ai.utils import clock
    assert clock.today() == another_day


def test_a_question_about_today_means_the_day_purchases_are_dated_by(another_day):
    from banko_ai.utils.intent_classifier import classify_aggregation, extract_window
    today = classify_aggregation("how much did I spend today?")
    assert (today.window_start, today.window_end) == (another_day, another_day + timedelta(days=1))
    recent = classify_aggregation("how much did I spend in the last 30 days")
    assert recent.window_end == another_day + timedelta(days=1)       # a purchase made a minute ago is inside
    assert extract_window("what did I buy yesterday") == (another_day - timedelta(days=1), another_day)


def test_the_planner_is_told_the_same_today(another_day):
    from banko_ai.assistant.prompts import planner_messages
    system = planner_messages("what did I buy today", history=[], memories=[], context_facts="",
                              tools="none")[0]["content"]
    assert f"Today is {another_day.isoformat()}" in system


def test_a_receipt_with_no_date_is_dated_today_in_utc(another_day):
    from banko_ai.agents.receipt_extraction_schema import ReceiptExtraction
    receipt = ReceiptExtraction.model_validate({"merchant": "Blue Bottle", "amount": 4.5, "category": "Coffee",
                                                "date": None})
    assert receipt.date == another_day


@pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")
def test_a_purchase_made_just_now_is_in_the_answer_about_today(another_day):
    """End to end: the purchase is posted, the question is asked, and the
    purchase is in the total."""
    from langgraph.checkpoint.memory import InMemorySaver

    from banko_ai.assistant import background
    from banko_ai.assistant import graph as G
    from banko_ai.services import ledger, threads
    from banko_ai.services.accounts import list_accounts
    from banko_ai.services.seeding import seed_user
    from tests._chat import ScriptedChat
    from tests._money import drop_user, make_user

    user = make_user("saver")
    seed_user(user, months=1, today=another_day - timedelta(days=3))     # accounts, and a history that ends days ago
    try:
        card = next(a for a in list_accounts(user)["accounts"] if a["kind"] == "credit")
        ledger.post_purchase(user, card["account_id"], amount="7.25", merchant="Evening Demo Espresso",
                             category="Coffee")
        thread = threads.create_thread(user)["thread_id"]
        try:
            for question in ("how much did I spend on coffee today?", "how much did I spend on coffee in the last 7 days?"):
                out = G.run_turn(user, thread, question, chat=ScriptedChat("- steady"),
                                 graph=G.build_graph(InMemorySaver()), on_event=lambda event: None)
                background.drain()
                assert "No matching transactions" not in out["answer"], out["answer"][:160]
                assert "7.25" in out["answer"] or "transaction" in out["answer"], out["answer"][:160]
        finally:
            threads.delete_thread(user, thread)
    finally:
        drop_user(user)
