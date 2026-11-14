"""Test users get a seeded history, and seeding draws each day's purchases
at random per user. About one diner in a thousand buys no groceries in 30
days, and every test that totals groceries for its module's member then reads
"No matching transactions". seeded_diner() posts one grocery purchase, so the
totals those tests ask for are always there."""

import os
import uuid
from datetime import date, timedelta

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from banko_ai.assistant import background
from banko_ai.assistant import graph as G
from banko_ai.config.catalog import get_catalog
from banko_ai.services import threads
from banko_ai.services.seed_plan import build_plan, history_start
from tests._chat import ScriptedChat
from tests._money import drop_user, seeded_diner

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")
QUESTION = "how much did I spend on groceries in the last 30 days"


def _diner_without_groceries(today: date) -> str:
    """A user id whose seeded month has no grocery purchase. The plan is a
    pure function of the id and the day, so the search needs no database."""
    catalog = get_catalog()
    profile, start = catalog.profiles["diner"], history_start(today, 1)
    for _ in range(100_000):
        user_id = str(uuid.uuid4())
        plan = build_plan(user_id, profile, catalog, start=start, end=today)
        if not any(e.category == "Groceries" and e.posted_at.date() >= today - timedelta(days=30) for e in plan.entries):
            return user_id
    raise AssertionError("no diner without groceries in 100,000 draws; has the seed profile changed?")


def _ask(user_id: str) -> str:
    thread = threads.create_thread(user_id)["thread_id"]
    try:
        out = G.run_turn(user_id, thread, QUESTION, chat=ScriptedChat("- steady"), graph=G.build_graph(InMemorySaver()),
                         on_event=lambda event: None)
        background.drain()
        return out["answer"]
    finally:
        threads.delete_thread(user_id, thread)


def test_a_diner_whose_seed_has_no_groceries_still_has_groceries_to_total():
    today = date.today()
    unlucky = _diner_without_groceries(today)
    uid = seeded_diner(1, today=today, user_id=unlucky)
    try:
        assert uid == unlucky
        answer = _ask(uid)
        assert answer.startswith("You spent **$23.45** on Groceries"), answer[:120]
    finally:
        drop_user(uid)
