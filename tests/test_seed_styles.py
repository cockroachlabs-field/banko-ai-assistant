import os
import uuid
from datetime import date

import pytest
from sqlalchemy import text

from banko_ai.db.engine import get_engine
from banko_ai.services.seeding import seed_user
from tests._money import drop_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")

# A fixed day and fixed user ids make the seeded history identical on every
# run, so the style assertions never depend on a lucky random draw.
TODAY = date(2026, 10, 11)


def _user(style: str) -> str:
    uid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"banko-test-seed-styles-{style}"))
    drop_user(uid)
    with get_engine().begin() as c:
        c.execute(text("INSERT INTO users (user_id, username, spending_style, demo_user) "
                       "VALUES (CAST(:u AS UUID), :name, :style, true)"),
                  {"u": uid, "name": f"t-{uid[:8]}", "style": style})
    return uid


@pytest.fixture
def diner():
    uid = _user("diner")
    yield uid
    drop_user(uid)


@pytest.fixture
def subscriber():
    uid = _user("subscriber")
    yield uid
    drop_user(uid)


def test_diner_style_is_restaurant_heavy(diner):
    n = seed_user(diner, months=1, today=TODAY)
    assert n > 40
    with get_engine().connect() as c:
        total, rest = c.execute(text("""
            SELECT count(*),
                   count(*) FILTER (WHERE shopping_type = 'Restaurant')
            FROM expenses WHERE user_id = CAST(:u AS UUID)"""), {"u": diner}).fetchone()
        assert rest / total > 0.3
        # embeddings present so vector search works immediately
        missing = c.execute(text(
            "SELECT count(*) FROM expenses WHERE user_id = CAST(:u AS UUID) "
            "AND embedding IS NULL"), {"u": diner}).scalar()
    assert missing == 0


def test_seeding_is_idempotent(subscriber):
    first = seed_user(subscriber, months=1, today=TODAY)
    second = seed_user(subscriber, months=1, today=TODAY)
    assert second == 0 and first > 0
