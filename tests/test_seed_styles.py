import os
import uuid
from datetime import date

import pytest
from sqlalchemy import text

from banko_ai.db.engine import get_engine
from banko_ai.services.seeding import seed_user
from tests._money import drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")

# A fixed day keeps runs comparable; the idempotency check also uses a
# fixed user id so its history is identical on every run.
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
def diners():
    ids = [make_user("diner") for _ in range(5)]
    yield ids
    for uid in ids:
        drop_user(uid)


@pytest.fixture
def subscriber():
    uid = _user("subscriber")
    yield uid
    drop_user(uid)


def test_diner_style_is_restaurant_heavy(diners):
    # What the demo relies on: for any diner, Restaurant is the category
    # they buy most often. Random ids, a six-month window.
    for uid in diners:
        assert seed_user(uid, months=6, today=TODAY) > 40
        with get_engine().connect() as c:
            top = c.execute(text("""
                SELECT shopping_type FROM expenses WHERE user_id = CAST(:u AS UUID)
                GROUP BY shopping_type ORDER BY count(*) DESC LIMIT 1"""), {"u": uid}).scalar()
            # embeddings present so vector search works immediately
            missing = c.execute(text(
                "SELECT count(*) FROM expenses WHERE user_id = CAST(:u AS UUID) "
                "AND embedding IS NULL"), {"u": uid}).scalar()
        assert top == "Restaurant", uid
        assert missing == 0


def test_seeding_is_idempotent(subscriber):
    first = seed_user(subscriber, months=1, today=TODAY)
    second = seed_user(subscriber, months=1, today=TODAY)
    assert second == 0 and first > 0
