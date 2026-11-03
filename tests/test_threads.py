"""Chat threads: per user, titled from the first message, RLS-friendly."""

import os
import uuid

import pytest

from banko_ai.services import threads as T
from tests._money import drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture
def user():
    uid = make_user()
    yield uid
    for t in T.list_threads(uid):
        T.delete_thread(uid, t["thread_id"])
    drop_user(uid)


def test_create_list_rename_delete(user):
    t = T.create_thread(user)
    assert t["title"] == "New chat" and t["message_count"] == 0
    T.touch_thread(user, t["thread_id"], added=2, title="Coffee spend")
    listed = T.list_threads(user)
    assert listed[0]["thread_id"] == t["thread_id"]
    assert listed[0]["title"] == "Coffee spend" and listed[0]["message_count"] == 2
    assert T.rename_thread(user, t["thread_id"], "Renamed")["title"] == "Renamed"
    assert T.delete_thread(user, t["thread_id"]) is True
    assert T.get_thread(user, t["thread_id"]) is None


def test_touch_keeps_a_chosen_title(user):
    t = T.create_thread(user, title="Mine")
    T.touch_thread(user, t["thread_id"], title="Something else")
    assert T.get_thread(user, t["thread_id"])["title"] == "Mine"


def test_other_users_thread_is_invisible(user):
    other = make_user()
    try:
        t = T.create_thread(other)
        assert T.get_thread(user, t["thread_id"]) is None
        assert T.rename_thread(user, t["thread_id"], "x") is None
        assert T.delete_thread(user, t["thread_id"]) is False
        T.delete_thread(other, t["thread_id"])
    finally:
        drop_user(other)


def test_bad_ids_are_none(user):
    assert T.get_thread(user, "not-a-uuid") is None
    assert T.get_thread(user, str(uuid.uuid4())) is None


def test_title_from():
    assert T.title_from("how much did I spend on coffee last month?") == "How much did I spend on coffee last month?"
    long = "please compare my grocery spending this month against the last three months and tell me why"
    title = T.title_from(long)
    assert len(title) <= 49 and title.endswith("…") and " " not in title[-2:]
    assert T.title_from("   ") == "New chat"
