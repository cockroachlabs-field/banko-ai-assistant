"""Per-user caches: never across users, never across data versions, and the
vector index really serves the answer cache lookup."""

import os
import uuid

import pytest
from sqlalchemy import text

from banko_ai.db.engine import get_engine
from banko_ai.services import caches as C
from banko_ai.services import ledger
from banko_ai.services._vector import embed, nearest
from tests._money import drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture
def two():
    a, b = make_user(), make_user()
    yield a, b
    with get_engine().begin() as c:
        for t in ("answer_cache", "search_cache"):
            c.execute(text(f"DELETE FROM {t} WHERE user_id = ANY(CAST(:u AS UUID[]))"), {"u": [a, b]})
    drop_user(a)
    drop_user(b)


def test_search_cache_key_changes_with_version_and_user():
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    payload = {"q": "coffee", "limit": 10}
    assert C.search_key(a, 1, payload) != C.search_key(a, 2, payload)
    assert C.search_key(a, 1, payload) != C.search_key(b, 1, payload)
    assert C.search_key(a, 1, payload) == C.search_key(a, 1, dict(reversed(list(payload.items()))))


def test_search_cache_round_trip_is_per_user(two):
    a, b = two
    key = C.search_key(a, 0, {"q": "coffee"})
    C.put_search(a, key, [{"merchant": "Blue Bottle"}])
    assert C.get_search(a, key) == [{"merchant": "Blue Bottle"}]
    assert C.get_search(b, key) is None


def test_answer_cache_is_per_user_and_version(two):
    a, b = two
    q = "where do I spend the most on coffee"
    qv = embed(q)
    v = ledger.data_version(a)
    C.put_answer(a, version=v, language="English", query=q, qvec=qv, answer="Mostly Blue Bottle.", tokens=40)
    hit = C.get_answer(a, version=v, language="English", qvec=embed("where do I spend most on coffee"))
    assert hit and hit["answer"] == "Mostly Blue Bottle." and hit["tokens"] == 40
    assert C.get_answer(b, version=v, language="English", qvec=qv) is None
    assert C.get_answer(a, version=v + 1, language="English", qvec=qv) is None
    assert C.get_answer(a, version=v, language="French", qvec=qv) is None


def test_answer_cache_ignores_unrelated_questions(two):
    a, _ = two
    v = ledger.data_version(a)
    C.put_answer(a, version=v, language="English", query="coffee spend", qvec=embed("coffee spend"),
                 answer="x", tokens=1)
    assert C.get_answer(a, version=v, language="English", qvec=embed("what is my rent")) is None


def test_nearest_uses_the_per_user_vector_index(two):
    a, _ = two
    with get_engine().connect() as c:
        plan = "\n".join(r[0] for r in c.execute(text(
            "EXPLAIN SELECT cache_id FROM answer_cache@answer_cache_user_embedding "
            f"WHERE user_id = '{a}' ORDER BY query_embedding <=> '{embed('x')}' LIMIT 5")))
    assert "vector search" in plan


def test_nearest_returns_nothing_for_a_user_without_rows(two):
    a, _ = two
    with get_engine().connect() as c:
        assert nearest(c, table="answer_cache", index="answer_cache_user_embedding",
                       exact_index="answer_cache_user_idx", id_col="cache_id", emb_col="query_embedding",
                       user_id=a, qvec=embed("x"), k=5) == []


def test_record_stat_feeds_cache_stats():
    with get_engine().connect() as c:
        before = c.execute(text("SELECT count(*) FROM cache_stats WHERE cache_type = 'answer'")).scalar()
    C.record_stat("answer", "hit", 12)
    with get_engine().connect() as c:
        after = c.execute(text("SELECT count(*) FROM cache_stats WHERE cache_type = 'answer'")).scalar()
    assert after == before + 1


def test_estimate_tokens():
    assert C.estimate_tokens("") == 0
    assert C.estimate_tokens("a" * 400) == 100
