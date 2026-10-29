"""On a multi-region cluster every statement the request path runs for a
signed-in user starts in that user's home region. Each area below opens its
pools through engine.routed, so an answer never pays a cross-region hop
because its connection happened to land on a far node (a totals query took
703 ms that way, against 12 ms through the home region)."""

import os
import sys
import threading
import uuid
from datetime import date, timedelta

import pytest

from banko_ai.db import engine as E
from banko_ai.db import regions
from tests._money import drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")

DB = os.getenv("DATABASE_URL")
HOME = "banko_gateway_region=eu-west-1"
# Per-user code; regions, schema and rls ask cluster-wide questions, cached per database.
SPIED = ("banko_ai.services", "banko_ai.assistant", "banko_ai.utils.aggregations", "banko_ai.db.engine")


@pytest.fixture
def user():
    u = make_user()
    yield u
    drop_user(u)


def _spy(monkeypatch) -> list:
    """The URL of every pool the request asks for from here on, on this
    thread (background writers are not the request). Most modules import
    get_engine by name, so each one's binding is replaced."""
    seen, real, request = [], E.get_engine, threading.current_thread()

    def spy(url=None, **kw):
        if threading.current_thread() is request:
            seen.append(url)
        return real(url, **kw)

    for mod in list(sys.modules.values()):
        name = getattr(mod, "__name__", None) or ""
        if name.startswith(SPIED) and getattr(mod, "get_engine", None) is real:
            monkeypatch.setattr(mod, "get_engine", spy)
    return seen


def _areas(user: str) -> dict:
    from banko_ai.assistant import graph
    from banko_ai.services import actions, caches, ledger, memories, profile, receipts
    from banko_ai.services import transactions as txns
    from banko_ai.services._vector import embed
    from banko_ai.utils.aggregations import explain_aggregation, resolve_merchant, run_aggregation
    from banko_ai.utils.intent_classifier import AggregationIntent

    q, today, nobody = "what did I spend on coffee", date.today(), str(uuid.uuid4())
    intent = AggregationIntent("sum", "Coffee", today - timedelta(days=30), today + timedelta(days=1))
    key = caches.search_key(user, 0, {"q": q})

    def answer_cache():
        caches.put_answer(user, version=0, language="English", query=q, qvec=embed(q), answer="$12.", tokens=5)
        caches.get_answer(user, version=0, language="English", query=q, qvec=embed(q))

    def memory():
        mid = memories.remember(user, "I prefer short answers")["memory_id"]
        memories.recall(user, q)
        memories.list_memories(user)
        memories.forget(user, mid)
        memories.forget_matching(user, "short answers")

    def receipt():
        doc = receipts._insert(user, "r.png", "image/png", None, None)
        receipts._finish(doc, user, None, status="failed")
        receipts.get_receipt(user, doc)
        receipts.latest_receipt(user)
        receipts.receipt_for_expense(user, nobody)
        receipts.thumbnail(user, doc)

    def proposal():
        made = actions.propose(user, "budget", {"category": "Coffee", "amount": 50}, "Coffee budget $50")
        actions.cancel(user, made["action_id"])

    return {
        "data version": lambda: ledger.data_version(user),
        "totals": lambda: run_aggregation(intent, user, DB),
        "totals plan": lambda: explain_aggregation(intent, user, DB),
        "merchant lookup": lambda: resolve_merchant("blue bottle", user, DB),
        "search plan": lambda: txns.explain_semantic(user, q),
        "visits plan": lambda: txns.explain_merchant_visits(user, "Blue Bottle"),
        "region badge": lambda: graph.home_region_of(user, None),
        "search cache": lambda: (caches.put_search(user, key, [1]), caches.get_search(user, key)),
        "answer cache": answer_cache,
        "cache counts": lambda: caches.cache_summary(user),
        "memories": memory,
        "receipts": receipt,
        "proposals": proposal,
        "profile": lambda: profile.get_profile(user),
    }


@pytest.mark.parametrize("area", ["data version", "totals", "totals plan", "merchant lookup", "search plan",
                                  "visits plan", "region badge", "search cache", "answer cache", "cache counts",
                                  "memories", "receipts", "proposals", "profile"])
def test_a_users_statements_start_in_their_home_region(user, monkeypatch, area):
    monkeypatch.setattr(regions, "is_regional_deployment", lambda url: True)
    monkeypatch.setattr(regions, "user_home_region", lambda user_id, url=None: "eu-west-1")
    run = _areas(user)[area]
    seen = _spy(monkeypatch)
    run()
    assert seen and all(u and HOME in u for u in seen), seen
