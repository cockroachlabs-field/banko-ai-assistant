"""Cached answers and cached search results never cross users."""

import os

import numpy as np
import pytest

from banko_ai.utils.cache_manager import BankoCacheManager
from banko_ai.utils.request_scope import user_scope

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")

MAYA = "00000000-0000-0000-0000-0000000000a1"
SAM = "00000000-0000-0000-0000-0000000000a2"
ROWS = [{"expense_id": "e1", "description": "coffee", "merchant": "Blue Bottle", "expense_amount": 6.5}]


@pytest.fixture(autouse=True)
def schema():
    from banko_ai.db.schema import ensure_schema
    ensure_schema()


def test_cached_answers_never_cross_users():
    cache = BankoCacheManager(strict_mode=False)
    question = f"what did I spend on coffee {os.getpid()}"
    with user_scope(MAYA):
        cache.cache_response(question, "Maya's answer", ROWS, "ollama", language="en")
        assert cache.get_cached_response(question, ROWS, "ollama", "en") == "Maya's answer"
    with user_scope(SAM):
        assert cache.get_cached_response(question, ROWS, "ollama", "en") is None


def test_cached_search_results_never_cross_users():
    cache = BankoCacheManager()
    vector = np.random.default_rng(os.getpid()).random(384).astype("float32")
    results = [{"expense_id": "e1", "user_id": MAYA, "description": "coffee", "merchant": "Blue Bottle",
                "expense_amount": 6.5, "expense_date": "2026-10-01", "similarity_score": 0.1,
                "shopping_type": "Coffee", "payment_method": "Credit Card", "recurring": False, "tags": []}]
    cache.cache_vector_search_results(vector, results, scope=f"{MAYA}|None|None|None|10")
    assert cache.get_cached_vector_search(vector, 1, scope=f"{MAYA}|None|None|None|10")
    assert cache.get_cached_vector_search(vector, 1, scope=f"{SAM}|None|None|None|10") is None
