"""The boot-time sample data load: few large batches, and one region lookup
per user rather than per row. On a region-survivable cluster every batch
commit waits on a cross-region quorum, so the batch count is the load time."""

import uuid
from contextlib import contextmanager
from datetime import date

import pandas as pd

from banko_ai.config.settings import Settings
from banko_ai.vector_search.generator import EnhancedExpenseGenerator


def _rows(count: int, users: list[str]) -> list[dict]:
    return [{
        "expense_id": str(uuid.uuid4()), "user_id": users[i % len(users)], "expense_date": date(2026, 10, 1),
        "expense_amount": 12.5, "shopping_type": "Coffee", "description": "latte", "merchant": "Blue Bottle",
        "payment_method": "Credit Card", "recurring": False, "tags": ["coffee"], "embedding": [0.0] * 384,
    } for i in range(count)]


def _fake_writes(monkeypatch, regional: bool, lookups: list[str]) -> list[int]:
    batches: list[int] = []

    class FakeEngine:
        @contextmanager
        def begin(self):
            yield object()

    monkeypatch.setattr(EnhancedExpenseGenerator, "engine", property(lambda self: FakeEngine()))
    monkeypatch.setattr(pd.DataFrame, "to_sql", lambda df, *a, **kw: batches.append(len(df)))
    monkeypatch.setattr("banko_ai.db.regions.regional_tables_ready", lambda url=None: regional)

    def lookup(user_id, url=None):
        lookups.append(user_id)
        return None  # the sample users do not exist in users

    monkeypatch.setattr("banko_ai.web.auth.resolve_user_region", lookup)
    return batches


def test_batches_default_to_500_rows():
    assert Settings().data_gen_batch_size == 500


def test_the_load_commits_in_few_large_batches(monkeypatch):
    lookups: list[str] = []
    batches = _fake_writes(monkeypatch, regional=False, lookups=lookups)
    gen = EnhancedExpenseGenerator("postgresql://root@localhost:26258/defaultdb?sslmode=disable")
    assert gen.save_expenses_to_database(_rows(1200, [str(uuid.uuid4())])) == 1200
    assert batches == [500, 500, 200]


def test_each_users_region_is_looked_up_once(monkeypatch):
    lookups: list[str] = []
    users = [str(uuid.uuid4()) for _ in range(3)]
    _fake_writes(monkeypatch, regional=True, lookups=lookups)
    gen = EnhancedExpenseGenerator("postgresql://root@localhost:26258/defaultdb?sslmode=disable")
    gen.save_expenses_to_database(_rows(120, users))
    assert sorted(lookups) == sorted(users)
