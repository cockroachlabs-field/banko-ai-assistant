"""The load simulator's workload: real banking operations through the
services, client-side idempotency, and a realistic skew across customers."""

import os
import random
import re
from collections import Counter

import psycopg
import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError

from banko_ai.config.catalog import get_catalog
from banko_ai.db.engine import get_engine
from banko_ai.services import ledger
from banko_ai.services.integrity import check_integrity
from banko_ai.services.load_users import LoadUser, clear, prepare
from banko_ai.workers import workload as W
from banko_ai.workers.workload import (
    OPS,
    PROFILES,
    OpContext,
    UserPicker,
    VectorCache,
    is_unique_violation,
    run_op,
    with_client_retries,
)

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")
NO_SLEEP = {"sleep": lambda seconds: None}


@pytest.fixture(scope="module")
def prepared():
    clear()
    users = prepare(2)
    yield users
    clear()


@pytest.fixture(scope="module")
def vectors():
    return VectorCache.build()


def _ambiguous() -> OperationalError:
    return OperationalError("COMMIT", {}, psycopg.errors.StatementCompletionUnknown("result is ambiguous"))


def _unique() -> IntegrityError:
    return IntegrityError("INSERT", {}, psycopg.errors.UniqueViolation("duplicate key value"))


def test_profiles_are_complete():
    assert set(PROFILES) == {"balanced", "payments", "browsing", "search"}
    assert all(sum(w.values()) == 100 and set(w) <= set(OPS) for w in PROFILES.values())


def test_users_are_picked_with_a_skew():
    users = [LoadUser(str(i), f"load-{i}", "saver", None, {}) for i in range(100)]
    picker, rng = UserPicker(users), random.Random(7)
    counts = Counter(picker.pick(rng).user_id for _ in range(10000))
    assert sum(n for _, n in counts.most_common(10)) > 3000
    assert len(counts) > 60  # the long tail still shows up


def test_an_ambiguous_commit_retries_and_a_duplicate_means_it_landed():
    raised = iter([_ambiguous(), _unique()])

    def fn():
        raise next(raised)

    assert with_client_retries(fn, committed=is_unique_violation, **NO_SLEEP) == (None, 1)


def test_a_duplicate_on_the_first_attempt_is_a_real_error():
    def fn():
        raise _unique()

    with pytest.raises(IntegrityError):
        with_client_retries(fn, committed=is_unique_violation, **NO_SLEEP)


def test_retries_give_up_after_the_attempts_and_say_how_many():
    calls = []

    def fn():
        calls.append(1)
        raise _ambiguous()

    with pytest.raises(OperationalError) as caught:
        with_client_retries(fn, committed=is_unique_violation, attempts=3, **NO_SLEEP)
    assert len(calls) == 3 and caught.value.client_retries == 2


def test_business_errors_are_never_retried():
    calls = []

    def fn():
        calls.append(1)
        raise ledger.InsufficientFunds("Not enough available on that account")

    with pytest.raises(ledger.InsufficientFunds):
        with_client_retries(fn, committed=is_unique_violation, **NO_SLEEP)
    assert len(calls) == 1


def test_the_vector_cache_covers_every_profile_purchase(vectors):
    for profile in get_catalog().profiles.values():
        for spec in profile.purchases:
            for merchant in spec.merchants:
                vec = vectors.get(merchant, spec.category)
                assert vec is not None and len(vec) == 384


def test_every_operation_runs_against_real_services(prepared, vectors):
    ctx, rng = OpContext(url=None, vectors=vectors), random.Random(1)
    for op in OPS:
        outcome, retries = run_op(op, prepared[0], ctx, rng)
        assert outcome in ("ok", "declined") and retries == 0, op
    assert check_integrity(follower_reads=False, user_id=prepared[0].user_id).ok


def test_a_purchase_posts_with_the_clients_id_and_a_cached_vector(prepared, vectors, monkeypatch):
    known = "11111111-2222-4333-8444-555555555555"
    monkeypatch.setattr(W, "new_id", lambda: known)
    assert run_op("purchase", prepared[1], OpContext(url=None, vectors=vectors), random.Random(3)) == ("ok", 0)
    with get_engine().connect() as c:
        row = c.execute(text("SELECT user_id, description, embedding IS NOT NULL FROM expenses "
                             "WHERE expense_id = CAST(:e AS UUID)"), {"e": known}).fetchone()
    assert str(row[0]) == prepared[1].user_id and row[2]
    assert re.fullmatch(r"Spent \$\d+\.\d{2} on .+ at .+ using .+\.", row[1])


def test_a_declined_purchase_is_a_business_outcome(prepared, vectors, monkeypatch):
    def broke(*args, **kwargs):
        raise ledger.InsufficientFunds("Not enough available on that account")

    monkeypatch.setattr(W.ledger, "post_purchase", broke)
    assert run_op("purchase", prepared[0], OpContext(url=None, vectors=vectors), random.Random(4)) == ("declined", 0)


def test_a_deposit_with_an_entry_id_posts_once(prepared):
    eid, acct = W.new_id(), prepared[0].accounts["checking"]
    ledger.record_income(prepared[0].user_id, acct, "10.00", entry_id=eid)
    with pytest.raises(IntegrityError):
        ledger.record_income(prepared[0].user_id, acct, "10.00", entry_id=eid)


def test_card_payments_pay_what_is_owed_then_decline(prepared, vectors):
    from decimal import Decimal

    from banko_ai.services import accounts
    user = prepared[1]
    ctx, rng = OpContext(url=None, vectors=vectors), random.Random(5)
    outcomes = [run_op("card_payment", user, ctx, rng)[0] for _ in range(12)]
    card = accounts.get_account(user.user_id, user.accounts["card"])
    assert Decimal(str(card["balance"])) == 0  # paid off exactly, never overpaid
    assert "ok" in outcomes and outcomes[-1] == "declined"
    assert check_integrity(follower_reads=False, user_id=user.user_id).ok
