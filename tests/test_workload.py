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


def _user(style: str = "saver", region: str | None = None) -> LoadUser:
    """A customer who exists only in memory, with every account the style uses."""
    accounts = {a.key: f"acct-{a.key}" for a in get_catalog().profiles[style].accounts}
    return LoadUser("u-1", "load-00001-aaaa", style, region, accounts)


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


def test_a_purchase_lets_the_database_pick_its_id_and_uses_a_cached_vector(prepared, vectors, monkeypatch):
    """A purchase sends no id of its own: on a REGIONAL BY ROW table a
    client-chosen key costs a uniqueness check in every region."""
    seen = {}
    real = ledger.post_purchase

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(W.ledger, "post_purchase", spy)
    assert run_op("purchase", prepared[1], OpContext(url=None, vectors=vectors), random.Random(3)) == ("ok", 0)
    assert seen.get("expense_id") is None and seen["embedding"] is not None
    with get_engine().connect() as c:
        row = c.execute(text("SELECT description, embedding IS NOT NULL FROM expenses WHERE user_id = CAST(:u AS UUID) "
                             "ORDER BY created_at DESC LIMIT 1"), {"u": prepared[1].user_id}).fetchone()
    assert row[1] and re.fullmatch(r"Spent \$\d+\.\d{2} on .+ at .+ using .+\.", row[0])


def test_a_deposit_lets_the_database_pick_its_entry_id(monkeypatch):
    seen = {}
    monkeypatch.setattr(W.ledger, "record_income", lambda *args, **kwargs: seen.update(kwargs) or "entry")
    assert run_op("deposit", _user(), OpContext(url=None, vectors=VectorCache({})), random.Random(2)) == ("ok", 0)
    assert seen.get("entry_id") is None


def _lost(at_commit: bool) -> OperationalError:
    exc = OperationalError("COMMIT", {}, psycopg.errors.AdminShutdown("server is shutting down"))
    if at_commit:
        exc.at_commit = True
    return exc


def test_a_purchase_whose_commit_was_lost_is_not_run_again(monkeypatch):
    """Without a key of its own a purchase cannot tell whether a lost commit
    landed, so it is reported, never posted twice."""
    calls = []

    def lost(*args, **kwargs):
        calls.append(1)
        raise _lost(at_commit=True)

    monkeypatch.setattr(W.ledger, "post_purchase", lost)
    with pytest.raises(OperationalError):
        run_op("purchase", _user(), OpContext(url=None, vectors=VectorCache({})), random.Random(1))
    assert len(calls) == 1


def test_work_that_is_not_idempotent_retries_only_what_cannot_have_committed():
    attempts = iter([_lost(at_commit=False), None])

    def fn():
        exc = next(attempts)
        if exc:
            raise exc
        return "posted"

    assert with_client_retries(fn, committed=lambda e: False, idempotent=False, **NO_SLEEP) == ("posted", 1)
    with pytest.raises(OperationalError) as err:
        with_client_retries(lambda: (_ for _ in ()).throw(_lost(at_commit=True)), committed=lambda e: False,
                            idempotent=False, **NO_SLEEP)
    assert err.value.client_retries == 0


def test_each_customer_is_served_through_their_home_region(monkeypatch):
    """Reads and writes go through the customer's home-region pool, and reads
    name the region so they touch only its partition."""
    calls = []
    monkeypatch.setattr(W.accounts, "list_accounts", lambda user_id, **kw: calls.append(kw))
    monkeypatch.setattr(W.transactions, "list_transactions", lambda user_id, **kw: calls.append(kw))
    monkeypatch.setattr(W.transactions, "semantic_search", lambda user_id, q, **kw: calls.append(kw))
    monkeypatch.setattr(W.ledger, "record_income", lambda *args, **kw: calls.append(kw) or "entry")
    homed = "postgresql://root@db:26257/defaultdb?banko_gateway_region=eu-west-1"
    ctx = OpContext(url="postgresql://root@db:26257/defaultdb", vectors=VectorCache({}),
                    region_urls={"eu-west-1": homed}, pin_regions=True)
    rng = random.Random(4)
    for op in ("page_load", "search", "deposit"):
        run_op(op, _user(region="eu-west-1"), ctx, rng)
    assert [c["url"] for c in calls] == [homed] * 4
    assert [c.get("region") for c in calls[:3]] == ["eu-west-1"] * 3
    calls.clear()
    run_op("page_load", _user(region="eu-west-1"), OpContext(url="plain", vectors=VectorCache({})), rng)
    assert [(c["url"], c.get("region")) for c in calls] == [("plain", None)] * 2
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


def test_card_payments_often_pay_the_whole_balance(monkeypatch):
    """Many cardholders pay the statement in full, which keeps busy
    customers' cards from piling up declines."""
    from decimal import Decimal

    paid = []
    monkeypatch.setattr(W.accounts, "get_account", lambda user_id, account_id, **kw: {"balance": "-1234.56"})
    monkeypatch.setattr(W.ledger, "card_payment", lambda user_id, src, card, amount, **kw: paid.append(amount))
    user = LoadUser("u", "load-00000-aaaa", "saver", None, {"checking": "c", "card": "k"})
    rng = random.Random(11)
    for _ in range(200):
        assert run_op("card_payment", user, OpContext(url=None, vectors=VectorCache({})), rng) == ("ok", 0)
    full = sum(1 for a in paid if a == Decimal("1234.56"))
    assert 70 <= full <= 130
    assert all(Decimal("100") <= a <= Decimal("600") for a in paid if a != Decimal("1234.56"))


@pytest.mark.parametrize("op", ["transfer", "card_payment"])
def test_a_transfer_the_ledger_refused_is_never_counted_as_done(op, monkeypatch):
    """ledger.transfer reports insufficient funds, a missing account or a
    reused key in its result instead of raising: not enough money is a
    decline, anything else an error, never a success."""
    monkeypatch.setattr(W.accounts, "get_account", lambda user_id, account_id, **kw: {"balance": "-50.00"})
    ctx, rng = OpContext(url=None, vectors=VectorCache({})), random.Random(6)
    for status, expect in (("insufficient_funds", "declined"), ("ok", "ok"), ("replay", "ok")):
        monkeypatch.setattr(W.ledger, op, lambda *a, _s=status, **kw: ledger.TransferResult(_s, None))
        assert run_op(op, _user(), ctx, rng)[0] == expect, status
    for status in ("not_found", "key_conflict"):
        monkeypatch.setattr(W.ledger, op, lambda *a, _s=status, **kw: ledger.TransferResult(_s, None))
        with pytest.raises(W.TransferRefused):
            run_op(op, _user(), ctx, rng)
