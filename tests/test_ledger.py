"""Money movement: atomic, idempotent, scoped to its user, never drifting."""

import os
import threading
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal

import pytest
from sqlalchemy import text

from banko_ai.db.engine import get_engine
from banko_ai.services import ledger as L
from tests._money import drop_user, invariant_violations, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture
def user():
    uid = make_user()
    yield uid
    drop_user(uid)


def _balance(account_id: str) -> Decimal:
    with get_engine().connect() as c:
        return c.execute(text("SELECT balance FROM accounts WHERE account_id = CAST(:a AS UUID)"),
                         {"a": account_id}).scalar()


def _checking(user_id: str, opening="100.00") -> str:
    return L.open_account(user_id, kind="checking", name="Checking", institution="Test Bank",
                          mask="0001", opening_balance=opening, payment_method="Debit Card")


def test_money_parsing():
    assert L.money("12.34") == Decimal("12.34")
    assert L.money("12.340") == Decimal("12.34")
    assert L.money(Decimal("1e2")) == Decimal("100.00")
    assert L.money(0.1) + L.money(0.2) == Decimal("0.30")
    assert L.money("-0") == Decimal("0.00")
    for bad in ("12.345", "abc", "NaN", "Infinity", None, [], "1,000", True, "1e400", "1e30", "1e26", 10**30):
        with pytest.raises(L.InvalidAmount):
            L.money(bad)
    with pytest.raises(L.InvalidAmount):
        L._positive("-0")


def test_open_account_starts_at_its_opening_balance(user):
    acct = _checking(user, "250.50")
    assert _balance(acct) == Decimal("250.50")
    assert invariant_violations(user) == []
    assert L.data_version(user) == 1


def test_purchase_writes_expense_and_ledger_atomically(user):
    acct = _checking(user)
    eid = L.post_purchase(user, acct, amount="12.50", merchant="Corner Cafe", category="Coffee")
    assert _balance(acct) == Decimal("87.50")
    with get_engine().connect() as c:
        exp = c.execute(text("SELECT expense_amount, shopping_type, merchant, payment_method FROM expenses "
                             "WHERE expense_id = CAST(:e AS UUID)"), {"e": eid}).fetchone()
        leg = c.execute(text("SELECT amount, kind FROM ledger_entries WHERE expense_id = CAST(:e AS UUID)"),
                        {"e": eid}).fetchone()
    assert exp == (Decimal("12.50"), "Coffee", "Corner Cafe", "Debit Card")
    assert leg == (Decimal("-12.50"), "purchase")
    assert invariant_violations(user) == []


def test_purchase_beyond_funds_writes_nothing(user):
    acct = _checking(user, "10.00")
    with pytest.raises(L.InsufficientFunds):
        L.post_purchase(user, acct, amount="10.01", merchant="Big Store", category="Shopping")
    with get_engine().connect() as c:
        assert c.execute(text("SELECT count(*) FROM expenses WHERE user_id = CAST(:u AS UUID)"),
                         {"u": user}).scalar() == 0
    assert _balance(acct) == Decimal("10.00")


def test_record_income_raises_balance_and_writes_income_leg(user):
    acct = _checking(user, "100.00")
    entry_id = L.record_income(user, acct, "500.00", description="Payroll")
    assert _balance(acct) == Decimal("600.00")
    with get_engine().connect() as c:
        leg = c.execute(text("SELECT amount, kind, description FROM ledger_entries WHERE entry_id = CAST(:e AS UUID)"),
                        {"e": entry_id}).fetchone()
    assert leg == (Decimal("500.00"), "income", "Payroll")
    assert invariant_violations(user) == []


def test_credit_card_respects_its_limit(user):
    card = L.open_account(user, kind="credit", name="Card", institution="Test Bank", mask="0002",
                          opening_balance="-90.00", credit_limit="100.00")
    L.post_purchase(user, card, amount="10.00", merchant="Shop", category="Shopping")
    assert _balance(card) == Decimal("-100.00")
    with pytest.raises(L.InsufficientFunds):
        L.post_purchase(user, card, amount="0.01", merchant="Shop", category="Shopping")


def test_transfer_moves_both_balances_with_two_legs(user):
    a = _checking(user)
    b = L.open_account(user, kind="savings", name="Savings", institution="Test Bank",
                       mask="0003", opening_balance="0")
    result = L.transfer(user, a, b, "40.00", idempotency_key="t-1")
    assert result.status == "ok"
    assert (_balance(a), _balance(b)) == (Decimal("60.00"), Decimal("40.00"))
    with get_engine().connect() as c:
        legs = c.execute(text("SELECT sum(amount), count(*) FROM ledger_entries WHERE transfer_id = CAST(:t AS UUID)"),
                         {"t": result.transfer_id}).fetchone()
    assert legs == (Decimal("0.00"), 2)
    assert invariant_violations(user) == []


def test_replayed_key_moves_money_once(user):
    a, b = _checking(user), _checking(user, "0")
    first = L.transfer(user, a, b, "25.00", idempotency_key="same")
    second = L.transfer(user, a, b, "25.00", idempotency_key="same")
    assert (first.status, second.status) == ("ok", "replay")
    assert second.transfer_id == first.transfer_id
    assert _balance(a) == Decimal("75.00")


def test_same_key_different_request_returns_key_conflict(user):
    a, b = _checking(user), _checking(user, "0")
    first = L.transfer(user, a, b, "25.00", idempotency_key="conflict")
    second = L.transfer(user, a, b, "30.00", idempotency_key="conflict")
    assert first.status == "ok"
    assert second.status == "key_conflict"
    assert _balance(a) == Decimal("75.00")
    assert _balance(b) == Decimal("25.00")


def test_transfer_refused_for_insufficient_funds_leaves_no_transfer_row(user):
    a, b = _checking(user, "10.00"), _checking(user, "0")
    result = L.transfer(user, a, b, "20.00", idempotency_key="poor")
    assert result.status == "insufficient_funds"
    with get_engine().connect() as c:
        assert c.execute(text("SELECT count(*) FROM transfers WHERE user_id = CAST(:u AS UUID)"),
                         {"u": user}).scalar() == 0
    L.record_income(user, a, "15.00")
    result2 = L.transfer(user, a, b, "20.00", idempotency_key="poor")
    assert result2.status == "ok"
    assert _balance(a) == Decimal("5.00")


def test_transfer_bumps_data_version(user):
    a, b = _checking(user), _checking(user, "0")
    assert L.data_version(user) == 2
    L.transfer(user, a, b, "10.00", idempotency_key="v1")
    assert L.data_version(user) == 3


def test_concurrent_same_key_moves_money_once(user):
    a, b = _checking(user), _checking(user, "0")
    results: list[L.TransferResult] = []
    lock = threading.Lock()

    def go():
        r = L.transfer(user, a, b, "30.00", idempotency_key="double-click")
        with lock:
            results.append(r)

    threads = [threading.Thread(target=go) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(r.status for r in results) == ["ok", "replay", "replay", "replay", "replay"]
    assert (_balance(a), _balance(b)) == (Decimal("70.00"), Decimal("30.00"))
    assert invariant_violations(user) == []


def test_concurrent_drain_never_overdraws(user):
    a, b = _checking(user, "100.00"), _checking(user, "0")
    results: list[str] = []
    lock = threading.Lock()

    def go(i):
        r = L.transfer(user, a, b, "30.00", idempotency_key=f"drain-{i}")
        with lock:
            results.append(r.status)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count("ok") == 3 and results.count("insufficient_funds") == 2
    assert _balance(a) == Decimal("10.00")
    assert invariant_violations(user) == []


def test_cross_user_account_is_not_found(user):
    other = make_user()
    try:
        mine, theirs = _checking(user), _checking(other)
        assert L.transfer(user, mine, theirs, "5.00", idempotency_key="x").status == "not_found"
        assert L.transfer(user, theirs, mine, "5.00", idempotency_key="y").status == "not_found"
        with pytest.raises(L.NotFound):
            L.post_purchase(user, theirs, amount="1.00", merchant="M", category="Other")
        assert (_balance(mine), _balance(theirs)) == (Decimal("100.00"), Decimal("100.00"))
    finally:
        drop_user(other)


def test_card_payment_cannot_overpay(user):
    chk = _checking(user, "500.00")
    card = L.open_account(user, kind="credit", name="Card", institution="Test Bank", mask="0004",
                          opening_balance="-50.00", credit_limit="1000.00")
    with pytest.raises(L.InvalidAmount):
        L.card_payment(user, chk, card, "50.01", idempotency_key="over")
    assert L.card_payment(user, chk, card, "50.00", idempotency_key="exact").status == "ok"
    assert _balance(card) == Decimal("0.00")


def test_goal_contribution_tags_both_legs(user):
    chk = _checking(user, "200.00")
    sav = L.open_account(user, kind="savings", name="Savings", institution="Test Bank", mask="0005")
    goal = L.create_goal(user, name="Trip", icon="✈️", target_amount="1000.00", account_id=sav)
    assert L.goal_contribution(user, goal, chk, "75.00", idempotency_key="g1").status == "ok"
    with get_engine().connect() as c:
        tagged = c.execute(text("SELECT count(*) FROM ledger_entries WHERE goal_id = CAST(:g AS UUID)"),
                           {"g": goal}).scalar()
    assert tagged == 2
    assert L.goal_contribution(user, str(uuid.uuid4()), chk, "1.00", idempotency_key="g2").status == "not_found"


def test_tagging_transfer_with_another_users_goal_returns_not_found(user):
    other = make_user()
    try:
        my_chk = _checking(user, "100.00")
        my_sav = L.open_account(user, kind="savings", name="Savings", institution="Test Bank", mask="0006")
        their_sav = L.open_account(other, kind="savings", name="Savings", institution="Test Bank", mask="0007")
        their_goal = L.create_goal(other, name="Trip", icon="✈️", target_amount="1000.00", account_id=their_sav)
        result = L.transfer(user, my_chk, my_sav, "50.00", idempotency_key="x", goal_id=their_goal)
        assert result.status == "not_found"
        assert _balance(my_chk) == Decimal("100.00")
    finally:
        drop_user(other)


def test_history_is_idempotent_and_rebalance_restores_the_invariant(user):
    acct = _checking(user, "0")
    day = date(2026, 1, 15)
    rows = [
        L.HistoryRow(entry_id=str(uuid.uuid4()), account_id=acct,
                     posted_at=datetime(2026, 1, 15, 9, tzinfo=timezone.utc), amount=Decimal("500.00"),
                     kind="income", description="Payroll deposit"),
        L.HistoryRow(entry_id=str(uuid.uuid4()), account_id=acct,
                     posted_at=datetime(2026, 1, 15, 12, tzinfo=timezone.utc), amount=Decimal("-20.00"),
                     kind="purchase", description="Spent $20.00 on coffee at Cafe using Debit Card.",
                     expense={"expense_id": str(uuid.uuid4()), "expense_date": day,
                              "expense_amount": Decimal("20.00"), "shopping_type": "Coffee",
                              "merchant": "Cafe", "payment_method": "Debit Card", "recurring": False,
                              "tags": ["coffee", "cafe"], "embedding": None}),
    ]
    assert L.post_history(user, rows) == 2
    assert L.post_history(user, rows) == 2  # same ids: a no-op write
    L.rebalance(user)
    assert _balance(acct) == Decimal("480.00")
    with get_engine().connect() as c:
        assert c.execute(text("SELECT count(*) FROM ledger_entries WHERE user_id = CAST(:u AS UUID)"),
                         {"u": user}).scalar() == 2
    assert invariant_violations(user) == []


MR_URL = os.getenv("BANKO_TEST_MR_URL")


@pytest.mark.skipif(not MR_URL, reason="BANKO_TEST_MR_URL not set")
def test_money_rows_live_in_the_users_home_region():
    from sqlalchemy.engine import make_url

    from banko_ai.db.schema import ensure_schema
    from banko_ai.utils.db_retry import normalize_db_url

    admin = get_engine(MR_URL)
    with admin.connect() as c:
        c.execution_options(isolation_level="AUTOCOMMIT")
        c.execute(text("DROP DATABASE IF EXISTS banko_ledger_mr_test CASCADE"))
        c.execute(text("CREATE DATABASE banko_ledger_mr_test"))
    url = make_url(normalize_db_url(MR_URL)).set(database="banko_ledger_mr_test").render_as_string(hide_password=False)
    try:
        ensure_schema(url)
        eng = get_engine(url)
        with eng.connect() as c:
            regions = [r[0] for r in c.execute(text("SELECT region FROM [SHOW REGIONS FROM DATABASE]"))]
        home = sorted(regions)[-1]
        uid = str(uuid.uuid4())
        with eng.begin() as c:
            c.execute(text("INSERT INTO users (user_id, username, spending_style, home_region) "
                           "VALUES (CAST(:u AS UUID), 'mr', 'saver', :r)"), {"u": uid, "r": home})
        acct = L.open_account(uid, kind="checking", name="C", institution="B", mask="1",
                              opening_balance="50", url=url)
        L.post_purchase(uid, acct, amount="5", merchant="M", category="Other", url=url)
        with eng.connect() as c:
            for table in ("accounts", "ledger_entries", "expenses", "user_data_versions"):
                got = {r[0] for r in c.execute(text(f"SELECT DISTINCT crdb_region::STRING FROM {table} "
                                                    "WHERE user_id = CAST(:u AS UUID)"), {"u": uid})}
                assert got == {home}, table
    finally:
        with admin.connect() as c:
            c.execution_options(isolation_level="AUTOCOMMIT")
            c.execute(text("DROP DATABASE IF EXISTS banko_ledger_mr_test CASCADE"))
