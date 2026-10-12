"""The integrity checker finds drift and stays quiet when there is none."""

import os
import threading

import pytest
from sqlalchemy import text

from banko_ai.config.settings import reset_config
from banko_ai.db.engine import get_engine
from banko_ai.services import integrity as I
from banko_ai.services import ledger as L
from tests._money import drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture
def account():
    uid = make_user()
    acct = L.open_account(uid, kind="checking", name="C", institution="B", mask="1", opening_balance="100")
    L.post_purchase(uid, acct, amount="10", merchant="M", category="Other")
    yield uid, acct
    drop_user(uid)


def test_clean_ledger_reports_no_drift(account):
    uid, _ = account
    report = I.check_integrity(follower_reads=False, user_id=uid)
    assert report.ok and report.accounts >= 1 and report.drifted == ()


def test_drift_is_reported_with_both_numbers(account):
    uid, acct = account
    with get_engine().begin() as c:
        c.execute(text("UPDATE accounts SET balance = balance + 1 WHERE account_id = CAST(:a AS UUID)"), {"a": acct})
    report = I.check_integrity(follower_reads=False, user_id=uid)
    assert not report.ok
    (drift,) = report.drifted
    assert drift["account_id"] == acct and drift["balance"] == 91.0 and drift["expected"] == 90.0


def test_skips_above_the_account_ceiling(account, monkeypatch):
    monkeypatch.setenv("INTEGRITY_MAX_ACCOUNTS", "0")
    reset_config()
    report = I.check_integrity(follower_reads=False)
    assert report.skipped and "more than 0 accounts" in report.skipped and not report.ok


def test_follower_reads_or_a_clean_fallback(account):
    report = I.check_integrity(follower_reads=True, user_id=account[0])
    assert report.skipped is None


def test_worker_reports_and_stops(account):
    stop = threading.Event()
    seen = []
    worker = I.IntegrityWorker(interval=3600, stop=stop, on_report=seen.append)
    thread = worker.start()
    for _ in range(100):
        if seen:
            break
        threading.Event().wait(0.1)
    stop.set()
    thread.join(timeout=5)
    assert seen and worker.latest() is seen[0] and I.latest_report() is seen[0]


def test_interval_zero_disables_the_worker(monkeypatch):
    monkeypatch.setenv("INTEGRITY_INTERVAL_SECONDS", "0")
    reset_config()
    assert I.IntegrityWorker().start() is None
