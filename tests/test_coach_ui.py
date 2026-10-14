"""The coach inbox API and presenter-only simulated signals, scoped to the
signed-in user and built from real data."""

import json
import os
import uuid
from datetime import date

import pytest
from sqlalchemy import text

from banko_ai.coach.simulate import SimulateUnavailable, build_signal
from banko_ai.db.engine import get_engine
from banko_ai.services import budgets
from banko_ai.services.seeding import seed_user
from tests._app import app_client, stop_apps
from tests._money import drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    stop_apps()


@pytest.fixture(scope="module")
def member():
    uid = make_user("diner")
    seed_user(uid, months=3, today=date.today())
    yield uid
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM coach_nudges WHERE user_id = CAST(:u AS UUID)"), {"u": uid})
        c.execute(text("DELETE FROM spending_signals WHERE user_id = CAST(:u AS UUID)"), {"u": uid})
    drop_user(uid)


def client_for(monkeypatch, uid):
    c = app_client(monkeypatch, REPORTING_FOLLOWER_READS="false")
    with c.session_transaction() as s:
        s["user_id"] = uid
    return c


def _nudge(uid, message="Dining is up 30%"):
    sid = str(uuid.uuid4())
    with get_engine().begin() as c:
        c.execute(text("INSERT INTO spending_signals (signal_id, user_id, signal_type, severity, payload, idempotency_key) "
                       "VALUES (CAST(:s AS UUID), CAST(:u AS UUID), 'budget_threshold', 'warn', CAST(:p AS JSONB), :k)"),
                  {"s": sid, "u": uid, "p": json.dumps({"category": "dining"}), "k": "test:" + sid})
        return str(c.execute(text("INSERT INTO coach_nudges (signal_id, user_id, message, tool_trace, provider_used) "
                                  "VALUES (CAST(:s AS UUID), CAST(:u AS UUID), :m, CAST(:t AS JSONB), 'ollama') RETURNING nudge_id"),
                             {"s": sid, "u": uid, "m": message, "t": json.dumps([{"tool": "get_user_budget"}])}).scalar())


def test_inbox_is_scoped_and_detailed(monkeypatch, member):
    nid = _nudge(member)
    c = client_for(monkeypatch, member)
    nudges = c.get("/api/v1/coach/nudges").get_json()["nudges"]
    assert nudges[0]["nudge_id"] == nid and nudges[0]["signal_type"] == "budget_threshold"
    detail = c.get(f"/api/v1/coach/nudges/{nid}").get_json()
    assert detail["tool_trace"][0]["tool"] == "get_user_budget" and detail["payload"]["category"] == "dining"
    other = make_user()
    try:
        c2 = client_for(monkeypatch, other)
        assert c2.get("/api/v1/coach/nudges").get_json()["nudges"] == []
        assert c2.get(f"/api/v1/coach/nudges/{nid}").status_code == 404
    finally:
        drop_user(other)


def test_persona_label_hides_the_actual_provider(monkeypatch, member):
    _nudge(member, "Persona check")
    c = app_client(monkeypatch, AI_DISPLAY_PROVIDER="watsonx", AI_DISPLAY_MODEL="fake-model", REPORTING_FOLLOWER_READS="false")
    with c.session_transaction() as s:
        s["user_id"] = member
    assert "ollama" not in json.dumps(c.get("/api/v1/coach/nudges").get_json()).lower()


def test_build_signal_uses_real_data(member):
    # A purchase today keeps the budget near its limit on any day of the month.
    from banko_ai.services import accounts, ledger
    checking = next(a for a in accounts.list_accounts(member)["accounts"] if a["kind"] != "credit")
    ledger.post_purchase(member, checking["account_id"], amount="45.00", merchant="Corner Bistro", category="Restaurant")
    budgets.set_budget(member, "Restaurant", "50")
    sig = build_signal(member, "budget_threshold")
    assert sig.user_id == member and sig.payload["monthly_budget"] == 50.0
    assert {"category", "pct_used", "spent_so_far", "days_remaining"} <= set(sig.payload)
    assert sig.idempotency_key.startswith("simulated:")
    anomaly = build_signal(member, "anomaly")
    assert anomaly.payload["amount"] > 0 and anomaly.payload["transaction_id"]


def test_build_signal_says_why_it_cannot(member):
    lonely = make_user()
    try:
        with pytest.raises(SimulateUnavailable):
            build_signal(lonely, "recurring_drift")
    finally:
        drop_user(lonely)


def test_a_budget_far_from_its_limit_is_not_near_its_limit(monkeypatch):
    lonely = make_user()
    try:
        budgets.set_budget(lonely, "Groceries", "500")  # nothing spent: 0%
        with pytest.raises(SimulateUnavailable):
            build_signal(lonely, "budget_threshold")
        resp = client_for(monkeypatch, lonely).post("/api/v1/coach/simulate", json={"signal_type": "budget_threshold"})
        assert resp.status_code == 409 and "near its limit" in resp.get_json()["error"]
    finally:
        drop_user(lonely)


def test_a_price_drift_signal_needs_a_bill_that_went_up(monkeypatch):
    from decimal import Decimal

    from banko_ai.coach import simulate
    from banko_ai.services.recurring import Recurring

    def bill(merchant, previous, amount, pct):
        return Recurring(merchant=merchant, category="Subscriptions", cadence="monthly", amount=Decimal(amount),
                         previous_amount=Decimal(previous), price_change_pct=pct, last_date=date.today(),
                         next_due=date.today(), occurrences=6)

    flat = [bill("Streamly", "15.99", "15.99", None), bill("Gym Co", "40.00", "35.00", -12.5)]
    monkeypatch.setattr(simulate.recurring, "detect_recurring", lambda user_id, url=None: flat)
    with pytest.raises(SimulateUnavailable):
        build_signal(str(uuid.uuid4()), "recurring_drift")
    monkeypatch.setattr(simulate.recurring, "detect_recurring",
                        lambda user_id, url=None: flat + [bill("Cloud Drive", "9.99", "11.99", 20.02)])
    sig = build_signal(str(uuid.uuid4()), "recurring_drift")
    assert sig.payload["subscription"] == "Cloud Drive" and sig.payload["pct_change"] > 0


def test_simulate_endpoint_queues_through_the_webhook_path(monkeypatch, member):
    c = client_for(monkeypatch, member)
    seen = []
    c.application.extensions["banko_coach"]["submit"] = lambda sig: seen.append(sig) or True
    resp = c.post("/api/v1/coach/simulate", json={"signal_type": "anomaly"})
    assert resp.status_code == 202 and seen and seen[0].user_id == member
    assert c.post("/api/v1/coach/simulate", json={"signal_type": "nonsense"}).status_code == 400
