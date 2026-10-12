"""API v1 contracts: shapes, status codes, scoping and validation."""

import os
import time
import uuid
from datetime import date

import pytest

from banko_ai.services.accounts import list_accounts
from banko_ai.services.ledger import open_account
from banko_ai.services.seeding import seed_user
from tests._money import drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")

_apps = []


def app_client(monkeypatch, **env):
    """A test client on a real app, built the way test_provider_persona's
    persona_client builds it: Ollama as the serving provider, whose init
    never reaches for the network, so no live LLM is needed."""
    from banko_ai.config.settings import reset_config
    from banko_ai.web.app import create_app

    monkeypatch.setenv("BANKO_SKIP_AUTOSETUP", "1")
    env.setdefault("AI_SERVICE", "ollama")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    reset_config()
    app = create_app()
    _apps.append(app)
    readiness = app.extensions["banko_readiness"]
    readiness.wait_for("schema", 60)
    readiness.wait_for("personas", 60)
    return app.test_client()


@pytest.fixture(autouse=True)
def _stop_warmup():
    yield
    while _apps:
        _apps.pop().extensions["banko_warmup_stop"].set()


@pytest.fixture(scope="module")
def member():
    uid = make_user("balanced")
    seed_user(uid, months=2, today=date.today())
    yield uid
    drop_user(uid)


@pytest.fixture
def client(member, monkeypatch):
    c = app_client(monkeypatch, REPORTING_FOLLOWER_READS="false")
    with c.session_transaction() as s:
        s["user_id"] = member
    return c


def _accounts(member):
    accts = list_accounts(member)["accounts"]
    return next(a for a in accts if a["kind"] == "checking"), next(a for a in accts if a["kind"] == "savings")


def test_me_and_dashboard(client, member):
    me = client.get("/api/v1/me").get_json()
    assert me["user_id"] == member and me["spending_style"] == "balanced"
    dash = client.get("/api/v1/dashboard")
    assert dash.status_code == 200
    assert {"net_worth", "budgets", "recent", "upcoming_bills"} <= set(dash.get_json())


def test_accounts_and_statement_csv(client, member):
    checking, _ = _accounts(member)
    assert client.get("/api/v1/accounts").get_json()["net_worth"] is not None
    month = date.today().strftime("%Y-%m")
    st = client.get(f"/api/v1/accounts/{checking['account_id']}/statements?month={month}")
    assert st.status_code == 200 and "closing" in st.get_json()
    csv = client.get(f"/api/v1/accounts/{checking['account_id']}/statements?month={month}&format=csv")
    assert csv.status_code == 200 and csv.mimetype == "text/csv"
    assert csv.data.decode().startswith("Date,Description,Kind,Amount")
    assert client.get(f"/api/v1/accounts/{checking['account_id']}/statements?month=2026-13").status_code == 400


def test_transactions_paginate_and_search(client):
    page = client.get("/api/v1/transactions?limit=5").get_json()
    assert len(page["items"]) == 5 and page["next_cursor"]
    nxt = client.get(f"/api/v1/transactions?limit=5&cursor={page['next_cursor']}").get_json()
    assert {i["id"] for i in page["items"]}.isdisjoint({i["id"] for i in nxt["items"]})
    hits = client.get("/api/v1/transactions?q=groceries").get_json()
    assert hits["items"] and "distance" in hits["items"][0]
    assert client.get(f"/api/v1/transactions/{page['items'][0]['id']}").status_code == 200


def test_bad_min_max_are_400(client):
    r = client.get("/api/v1/transactions?min=abc")
    assert r.status_code == 400 and r.get_json()["error"]
    assert client.get("/api/v1/transactions?max=1234567890123").status_code == 400


def test_transfer_double_submit_is_a_replay(client, member):
    checking, savings = _accounts(member)
    body = {"from_account": checking["account_id"], "to_account": savings["account_id"],
            "amount": "10.00", "idempotency_key": f"api-{uuid.uuid4()}"}
    first = client.post("/api/v1/transfers", json=body)
    second = client.post("/api/v1/transfers", json=body)
    assert (first.status_code, first.get_json()["status"]) == (201, "ok")
    assert (second.status_code, second.get_json()["status"]) == (200, "replay")


def test_same_key_different_amount_is_a_409_key_conflict(client, member):
    checking, savings = _accounts(member)
    body = {"from_account": checking["account_id"], "to_account": savings["account_id"],
            "amount": "3.00", "idempotency_key": f"kc-{uuid.uuid4()}"}
    assert client.post("/api/v1/transfers", json=body).status_code == 201
    r = client.post("/api/v1/transfers", json={**body, "amount": "4.00"})
    assert r.status_code == 409 and r.get_json() == {"error": "key_conflict"}


@pytest.mark.parametrize("amount", ["0", "-5", "1e9", "12.345", "abc", "NaN", None, True])
def test_bad_amounts_are_400(client, member, amount):
    checking, savings = _accounts(member)
    r = client.post("/api/v1/transfers", json={"from_account": checking["account_id"],
                                               "to_account": savings["account_id"], "amount": amount,
                                               "idempotency_key": f"bad-{uuid.uuid4()}"})
    assert r.status_code == 400 and r.get_json()["error"]


def test_insufficient_funds_is_409(client, member):
    checking, savings = _accounts(member)
    r = client.post("/api/v1/transfers", json={"from_account": savings["account_id"],
                                               "to_account": checking["account_id"], "amount": "999999.00",
                                               "idempotency_key": f"big-{uuid.uuid4()}"})
    assert r.status_code == 409 and r.get_json()["error"] == "insufficient_funds"


def test_other_users_account_is_404(client, member):
    other = make_user()
    try:
        theirs = open_account(other, kind="checking", name="C", institution="B", mask="1", opening_balance="50")
        checking, _ = _accounts(member)
        assert client.get(f"/api/v1/accounts/{theirs}").status_code == 404
        r = client.post("/api/v1/transfers", json={"from_account": theirs, "to_account": checking["account_id"],
                                                   "amount": "5", "idempotency_key": f"x-{uuid.uuid4()}"})
        assert r.status_code == 404
        assert client.get("/api/v1/accounts/not-a-uuid").status_code == 404
        assert client.get(f"/api/v1/transactions/{uuid.uuid4()}").status_code == 404
    finally:
        drop_user(other)


def test_budgets_put_and_validation(client):
    r = client.put("/api/v1/budgets", json={"category": "groceries", "amount": "425"})
    assert r.status_code == 200 and r.get_json()["amount"] == 425.0
    assert client.put("/api/v1/budgets", json={"category": "income", "amount": "5"}).status_code == 400
    assert client.put("/api/v1/budgets", data="not json", content_type="application/json").status_code == 400


def test_goals_recurring_cashflow(client, member):
    goals = client.get("/api/v1/goals").get_json()["items"]
    assert goals
    checking, _ = _accounts(member)
    r = client.post(f"/api/v1/goals/{goals[0]['goal_id']}/contribute",
                    json={"from_account": checking["account_id"], "amount": "5.00",
                          "idempotency_key": f"g-{uuid.uuid4()}"})
    assert r.status_code == 201
    assert client.post(f"/api/v1/goals/{uuid.uuid4()}/contribute",
                       json={"from_account": checking["account_id"], "amount": "5.00",
                             "idempotency_key": "nope"}).status_code == 404
    assert "items" in client.get("/api/v1/recurring").get_json()
    assert len(client.get("/api/v1/cashflow?months=3").get_json()["months"]) == 3


def test_ai_shows_the_persona(monkeypatch, member):
    c = app_client(monkeypatch, AI_SERVICE="ollama", AI_DISPLAY_PROVIDER="watsonx",
                   AI_DISPLAY_MODEL="ibm/granite-3-3-8b-instruct")
    body = c.get("/api/v1/ai").get_json()
    assert body["provider"] == "watsonx" and body["model"] == "ibm/granite-3-3-8b-instruct"
    assert "ollama" not in str(body).lower()


def test_ai_model_switch_goes_through_the_shared_switch(monkeypatch):
    c = app_client(monkeypatch, AI_SERVICE="ollama", AI_DISPLAY_PROVIDER="watsonx",
                   AI_DISPLAY_MODEL="ibm/granite-3-3-8b-instruct")
    ok = c.post("/api/v1/ai/model", json={"model": "ibm/granite-3-3-8b-instruct"})
    assert ok.status_code == 200 and ok.get_json()["success"] is True
    assert c.post("/api/v1/ai/model", json={"model": "not-a-model"}).status_code == 400
    assert c.post("/api/v1/ai/model", json={}).status_code == 400


def test_admin_reset_all_needs_confirmation(client):
    assert client.post("/api/v1/admin/reset", json={"all": True}).status_code == 400
    assert client.post("/api/v1/admin/reset", json={"persona": "nobody-here"}).status_code == 404
    assert client.post("/api/v1/admin/seed", json={"users": 0}).status_code == 400


def test_api_v1_is_gated_with_json_503_while_warming_up(monkeypatch):
    from banko_ai.config.settings import reset_config
    from banko_ai.web.app import create_app

    monkeypatch.setenv("AI_SERVICE", "ollama")
    monkeypatch.setenv("DATABASE_URL", "postgresql://root@203.0.113.1:26257/defaultdb?sslmode=disable")
    monkeypatch.setenv("DB_CONNECT_TIMEOUT", "1")
    monkeypatch.setenv("WARMUP_WAIT_SECONDS", "1")
    reset_config()
    started = time.time()
    app = create_app()
    _apps.append(app)
    assert time.time() - started < 5
    r = app.test_client().get("/api/v1/accounts")
    assert r.status_code == 503 and r.get_json()["error"] == "warming_up"
