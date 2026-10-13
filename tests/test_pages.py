"""Every page renders for a signed-in persona, inside the shell, with its
own component and no external assets."""

import os
import re
from datetime import date

import pytest
from sqlalchemy import text

from banko_ai.db.engine import get_engine
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
    uid = make_user("balanced")
    seed_user(uid, months=2, today=date.today())
    yield uid
    drop_user(uid)


def signed_in(monkeypatch, user_id):
    client = app_client(monkeypatch, REPORTING_FOLLOWER_READS="false")
    with client.session_transaction() as s:
        s["user_id"] = user_id
        s["username"] = "tester"
    return client


def assert_page(resp, page: str):
    assert resp.status_code == 200, resp.status_code
    html = resp.get_data(as_text=True)
    assert f'data-page="{page}"' in html
    assert f"js/pages/{page}.js" in html
    assert not re.search(r'(src|href)="https?://', html)
    assert "Math.random" not in html
    return html


def test_dashboard_page(monkeypatch, member):
    html = assert_page(signed_in(monkeypatch, member).get("/"), "dashboard")
    assert 'x-data="dashboardPage"' in html


def test_dashboard_for_a_user_still_seeding(monkeypatch):
    uid = make_user("balanced")
    try:
        client = signed_in(monkeypatch, uid)
        html = assert_page(client.get("/"), "dashboard")
        assert "Setting up your accounts" in html and "No accounts yet" in html
        data = client.get("/api/v1/dashboard").get_json()
        assert data["recent"] == [] and float(data["net_worth"]) == 0
    finally:
        drop_user(uid)


def test_signed_out_dashboard_goes_to_login(monkeypatch):
    resp = app_client(monkeypatch).get("/")
    assert resp.status_code == 302 and "/login" in resp.headers["Location"]


def test_latest_insight_carries_the_nudge_id(member):
    from banko_ai.services.reports import latest_insight
    with get_engine().begin() as c:
        nid = c.execute(text("INSERT INTO coach_nudges (user_id, message) VALUES (CAST(:u AS UUID), 'Dining is up') "
                             "RETURNING nudge_id"), {"u": member}).scalar()
    try:
        got = latest_insight(member)
        assert got["nudge_id"] == str(nid) and got["message"] == "Dining is up"
    finally:
        with get_engine().begin() as c:
            c.execute(text("DELETE FROM coach_nudges WHERE nudge_id = :n"), {"n": nid})


def test_accounts_pages(monkeypatch, member):
    client = signed_in(monkeypatch, member)
    html = assert_page(client.get("/accounts"), "accounts")
    assert 'x-data="moveMoney"' in html
    acct = client.get("/api/v1/accounts").get_json()["accounts"][0]["account_id"]
    detail = assert_page(client.get(f"/accounts/{acct}"), "account")
    # The controller ruling: tojson inside double quotes breaks the attribute.
    assert f"x-data='accountPage(\"{acct}\"" in detail
    assert client.get("/accounts/00000000-0000-0000-0000-000000000000").status_code == 404


def test_account_history_api(monkeypatch, member):
    client = signed_in(monkeypatch, member)
    items = client.get("/api/v1/accounts/history?months=3").get_json()["items"]
    assert len(items) == 3 and {"month", "net_worth"} <= set(items[0])
    assert client.get("/api/v1/accounts/history?months=abc").status_code == 400


def test_transactions_page(monkeypatch, member):
    html = assert_page(signed_in(monkeypatch, member).get("/transactions"), "transactions")
    assert 'x-data="txDrawer"' in html


def test_cashflow_and_budget_pages(monkeypatch, member):
    client = signed_in(monkeypatch, member)
    assert_page(client.get("/cashflow"), "cashflow")
    html = assert_page(client.get("/budget"), "budget")
    assert "projected" in html.lower()
