"""What the transaction drawer needs: categories, a direct recategorize,
the linked receipt and its thumbnail, and the agent checks."""

import io
import json
import os
from datetime import date

import pytest
from PIL import Image
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
    uid = make_user("diner")
    seed_user(uid, months=1, today=date.today())
    yield uid
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM receipts WHERE user_id = CAST(:u AS UUID)"), {"u": uid})
    drop_user(uid)


def client_for(monkeypatch, uid):
    c = app_client(monkeypatch, REPORTING_FOLLOWER_READS="false")
    with c.session_transaction() as s:
        s["user_id"] = uid
    return c


def _an_expense(uid):
    with get_engine().connect() as c:
        return str(c.execute(text("SELECT expense_id FROM expenses WHERE user_id = CAST(:u AS UUID) LIMIT 1"),
                             {"u": uid}).scalar())


def test_categories(monkeypatch, member):
    items = client_for(monkeypatch, member).get("/api/v1/categories").get_json()["items"]
    assert any(i["label"] == "Groceries" for i in items)
    assert {"key", "label", "icon", "color", "group"} <= set(items[0])


def test_recategorize_from_the_drawer(monkeypatch, member):
    c = client_for(monkeypatch, member)
    eid = _an_expense(member)
    got = c.patch(f"/api/v1/transactions/{eid}", json={"category": "Entertainment"})
    assert got.status_code == 200 and got.get_json()["category"] == "Entertainment"
    assert c.patch(f"/api/v1/transactions/{eid}", json={"category": "Nope"}).status_code == 400
    other = make_user()
    try:
        assert client_for(monkeypatch, other).patch(f"/api/v1/transactions/{eid}", json={"category": "Groceries"}).status_code == 404
    finally:
        drop_user(other)


def test_detail_carries_receipt_and_checks(monkeypatch, member):
    eid = _an_expense(member)
    buf = io.BytesIO()
    Image.new("RGB", (40, 50), "white").save(buf, "JPEG")
    with get_engine().begin() as c:
        doc = c.execute(text("INSERT INTO receipts (user_id, status, expense_id, extracted, thumbnail) VALUES "
                             "(CAST(:u AS UUID), 'posted', CAST(:x AS UUID), CAST(:e AS JSONB), :t) RETURNING document_id"),
                        {"u": member, "x": eid, "e": json.dumps({"merchant": "Corner Market", "items": ["Bread"]}),
                         "t": buf.getvalue()}).scalar()
    c = client_for(monkeypatch, member)
    detail = c.get(f"/api/v1/transactions/{eid}").get_json()
    assert detail["receipt"]["document_id"] == str(doc) and detail["receipt"]["has_thumbnail"]
    assert set(detail["checks"]) == {"duplicates", "decisions"}
    thumb = c.get(f"/api/v1/receipts/{doc}/thumbnail")
    assert thumb.status_code == 200 and thumb.mimetype == "image/jpeg"
    other = make_user()
    try:
        assert client_for(monkeypatch, other).get(f"/api/v1/receipts/{doc}/thumbnail").status_code == 404
    finally:
        drop_user(other)


def test_merchant_markup_is_text_not_html():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "banko_ai" / "templates"
    for name in ("pages/transactions.html", "components/transaction_drawer.html"):
        assert "x-html" not in (root / name).read_text(), name
