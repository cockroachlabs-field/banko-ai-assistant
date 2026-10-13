"""Receipts post through the ledger, catch duplicates, and fail honestly."""

import io
import os
from datetime import date

import pytest
from PIL import Image
from sqlalchemy import text

from banko_ai.db.engine import get_engine
from banko_ai.services import accounts, receipts
from banko_ai.services._sql import vector_floats, vector_text
from banko_ai.services.seeding import seed_user
from tests._money import drop_user, invariant_violations, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")

FIELDS = {"merchant": "Trader Joe's", "amount": 23.47, "date": str(date.today()), "category": "groceries",
          "items": ["Pinot Noir wine", "Sourdough bread"], "payment_method": "credit card"}


def _hooks(fields=None, ocr_text="TRADER JOE'S TOTAL 23.47"):
    return receipts.ReceiptHooks(ocr=lambda path: ocr_text, extract=lambda text_: dict(fields or FIELDS),
                                 fraud=lambda u, e: "No issues detected", budget=lambda u: "Within budget")


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "r.png"
    buf = io.BytesIO()
    Image.new("RGB", (800, 1200), "white").save(buf, "PNG")
    path.write_bytes(buf.getvalue())
    return str(path)


@pytest.fixture(scope="module")
def member():
    uid = make_user("balanced")
    seed_user(uid, months=1, today=date.today())
    yield uid
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM receipts WHERE user_id = CAST(:u AS UUID)"), {"u": uid})
    drop_user(uid)


def _run(user, path, hooks):
    return list(receipts.process_receipt(user, path, "r.png", hooks=hooks))


def test_receipt_posts_through_the_ledger(member, image):
    before = {a["account_id"]: a["balance"] for a in accounts.list_accounts(member)["accounts"]}
    events = _run(member, image, _hooks())
    steps = [(e["step"], e["status"]) for e in events if e["type"] == "step" and e["status"] != "start"]
    assert steps == [("ocr", "done"), ("extract", "done"), ("category", "done"), ("duplicate", "done"),
                     ("post", "done"), ("fraud", "done"), ("budget", "done")]
    final = events[-1]
    assert final["type"] == "receipt" and final["status"] == "posted" and final["merchant"] == "Trader Joe's"
    assert final["category"] == "Groceries"
    with get_engine().connect() as c:
        entry = c.execute(text("SELECT amount, kind FROM ledger_entries WHERE expense_id = CAST(:x AS UUID)"),
                          {"x": final["expense_id"]}).fetchone()
        thumb = c.execute(text("SELECT length(thumbnail) FROM receipts WHERE document_id = CAST(:d AS UUID)"),
                          {"d": final["document_id"]}).scalar()
    assert (float(entry[0]), entry[1]) == (-23.47, "purchase")
    assert 0 < thumb <= 256 * 1024
    after = {a["account_id"]: a["balance"] for a in accounts.list_accounts(member)["accounts"]}
    assert sum(before.values()) - sum(after.values()) == pytest.approx(23.47, abs=0.01)
    assert invariant_violations(member) == []


def test_the_same_receipt_twice_is_a_duplicate(member, image):
    fields = dict(FIELDS, merchant="Twin Grocer", amount=31.10)
    first = _run(member, image, _hooks(fields))[-1]
    second = _run(member, image, _hooks(fields))
    final = second[-1]
    assert final["status"] == "duplicate" and final["expense_id"] == first["expense_id"]
    assert ("post", "skipped") in [(e["step"], e["status"]) for e in second if e["type"] == "step"]


def test_placeholder_payload_fails_neutrally(member, image):
    junk = {"merchant": "store name", "amount": 0.0, "date": "YYYY-MM-DD", "category": "food|transport"}
    events = _run(member, image, _hooks(junk))
    assert events[-1]["type"] == "error" and "template" in events[-1]["message"]
    with get_engine().connect() as c:
        status = c.execute(text("SELECT status FROM receipts WHERE user_id = CAST(:u AS UUID) "
                                "ORDER BY created_at DESC LIMIT 1"), {"u": member}).scalar()
    assert status == "failed"


def test_unreadable_image_fails(member, image):
    events = _run(member, image, _hooks(ocr_text="   "))
    assert events[-1] == {"type": "error", "message": "We couldn't read any text on that receipt."}


def test_a_user_without_accounts_gets_a_card(image):
    uid = make_user("balanced")
    try:
        final = _run(uid, image, _hooks(dict(FIELDS, merchant="First Shop", amount=12.00)))[-1]
        assert final["status"] == "posted"
        names = [a["name"] for a in accounts.list_accounts(uid)["accounts"]]
        assert names == ["Banko Card"]
        assert invariant_violations(uid) == []
    finally:
        with get_engine().begin() as c:
            c.execute(text("DELETE FROM receipts WHERE user_id = CAST(:u AS UUID)"), {"u": uid})
        drop_user(uid)


def test_failed_side_checks_do_not_undo_the_purchase(member, image):
    def broken_fraud(user_id, expense_id):
        raise RuntimeError("fraud service down")

    hooks = _hooks(dict(FIELDS, merchant="Side Check Shop", amount=7.25))
    hooks = receipts.ReceiptHooks(ocr=hooks.ocr, extract=hooks.extract, fraud=broken_fraud, budget=hooks.budget)
    events = _run(member, image, hooks)
    assert ("fraud", "failed") in [(e["step"], e["status"]) for e in events if e["type"] == "step"]
    assert events[-1]["status"] == "posted"


def test_vector_floats_is_the_inverse_of_vector_text():
    assert vector_floats(vector_text([0.5, 0.25])) == [0.5, 0.25]
