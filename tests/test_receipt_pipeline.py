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


def test_default_hooks_extract_reaches_the_llm():
    """Issue #1: default_hooks().extract must work with a real LLM (monkeypatched here)."""
    from unittest.mock import MagicMock

    from banko_ai.agents import llm_factory

    class StubMessage:
        def __init__(self, text):
            self.content = text

    stub_llm = MagicMock()
    stub_llm.invoke.return_value = StubMessage('{"merchant": "Test Cafe", "amount": 12.50, "date": "2026-10-13", '
                                                '"category": "food", "items": ["coffee"], "payment_method": "cash"}')

    original_get_llm = llm_factory.get_llm_for_agent
    try:
        llm_factory.get_llm_for_agent = lambda temp, model_override=None: stub_llm
        hooks = receipts.default_hooks()
        result = hooks.extract("CAFE RECEIPT TOTAL 12.50")
        assert result["merchant"] == "Test Cafe"
        assert result["amount"] == 12.50
    finally:
        llm_factory.get_llm_for_agent = original_get_llm


def test_exception_after_posting_reports_success(member, image):
    """Issue #2: an exception after ledger.post_purchase must not mark the row failed."""
    from unittest.mock import patch

    fields = dict(FIELDS, merchant="Post Then Fail", amount=19.99)
    hooks = _hooks(fields)

    original_get_receipt = receipts.get_receipt
    call_count = {"get_receipt": 0}

    def raise_on_final_call(user_id, document_id, url=None):
        call_count["get_receipt"] += 1
        if call_count["get_receipt"] == 1:
            raise RuntimeError("simulated get_receipt failure after posting")
        return original_get_receipt(user_id, document_id, url=url)

    with patch("banko_ai.services.receipts.get_receipt", side_effect=raise_on_final_call):
        events = _run(member, image, hooks)

    final = events[-1]
    assert final["type"] == "receipt" and final["status"] == "posted"
    assert final["merchant"] == "Post Then Fail"

    with get_engine().connect() as c:
        entry = c.execute(text("SELECT amount FROM ledger_entries WHERE expense_id = CAST(:x AS UUID)"),
                          {"x": final["expense_id"]}).scalar()
    assert float(entry) == -19.99


def test_generator_close_marks_row_failed(member, image):
    """Issue #3: GeneratorExit (client disconnect) must mark the row failed."""
    gen = receipts.process_receipt(member, image, "r.png", hooks=_hooks(dict(FIELDS, merchant="Closed Early", amount=5.00)))
    events = []
    for event in gen:
        events.append(event)
        if event.get("step") == "post" and event.get("status") == "start":
            gen.close()
            break

    with get_engine().connect() as c:
        status = c.execute(text("SELECT status FROM receipts WHERE user_id = CAST(:u AS UUID) "
                                "ORDER BY created_at DESC LIMIT 1"), {"u": member}).scalar()
        ledger_count = c.execute(text("SELECT count(*) FROM ledger_entries WHERE user_id = CAST(:u AS UUID) "
                                      "AND description LIKE '%Closed Early%'"), {"u": member}).scalar()
    assert status == "failed"
    assert ledger_count == 0


def test_fail_itself_raising_still_ends_cleanly(member, image):
    """Issue #3 (optional): if _finish in fail() raises, the finally cleanup still runs."""
    from unittest.mock import patch

    fields = dict(FIELDS, merchant="Fail Raises", amount=3.00)
    hooks = _hooks(fields, ocr_text="   ")

    original_finish = receipts._finish
    call_count = {"finish": 0}

    def raise_twice_then_work(doc_id, user_id, url, **cols):
        call_count["finish"] += 1
        if call_count["finish"] <= 2:
            raise RuntimeError("simulated _finish failure")
        return original_finish(doc_id, user_id, url, **cols)

    with patch("banko_ai.services.receipts._finish", side_effect=raise_twice_then_work):
        events = _run(member, image, hooks)

    assert events[-1]["type"] == "error"
    with get_engine().connect() as c:
        status = c.execute(text("SELECT status FROM receipts WHERE user_id = CAST(:u AS UUID) "
                                "ORDER BY created_at DESC LIMIT 1"), {"u": member}).scalar()
    assert status == "failed"
