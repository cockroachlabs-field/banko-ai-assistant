"""Tests for the ReceiptExtraction Pydantic schema and the
is_placeholder_payload() heuristic."""
import datetime

import pytest
from pydantic import ValidationError

from banko_ai.agents.receipt_extraction_schema import (
    ReceiptExtraction,
    is_placeholder_payload,
)


def test_valid_extraction_parses():
    rx = ReceiptExtraction(
        merchant="Whole Foods Market",
        amount=42.17,
        date=datetime.date(2026, 5, 21),
        category="Groceries",
        items=["organic bananas", "almond milk"],
        payment_method="credit card",
    )
    assert rx.amount == 42.17
    assert rx.date.year == 2026
    assert rx.merchant == "Whole Foods Market"


def test_valid_extraction_with_iso_date_string():
    """LLMs often return dates as ISO strings; pydantic should coerce."""
    rx = ReceiptExtraction(
        merchant="Trader Joes",
        amount=15.42,
        date="2026-05-21",
        category="Groceries",
        items=["bread"],
        payment_method="debit card",
    )
    assert rx.date == datetime.date(2026, 5, 21)


def test_placeholder_date_string_rejected():
    """The 'YYYY-MM-DD' placeholder from prompt templates must not slip past."""
    with pytest.raises(ValidationError, match="placeholder"):
        ReceiptExtraction(
            merchant="Real Merchant",
            amount=10.0,
            date="YYYY-MM-DD",
            category="Groceries",
            items=["item"],
            payment_method="credit card",
        )


def test_zero_amount_rejected():
    """Receipts with $0.00 are almost certainly extraction failures."""
    with pytest.raises(ValidationError):
        ReceiptExtraction(
            merchant="Real Merchant",
            amount=0.0,
            date=datetime.date(2026, 5, 21),
            category="Groceries",
            items=["item"],
            payment_method="credit card",
        )


def test_placeholder_merchant_rejected():
    with pytest.raises(ValidationError, match="placeholder"):
        ReceiptExtraction(
            merchant="store name",
            amount=10.0,
            date=datetime.date(2026, 5, 21),
            category="Groceries",
            items=["item"],
            payment_method="credit card",
        )


def test_empty_items_accepted_when_other_fields_real():
    """A real receipt can legitimately have no parsed line items: tip slips,
    single-line totals, and lower-res scans where the model only recovers the
    grand total. Observed 2026-05-22 with OpenAI on a parking receipt
    ('FL-POINTE ORLANDO - PBR', $96.82, real date, real payment_method) — the
    extraction was correct, items was just []. Empty items alone is too weak a
    signal to reject; the is_placeholder_payload heuristic still flags
    template-echo cases via the _PLACEHOLDER_ITEMS set + the 2-hit threshold."""
    rx = ReceiptExtraction(
        merchant="FL-POINTE ORLANDO - PBR",
        amount=96.82,
        date=datetime.date(2025, 10, 6),
        category="Transport",
        items=[],
        payment_method="credit card",
    )
    assert rx.items == []
    assert rx.amount == 96.82


def test_items_field_defaults_to_empty_list_when_omitted():
    """Some provider response shapes omit the key entirely rather than
    returning []. Default factory keeps the schema permissive without making
    the field optional in the type sense."""
    rx = ReceiptExtraction(
        merchant="Quick Stop",
        amount=5.00,
        date=datetime.date(2026, 5, 22),
        category="Groceries",
        payment_method="cash",
    )
    assert rx.items == []


def test_items_null_coerced_to_empty_list():
    """Claude on Bedrock (observed 2026-05-22 with Haiku 4.5) returns
    explicit JSON null for items when no line items were parsed, instead of
    omitting the key or returning []. The default_factory only fires on
    omission, so we need a before-validator that coerces None -> []. Without
    this, legitimate receipts with no parseable line items 422 on a
    list_type ValidationError."""
    rx = ReceiptExtraction(
        merchant="FL-POINTE ORLANDO - PBR",
        amount=96.82,
        date=datetime.date(2025, 10, 6),
        category="Entertainment",
        items=None,
        payment_method="credit card",
    )
    assert rx.items == []


def test_is_placeholder_payload_full_template_echo():
    """The exact payload pattern observed during the 2026-05-21 watsonx smoke
    with ibm/granite-8b-code-instruct."""
    payload = {
        "merchant": "store name",
        "amount": 0.0,
        "date": "YYYY-MM-DD",
        "category": "food or transportation or entertainment or shopping or services or other",
        "items": ["item1", "item2"],
        "payment_method": "credit card or debit card or cash or other",
    }
    assert is_placeholder_payload(payload) is True


def test_is_placeholder_payload_real_receipt_not_flagged():
    payload = {
        "merchant": "Whole Foods",
        "amount": 42.17,
        "date": "2026-05-21",
        "category": "Groceries",
        "items": ["organic bananas", "almond milk"],
        "payment_method": "credit card",
    }
    assert is_placeholder_payload(payload) is False


def test_is_placeholder_payload_single_hit_not_flagged():
    """One placeholder field could just be a weird real receipt. Two or more
    is the threshold."""
    payload = {
        "merchant": "Real Store",
        "amount": 15.42,
        "date": "YYYY-MM-DD",  # only this is a placeholder
        "category": "Groceries",
        "items": ["bread"],
        "payment_method": "credit card",
    }
    assert is_placeholder_payload(payload) is False


def test_is_placeholder_payload_catches_the_catalog_template_echo():
    """The prompt now lists the catalog's labels; a model that echoes the
    template back gives that list as the category."""
    from banko_ai.config.catalog import get_catalog
    labels = ", ".join(get_catalog().spending_labels())
    payload = {
        "merchant": "store name",
        "amount": 0.0,
        "date": "YYYY-MM-DD",
        "category": f"one of: {labels}",
        "items": ["item1", "item2"],
        "payment_method": "credit card or debit card or cash or other",
    }
    assert is_placeholder_payload(payload) is True
    # The category list alone is one signal; with the item placeholders it is two.
    assert is_placeholder_payload({"merchant": "Real Store", "amount": 9.5, "date": "2026-10-01",
                                   "category": labels, "items": ["item1"], "payment_method": "cash"}) is True
    assert is_placeholder_payload({"merchant": "Real Store", "amount": 9.5, "date": "2026-10-01",
                                   "category": labels, "items": ["bread"], "payment_method": "cash"}) is False


def test_a_missing_date_or_payment_method_is_filled_in():
    """Receipts often print no payment method, and the prompt asks for null
    when a field is not found; neither may fail the receipt."""
    for extra in ({}, {"date": None, "payment_method": None}, {"date": "", "payment_method": "  "}):
        rx = ReceiptExtraction(merchant="Corner Bakery", amount=6.4, category="Restaurant", **extra)
        assert rx.date == datetime.date.today() and rx.payment_method is None
    with pytest.raises(ValidationError, match="placeholder"):
        ReceiptExtraction(merchant="Corner Bakery", amount=6.4, category="Restaurant", date="YYYY-MM-DD")
