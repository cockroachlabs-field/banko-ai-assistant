"""Validation layer for LLM-extracted receipt fields.

Sits between the receipt agent's LLM response and DB persistence. Catches the
common failure mode where a code-tuned or otherwise unsuitable model echoes
the prompt template back instead of extracting real values.
"""
from __future__ import annotations

import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

_PLACEHOLDER_DATE_STRINGS = frozenset({
    "YYYY-MM-DD",
    "yyyy-mm-dd",
    "2024-01-01",
})
_PLACEHOLDER_MERCHANTS = frozenset({
    "store name",
    "merchant",
    "merchant name",
    "store",
    "unknown",
})
_PLACEHOLDER_CATEGORY_FRAGMENTS = (
    "or transportation or",
    "or entertainment or",
    "category1",
    "categoryname",
    "one of:",  # the prompt's "category": "one of: <the catalog's labels>"
)


def _category_echo(category: str) -> bool:
    """The category is the prompt's template, not a choice: a known
    fragment, or the list of labels itself (two commas or more)."""
    return any(f in category for f in _PLACEHOLDER_CATEGORY_FRAGMENTS) or category.count(",") >= 2
_PLACEHOLDER_PAYMENT_FRAGMENTS = (
    "or debit card or cash or",
    "or cash or other",
    "paymentmethod",
)
_PLACEHOLDER_ITEMS = frozenset({"item1", "item2", "itemname"})


class ReceiptExtraction(BaseModel):
    """Strict shape for receipt fields the LLM returned.

    Designed to reject prompt-template echoes (`"date": "YYYY-MM-DD"`,
    `"amount": 0.0`, `"merchant": "store name"`) BEFORE they reach the DB
    layer, where they would surface as opaque driver errors.
    """

    merchant: str = Field(min_length=1)
    amount: float = Field(gt=0)
    # Many receipts print no payment method, and some no date a model can
    # read; the prompt asks for null then. A missing date is today, a
    # missing payment method None (the ledger uses the account's own).
    date: datetime.date = Field(default_factory=datetime.date.today)
    category: str = Field(min_length=1)
    items: list[str] = Field(default_factory=list)
    payment_method: str | None = None

    @field_validator("items", mode="before")
    @classmethod
    def normalize_null_items(cls, v: Any) -> Any:
        # Some providers (Claude on Bedrock observed 2026-05-22) emit explicit
        # JSON null when no line items were parsed instead of omitting the key
        # or returning []. Treat null the same as an empty list.
        if v is None:
            return []
        return v

    @field_validator("date", mode="before")
    @classmethod
    def reject_placeholder_date(cls, v: Any) -> Any:
        if v is None or (isinstance(v, str) and not v.strip()):
            return datetime.date.today()
        if isinstance(v, str) and v.strip() in _PLACEHOLDER_DATE_STRINGS:
            raise ValueError(f"date {v!r} is a prompt-template placeholder")
        return v

    @field_validator("payment_method", mode="before")
    @classmethod
    def blank_payment_method_is_none(cls, v: Any) -> Any:
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("merchant")
    @classmethod
    def reject_placeholder_merchant(cls, v: str) -> str:
        if v.strip().lower() in _PLACEHOLDER_MERCHANTS:
            raise ValueError(f"merchant {v!r} is a prompt-template placeholder")
        return v


def is_placeholder_payload(payload: dict[str, Any]) -> bool:
    """Heuristic: does this whole payload look like the LLM regurgitated the
    prompt scaffold? True when two or more fields match known placeholder
    signatures. Cheaper than catching ValidationError per field and lets the
    caller log one clean diagnostic instead of a stack trace."""
    hits = 0
    if str(payload.get("date", "")).strip() in _PLACEHOLDER_DATE_STRINGS:
        hits += 1
    if str(payload.get("merchant", "")).strip().lower() in _PLACEHOLDER_MERCHANTS:
        hits += 1
    if _category_echo(str(payload.get("category", "")).lower()):
        hits += 1
    pm = str(payload.get("payment_method", "")).lower()
    if any(f in pm for f in _PLACEHOLDER_PAYMENT_FRAGMENTS):
        hits += 1
    items = payload.get("items") or []
    if isinstance(items, list) and any(
        str(i).strip().lower() in _PLACEHOLDER_ITEMS for i in items
    ):
        hits += 1
    return hits >= 2
