"""Receipts: the store behind uploads and the assistant's receipt
follow-ups. Task 10 adds the upload pipeline to this module."""

from __future__ import annotations

import json

from sqlalchemy import text

from ..db.engine import get_engine
from ..db.rls import request_url
from ._sql import jsonable, valid_uuid

_COLS = "document_id, status, filename, extracted, expense_id, created_at, error"


def _row(r) -> dict:
    fields = r[3] if isinstance(r[3], dict) else (json.loads(r[3]) if r[3] else {})
    return jsonable({"document_id": str(r[0]), "status": r[1], "filename": r[2],
                     "merchant": fields.get("merchant"), "amount": fields.get("amount"),
                     "date": fields.get("date"), "category": fields.get("category"),
                     "items": fields.get("items") or [], "payment_method": fields.get("payment_method"),
                     "expense_id": str(r[4]) if r[4] else None, "created_at": r[5], "error": r[6]})


def get_receipt(user_id: str, document_id: str, *, url: str | None = None) -> dict | None:
    if not valid_uuid(document_id):
        return None
    url = request_url(url)
    with get_engine(url).connect() as conn:
        r = conn.execute(text(f"SELECT {_COLS} FROM receipts WHERE document_id = CAST(:d AS UUID) "
                              "AND user_id = CAST(:u AS UUID)"), {"d": document_id, "u": user_id}).fetchone()
    return _row(r) if r else None


def latest_receipt(user_id: str, *, url: str | None = None) -> dict | None:
    url = request_url(url)
    with get_engine(url).connect() as conn:
        r = conn.execute(text(f"""
            SELECT {_COLS} FROM receipts@receipts_user_idx
            WHERE user_id = CAST(:u AS UUID) AND status IN ('posted', 'duplicate')
            ORDER BY created_at DESC LIMIT 1
        """), {"u": user_id}).fetchone()
    return _row(r) if r else None
