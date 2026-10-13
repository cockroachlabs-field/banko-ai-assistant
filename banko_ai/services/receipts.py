"""Receipts: the store behind uploads and the assistant's receipt
follow-ups. Task 10 adds the upload pipeline to this module."""

from __future__ import annotations

import io
import json
import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from sqlalchemy import text

from ..assistant import narration
from ..config.catalog import get_catalog
from ..config.settings import get_config
from ..db.engine import get_engine, run_transaction
from ..db.regions import money_region, region_columns
from ..db.rls import request_url
from . import accounts as accounts_service
from . import ledger
from . import transactions as txns
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


log = logging.getLogger("banko.receipts")

ALLOWED_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
                 ".gif": "image/gif", ".bmp": "image/bmp", ".tif": "image/tiff", ".tiff": "image/tiff",
                 ".pdf": "application/pdf"}
UNREADABLE = "We couldn't read any text on that receipt."
TEMPLATE_ECHO = ("The selected model returned the prompt template instead of real receipt values. "
                 "Switch to a chat or instruct model (not a code model) and upload again.")
INVALID = "The receipt fields didn't look right. Try a clearer photo."
NO_FUNDS = "That account doesn't have enough available for this receipt."
FAILED = "Something went wrong on our side. Please try again."
THUMB_MAX = 256 * 1024
_CREDIT_WORDS = ("credit", "visa", "mastercard", "amex", "american express", "discover")


@dataclass
class ReceiptHooks:
    ocr: Callable[[str], str]
    extract: Callable[[str], dict]
    fraud: Callable[[str, str], str]
    budget: Callable[[str], str]


def default_hooks(model: str | None = None, url: str | None = None) -> ReceiptHooks:
    """The real agents: tesseract OCR, LLM field extraction, fraud and budget checks."""
    import json as _json

    from ..agents.llm_factory import get_embedding_model, get_llm_for_agent
    from ..agents.tools.document_tools import create_document_tools, parse_receipt_fields
    from ..db.regions import agent_home_region
    db_url = url or get_config().database_url
    tools = {t.name: t for t in create_document_tools(db_url, get_embedding_model())}
    region = agent_home_region(db_url)

    def ocr(path: str) -> str:
        name = "extract_text_from_pdf" if path.lower().endswith(".pdf") else "extract_text_from_image"
        out = _json.loads(tools[name].func(path))
        return out.get("text", "") if out.get("success") else ""

    def extract(text_value: str) -> dict:
        out = parse_receipt_fields(text_value, get_llm_for_agent(0.1, model_override=model))
        return out.get("fields", {}) if out.get("success") else {}

    def fraud(user_id: str, expense_id: str) -> str:
        from ..agents.fraud_agent import FraudAgent
        agent = FraudAgent(region=region, llm=get_llm_for_agent(0.2, model_override=model), database_url=db_url,
                           embedding_model=get_embedding_model(), fraud_threshold=0.7,
                           duplicate_window_days=get_config().fraud_duplicate_window_days)
        check = agent.analyze_expense(expense_id)
        if not check.get("fraud_detected"):
            return "No issues detected"
        return f"Suspicious transaction (confidence {check.get('confidence', 0):.0%})"

    def budget(user_id: str) -> str:
        from ..agents.budget_agent import BudgetAgent
        agent = BudgetAgent(region=region, llm=get_llm_for_agent(0.2, model_override=model), database_url=db_url,
                            alert_threshold=0.8)
        status = agent.check_budget_status(user_id=user_id,
                                           monthly_budget=get_config().monthly_budget_default).get("status")
        return {"over_budget": "Over budget", "on_pace_to_exceed": "On pace to exceed"}.get(str(status or ""), "Within budget")

    return ReceiptHooks(ocr=ocr, extract=extract, fraud=fraud, budget=budget)


def _thumbnail(path: str) -> bytes | None:
    if path.lower().endswith(".pdf"):
        return None
    try:
        from PIL import Image
        with Image.open(path) as img:
            img = img.convert("RGB")
            img.thumbnail((512, 512))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=70)
        data = buf.getvalue()
        return data if len(data) <= THUMB_MAX else None
    except Exception:
        return None


def _insert(user_id: str, filename: str, mime: str, thumb: bytes | None, url: str | None) -> str:
    def fn(conn) -> str:
        region = money_region(conn, user_id, url)
        rcol, rval = region_columns(region)
        return str(conn.execute(text(f"""
            INSERT INTO receipts ({rcol}user_id, filename, mime, status, thumbnail)
            VALUES ({rval}CAST(:u AS UUID), :f, :m, 'processing', :t) RETURNING document_id
        """), {"u": user_id, "f": filename[:200], "m": mime, "t": thumb, "crdb_region": region}).scalar())
    return run_transaction(fn, url=url, idempotent=False)


_CASTS = {"extracted": "CAST(:extracted AS JSONB)", "expense_id": "CAST(:expense_id AS UUID)"}


def _finish(document_id: str, user_id: str, url: str | None, **cols) -> None:
    """Close out a receipt row. Column names come from this module, never from input."""
    sets = ", ".join(f"{k} = {_CASTS.get(k, ':' + k)}" for k in cols)
    params = {**cols, "d": document_id, "u": user_id}
    if "extracted" in params:
        params["extracted"] = json.dumps(params["extracted"], default=str)
    run_transaction(lambda conn: conn.execute(text(
        f"UPDATE receipts SET {sets} WHERE document_id = CAST(:d AS UUID) AND user_id = CAST(:u AS UUID)"),
        params), url=url)


def _account_for(user_id: str, payment_method: str | None, url: str | None) -> str:
    accts = accounts_service.list_accounts(user_id, url=url)["accounts"]
    if not accts:
        return ledger.open_account(user_id, kind="credit", name="Banko Card", institution="Banko", mask="0000",
                                   opening_balance=0, credit_limit=get_config().receipt_default_card_limit,
                                   payment_method="Credit Card", url=url)
    wants_credit = any(w in (payment_method or "").lower() for w in _CREDIT_WORDS)
    for kind in (("credit", "checking") if wants_credit else ("checking", "credit")):
        for a in accts:
            if a["kind"] == kind:
                return a["account_id"]
    return accts[0]["account_id"]


def process_receipt(user_id: str, path: str, filename: str, *, hooks: ReceiptHooks,
                    url: str | None = None) -> Iterator[dict]:
    from ..agents.receipt_extraction_schema import ReceiptExtraction, is_placeholder_payload
    from ._vector import embed

    url = request_url(url)
    cfg = get_config()
    narration.receipt_uploaded(filename)
    mime = ALLOWED_TYPES.get(Path(filename).suffix.lower(), "application/octet-stream")
    doc = _insert(user_id, filename, mime, _thumbnail(path), url)
    state = {"step": "ocr", "t0": 0.0, "terminal": None, "posted_data": None}

    def start(step: str) -> dict:
        state.update(step=step, t0=time.perf_counter())
        return {"type": "step", "step": step, "status": "start", "detail": "", "ms": None}

    def end(status: str, detail: str = "") -> dict:
        return {"type": "step", "step": state["step"], "status": status, "detail": detail,
                "ms": round((time.perf_counter() - float(state["t0"])) * 1000, 1)}

    def fail(message: str, ocr_text: str | None = None) -> Iterator[dict]:
        narration.receipt_failed(message)
        yield end("failed", message)
        try:
            _finish(doc, user_id, url, status="failed", error=message,
                    **({"ocr_text": ocr_text[:20000]} if ocr_text else {}))
            state["terminal"] = "failed"
        except Exception:
            log.warning("_finish for failed status raised, retrying once", exc_info=True)
            try:
                _finish(doc, user_id, url, status="failed", error=message,
                        **({"ocr_text": ocr_text[:20000]} if ocr_text else {}))
                state["terminal"] = "failed"
            except Exception:
                log.error("_finish retry failed, row may stay processing", exc_info=True)
        yield {"type": "error", "message": message}

    try:
        yield start("ocr")
        ocr_text = (hooks.ocr(path) or "").strip()
        if not ocr_text:
            yield from fail(UNREADABLE)
            return
        yield end("done", f"{len(ocr_text)} characters")

        yield start("extract")
        fields = hooks.extract(ocr_text) or {}
        narration.receipt_fields(fields)
        if is_placeholder_payload(fields):
            yield from fail(TEMPLATE_ECHO, ocr_text)
            return
        try:
            valid = ReceiptExtraction(**fields)
        except Exception:
            yield from fail(INVALID, ocr_text)
            return
        yield end("done", f"{valid.merchant}, ${valid.amount:,.2f}")

        yield start("category")
        category = get_catalog().category_for(valid.category).label
        day = valid.date if valid.date <= datetime.now(timezone.utc).date() else date.today()
        extracted = {"merchant": valid.merchant, "amount": round(valid.amount, 2), "date": str(day),
                     "category": category, "items": valid.items, "payment_method": valid.payment_method}
        yield end("done", category)

        yield start("duplicate")
        emb_text = f"{valid.merchant} {category} " + ", ".join(valid.items)
        qvec = embed(emb_text)
        dupes = txns.duplicate_candidates(user_id, amount=f"{valid.amount:.2f}", day=day, embedding=qvec,
                                          window_days=cfg.duplicate_window_days,
                                          max_distance=cfg.receipt_duplicate_max_distance, url=url)
        if dupes:
            narration.receipt_duplicate(f"matches {dupes[0]['merchant']} on {dupes[0]['date']}")
            yield end("done", f"matches {dupes[0]['merchant']} on {dupes[0]['date']}")
            yield {"type": "step", "step": "post", "status": "skipped", "detail": "duplicate, nothing posted",
                   "ms": None}
            _finish(doc, user_id, url, status="duplicate", extracted=extracted, ocr_text=ocr_text[:20000],
                    expense_id=dupes[0]["expense_id"], embedding=qvec)
            state["terminal"] = "duplicate"
            yield {"type": "receipt", **(get_receipt(user_id, doc, url=url) or {})}
            return
        yield end("done", "no duplicate")

        yield start("post")
        account_id = _account_for(user_id, valid.payment_method, url)
        try:
            from ._sql import vector_floats
            expense_id = ledger.post_purchase(
                user_id, account_id, amount=f"{valid.amount:.2f}", merchant=valid.merchant, category=category,
                description=f"{valid.merchant} receipt: " + (", ".join(valid.items) or category),
                payment_method=valid.payment_method, expense_date=day, embedding=vector_floats(qvec), url=url)
        except ledger.InsufficientFunds:
            yield from fail(NO_FUNDS, ocr_text)
            return

        state["posted_data"] = {"document_id": doc, "status": "posted", "merchant": valid.merchant,
                                "amount": extracted["amount"], "date": extracted["date"], "category": category,
                                "items": extracted["items"], "payment_method": extracted["payment_method"],
                                "expense_id": expense_id, "created_at": datetime.now(timezone.utc)}
        narration.receipt_posted(expense_id)

        try:
            _finish(doc, user_id, url, status="posted", extracted=extracted, ocr_text=ocr_text[:20000],
                    expense_id=expense_id, embedding=qvec)
            state["terminal"] = "posted"
        except Exception:
            log.warning("_finish for posted status raised, retrying once", exc_info=True)
            try:
                _finish(doc, user_id, url, status="posted", extracted=extracted, ocr_text=ocr_text[:20000],
                        expense_id=expense_id, embedding=qvec)
                state["terminal"] = "posted"
            except Exception:
                log.error("_finish retry failed after posting, but money moved", exc_info=True)
                state["terminal"] = "posted"

        yield end("done", f"${valid.amount:,.2f} on {category}")

        for step, check in (("fraud", lambda: hooks.fraud(user_id, expense_id)),
                            ("budget", lambda: hooks.budget(user_id))):
            yield start(step)
            narration.receipt_check(step)
            try:
                result = check()
            except Exception:
                log.warning("receipt %s check failed", step, exc_info=True)
                narration.receipt_check_result("check failed")
                yield end("failed", "check failed")
                continue
            narration.receipt_check_result(result)
            yield end("done", result)

        try:
            yield {"type": "receipt", **(get_receipt(user_id, doc, url=url) or {})}
        except Exception:
            log.warning("get_receipt failed after posting, building from known fields", exc_info=True)
            yield {"type": "receipt", **jsonable(state["posted_data"])}
    except Exception:
        if state.get("posted_data"):
            log.error("exception after posting, money moved, reporting success", exc_info=True)
            try:
                yield {"type": "receipt", **jsonable(state["posted_data"])}
            except Exception:
                log.error("failed to yield posted receipt data", exc_info=True)
        else:
            log.warning("receipt pipeline failed at %s", state["step"], exc_info=True)
            yield from fail(FAILED)
    finally:
        if state["terminal"] is None:
            log.warning("receipt row %s left processing, marking failed", doc)
            try:
                _finish(doc, user_id, url, status="failed", error=FAILED)
            except Exception:
                log.error("finally cleanup failed, row may stay processing", exc_info=True)
