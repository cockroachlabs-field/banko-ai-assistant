"""Versioned JSON API over the money services. Every route answers for the
signed-in persona, or the default persona when there is no session (curl
demos and the pipeline). Money leaves as numbers rounded to cents, dates as
ISO strings; errors are neutral and never include exception text."""

from __future__ import annotations

import logging
import threading
from datetime import date

from flask import Blueprint, Response, current_app, jsonify, request, session
from sqlalchemy import text

from ..config.catalog import get_catalog
from ..config.settings import get_config
from ..db.engine import get_engine
from ..services import accounts, budgets, goals, ledger, recurring, reports, seeding, transactions
from ..services._sql import jsonable, valid_uuid

api_v1 = Blueprint("api_v1", __name__, url_prefix="/api/v1")
log = logging.getLogger("banko.api")


class _BadRequest(Exception):
    pass


def _uid() -> str:
    return session.get("user_id") or get_config().coach_default_user_id


def _body() -> dict:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise _BadRequest("Send a JSON object")
    return data


def _date_arg(name: str) -> date | None:
    raw = request.args.get(name)
    if not raw:
        return None
    try:
        day = date.fromisoformat(raw)
    except ValueError:
        raise _BadRequest(f"{name} must look like 2026-10-01") from None
    # Far-off years overflow the date math downstream (to + 1 day).
    if not 1900 <= day.year <= 2999:
        raise _BadRequest(f"{name} must be a date between 1900 and 2999")
    return day


def _int_arg(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(request.args.get(name, default))))
    except ValueError:
        raise _BadRequest(f"{name} must be a whole number") from None


@api_v1.errorhandler(_BadRequest)
def _bad(err):
    return jsonify({"error": str(err)}), 400


@api_v1.errorhandler(ledger.InsufficientFunds)
def _poor(err):
    return jsonify({"error": "insufficient_funds"}), 409


@api_v1.errorhandler(ledger.NotFound)
def _missing(err):
    return jsonify({"error": "not_found"}), 404


@api_v1.errorhandler(ledger.LedgerError)
def _refused(err):
    return jsonify({"error": str(err)}), 400


def _transfer_response(result: ledger.TransferResult):
    if result.status == "ok":
        return jsonify({"status": "ok", "transfer_id": result.transfer_id}), 201
    if result.status == "replay":
        return jsonify({"status": "replay", "transfer_id": result.transfer_id}), 200
    if result.status == "insufficient_funds":
        return jsonify({"error": "insufficient_funds"}), 409
    if result.status == "key_conflict":
        # The key was already used for a different amount, account pair or kind.
        return jsonify({"error": "key_conflict"}), 409
    return jsonify({"error": "not_found"}), 404


@api_v1.get("/me")
def me():
    uid = _uid()
    with get_engine().connect() as conn:
        row = conn.execute(text("""
            SELECT user_id, username, display_name, avatar_color, spending_style, home_region
            FROM users WHERE user_id = CAST(:u AS UUID)
        """), {"u": uid}).fetchone()
    if row is None:
        return jsonify({"error": "not_found"}), 404
    return jsonify(jsonable({"user_id": row[0], "username": row[1], "display_name": row[2] or row[1],
                             "avatar_color": row[3], "spending_style": row[4], "home_region": row[5],
                             "seeding": seeding.seeding_status(uid)}))


@api_v1.get("/ai")
def ai():
    shared = current_app.extensions["banko_ai"]
    shown = shared["shown"]()
    try:
        models = shared["models"]()
    except Exception:  # a live model list can fail; the identity still answers
        log.warning("model list unavailable for /api/v1/ai", exc_info=True)
        models = []
    return jsonify({"provider": shown.provider, "name": shown.name, "model": shown.model,
                    "icon": shown.icon_file, "models": models})


@api_v1.post("/ai/model")
def ai_model():
    model = _body().get("model")
    if not isinstance(model, str) or not model:
        raise _BadRequest("Pick a model")
    payload, status = current_app.extensions["banko_ai"]["switch_model"](model)
    return jsonify(payload), status


@api_v1.get("/dashboard")
def dashboard():
    return jsonify(reports.dashboard(_uid()))


@api_v1.get("/accounts")
def list_accounts():
    return jsonify(accounts.list_accounts(_uid()))


@api_v1.get("/accounts/<account_id>")
def get_account(account_id: str):
    found = accounts.get_account(_uid(), account_id)
    return (jsonify(found), 200) if found else (jsonify({"error": "not_found"}), 404)


@api_v1.get("/accounts/<account_id>/statements")
def statement(account_id: str):
    uid = _uid()
    month = request.args.get("month")
    if not month:
        if accounts.get_account(uid, account_id) is None:
            return jsonify({"error": "not_found"}), 404
        return jsonify({"months": accounts.statement_months(uid, account_id)})
    try:
        st = accounts.statement(uid, account_id, month)
    except ValueError:
        raise _BadRequest("month must look like 2026-09") from None
    if st is None:
        return jsonify({"error": "not_found"}), 404
    if request.args.get("format") == "csv":
        name = f"statement-{st['account']['mask']}-{month}.csv"
        return Response(accounts.statement_csv(st), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={name}"})
    return jsonify(st)


@api_v1.get("/transactions")
def list_transactions():
    uid = _uid()
    q = request.args.get("q")
    if q:
        return jsonify(transactions.semantic_search(
            uid, q, limit=_int_arg("limit", 20, 1, 200), category=request.args.get("category"),
            date_from=_date_arg("from"), date_to=_date_arg("to")))
    amounts = {name: ledger.money(request.args[name]) for name in ("min", "max") if request.args.get(name)}
    return jsonify(transactions.list_transactions(
        uid, limit=_int_arg("limit", 50, 1, 200), cursor=request.args.get("cursor"),
        category=request.args.get("category"), account_id=request.args.get("account"),
        kind=request.args.get("kind"), date_from=_date_arg("from"), date_to=_date_arg("to"),
        min_amount=amounts.get("min"), max_amount=amounts.get("max")))


@api_v1.get("/transactions/<txn_id>")
def transaction_detail(txn_id: str):
    found = transactions.transaction_detail(_uid(), txn_id)
    return (jsonify(found), 200) if found else (jsonify({"error": "not_found"}), 404)


@api_v1.get("/cashflow")
def cashflow():
    return jsonify(reports.cashflow(_uid(), months=_int_arg("months", 6, 1, 24)))


@api_v1.get("/budgets")
def get_budgets():
    return jsonify(budgets.budget_progress(_uid()))


@api_v1.put("/budgets")
def put_budget():
    body = _body()
    category = body.get("category")
    if not isinstance(category, str) or not category.strip():
        raise _BadRequest("Pick a category")
    return jsonify(budgets.set_budget(_uid(), category, body.get("amount")))


@api_v1.get("/recurring")
def get_recurring():
    return jsonify({"items": recurring.recurring_json(recurring.detect_recurring(_uid()))})


@api_v1.get("/goals")
def get_goals():
    return jsonify({"items": goals.list_goals(_uid())})


def _key(body: dict) -> str:
    key = body.get("idempotency_key")
    if not isinstance(key, str) or not key or len(key) > 128:
        raise _BadRequest("Send an idempotency_key of at most 128 characters")
    return key


def _account_arg(body: dict, name: str) -> str:
    value = body.get(name)
    if not isinstance(value, str) or not valid_uuid(value):
        raise ledger.NotFound("That account was not found")
    return value


@api_v1.post("/goals/<goal_id>/contribute")
def contribute(goal_id: str):
    body = _body()
    return _transfer_response(goals.contribute(_uid(), goal_id, _account_arg(body, "from_account"),
                                               body.get("amount"), idempotency_key=_key(body)))


@api_v1.post("/transfers")
def make_transfer():
    body = _body()
    kind = body.get("kind", "transfer")
    if kind not in ("transfer", "card_payment"):
        raise _BadRequest("kind must be transfer or card_payment")
    return _transfer_response(ledger.transfer(_uid(), _account_arg(body, "from_account"),
                                              _account_arg(body, "to_account"), body.get("amount"),
                                              idempotency_key=_key(body), kind=kind))


@api_v1.post("/admin/reset")
def admin_reset():
    body = _body()
    if body.get("all"):
        if body.get("confirm") != "all users":
            raise _BadRequest('Resetting every demo persona needs "confirm": "all users"')
        targets = [p.user_id for p in get_catalog().personas]
    else:
        targets = [p.user_id for p in get_catalog().personas if p.username == body.get("persona")]
        if not targets:
            return jsonify({"error": "not_found"}), 404
    url = get_config().database_url

    def run():
        for uid in targets:
            try:
                seeding.reset_user(uid, url=url)
            except Exception:
                log.exception("reset of %s failed; moving on to the next persona", uid)

    threading.Thread(target=run, daemon=True, name="banko-reset").start()
    return jsonify({"status": "started", "users": len(targets)}), 202


@api_v1.post("/admin/seed")
def admin_seed():
    body = _body()
    users = body.get("users", 0)
    months = body.get("months", get_config().seed_history_months)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (users, months)):
        raise _BadRequest("users and months must be whole numbers")
    if not 1 <= users <= 200 or not 1 <= months <= 24:
        raise _BadRequest("users must be 1 to 200 and months 1 to 24")
    url = get_config().database_url
    threading.Thread(target=lambda: seeding.seed_generated_users(users, months=months, url=url),
                     daemon=True, name="banko-seed-users").start()
    return jsonify({"status": "started", "users": users}), 202
