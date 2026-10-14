"""Presenter-only simulated pipeline signals. Each one is built from the
signed-in user's real data, so the coach agent that reacts to it reasons
over real budgets, purchases and bills. Payloads follow PIPELINE_CONTRACT.md."""

from __future__ import annotations

import uuid
from datetime import date, timedelta

from ..services import budgets, recurring, transactions
from .signals import Signal, SignalType

# A budget is near its limit from 80% spent, where the pipeline's warn level starts.
NEAR_LIMIT = 0.8


class SimulateUnavailable(Exception):
    """This user's data cannot support the requested signal; the message says why."""


def _budget_threshold(user_id: str, url: str | None) -> tuple[str, dict]:
    progress = budgets.budget_progress(user_id, url=url)
    items = sorted(progress["items"], key=lambda i: i["pct"], reverse=True)
    if not items:
        raise SimulateUnavailable("Set a budget first; the coach reacts to a budget near its limit.")
    top = items[0]
    if top["pct"] < NEAR_LIMIT:
        raise SimulateUnavailable(f"No budget is near its limit yet. The closest is {top['category']}, "
                                  f"at {round(top['pct'] * 100)}% spent.")
    days = progress["days_in_month"] - progress["day"]
    severity = "critical" if top["pct"] >= 1 else "warn"
    return severity, {"category": top["key"], "pct_used": round(float(top["pct"]), 4),
                      "monthly_budget": float(top["budget"]), "spent_so_far": float(top["spent"]),
                      "days_remaining": days}


def _anomaly(user_id: str, url: str | None) -> tuple[str, dict]:
    since = (date.today() - timedelta(days=60)).isoformat()
    items = [t for t in transactions.list_transactions(user_id, limit=200, kind="purchase",
                                                       date_from=date.fromisoformat(since), url=url)["items"]
             if t.get("expense_id")]
    if len(items) < 3:
        raise SimulateUnavailable("There aren't enough recent purchases to call one unusual.")
    amounts = sorted(abs(float(t["amount"])) for t in items)
    biggest = max(items, key=lambda t: abs(float(t["amount"])))
    median = amounts[len(amounts) // 2]
    spread = (sum((a - median) ** 2 for a in amounts) / len(amounts)) ** 0.5 or 1.0
    amount = abs(float(biggest["amount"]))
    return "warn", {"merchant": biggest.get("merchant") or biggest.get("description"), "amount": amount,
                    "expected_max": round(median + 2 * spread, 2), "z_score": round((amount - median) / spread, 2),
                    "transaction_id": biggest["expense_id"]}


def _recurring_drift(user_id: str, url: str | None) -> tuple[str, dict]:
    bills = recurring.detect_recurring(user_id, url=url)
    if not bills:
        raise SimulateUnavailable("No recurring bills were found in this user's history yet.")
    raised = [b for b in bills if (b.price_change_pct or 0) > 0 and b.previous_amount > 0]
    if not raised:
        raise SimulateUnavailable("None of the recurring bills went up in price.")
    bill = max(raised, key=lambda b: b.price_change_pct or 0)
    old, new = float(bill.previous_amount), float(bill.amount)
    pct = round((new - old) / old, 4)
    return "info", {"subscription": bill.merchant, "old_amount": old, "new_amount": new, "pct_change": pct,
                    "merchant_id": str(uuid.uuid5(uuid.NAMESPACE_URL, "banko-merchant:" + bill.merchant))}


_BUILDERS = {SignalType.BUDGET_THRESHOLD: _budget_threshold, SignalType.ANOMALY: _anomaly,
             SignalType.RECURRING_DRIFT: _recurring_drift}


def build_signal(user_id: str, signal_type: str, *, url: str | None = None) -> Signal:
    kind = SignalType(signal_type)  # ValueError for an unknown type
    severity, payload = _BUILDERS[kind](user_id, url)
    signal_id = str(uuid.uuid4())
    return Signal.from_dict({"signal_id": signal_id, "user_id": user_id, "signal_type": kind.value,
                             "severity": severity, "payload": payload,
                             "idempotency_key": f"simulated:{signal_id}"})
