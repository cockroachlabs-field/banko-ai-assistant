"""Deterministic demo history for one user, built in memory.

Every random draw comes from an RNG seeded with the user id and the day, so
the same user always gets the same history, and a catch-up run regenerates
the earlier days identically before adding new ones. Nothing here touches
the database; seeding.py writes the plan."""

from __future__ import annotations

import calendar
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal

from ..config.catalog import AccountSpec, Catalog, SeedProfile

CENTS = Decimal("0.01")
NAMESPACE = uuid.UUID("6f1c2a52-6f0e-4a8e-9a8a-7b0d1f3c2e11")


@dataclass(frozen=True)
class PlannedAccount:
    key: str
    kind: str
    name: str
    institution: str
    mask: str
    payment_method: str
    opening_balance: Decimal
    credit_limit: Decimal | None
    apr: Decimal | None


@dataclass(frozen=True)
class PlannedEntry:
    entry_id: str
    account_key: str
    posted_at: datetime
    amount: Decimal          # signed, from the account's point of view
    kind: str                # ledger kind
    description: str
    merchant: str | None = None
    category: str | None = None
    recurring: bool = False
    goal_key: str | None = None
    transfer_ref: str | None = None  # shared by both legs of one movement

    @property
    def is_purchase(self) -> bool:
        return self.kind == "purchase"

    @property
    def expense_id(self) -> str | None:
        return str(uuid.uuid5(NAMESPACE, f"expense:{self.entry_id}")) if self.is_purchase else None


@dataclass(frozen=True)
class PlannedGoal:
    key: str
    name: str
    icon: str
    target_amount: Decimal
    target_date: date
    account_key: str


@dataclass
class SeedPlan:
    accounts: list[PlannedAccount]
    entries: list[PlannedEntry]
    budgets: dict[str, Decimal] = field(default_factory=dict)
    goals: list[PlannedGoal] = field(default_factory=list)

    def entries_after(self, day: date) -> list[PlannedEntry]:
        return [e for e in self.entries if e.posted_at.date() > day]


def _money(value) -> Decimal:
    return Decimal(str(value)).quantize(CENTS, rounding=ROUND_HALF_UP)


def _draw(rng: random.Random, amount) -> Decimal:
    if isinstance(amount, tuple):
        low, high = amount
        return _money(rng.uniform(float(low), float(high)))
    return _money(amount)


def _clamp(year: int, month: int, day: int) -> date:
    return date(year, month, min(day, calendar.monthrange(year, month)[1]))


def add_months(day: date, months: int) -> date:
    total = day.year * 12 + day.month - 1 + months
    return _clamp(total // 12, total % 12 + 1, day.day)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def history_start(today: date, months: int) -> date:
    return add_months(today, -months)


def _due(day: date, day_of_month: int) -> bool:
    return day.day == min(day_of_month, calendar.monthrange(day.year, day.month)[1])


def _months_before(day: date, end: date) -> int:
    return (end.year - day.year) * 12 + end.month - day.month


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _planned_account(user_id: str, spec: AccountSpec) -> PlannedAccount:
    mask = f"{random.Random(f'{user_id}:{spec.key}:mask').randint(1000, 9999)}"
    default_pm = "Credit Card" if spec.kind == "credit" else "Debit Card"
    return PlannedAccount(spec.key, spec.kind, spec.name, spec.institution, mask,
                          spec.payment_method or default_pm, _money(spec.opening_balance),
                          _money(spec.credit_limit) if spec.credit_limit is not None else None,
                          spec.apr)


def build_plan(user_id: str, profile: SeedProfile, catalog: Catalog, *, start: date, end: date) -> SeedPlan:
    accounts = [_planned_account(user_id, a) for a in profile.accounts]
    by_key = {a.key: a for a in accounts}
    balance = {a.key: a.opening_balance for a in accounts}
    goals = [PlannedGoal(_slug(g.name), g.name, g.icon, _money(g.target_amount),
                         add_months(start, g.target_months), g.account) for g in profile.goals]
    entries: list[PlannedEntry] = []
    now = _utcnow()
    live_today = end == now.date()

    def can_spend(key: str, amount: Decimal) -> bool:
        acct = by_key[key]
        floor = -(acct.credit_limit or Decimal("0")) if acct.kind == "credit" else Decimal("0")
        return balance[key] - amount >= floor

    day = start
    while day <= end:
        rng = random.Random(f"{user_id}:{day.isoformat()}")
        counter = 0

        def add(key: str, amount: Decimal, kind: str, description: str, hour: int, **extra) -> None:
            nonlocal counter
            entry_id = str(uuid.uuid5(NAMESPACE, f"{user_id}:{day.isoformat()}:{counter}"))
            counter += 1
            posted = datetime.combine(day, time(hour, rng.randint(0, 59)), tzinfo=timezone.utc)
            if day == end and live_today:
                # Today's entries never land in the future. The draw above
                # still happens, so ids and amounts stay deterministic.
                posted = min(posted, now)
            entries.append(PlannedEntry(entry_id, key, posted, amount, kind, description, **extra))
            balance[key] += amount

        def ref() -> str:
            return str(uuid.uuid5(NAMESPACE, f"{user_id}:{day.isoformat()}:ref:{counter}"))

        if (day - start).days % profile.income.every_days == 0:
            add(profile.income.account, _money(profile.income.amount), "income",
                profile.income.description, hour=6)

        for bill in profile.bills:
            if not _due(day, bill.day_of_month):
                continue
            amount = bill.amount
            if bill.price_change and _months_before(day, end) < bill.price_change.months_ago:
                amount = bill.price_change.to
            amt = _draw(rng, amount)
            label = catalog.category_for(bill.category).label
            pay = by_key[bill.account].payment_method
            if can_spend(bill.account, amt):
                add(bill.account, -amt, "purchase",
                    f"Spent ${amt:.2f} on {label.lower()} at {bill.merchant} using {pay}.",
                    hour=9, merchant=bill.merchant, category=label, recurring=True)

        for spec in profile.purchases:
            if rng.random() >= spec.per_month / 30.0:
                continue
            amt = _draw(rng, spec.amount)
            merchant = rng.choice(spec.merchants)
            label = catalog.category_for(spec.category).label
            pay = by_key[spec.account].payment_method
            if can_spend(spec.account, amt):
                add(spec.account, -amt, "purchase",
                    f"Spent ${amt:.2f} on {label.lower()} at {merchant} using {pay}.",
                    hour=rng.randint(10, 21), merchant=merchant, category=label)

        cp = profile.card_payment
        if cp and _due(day, cp.day_of_month):
            owed = -balance[cp.to_account]
            payment = min(owed, balance[cp.from_account]) if owed > 0 else Decimal("0")
            if payment > 0:
                r = ref()
                add(cp.from_account, -payment, "card_payment", "Card payment", hour=8, transfer_ref=r)
                add(cp.to_account, payment, "card_payment", "Card payment", hour=8, transfer_ref=r)

        for goal_spec, goal in zip(profile.goals, goals):
            contribution = _money(goal_spec.monthly_contribution)
            if _due(day, goal_spec.day_of_month) and can_spend(goal_spec.from_account, contribution):
                r = ref()
                note = f"Goal contribution: {goal.name}"
                add(goal_spec.from_account, -contribution, "transfer_out", note, hour=7,
                    goal_key=goal.key, transfer_ref=r)
                add(goal.account_key, contribution, "transfer_in", note, hour=7,
                    goal_key=goal.key, transfer_ref=r)

        day += timedelta(days=1)

    budgets = {catalog.category_for(k).label: _money(v) for k, v in profile.budgets.items()}
    return SeedPlan(accounts, entries, budgets, goals)
