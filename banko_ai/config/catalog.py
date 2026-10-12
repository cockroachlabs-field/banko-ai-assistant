"""Categories, personas and seed profiles: the demo's editable content.

Each list ships as YAML under banko_ai/config/defaults/. An SE points
CATEGORIES_FILE, PERSONAS_FILE or SEED_PROFILES_FILE at their own file to
rebrand the demo without touching code. The catalog is validated as a whole
when it loads, so a typo fails at boot with a message that names it."""

from __future__ import annotations

import threading
from decimal import Decimal
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .settings import get_config

GROUPS = ("needs", "wants", "savings", "income", "transfer")
ACCOUNT_KINDS = ("checking", "savings", "credit")


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="forbid")


class Category(_Model):
    key: str
    label: str
    icon: str
    color: str
    grp: str
    sort: int = 0
    aliases: tuple[str, ...] = ()

    @field_validator("grp")
    @classmethod
    def _known_group(cls, value: str) -> str:
        if value not in GROUPS:
            raise ValueError(f"group '{value}' is not one of {', '.join(GROUPS)}")
        return value


class Persona(_Model):
    user_id: str
    username: str
    display_name: str
    avatar_color: str
    spending_style: str
    home_region: str = "auto"


class AccountSpec(_Model):
    key: str
    kind: str
    name: str
    institution: str
    opening_balance: Decimal = Decimal("0")
    credit_limit: Decimal | None = None
    apr: Decimal | None = None
    payment_method: str | None = None

    @field_validator("kind")
    @classmethod
    def _known_kind(cls, value: str) -> str:
        if value not in ACCOUNT_KINDS:
            raise ValueError(f"account kind '{value}' is not one of {', '.join(ACCOUNT_KINDS)}")
        return value


class IncomeSpec(_Model):
    account: str
    amount: Decimal
    every_days: int = 14
    description: str = "Payroll deposit"


class CardPaymentSpec(_Model):
    from_account: str = Field(alias="from")
    to_account: str = Field(alias="to")
    day_of_month: int = 25


class PurchaseSpec(_Model):
    category: str
    merchants: tuple[str, ...]
    per_month: float
    amount: tuple[Decimal, Decimal]
    account: str


class PriceChange(_Model):
    to: Decimal
    months_ago: int


class BillSpec(_Model):
    merchant: str
    category: str
    amount: Decimal | tuple[Decimal, Decimal]
    day_of_month: int
    account: str
    price_change: PriceChange | None = None


class GoalSpec(_Model):
    name: str
    icon: str
    target_amount: Decimal
    target_months: int
    account: str
    monthly_contribution: Decimal
    from_account: str = Field(alias="from")
    day_of_month: int = 1


class SeedProfile(_Model):
    style: str
    months: int | None = None
    accounts: tuple[AccountSpec, ...]
    income: IncomeSpec
    card_payment: CardPaymentSpec | None = None
    purchases: tuple[PurchaseSpec, ...] = ()
    bills: tuple[BillSpec, ...] = ()
    budgets: dict[str, Decimal] = Field(default_factory=dict)
    goals: tuple[GoalSpec, ...] = ()

    @model_validator(mode="after")
    def _accounts_line_up(self) -> SeedProfile:
        keys = [a.key for a in self.accounts]
        if len(keys) != len(set(keys)):
            raise ValueError(f"profile '{self.style}' repeats an account key")
        if not any(a.kind == "checking" for a in self.accounts):
            raise ValueError(f"profile '{self.style}' needs a checking account")
        by_key = {a.key: a for a in self.accounts}
        refs = [self.income.account]
        refs += [p.account for p in self.purchases] + [b.account for b in self.bills]
        refs += [g.account for g in self.goals] + [g.from_account for g in self.goals]
        if self.card_payment:
            refs += [self.card_payment.from_account, self.card_payment.to_account]
        for ref in refs:
            if ref not in by_key:
                raise ValueError(f"profile '{self.style}' has no account '{ref}'")
        if self.card_payment and by_key[self.card_payment.to_account].kind != "credit":
            raise ValueError(f"profile '{self.style}' card_payment must pay a credit account")
        return self


class Catalog(_Model):
    categories: tuple[Category, ...]
    personas: tuple[Persona, ...]
    profiles: dict[str, SeedProfile]

    @model_validator(mode="after")
    def _cross_check(self) -> Catalog:
        if not any(c.key == "other" for c in self.categories):
            raise ValueError("categories need an 'other' entry")
        for persona in self.personas:
            if persona.spending_style not in self.profiles:
                raise ValueError(f"persona '{persona.username}' uses unknown spending style "
                                 f"'{persona.spending_style}'")
        known = self._lookup()
        for profile in self.profiles.values():
            names = [p.category for p in profile.purchases] + [b.category for b in profile.bills]
            names += list(profile.budgets)
            for name in names:
                if name.lower() not in known:
                    raise ValueError(f"profile '{profile.style}' uses unknown category '{name}'")
        return self

    def _lookup(self) -> dict[str, Category]:
        table: dict[str, Category] = {}
        for c in self.categories:
            for name in (c.key, c.label, *c.aliases):
                table.setdefault(name.lower(), c)
        return table

    def category_for(self, label: str | None) -> Category:
        """The category a stored label means, by key, label or alias, ignoring case."""
        table = self._lookup()
        return table.get((label or "").strip().lower()) or table["other"]


_lock = threading.Lock()
_cache: dict[tuple[str, str, str], Catalog] = {}


def _read(path_setting: str, default_name: str) -> Any:
    if path_setting:
        raw = Path(path_setting).read_text(encoding="utf-8")
    else:
        raw = resources.files("banko_ai.config.defaults").joinpath(default_name).read_text(encoding="utf-8")
    return yaml.safe_load(raw)


def get_catalog() -> Catalog:
    cfg = get_config()
    key = (cfg.categories_file, cfg.personas_file, cfg.seed_profiles_file)
    with _lock:
        cached = _cache.get(key)
        if cached is None:
            profiles = _read(cfg.seed_profiles_file, "seed_profiles.yaml")["profiles"]
            cached = Catalog.model_validate({
                "categories": _read(cfg.categories_file, "categories.yaml")["categories"],
                "personas": _read(cfg.personas_file, "personas.yaml")["personas"],
                "profiles": {style: {**body, "style": style} for style, body in profiles.items()},
            })
            _cache[key] = cached
        return cached


def reset_catalog() -> None:
    with _lock:
        _cache.clear()
