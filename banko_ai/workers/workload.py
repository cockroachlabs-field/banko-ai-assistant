"""The load simulator's workload: what a population of banking customers
does, through the same services the app uses, so every simulated purchase
is a real ledger movement and every search a real per-user vector query.

Each customer is served through their home region's own pool and reads
name that region, so on a multi-region cluster every statement stays in
the customer's region and only a write's commit crosses to another one.

Transfers and card payments carry an idempotency key, so a retry after a
dropped connection or an ambiguous commit finds its own row instead of
moving money twice. Purchases and deposits let the database pick their ids
(a client-chosen key on a REGIONAL BY ROW table costs a uniqueness check in
every region), so they retry only what cannot have committed and report a
lost commit. Declined purchases and transfers (not enough money) are a
normal business outcome, not errors.
"""

from __future__ import annotations

import bisect
import random
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TypeVar

from sqlalchemy.exc import DBAPIError

from ..config.catalog import get_catalog
from ..db.engine import is_retryable
from ..services import accounts, ledger, reports, transactions
from ..services.load_users import LoadUser

T = TypeVar("T")

OPS = ("purchase", "transfer", "card_payment", "deposit", "page_load", "search", "report")
PROFILES: dict[str, dict[str, int]] = {
    "balanced": {"purchase": 35, "page_load": 30, "search": 15, "transfer": 8, "card_payment": 6,
                 "deposit": 4, "report": 2},
    "payments": {"purchase": 70, "transfer": 14, "card_payment": 10, "deposit": 6},
    "browsing": {"page_load": 55, "search": 25, "report": 10, "purchase": 10},
    "search": {"search": 70, "page_load": 20, "purchase": 10},
}
QUERIES = ("coffee", "groceries", "dinner out", "ride share", "subscriptions", "electronics",
           "gas station", "movie night")
_BACKOFF = (0.1, 0.2, 0.4, 0.8)


class Declined(Exception):
    """The bank said no (nothing owed, not enough money): a business outcome."""

    def __init__(self, client_retries: int = 0):
        super().__init__("declined")
        self.client_retries = client_retries


def new_id() -> str:
    return str(uuid.uuid4())


def is_unique_violation(exc: BaseException) -> bool:
    orig = getattr(exc, "orig", None)
    return str(getattr(orig, "sqlstate", None) or "") == "23505"


def _never(exc: BaseException) -> bool:
    return False


def with_client_retries(fn: Callable[[], T], *, committed: Callable[[BaseException], bool],
                        attempts: int = 5, sleep: Callable[[float], None] = time.sleep,
                        idempotent: bool = True) -> tuple[T | None, int]:
    """Run fn, retrying what is safe to retry (connection loss, an ambiguous
    commit, a 40001 run_transaction gave up on) with jittered backoff. A
    retry that fails because committed(exc) says the first attempt landed
    is a success. Work that is not idempotent never retries an error from
    its COMMIT (at_commit), which may have landed. Returns (result,
    retries); an error that ends the attempt carries .client_retries."""
    retries = 0
    while True:
        try:
            return fn(), retries
        except Exception as exc:
            if retries and committed(exc):
                return None, retries
            if (retries + 1 < attempts and isinstance(exc, DBAPIError) and is_retryable(exc, idempotent=True)
                    and (idempotent or not getattr(exc, "at_commit", False))):
                delay = _BACKOFF[min(retries, len(_BACKOFF) - 1)]
                sleep(delay + random.uniform(0, delay / 2))
                retries += 1
                continue
            exc.client_retries = retries  # type: ignore[attr-defined]
            raise


def _template(merchant: str, label: str) -> str:
    # The purchase without its amount: one vector per merchant and category.
    return f"Spent on {label.lower()} at {merchant}."


class VectorCache:
    """Every purchase description's vector, encoded once in one batch, so a
    purchase at 2,000 operations a second never waits on the model."""

    def __init__(self, vectors: dict[tuple[str, str], list[float]]):
        self._vectors = vectors

    @classmethod
    def build(cls) -> VectorCache:
        from ..utils.embeddings import load_embedding_model
        catalog = get_catalog()
        keys = sorted({(m, s.category) for p in catalog.profiles.values() for s in p.purchases for m in s.merchants})
        if not keys:
            return cls({})
        encoded = load_embedding_model(wait_seconds=600).encode(
            [_template(m, catalog.category_for(c).label) for m, c in keys])
        return cls({k: [float(x) for x in v] for k, v in zip(keys, encoded)})

    def get(self, merchant: str, category: str) -> list[float] | None:
        return self._vectors.get((merchant, category))


class UserPicker:
    """Customers with a Zipf-like skew: the busiest few are much busier than
    the rest, as in real traffic, and the long tail still shows up."""

    def __init__(self, users: Sequence[LoadUser], s: float = 1.1):
        if not users:
            raise ValueError("no load users to pick from")
        self.users = list(users)
        self._cumulative: list[float] = []
        total = 0.0
        for rank in range(1, len(self.users) + 1):
            total += 1.0 / rank ** s
            self._cumulative.append(total)

    def pick(self, rng: random.Random) -> LoadUser:
        return self.users[bisect.bisect_left(self._cumulative, rng.random() * self._cumulative[-1])]


@dataclass
class OpContext:
    url: str | None
    vectors: VectorCache
    region_urls: dict[str, str] = field(default_factory=dict)  # home region -> its own pool (with_gateway_region)
    pin_regions: bool = False  # reads name the home region: every load row lives there

    def url_for(self, user: LoadUser) -> str | None:
        return self.region_urls.get(user.region or "", self.url)

    def region_for(self, user: LoadUser) -> str | None:
        return user.region if self.pin_regions else None


def _money(rng: random.Random, low, high) -> Decimal:
    return Decimal(rng.randint(int(Decimal(low) * 100), int(Decimal(high) * 100))) / 100


def _purchase(user: LoadUser, ctx: OpContext, rng: random.Random) -> int:
    catalog = get_catalog()
    profile = catalog.profiles[user.style]
    spec = rng.choices(profile.purchases, weights=[s.per_month for s in profile.purchases])[0]
    amount, merchant = _money(rng, *spec.amount), rng.choice(spec.merchants)
    label = catalog.category_for(spec.category).label
    account = next(a for a in profile.accounts if a.key == spec.account)
    pay = account.payment_method or ("Credit Card" if account.kind == "credit" else "Debit Card")
    _, retries = with_client_retries(lambda: ledger.post_purchase(
        user.user_id, user.accounts[spec.account], amount=amount, merchant=merchant, category=label,
        description=f"Spent ${amount:.2f} on {label.lower()} at {merchant} using {pay}.", payment_method=pay,
        embedding=ctx.vectors.get(merchant, spec.category), url=ctx.url_for(user)),
        committed=_never, idempotent=False)
    return retries


def _transfer(user: LoadUser, ctx: OpContext, rng: random.Random) -> int:
    pair = ("checking", "savings") if rng.random() < 0.5 else ("savings", "checking")
    amount, key = _money(rng, 20, 400), new_id()
    _, retries = with_client_retries(lambda: ledger.transfer(
        user.user_id, user.accounts[pair[0]], user.accounts[pair[1]], amount, idempotency_key=key,
        url=ctx.url_for(user)), committed=is_unique_violation)
    return retries


def _card_payment(user: LoadUser, ctx: OpContext, rng: random.Random) -> int:
    """Pay the statement in full half the time, else part of it, as
    cardholders do; never more than the card owes (the ledger refuses)."""
    card = with_client_retries(lambda: accounts.get_account(user.user_id, user.accounts["card"], url=ctx.url_for(user),
                                                            region=ctx.region_for(user)), committed=_never)[0]
    owed = -Decimal(str((card or {}).get("balance") or 0))
    if owed < 1:
        raise Declined(0)
    amount = owed if rng.random() < 0.5 else min(owed, _money(rng, 100, 600))
    key = new_id()
    try:
        _, retries = with_client_retries(lambda: ledger.card_payment(
            user.user_id, user.accounts["checking"], user.accounts["card"], amount, idempotency_key=key,
            url=ctx.url_for(user)), committed=is_unique_violation)
    except ledger.InvalidAmount as exc:  # a concurrent payment got there first
        raise Declined(getattr(exc, "client_retries", 0)) from exc
    return retries


def _deposit(user: LoadUser, ctx: OpContext, rng: random.Random) -> int:
    income = get_catalog().profiles[user.style].income
    _, retries = with_client_retries(lambda: ledger.record_income(
        user.user_id, user.accounts[income.account], income.amount, description=income.description,
        url=ctx.url_for(user)), committed=_never, idempotent=False)
    return retries


def _page_load(user: LoadUser, ctx: OpContext, rng: random.Random) -> int:
    url, region = ctx.url_for(user), ctx.region_for(user)

    def visit() -> None:
        accounts.list_accounts(user.user_id, url=url, region=region)
        transactions.list_transactions(user.user_id, limit=20, url=url, region=region)
    return with_client_retries(visit, committed=_never)[1]


def _search(user: LoadUser, ctx: OpContext, rng: random.Random) -> int:
    q = rng.choice(QUERIES)
    return with_client_retries(lambda: transactions.semantic_search(
        user.user_id, q, limit=10, url=ctx.url_for(user), region=ctx.region_for(user)), committed=_never)[1]


def _report(user: LoadUser, ctx: OpContext, rng: random.Random) -> int:
    return with_client_retries(lambda: reports.cashflow(user.user_id, months=3, url=ctx.url_for(user)),
                               committed=_never)[1]


_HANDLERS: dict[str, Callable[[LoadUser, OpContext, random.Random], int]] = {
    "purchase": _purchase, "transfer": _transfer, "card_payment": _card_payment, "deposit": _deposit,
    "page_load": _page_load, "search": _search, "report": _report,
}


def run_op(op: str, user: LoadUser, ctx: OpContext, rng: random.Random) -> tuple[str, int]:
    """Run one operation for one customer: ("ok" or "declined", client
    retries). Any other failure raises, carrying .client_retries."""
    handler = _HANDLERS.get(op)
    if handler is None:
        raise ValueError(f"unknown operation {op!r}")
    try:
        return "ok", handler(user, ctx, rng)
    except (ledger.InsufficientFunds, Declined) as exc:
        return "declined", getattr(exc, "client_retries", 0)
