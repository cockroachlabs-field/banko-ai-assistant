"""Load users: the load simulator's customers.

Ordinary demo users named load-<sequence>-<hex>, each with a spending
style from the seed profiles and that profile's accounts at their opening
balances, homed round robin across the database's regions so REGIONAL BY
ROW rows land in every region. A batch of users and their accounts is one
transaction (no user ever exists without its accounts) and costs two
set-based statements, whatever its size. clear() removes them with every
row they own.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from ..config.catalog import get_catalog
from ..config.settings import get_config
from ..db.engine import get_engine, run_transaction
from ..db.regions import detect_database_regions, has_region_column
from .users import delete_users

LOAD_PREFIX = "load-"
_BATCH = 500


@dataclass(frozen=True)
class LoadUser:
    user_id: str
    username: str
    style: str
    region: str | None
    accounts: dict[str, str] = field(default_factory=dict)  # profile account key -> account id


def load_users(*, url: str | None = None) -> list[LoadUser]:
    """Every load user, in the order they were created."""
    url = url or get_config().database_url
    with get_engine(url).connect() as conn:
        users = conn.execute(text(
            "SELECT user_id, username, spending_style, home_region FROM users "
            "WHERE username LIKE :prefix ORDER BY username"), {"prefix": LOAD_PREFIX + "%"}).fetchall()
        if not users:
            return []
        owned: dict[str, dict[tuple[str, str], str]] = {}
        for uid, aid, kind, name in conn.execute(text(
                "SELECT user_id, account_id, kind, name FROM accounts WHERE user_id = ANY(CAST(:ids AS UUID[]))"),
                {"ids": [str(u[0]) for u in users]}):
            owned.setdefault(str(uid), {})[(kind, name)] = str(aid)
    profiles = get_catalog().profiles
    out = []
    for uid, username, style, region in users:
        mine = owned.get(str(uid), {})
        specs = profiles[style].accounts if style in profiles else ()
        out.append(LoadUser(str(uid), username, style, region,
                            {a.key: mine[(a.kind, a.name)] for a in specs if (a.kind, a.name) in mine}))
    return out


def _insert_batch(conn, sequence: list[int], styles: list[str], regions: list[str]) -> None:
    profiles = get_catalog().profiles
    names = [f"{LOAD_PREFIX}{i:05d}-{secrets.token_hex(2)}" for i in sequence]
    user_styles = [styles[i % len(styles)] for i in sequence]
    user_regions = [regions[i % len(regions)] if regions else None for i in sequence]
    created = conn.execute(text("""
        INSERT INTO users (username, spending_style, home_region, demo_user)
        SELECT t.name, t.style, t.region, true
        FROM unnest(CAST(:names AS STRING[]), CAST(:styles AS STRING[]), CAST(:regions AS STRING[]))
             AS t(name, style, region)
        RETURNING user_id, spending_style, home_region
    """), {"names": names, "styles": user_styles, "regions": user_regions}).fetchall()
    cols: dict[str, list] = {k: [] for k in ("u", "region", "kind", "name", "inst", "mask", "pm",
                                             "opening", "limit", "apr")}
    for uid, style, region in created:
        for spec in profiles[style].accounts:
            for key, value in (("u", str(uid)), ("region", region), ("kind", spec.kind), ("name", spec.name),
                               ("inst", spec.institution), ("mask", f"{secrets.randbelow(10000):04d}"),
                               ("pm", spec.payment_method), ("opening", spec.opening_balance),
                               ("limit", spec.credit_limit), ("apr", spec.apr)):
                cols[key].append(value)
    pinned = bool(regions) and has_region_column(conn, "accounts")
    conn.execute(text(f"""
        INSERT INTO accounts ({"crdb_region, " if pinned else ""}user_id, kind, name, institution, mask,
                              payment_method, opening_balance, balance, credit_limit, apr, opened_at)
        SELECT {"CAST(t.region AS crdb_internal_region), " if pinned else ""}t.u, t.kind, t.name, t.inst,
               t.mask, t.pm, t.opening, t.opening, t.lim, t.apr, :today
        FROM unnest(CAST(:u AS UUID[]), CAST(:region AS STRING[]), CAST(:kind AS STRING[]),
                    CAST(:name AS STRING[]), CAST(:inst AS STRING[]), CAST(:mask AS STRING[]),
                    CAST(:pm AS STRING[]), CAST(:opening AS DECIMAL[]), CAST(:limit AS DECIMAL[]),
                    CAST(:apr AS DECIMAL[]))
             AS t(u, region, kind, name, inst, mask, pm, opening, lim, apr)
    """), {**cols, "today": datetime.now(timezone.utc).date()})


def prepare(count: int, *, url: str | None = None) -> list[LoadUser]:
    """Top the pool up to count load users (it never shrinks) and return
    all of them."""
    url = url or get_config().database_url  # the admin role: users is written before any RLS session
    have = load_users(url=url)
    missing = count - len(have)
    if missing <= 0:
        return have
    styles = sorted(get_catalog().profiles)
    regions = sorted(detect_database_regions(url))
    start = len(have)
    for offset in range(0, missing, _BATCH):
        sequence = list(range(start + offset, start + min(offset + _BATCH, missing)))
        for attempt in range(3):
            try:
                # Not idempotent (a replay after an ambiguous commit would add
                # users), so only a definite rollback is retried.
                run_transaction(lambda conn: _insert_batch(conn, sequence, styles, regions),
                                url=url, idempotent=False)
                break
            except IntegrityError:  # a name collided with a concurrent prepare; new names, try again
                if attempt == 2:
                    raise
    return load_users(url=url)


def clear(*, url: str | None = None) -> int:
    """Remove every load user and every row they own."""
    url = url or get_config().database_url
    return delete_users([u.user_id for u in load_users(url=url)], url=url)
