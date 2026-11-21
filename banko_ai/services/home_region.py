"""An account's home region, and moving an account to another one.

On a multi-region database every table that holds a person's rows is
REGIONAL BY ROW: a row lives in the region its crdb_region names. An
account's rows belong together in one region, its home. Writes pin them
there (db/regions.money_region) and reads name that region (read_region), so
a statement stays in it. An account without a home region gets neither: its
rows land wherever the load balancer sent each write, and every read looks
in all regions. The accounts the Google sign-in makes had none."""

from __future__ import annotations

import logging

from sqlalchemy import text

from ..config.catalog import get_catalog
from ..db import regions, schema
from ..db.engine import run_transaction

log = logging.getLogger("banko.home_region")

BATCH = 200  # rows moved per transaction: short ones, so a purchase made meanwhile is not kept waiting
PASSES = 4   # over every table; the last one is expected to find nothing


class NotRegional(Exception):
    """The database has one region: there is no home region to choose."""


class UnknownRegion(Exception):
    """The database has no such region."""


class NotTheirs(Exception):
    """A persona's region is the personas file's to give: boot sets it again."""


def regions_of(url: str | None = None) -> list[str]:
    """The regions an account can live in, primary first: the database's
    own, once its tables are REGIONAL BY ROW. Empty on one region."""
    base = regions._owner(url)
    return list(regions.detect_database_regions(base)) if regions.regional_tables_ready(base) else []


def default_region(url: str | None = None) -> str | None:
    """Where an account starts when nobody chose: the database's primary
    region. None on one region."""
    found = regions_of(url)
    return found[0] if found else None


def _tables() -> list[str]:
    """Every table that keeps a person's rows by region."""
    return [t.name for t in schema.tables()
            if t.locality == "REGIONAL BY ROW" and any(c.name == "user_id" for c in t.columns)]


def _move(table: str, user_id: str, region: str, url: str) -> int:
    """This account's rows of one table that live elsewhere, brought to the
    region, a batch per transaction. Safe to run again: it moves what is
    still away."""
    moved = 0
    while True:
        def batch(conn) -> int:
            if not regions.has_region_column(conn, table):  # an older database still converting this table
                return 0
            return int(conn.execute(text(f"""
                UPDATE {table} SET crdb_region = CAST(:r AS crdb_internal_region)
                WHERE user_id = CAST(:u AS UUID) AND crdb_region != CAST(:r AS crdb_internal_region)
                LIMIT {BATCH}
            """), {"u": user_id, "r": region}).rowcount)
        n = run_transaction(batch, url=url)
        moved += n
        if n < BATCH:
            return moved


def set_home_region(user_id: str, region: str, *, url: str | None = None) -> dict:
    """Make this region the account's home and bring its rows there.

    The account is pointed at the region first, so whatever it writes from
    then on is born there, and reads stop naming a region until the last
    row has arrived. Run as the owner, never as the row-level security
    role. Asked again, or after a run that stopped halfway, it moves what is
    still away."""
    user_id = str(user_id)
    if user_id in {p.user_id for p in get_catalog().personas}:
        raise NotTheirs(user_id)
    base = regions._owner(url)
    available = regions_of(base)
    if not available:
        raise NotRegional()
    if region not in available:
        raise UnknownRegion(region)

    def point(conn) -> int:
        return int(conn.execute(text("UPDATE users SET home_region = :r WHERE user_id = CAST(:u AS UUID)"),
                                {"r": region, "u": user_id}).rowcount)
    if not run_transaction(point, url=base):
        raise LookupError(f"no account {user_id}")
    regions.home_region_changed(user_id)
    # A write that began before the account was pointed at its new region
    # commits in the old one, and can do so after its table was gone
    # through: go again until a pass finds nothing.
    rows = 0
    for _ in range(PASSES):
        found = sum(_move(table, user_id, region, base) for table in _tables())
        rows += found
        if not found:
            break
    regions.home_region_changed(user_id)   # what was learned during the move ("rows away") is stale now
    log.info("account %s lives in %s now: %d rows moved", user_id, region, rows)
    return {"home_region": region, "rows": rows}
