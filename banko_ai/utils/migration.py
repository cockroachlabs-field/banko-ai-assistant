"""
Database migration utilities.

This module provides migration scripts to update the database schema
for user-specific vector indexing and other enhancements.
"""

import logging

from .db_retry import get_database_url

log = logging.getLogger("banko.migration")

from ..db.regions import detect_regions
from ..db.schema import ensure_schema, run_pending_conversions


def migrate_regional_tables(database_url: str, primary_region: str | None) -> bool:
    """Older entry point, kept for tests and scripts. The schema registry
    owns topology and localities now; this applies it and runs any pending
    locality conversions right away instead of in the background."""
    if len(detect_regions(database_url)) < 2:
        log.info("single-region deployment detected, skipping REGIONAL BY ROW migration")
        return False
    if not primary_region:
        log.error("primary_region required for multi-region migration but was None")
        return False
    try:
        # Force schema reapply so region bootstrap still happens
        ensure_schema(database_url, force=True, primary_region=primary_region)
        run_pending_conversions(database_url)
        return True
    except Exception as e:
        log.exception("multi-region migration failed: %s", e)
        return False


class DatabaseMigration:
    """Older migration entry points. Each one now runs the schema registry,
    which is a single SELECT when the schema is already current."""

    def __init__(self, database_url: str | None = None):
        self.database_url = get_database_url(database_url)

    @property
    def engine(self):
        from ..db.engine import get_engine
        return get_engine(self.database_url)

    def _ensure(self, force: bool = False) -> bool:
        try:
            ensure_schema(self.database_url, force=force)
            return True
        except Exception as e:
            print(f"schema setup failed: {e}")
            return False

    def migrate_timestamptz(self) -> bool:
        return self._ensure()

    def add_created_at_column(self) -> bool:
        return self._ensure()

    def migrate_users_table(self) -> bool:
        ok = self._ensure()
        self.expire_demo_users()
        return ok

    def migrate_to_coach_v1(self) -> bool:
        return self._ensure()

    def run_all_migrations(self) -> bool:
        return self._ensure(force=True)

    def expire_demo_users(self) -> int:
        """Delete demo users older than DEMO_USER_TTL_DAYS along with their
        dependent rows, sparing the legacy personas. Runs at boot. Row-level
        TTL could not do this: it cannot cascade to expenses/signals/nudges
        and it cannot exempt personas or real accounts. Returns the number
        of users removed (0 when the knob is off, the default)."""
        from ..config.settings import get_config

        ttl_days = get_config().demo_user_ttl_days
        if ttl_days <= 0:
            return 0

        from datetime import datetime, timedelta, timezone

        from sqlalchemy import text

        from ..config.catalog import get_catalog

        personas = [p.user_id for p in get_catalog().personas]
        cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
        limit = 1000
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT user_id FROM users
                    WHERE demo_user AND created_at < :cutoff
                      AND user_id <> ALL(CAST(:personas AS UUID[]))
                """), {"cutoff": cutoff, "personas": personas}).fetchall()
            ids = [str(r[0]) for r in rows]
            if not ids:
                return 0

            # Bounded, set-based deletes, children before parents, each in
            # its own short transaction so no sweep holds intents on every
            # row an expired user ever wrote.
            for i in range(0, len(ids), limit):
                chunk = ids[i:i + limit]
                for table in ("coach_nudges", "spending_signals", "ledger_entries", "transfers",
                              "goals", "accounts", "user_budgets", "user_data_versions", "expenses"):
                    while True:
                        with self.engine.begin() as conn:
                            deleted = conn.execute(text(
                                f"DELETE FROM {table} WHERE user_id = ANY(CAST(:ids AS UUID[])) LIMIT {limit}"),
                                {"ids": chunk}).rowcount
                        if deleted < limit:
                            break
                keys = [f"seed:{u}" for u in chunk] + [f"seed_lock:{u}" for u in chunk]
                with self.engine.begin() as conn:
                    conn.execute(text("DELETE FROM banko_meta WHERE key = ANY(CAST(:keys AS STRING[]))"),
                                 {"keys": keys})
                    conn.execute(text("DELETE FROM users WHERE user_id = ANY(CAST(:ids AS UUID[]))"),
                                 {"ids": chunk})
            print(f"Expired {len(ids)} demo users older than {ttl_days} days")
            return len(ids)
        except Exception as e:
            print(f"Demo user expiry sweep failed (non-fatal): {e}")
            return 0
