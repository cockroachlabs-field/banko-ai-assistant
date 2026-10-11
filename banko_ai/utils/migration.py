"""
Database migration utilities.

This module provides migration scripts to update the database schema
for user-specific vector indexing and other enhancements.
"""

import logging

from .db_retry import get_database_url

log = logging.getLogger("banko.migration")

from ..db.regions import (  # noqa: F401  (re-exported for older imports)
    agent_home_region,
    clear_region_caches,
    detect_database_regions,
    detect_regions,
    is_regional_deployment,
    regional_tables_ready,
    resolve_primary_region,
    resolve_user_region_cache_clear,
)
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
        ensure_schema(database_url, primary_region=primary_region)
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

        personas = ("00000000-0000-0000-0000-0000000000a1",
                    "00000000-0000-0000-0000-0000000000a2",
                    "00000000-0000-0000-0000-0000000000a3")
        cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
        try:
            with self.engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT user_id FROM users
                    WHERE demo_user AND created_at < :cutoff
                      AND user_id NOT IN (:p1, :p2, :p3)
                """), {"cutoff": cutoff, "p1": personas[0],
                       "p2": personas[1], "p3": personas[2]}).fetchall()
                ids = [str(r[0]) for r in rows]
                if not ids:
                    return 0

                placeholders = ", ".join(f":u{i}" for i in range(len(ids)))
                params = {f"u{i}": uid for i, uid in enumerate(ids)}
                for table in ("coach_nudges", "spending_signals",
                              "user_budgets", "expenses", "users"):
                    conn.execute(text(
                        f"DELETE FROM {table} WHERE user_id IN ({placeholders})"),
                        params)
                conn.commit()
                print(f"Expired {len(ids)} demo users older than {ttl_days} days")
                return len(ids)
        except Exception as e:
            print(f"Demo user expiry sweep failed (non-fatal): {e}")
            return 0
