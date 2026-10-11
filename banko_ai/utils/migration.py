"""
Database migration utilities.

This module provides migration scripts to update the database schema
for user-specific vector indexing and other enhancements.
"""

import logging

from .db_retry import get_database_url

log = logging.getLogger("banko.migration")

# Success-only caches keyed by database URL. lru_cache used to memoize
# failures too: booting before the DB was reachable pinned an empty
# region list (or a False readiness) for the process lifetime, which
# killed the region story in long-running fault demos. These remember
# only answers a live database actually gave; errors fall through and
# retry on the next call.
_cluster_regions_cache: dict[str, list[str]] = {}
_database_regions_cache: dict[str, list[str]] = {}
_tables_ready_cache: dict[str, bool] = {}
_gateway_region_cache: dict[str, str] = {}


def is_regional_deployment(database_url: str) -> bool:
    """Check if database is multi-region (cached wrapper around detect_regions)."""
    return len(detect_regions(database_url)) >= 2


def regional_tables_ready(database_url: str) -> bool:
    """True only when the cluster is multi-region AND the RBR migration
    actually applied (expenses carries crdb_region). Readers and writers
    gate on this, not on is_regional_deployment, so a failed or skipped
    migration degrades to single-region behavior instead of generating
    SQL against a column that does not exist. Only definitive answers
    are cached; an unreachable database is retried on the next call."""
    if database_url in _tables_ready_cache:
        return _tables_ready_cache[database_url]

    regions = detect_regions(database_url)
    if database_url not in _cluster_regions_cache:
        return False  # detection itself failed; do not cache
    if not regions:
        _tables_ready_cache[database_url] = False
        return False

    from sqlalchemy import text

    from ..db.engine import get_engine

    try:
        engine = get_engine(database_url)
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'expenses' AND column_name = 'crdb_region'
                LIMIT 1
            """)).fetchone()
            ready = row is not None
            _tables_ready_cache[database_url] = ready
            return ready
    except Exception as e:
        log.debug("regional readiness check failed, treating as single-region: %s", e)
        return False


regional_tables_ready.cache_clear = _tables_ready_cache.clear  # type: ignore[attr-defined]


def detect_regions(database_url: str) -> list[str]:
    """Detect cluster regions from SHOW REGIONS FROM CLUSTER.

    Returns empty list on single-region deployments, unreachable databases,
    or when fewer than two regions exist.

    Successful answers are cached per URL; failures are not, so a database
    that comes up after the app does is still detected.
    """
    if database_url in _cluster_regions_cache:
        return _cluster_regions_cache[database_url]

    from sqlalchemy import text

    try:
        from ..db.engine import get_engine
        engine = get_engine(database_url)
        with engine.connect() as conn:
            result = conn.execute(text("SHOW REGIONS FROM CLUSTER"))
            rows = result.fetchall()
            regions = [str(row[0]) for row in rows]
            answer = regions if len(regions) >= 2 else []
            _cluster_regions_cache[database_url] = answer
            return answer
    except Exception as e:
        log.debug("region detection failed (unreachable, will retry): %s", e)
        return []


detect_regions.cache_clear = _cluster_regions_cache.clear  # type: ignore[attr-defined]


def agent_home_region(database_url: str) -> str:
    """Region label for agent_state rows and dashboard activity events:
    the real gateway region on a multi-region deployment, 'local'
    everywhere else. Replaces the hardcoded AWS region names that used to
    show on the agent dashboard no matter what cluster was underneath."""
    if database_url in _gateway_region_cache:
        return _gateway_region_cache[database_url]

    regions = detect_regions(database_url)
    if not regions:
        if database_url in _cluster_regions_cache:
            _gateway_region_cache[database_url] = "local"
        return "local"

    from sqlalchemy import text

    try:
        from ..db.engine import get_engine
        engine = get_engine(database_url)
        with engine.connect() as conn:
            row = conn.execute(text("SELECT gateway_region()")).fetchone()
            gateway = str(row[0]) if row and row[0] else None
        label = gateway if gateway in regions else regions[0]
        _gateway_region_cache[database_url] = label
        return label
    except Exception as e:
        log.debug("gateway region lookup failed, using first region: %s", e)
        return regions[0]


def detect_database_regions(database_url: str) -> list[str]:
    """Regions configured on the connected database (SHOW REGIONS FROM
    DATABASE), primary first.

    This is the list signup may offer. An operator-configured database can
    legitimately carry fewer regions than the cluster, and a region missing
    from the database's crdb_internal_region enum can never be written as
    crdb_region, so offering cluster regions here would let a user pick a
    home their rows cannot live in.

    Returns empty list when the database has no regions configured or is
    unreachable. Successful answers are cached per URL (failures are not);
    migrate_regional_tables clears the cache after changing topology.
    """
    if database_url in _database_regions_cache:
        return _database_regions_cache[database_url]

    from sqlalchemy import text


    try:
        from ..db.engine import get_engine
        engine = get_engine(database_url)
        with engine.connect() as conn:
            rows = conn.execute(text(
                'SELECT region, "primary" FROM [SHOW REGIONS FROM DATABASE]'
            )).fetchall()
            answer = [str(r[0]) for r in sorted(rows, key=lambda r: not r[1])]
            _database_regions_cache[database_url] = answer
            return answer
    except Exception as e:
        log.debug("database region detection failed (will retry): %s", e)
        return []


detect_database_regions.cache_clear = _database_regions_cache.clear  # type: ignore[attr-defined]


def resolve_primary_region(database_url: str) -> str | None:
    """Resolve the primary region for migrate_regional_tables.

    Returns the database's current PRIMARY REGION if one is already set,
    else the gateway region, else the first detected region. Returns None
    if the deployment is single-region or unreachable.

    Never re-homes an existing primary: if the DB already has a primary
    region, that region is returned to make SET PRIMARY REGION idempotent.
    """
    from sqlalchemy import text

    regions = detect_regions(database_url)
    if not regions:
        return None

    try:
        from ..db.engine import get_engine
        engine = get_engine(database_url)
        with engine.connect() as conn:
            result = conn.execute(text('SELECT region FROM [SHOW REGIONS FROM DATABASE] WHERE "primary"'))
            row = result.fetchone()
            if row:
                return str(row[0])

            result = conn.execute(text("SELECT gateway_region()"))
            row = result.fetchone()
            if row and row[0]:
                gateway = str(row[0])
                if gateway in regions:
                    return gateway

            return regions[0]
    except Exception as e:
        log.debug("primary region resolution failed, falling back to first detected: %s", e)
        return regions[0]


def resolve_user_region_cache_clear() -> None:
    """Drop auth.resolve_user_region's cache. Local import because auth
    imports from this module."""
    try:
        from ..web.auth import resolve_user_region
        resolve_user_region.cache_clear()
    except Exception:
        pass


def migrate_regional_tables(database_url: str, primary_region: str | None) -> bool:
    """Idempotently apply REGIONAL BY ROW locality to expenses, spending_signals,
    and coach_nudges tables.

    Returns False immediately when fewer than two regions detected (no-op).
    Returns True when multi-region DDL succeeds or is already applied.

    Args:
        database_url: Database connection string
        primary_region: Region to designate as PRIMARY REGION (required when
                        multi-region is detected)
    """
    import re

    from sqlalchemy import text

    regions = detect_regions(database_url)
    if len(regions) < 2:
        log.info("single-region deployment detected, skipping REGIONAL BY ROW migration")
        return False

    if not primary_region:
        log.error("primary_region required for multi-region migration but was None")
        return False

    db_name = database_url.split("/")[-1].split("?")[0]
    identifier_pattern = re.compile(r"^[a-zA-Z0-9_-]+$")

    if not identifier_pattern.match(db_name):
        log.error("invalid database name for DDL interpolation: %s", db_name)
        return False

    if not identifier_pattern.match(primary_region):
        log.error("invalid primary_region for DDL interpolation: %s", primary_region)
        return False

    for region in regions:
        if not identifier_pattern.match(region):
            log.error("invalid region name for DDL interpolation: %s", region)
            return False

    from ..db.engine import get_engine
    engine = get_engine(database_url)
    try:
        with engine.connect() as conn:
            # Database-level topology (primary region, region list,
            # survival goal) belongs to whoever configured it first. The
            # chaos demo's init.sql owns its cluster's defaultdb, and
            # banko fighting it for control broke a working demo: both
            # ran the same ALTERs, the loser died on duplicates, and
            # init.sql never reached its SURVIVE REGION FAILURE line.
            # If a primary region exists, banko leaves the database DDL
            # alone and only manages its own tables below.
            existing_primary = conn.execute(text(
                'SELECT region FROM [SHOW REGIONS FROM DATABASE] '
                'WHERE "primary"')).fetchone()

            if existing_primary:
                log.info("database %s topology already configured "
                         "(primary: %s); leaving database DDL alone",
                         db_name, existing_primary[0])
                # Respecting the operator's topology means the database can
                # lag the cluster: a region added to the cluster later is
                # never joined here. Rows cannot pin to it and signup will
                # not offer it, so say so where an operator will see it.
                db_regions = {str(r[0]) for r in conn.execute(text(
                    'SELECT region FROM [SHOW REGIONS FROM DATABASE]'
                )).fetchall()}
                missing = [r for r in regions if r not in db_regions]
                if missing:
                    log.warning(
                        "cluster has regions the database does not: %s. "
                        "Signup only offers the database's regions; run "
                        "ALTER DATABASE %s ADD REGION to join them.",
                        ", ".join(missing), db_name)
            else:
                log.info("bootstrapping multi-region config on %s "
                         "(primary: %s)", db_name, primary_region)
                conn.execute(text(
                    f"ALTER DATABASE {db_name} SET PRIMARY REGION '{primary_region}'"))
                log.info("set PRIMARY REGION to %s", primary_region)

                for region in regions:
                    if region == primary_region:
                        continue
                    conn.execute(text(
                        f"ALTER DATABASE {db_name} ADD REGION '{region}'"))
                    log.info("added region %s", region)

                # Region survival is the point of the multi-region demo:
                # the zone default cannot keep quorum through a whole
                # region dying. Needs three or more regions.
                if len(regions) >= 3:
                    conn.execute(text(
                        f"ALTER DATABASE {db_name} SURVIVE REGION FAILURE"))
                    log.info("survival goal set to REGION FAILURE")

            # SET LOCALITY is not a no-op on a table that already has that
            # locality: CockroachDB still runs a full schema-change job,
            # about 9s apiece on the 9-node chaos cluster, and boot used to
            # pay for four of them every time. Read what each table has and
            # only alter the ones that differ.
            localities = {str(r[0]): str(r[1]) for r in conn.execute(text(
                "SELECT table_name, locality FROM [SHOW TABLES] "
                "WHERE schema_name = 'public'")).fetchall()}

            # Table names are hardcoded here (not user input), safe to interpolate
            for table in ["expenses", "spending_signals", "coach_nudges"]:
                if localities.get(table) == "REGIONAL BY ROW":
                    log.info("%s already REGIONAL BY ROW", table)
                    continue
                log.info("converting %s to REGIONAL BY ROW (a real schema "
                         "change, can take a minute or two on a multi-node "
                         "cluster)", table)
                try:
                    conn.execute(text(f"ALTER TABLE {table} SET LOCALITY REGIONAL BY ROW"))
                    log.info("set %s to REGIONAL BY ROW", table)
                except Exception as e:
                    if "already regional by row" in str(e).lower() or "locality already set" in str(e).lower():
                        log.debug("%s already REGIONAL BY ROW: %s", table, e)
                    else:
                        raise

            # users is read on every request (identity pill, region
            # resolution) and written only at signup. GLOBAL gives every
            # region a fast local read and keeps auth instant while a
            # region is down, per the multi-region skill's guidance for
            # read-mostly reference tables.
            if localities.get("users") == "GLOBAL":
                log.info("users already GLOBAL")
            else:
                try:
                    conn.execute(text("ALTER TABLE users SET LOCALITY GLOBAL"))
                    log.info("set users to GLOBAL")
                except Exception as e:
                    if "already" in str(e).lower():
                        log.debug("users already GLOBAL: %s", e)
                    else:
                        log.warning("could not set users GLOBAL: %s", e)

            conn.commit()
            log.info("multi-region migration completed successfully")
            # Anything that asked "are the tables regional yet" before this
            # point cached a stale no (on a fresh cluster, sample-data setup
            # runs before this migration). Drop those answers so writers
            # start pinning and readers start pruning immediately.
            regional_tables_ready.cache_clear()
            detect_database_regions.cache_clear()
            _gateway_region_cache.clear()
            resolve_user_region_cache_clear()
            return True

    except Exception as e:
        log.exception("multi-region migration failed: %s", e)
        return False


class DatabaseMigration:
    """Database migration utilities."""
    
    def __init__(self, database_url: str | None = None):
        """Initialize migration manager."""
        self.database_url = get_database_url(database_url)

    @property
    def engine(self):
        """Get SQLAlchemy engine (lazy import)."""
        from ..db.engine import get_engine
        return get_engine(self.database_url)
    
    def migrate_timestamptz(self) -> bool:
        """Convert naive TIMESTAMP columns on banko's tables to
        TIMESTAMPTZ. Naive timestamps in a multi-region cluster invite
        interpretation bugs; the schema DDL now creates TIMESTAMPTZ and
        this brings existing databases along. Sessions run in UTC so the
        stored instants do not move."""
        tables = ("expenses", "agent_state", "agent_memory", "agent_tasks",
                  "agent_decisions", "documents", "query_cache",
                  "embedding_cache", "vector_search_cache", "cache_stats")
        try:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                rows = conn.execute(text("""
                    SELECT table_name, column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND data_type = 'timestamp without time zone'
                """)).fetchall()
                todo = [(t, c) for t, c in rows if t in tables]
                if not todo:
                    return True
                conn.execute(text(
                    "SET enable_experimental_alter_column_type_general = true"))
                for table, column in todo:
                    # Table and column names come from information_schema
                    # filtered to our own hardcoded table list.
                    conn.execute(text(
                        f'ALTER TABLE {table} ALTER COLUMN "{column}" '
                        f"TYPE TIMESTAMPTZ"))
                    print(f"   {table}.{column} -> TIMESTAMPTZ")
                conn.commit()
                return True
        except Exception as e:
            print(f"timestamptz migration skipped: {e}")
            return False

    def add_created_at_column(self) -> bool:
        """Add created_at timestamp column."""
        try:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                # Check if created_at column exists
                result = conn.execute(text("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'expenses' AND column_name = 'created_at'
                """))

                if not result.fetchone():
                    conn.execute(text("""
                        ALTER TABLE expenses
                        ADD COLUMN created_at TIMESTAMP DEFAULT now()
                    """))
                    print("Added created_at column to expenses table")
                    conn.commit()
                    return True
                else:
                    print("created_at column already exists")
                    return True

        except Exception as e:
            print(f"Failed to add created_at column: {e}")
            return False

    def migrate_users_table(self) -> bool:
        """Create users table for real user signup."""
        try:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS users (
                      user_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                      username        STRING NOT NULL UNIQUE,
                      spending_style  STRING NOT NULL,
                      home_region     STRING,
                      created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                      demo_user       BOOL NOT NULL DEFAULT true
                    )
                """))
                print("Created users table")
                conn.commit()

                # Row-level TTL was the wrong tool for demo-user expiry:
                # it swept personas and non-demo rows too, and orphaned
                # their expenses. Remove it where an older boot applied
                # it; expiry is now the boot-time sweep below.
                try:
                    ddl = conn.execute(text(
                        "SHOW CREATE TABLE users")).fetchone()[1]
                    if "ttl_expire_after" in str(ddl):
                        conn.execute(text("ALTER TABLE users RESET (ttl)"))
                        conn.commit()
                        print("Removed users table TTL (replaced by boot sweep)")
                except Exception as e:
                    print(f"users TTL reset skipped (non-fatal): {e}")

            self.expire_demo_users()
            return True

        except Exception as e:
            print(f"Users table migration failed: {e}")
            return False

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

    def migrate_to_coach_v1(self) -> bool:
        """Create spending_signals and coach_nudges tables for the Coach v1
        feature. Both tables use row-level TTL matching the LangGraph
        checkpoint pattern."""
        try:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS spending_signals (
                      signal_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                      user_id         UUID NOT NULL,
                      signal_type     STRING NOT NULL,
                      severity        STRING NOT NULL,
                      payload         JSONB NOT NULL,
                      produced_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                      consumed_at     TIMESTAMPTZ,
                      idempotency_key STRING NOT NULL UNIQUE,
                      INDEX (user_id, produced_at DESC)
                    ) WITH (ttl_expire_after = '30 days')
                """))
                print("Created spending_signals table")

                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS coach_nudges (
                      nudge_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                      signal_id      UUID REFERENCES spending_signals(signal_id),
                      user_id        UUID NOT NULL,
                      message        STRING NOT NULL,
                      tool_trace     JSONB,
                      provider_used  STRING,
                      trace_id       STRING,
                      created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                      INDEX (user_id, created_at DESC)
                    ) WITH (ttl_expire_after = '90 days')
                """))
                print("Created coach_nudges table")

                # set_budget creates this lazily, but get_user_budget can run
                # first (the nudge planner reads budgets), so it has to exist
                # from boot like the other coach tables.
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS user_budgets (
                      user_id   UUID NOT NULL,
                      category  STRING NOT NULL,
                      amount    DECIMAL NOT NULL,
                      updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                      PRIMARY KEY (user_id, category)
                    )
                """))
                print("Created user_budgets table")

                conn.commit()
                return True

        except Exception as e:
            print(f"Coach v1 migration failed: {e}")
            return False

    def run_all_migrations(self) -> bool:
        """Run all pending migrations."""
        print("Running database migrations...")

        success = True
        success &= self.add_created_at_column()
        success &= self.migrate_users_table()
        success &= self.migrate_to_coach_v1()
        success &= self.migrate_timestamptz()

        if success:
            print("All migrations completed successfully")
        else:
            print("Some migrations failed")

        return success
