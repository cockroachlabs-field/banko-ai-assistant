"""Region detection and topology helpers for multi-region deployments."""

import logging

from sqlalchemy import text

from .engine import get_engine

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

    try:
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

    try:
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

    try:
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
    regions = detect_regions(database_url)
    if not regions:
        return None

    try:
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
        from ..web.auth import _user_region_cache
        _user_region_cache.clear()
    except Exception:
        pass


def clear_region_caches() -> None:
    """Forget every cached topology answer after the schema changes it."""
    _cluster_regions_cache.clear()
    _database_regions_cache.clear()
    _tables_ready_cache.clear()
    _gateway_region_cache.clear()
    resolve_user_region_cache_clear()


_region_columns_seen: set[tuple[str, str]] = set()


def money_region(conn, user_id: str, url: str | None = None) -> str | None:
    """The user's home region when REGIONAL BY ROW tables are live, else None.

    Inserts into REGIONAL BY ROW money tables pin crdb_region to this value,
    so a user's accounts and ledger live where the user lives instead of
    wherever the gateway node happened to be."""
    from ..config.settings import get_config

    if not regional_tables_ready(url or get_config().database_url):
        return None
    row = conn.execute(text("SELECT home_region FROM users WHERE user_id = CAST(:u AS UUID)"),
                       {"u": str(user_id)}).fetchone()
    return str(row[0]) if row and row[0] else None


def region_columns(region: str | None) -> tuple[str, str]:
    """The column-list prefix and value prefix that pin crdb_region on an
    INSERT, or two empty strings when nothing should be pinned. The value
    prefix reads the :crdb_region bind parameter."""
    if not region:
        return "", ""
    return "crdb_region, ", "CAST(:crdb_region AS crdb_internal_region), "


def has_region_column(conn, table: str) -> bool:
    """Whether the table has crdb_region yet. user_budgets converts to
    REGIONAL BY ROW in the background on existing databases, so writers
    check before pinning. Positive answers are cached per database."""
    key = (str(conn.engine.url), table)
    if key in _region_columns_seen:
        return True
    row = conn.execute(text("SELECT 1 FROM information_schema.columns "
                            "WHERE table_name = :t AND column_name = 'crdb_region' LIMIT 1"),
                       {"t": table}).fetchone()
    if row is not None:
        _region_columns_seen.add(key)
    return row is not None
