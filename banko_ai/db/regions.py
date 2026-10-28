"""Region detection and topology helpers for multi-region deployments."""

import logging
import time

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
    _scoped_keys.clear()
    _home_regions.clear()
    _rows_home.clear()
    _cluster_regions_cache.clear()
    _database_regions_cache.clear()
    _tables_ready_cache.clear()
    _gateway_region_cache.clear()
    _money_tables_regional.clear()
    resolve_user_region_cache_clear()


_region_columns_seen: set[tuple[str, str]] = set()

# Every table a money write may pin crdb_region on.
MONEY_TABLES = ("accounts", "ledger_entries", "transfers", "goals", "user_budgets",
                "user_data_versions", "expenses")
_money_tables_regional: set[str] = set()  # database URLs (no password) where all have it


def _all_money_tables_regional(conn) -> bool:
    """Whether every money table has crdb_region yet. On an existing
    database the conversion to REGIONAL BY ROW runs in the background, one
    table at a time, and pinning crdb_region on a table that has not
    converted fails the insert. Positive answers are cached per database."""
    key = conn.engine.url.render_as_string(hide_password=True)
    if key in _money_tables_regional:
        return True
    found = conn.execute(text("SELECT count(DISTINCT table_name) FROM information_schema.columns "
                              "WHERE column_name = 'crdb_region' AND table_name = ANY(:t)"),
                         {"t": list(MONEY_TABLES)}).scalar()
    if found == len(MONEY_TABLES):
        _money_tables_regional.add(key)
        return True
    return False


def money_tables_regional(url: str) -> bool:
    """Whether REGIONAL BY ROW is live on every money table of this database
    (cached once it is), the condition under which writes pin crdb_region."""
    if not regional_tables_ready(url):
        return False
    try:
        with get_engine(url).connect() as conn:
            return _all_money_tables_regional(conn)
    except Exception as e:
        log.debug("money table locality check failed: %s", e)
        return False


def money_region(conn, user_id: str, url: str | None = None) -> str | None:
    """The user's home region when REGIONAL BY ROW tables are live, else None.

    Inserts into REGIONAL BY ROW money tables pin crdb_region to this value,
    so a user's accounts and ledger live where the user lives instead of
    wherever the gateway node happened to be. Until every money table has
    finished converting, nothing is pinned."""
    from ..config.settings import get_config

    if not regional_tables_ready(url or get_config().database_url):
        return None
    if not _all_money_tables_regional(conn):
        return None
    row = conn.execute(text("SELECT home_region FROM users WHERE user_id = CAST(:u AS UUID)"),
                       {"u": str(user_id)}).fetchone()
    return str(row[0]) if row and row[0] else None


# The unique keys a database may have scoped to the region (schema.py):
# expenses created multi-region by Banko 2.0, transfers after the upgrade.
_SCOPED = {"expenses": ["crdb_region", "expense_id"], "transfers": ["crdb_region", "user_id", "idempotency_key"]}
SCOPED_RECHECK_S = 60.0  # a "not scoped" answer is asked again after this long (an upgrade may have run elsewhere)
_scoped_keys: dict[tuple[str, str], tuple[bool, float]] = {}


def scoped_key(conn, table: str) -> bool:
    """Whether this table's unique key leads with crdb_region, so ON CONFLICT
    must name the region too and the row's region must not depend on the
    gateway. Positive answers are kept per database; negative ones for
    SCOPED_RECHECK_S."""
    key = (conn.engine.url.render_as_string(hide_password=True), table)
    now = time.monotonic()
    known = _scoped_keys.get(key)
    if known and (known[0] or now - known[1] < SCOPED_RECHECK_S):
        return known[0]
    keys: dict[str, list[str]] = {}
    for name, column in conn.execute(text(
            f"SELECT index_name, column_name FROM [SHOW INDEXES FROM {table}] "
            "WHERE NOT non_unique AND NOT storing AND NOT implicit ORDER BY index_name, seq_in_index")):
        keys.setdefault(str(name), []).append(str(column))
    found = _SCOPED[table] in keys.values()
    _scoped_keys[key] = (found, now)
    return found


HOME_REGION_TTL_S = 600.0
_home_regions: dict[tuple[str, str], tuple[str | None, float]] = {}


def _owner(url: str | None) -> str:
    """The URL to read region facts with: the one a call targets, or the
    configured database; never the row-level security role's."""
    from ..config.settings import get_config
    from . import rls

    cfg = get_config()
    return cfg.database_url if not url or url == rls.app_url() else url


def _db_key(url: str) -> str:
    """One cache key per database, whatever pool options the URL carries."""
    from sqlalchemy.engine import make_url

    from ..utils.db_retry import normalize_db_url

    parsed = make_url(normalize_db_url(url))
    return parsed.set(query={k: v for k, v in parsed.query.items() if k == "sslmode"}).render_as_string(
        hide_password=True)


def user_home_region(user_id: str, url: str | None = None) -> str | None:
    """The user's home region in the database url targets, remembered for
    HOME_REGION_TTL_S. users is GLOBAL, so a miss is a local read."""
    base = _owner(url)
    key = (_db_key(base), str(user_id))
    now = time.monotonic()
    known = _home_regions.get(key)
    if known and now - known[1] < HOME_REGION_TTL_S:
        return known[0]
    with get_engine(base).connect() as conn:
        region = home_region(conn, user_id)
    _home_regions[key] = (region, now)
    return region


ROWS_HOME_TTL_S = 600.0
_rows_home: dict[tuple[str, str], tuple[bool, float]] = {}
_READ_TABLES = ("accounts", "ledger_entries", "expenses")  # the tables region-pinned reads touch


def _users_with_rows_away(user_ids: list[str], url: str | None) -> set[str]:
    """Which of these users have a money row outside their home region."""
    away: set[str] = set()
    with get_engine(_owner(url)).connect() as conn:
        for table in _READ_TABLES:
            away |= {str(r[0]) for r in conn.execute(text(f"""
                SELECT DISTINCT t.user_id FROM {table} AS t JOIN users AS u ON u.user_id = t.user_id
                WHERE t.user_id = ANY(CAST(:ids AS UUID[]))
                  AND t.crdb_region::STRING IS DISTINCT FROM u.home_region"""), {"ids": list(user_ids)})}
    return away


def rows_home(user_id: str, region: str, url: str | None = None) -> bool:
    """Whether every money row this user's reads touch lives in their home
    region, in the database url targets. A yes is kept (new rows are written
    there); a no is checked again after ROWS_HOME_TTL_S."""
    key = (_db_key(_owner(url)), str(user_id))
    now = time.monotonic()
    known = _rows_home.get(key)
    if known and (known[0] or now - known[1] < ROWS_HOME_TTL_S):
        return known[0]
    home = str(user_id) not in _users_with_rows_away([str(user_id)], url)
    _rows_home[key] = (home, now)
    return home


def prime_rows_home(user_ids: list[str], url: str | None = None) -> None:
    """rows_home for many users in one pass (the load simulator's customers,
    when a run starts)."""
    if not user_ids:
        return
    away = _users_with_rows_away(user_ids, url)
    db, now = _db_key(_owner(url)), time.monotonic()
    for uid in user_ids:
        _rows_home[(db, str(uid))] = (str(uid) not in away, now)


def read_region(user_id: str, url: str | None) -> str | None:
    """The region this user's reads may name (region_pin), or None. Only
    with routing by home region on, on a connection without row-level
    security (CockroachDB 26.3 keeps the region filter out of an RLS scan),
    when every money table is REGIONAL BY ROW, and for a user whose money
    rows all live in their home region."""
    from ..config.settings import get_config
    from . import rls

    if not get_config().route_by_home_region:
        return None
    app = rls.app_url()
    if url is not None and app is not None and url == app:
        return None
    base = _owner(url)
    if not is_regional_deployment(base) or not money_tables_regional(base):
        return None
    region = user_home_region(str(user_id), base)
    if not region or not rows_home(str(user_id), region, base):
        return None
    return region


def home_region(conn, user_id: str) -> str | None:
    """The user's home region from the GLOBAL users table, or None."""
    row = conn.execute(text("SELECT home_region FROM users WHERE user_id = CAST(:u AS UUID)"),
                       {"u": str(user_id)}).fetchone()
    return str(row[0]) if row and row[0] else None


# A region-scoped row needs a region that never depends on the gateway, or a
# retry through another region could write it twice: the home region, or the
# database's primary region for a user without one. Reads :scoped_region.
SCOPED_REGION_VALUE = "default_to_database_primary_region(:scoped_region)::crdb_internal_region, "


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
    # Cache key never carries the password (explicit hide_password=True)
    key = (conn.engine.url.render_as_string(hide_password=True), table)
    if key in _region_columns_seen:
        return True
    row = conn.execute(text("SELECT 1 FROM information_schema.columns "
                            "WHERE table_name = :t AND column_name = 'crdb_region' LIMIT 1"),
                       {"t": table}).fetchone()
    if row is not None:
        _region_columns_seen.add(key)
    return row is not None
