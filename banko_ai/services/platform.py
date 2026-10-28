"""SQL behind the Platform page: lease holders and regions (when neither
the chaos control plane nor CockroachDB's HTTP API answers), users per
region, lease distribution, residency, vectors, CDC, and which regions
serve a customer's read. Statements run on the admin URL with a short
statement timeout, so a struggling cluster costs the page a stale card,
never a hung request."""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import cast
from urllib.parse import urlsplit

from sqlalchemy import text
from sqlalchemy.engine import Connection

from ..config.catalog import get_catalog
from ..config.settings import get_config
from ..db import rls
from ..db.engine import get_engine
from ..utils.request_trace import serving_regions

log = logging.getLogger("banko.platform")

LEASE_TABLES = ("accounts", "ledger_entries", "expenses")
VECTOR_INDEX = "idx_expenses_user_embedding: C-SPANN vector index on (user_id, embedding), cosine distance"


@contextmanager
def _session(url: str | None, timeout: str = "2s") -> Iterator[Connection]:
    """Autocommit, so one failed statement never poisons the next, with a
    statement timeout for the session that is reset before the connection
    goes back to the pool."""
    with get_engine(url or get_config().database_url).connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        conn.execute(text(f"SET statement_timeout = '{timeout}'"))
        try:
            yield conn
        finally:
            try:
                conn.execute(text("RESET statement_timeout"))
            except Exception:  # a session that cannot reset must not go back to the pool
                conn.invalidate()


def _locality(value: str | None) -> dict[str, str]:
    return dict(part.split("=", 1) for part in (value or "").split(",") if "=" in part)


def _regions(conn: Connection) -> list[str]:
    try:
        return sorted(str(r[0]) for r in conn.execute(text("SELECT region FROM [SHOW REGIONS FROM DATABASE]")))
    except Exception:
        return []


def cluster_regions(*, url: str | None = None) -> list[str]:
    """The cluster's regions (empty without localities): how a derived HTTP
    API address is checked against the database Banko is connected to."""
    with _session(url) as conn:
        return sorted(str(r[0]) for r in conn.execute(text("SELECT region FROM [SHOW REGIONS FROM CLUSTER]")))


def _lease_holders(conn: Connection, table: str) -> list[tuple[int, dict[str, str]]]:
    # WITH DETAILS asks every range for its span stats: about 7 s per table
    # on a three-region cluster with realistic latency, so only the
    # background refresh calls this, and only without a node-level source.
    rows = conn.execute(text(
        f"SELECT lease_holder, lease_holder_locality FROM [SHOW RANGES FROM TABLE {table} WITH DETAILS]"))
    return [(int(holder), _locality(locality)) for holder, locality in rows if holder is not None]


def topology(*, url: str | None = None) -> dict:
    """Regions and the nodes holding replicas of Banko's accounts, for a
    cluster where neither the chaos plane nor the HTTP API answers. SQL
    cannot say which nodes are live (crdb_internal is restricted) or hold
    leases (that needs the slow WITH DETAILS), so live and leases are None."""
    nodes: dict[int, dict] = {}
    with _session(url, "5s") as conn:
        regions = _regions(conn)
        for replicas, localities in conn.execute(text(
                "SELECT replicas, replica_localities FROM [SHOW RANGES FROM TABLE accounts]")):
            for node_id, locality in zip(replicas or [], localities or []):
                tiers = _locality(locality)
                node = nodes.setdefault(int(node_id), {
                    "node_id": int(node_id), "address": None, "region": tiers.get("region"),
                    "zone": tiers.get("zone"), "live": None, "status": None, "replicas": 0, "leases": None})
                node["replicas"] += 1
    return {"source": "sql", "regions": regions, "nodes": sorted(nodes.values(), key=lambda n: n["node_id"])}


def users_by_region(*, url: str | None = None) -> list[dict]:
    personas = [p.user_id for p in get_catalog().personas]
    with _session(url) as conn:
        rows = conn.execute(text("""
            SELECT home_region,
                   count(*) FILTER (WHERE user_id = ANY(CAST(:personas AS UUID[]))),
                   count(*) FILTER (WHERE username LIKE 'load-%'),
                   count(*)
            FROM users GROUP BY home_region ORDER BY home_region NULLS FIRST
        """), {"personas": personas}).fetchall()
    return [{"region": r[0], "personas": int(r[1]), "load": int(r[2]), "demo": int(r[3] - r[1] - r[2]),
             "total": int(r[3])} for r in rows]


def lease_distribution(tables: Sequence[str] = ("accounts",), *, url: str | None = None,
                       timeout: str = "20s") -> dict[str, dict[str, int]]:
    """Ranges per lease holder region, per table ("all" on a database
    without regions). Slow on a multi-region cluster: for the background
    refresh when no node-level source reports leases."""
    out: dict[str, dict[str, int]] = {}
    with _session(url, timeout) as conn:
        for table in tables:
            counts: dict[str, int] = {}
            for _, tiers in _lease_holders(conn, table):
                region = tiers.get("region") or "all"
                counts[region] = counts.get(region, 0) + 1
            out[table] = counts
    return out


def residency(*, url: str | None = None) -> dict:
    with _session(url) as conn:
        rows = conn.execute(text(
            "SELECT table_name, locality FROM [SHOW TABLES] WHERE schema_name = 'public' ORDER BY table_name")).fetchall()
        try:
            protected = rls.policy_tables(conn)
        except Exception:
            protected = set()
    status = rls.status()
    tables = [{"name": str(name), "locality": locality, "rls": name in protected} for name, locality in rows]
    users = next((t for t in tables if t["name"] == "users"), None)
    return {"tables": tables, "users_global": bool(users and (users["locality"] or "").upper() == "GLOBAL"),
            "rls": {"enabled": status.enabled, "reason": status.reason}}


def vector_stats(*, url: str | None = None) -> dict:
    sql = "SELECT count(*), count(DISTINCT user_id) FROM expenses{aost} WHERE embedding IS NOT NULL"
    with _session(url, "5s") as conn:
        try:
            row = conn.execute(text(sql.format(aost=" AS OF SYSTEM TIME follower_read_timestamp()"))).fetchone()
        except Exception:  # no follower reads on this cluster: read the current data
            row = conn.execute(text(sql.format(aost=""))).fetchone()
    vectors, users = (row[0], row[1]) if row is not None else (0, 0)
    return {"vectors": int(vectors), "users_with_vectors": int(users), "index": VECTOR_INDEX}


def _sink(uri: str | None) -> str | None:
    """The sink's kind and host only: a sink URI can carry credentials."""
    if not uri:
        return None
    parts = urlsplit(uri)
    return f"{parts.scheme}://{parts.hostname}" if parts.hostname else parts.scheme or None


def cdc_status(*, url: str | None = None) -> dict:
    with _session(url) as conn:
        try:
            feeds = [{"job_id": str(r[0]), "status": r[1], "running_status": r[2],
                      "lag_s": float(r[3]) if r[3] is not None else None, "sink": _sink(r[4])}
                     for r in conn.execute(text("""
                         SELECT job_id, status, running_status,
                                round(extract(epoch FROM now()) - high_water_timestamp / 1e9, 1), sink_uri
                         FROM [SHOW CHANGEFEED JOBS] ORDER BY created DESC LIMIT 10"""))]
        except Exception:
            feeds = []
        signals = conn.execute(text(
            "SELECT count(*) FROM spending_signals WHERE produced_at > now() - INTERVAL '10 minutes'")).scalar() or 0
        nudges = conn.execute(text(
            "SELECT count(*) FROM coach_nudges WHERE created_at > now() - INTERVAL '10 minutes'")).scalar() or 0
    return {"changefeeds": feeds, "signals_per_min": round(int(signals) / 10, 1),
            "nudges_per_min": round(int(nudges) / 10, 1)}


def sample_serving(user_id: str, *, url: str | None = None) -> str | None:
    """Which regions served one of this customer's page-load reads, from its
    EXPLAIN ANALYZE ("regions: ..."), or None on a cluster without regions."""
    import psycopg

    with get_engine(url or get_config().database_url).connect() as conn:
        with psycopg.ClientCursor(cast(psycopg.Connection, conn.connection.driver_connection)) as cur:
            # Local to the implicit transaction, which the pool rolls back.
            cur.execute("SELECT set_config('banko.user_id', %s, true)", (user_id,))
            cur.execute("EXPLAIN ANALYZE SELECT account_id, balance FROM accounts WHERE user_id = %s", (user_id,))
            plan = "\n".join(str(r[0]) for r in cur.fetchall())
    return serving_regions(plan)
