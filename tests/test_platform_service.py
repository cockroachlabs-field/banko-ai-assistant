"""Platform SQL on a single node: lease holders, users per region, leases,
residency, vectors and CDC answer, and a serving-region sample never raises."""

import os

import pytest

from banko_ai.db.rls import RLS_TABLES
from banko_ai.services import platform as P
from banko_ai.services.load_users import clear, prepare

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


def test_sql_topology_reports_replica_holders_without_liveness():
    t = P.topology()
    assert t["source"] == "sql" and isinstance(t["regions"], list)
    assert t["nodes"] and all(n["live"] is None and n["leases"] is None and n["replicas"] >= 1 for n in t["nodes"])
    assert {n["node_id"] for n in t["nodes"]} == {1}


def test_users_by_region_counts_the_personas():
    rows = P.users_by_region()
    assert sum(r["personas"] for r in rows) == 3
    assert all(r["total"] == r["personas"] + r["demo"] + r["load"] for r in rows)


def test_lease_distribution_counts_ranges_per_region():
    assert set(P.lease_distribution()) == {"accounts"}
    leases = P.lease_distribution(("accounts", "ledger_entries", "expenses"))
    assert set(leases) == {"accounts", "ledger_entries", "expenses"}
    assert all(sum(by_region.values()) >= 1 for by_region in leases.values())


def test_residency_lists_localities_and_rls():
    r = P.residency()
    rows = {t["name"]: t for t in r["tables"]}
    assert set(RLS_TABLES) <= set(rows) and "expenses" in rows and "users" in rows
    assert all(isinstance(t["rls"], bool) for t in r["tables"])
    assert not any(rows[name]["rls"] for name in ("expenses", "memories", "users"))  # vector and GLOBAL tables stay policy free
    assert isinstance(r["users_global"], bool)


def test_vector_stats_count_vectors():
    v = P.vector_stats()
    assert v["vectors"] >= 0 and v["users_with_vectors"] >= 0 and "idx_expenses_user_embedding" in v["index"]


def test_cdc_status_never_raises():
    c = P.cdc_status()
    assert isinstance(c["changefeeds"], list) and c["signals_per_min"] >= 0 and c["nudges_per_min"] >= 0


def test_sample_serving_runs_an_explain():
    clear()
    try:
        user = prepare(1)[0]
        served = P.sample_serving(user.user_id)
        assert served is None or isinstance(served, str)
    finally:
        clear()
