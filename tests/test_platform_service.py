"""Platform SQL on a single node: lease holders, users per region, leases,
residency, vectors and CDC answer, and a serving-region sample never raises."""

import os
import time
import uuid

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


def test_the_changefeed_statement_runs_on_this_cluster():
    """It subtracted a DECIMAL from a FLOAT, which fails when the statement
    is planned, with or without a changefeed. The card swallowed that and
    said there were none."""
    with P._session(None) as conn:
        assert isinstance(P.changefeeds(conn), list)


def test_a_running_changefeed_is_listed_with_its_lag():
    from sqlalchemy import text

    from banko_ai.db.engine import get_engine
    table = f"cdc_probe_{uuid.uuid4().hex[:8]}"
    job = None
    with get_engine().connect() as conn:
        conn = conn.execution_options(isolation_level="AUTOCOMMIT")
        was = conn.execute(text("SHOW CLUSTER SETTING kv.rangefeed.enabled")).scalar()
        try:
            conn.execute(text("SET CLUSTER SETTING kv.rangefeed.enabled = true"))
            conn.execute(text(f"CREATE TABLE {table} (id INT PRIMARY KEY)"))
            try:
                job = conn.execute(text(f"CREATE CHANGEFEED FOR TABLE {table} INTO 'null://' "
                                        "WITH resolved = '1s', min_checkpoint_frequency = '1s'")).scalar()
            except Exception as exc:
                pytest.skip(f"this cluster will not start a changefeed ({exc.__class__.__name__})")
            deadline, mine = time.monotonic() + 60, None
            while time.monotonic() < deadline:
                mine = next((f for f in P.cdc_status()["changefeeds"] if f["job_id"] == str(job)), None)
                if mine and mine["lag_s"] is not None:
                    break
                time.sleep(0.5)
            assert mine is not None, "the changefeed is running and the card does not list it"
            assert mine["status"] == "running" and mine["sink"] == "null"
            assert mine["lag_s"] is not None and 0 <= mine["lag_s"] < 120
        finally:
            if job is not None:
                conn.execute(text(f"CANCEL JOB {int(job)}"))
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
            conn.execute(text(f"SET CLUSTER SETTING kv.rangefeed.enabled = {'true' if was else 'false'}"))


def test_changefeeds_that_cannot_be_listed_are_not_reported_as_none(monkeypatch, caplog):
    """Listing can fail for a real reason (a role that may not see jobs). The
    card then says so, and the log says why, once."""
    def refused(conn):
        raise RuntimeError("user banko_app does not have VIEWJOB privilege")

    monkeypatch.setattr(P, "changefeeds", refused)
    monkeypatch.setattr(P, "_unlisted", set())
    with caplog.at_level("WARNING", logger="banko.platform"):
        first, second = P.cdc_status(), P.cdc_status()
    assert first["changefeeds"] is None and second["changefeeds"] is None
    assert first["signals_per_min"] >= 0  # the rest of the card still answers
    said = [r.getMessage() for r in caplog.records if "changefeed" in r.getMessage()]
    assert len(said) == 1 and "VIEWJOB" in said[0]


def test_sample_serving_runs_an_explain():
    clear()
    try:
        user = prepare(1)[0]
        served = P.sample_serving(user.user_id)
        assert served is None or isinstance(served, str)
    finally:
        clear()
