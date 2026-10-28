"""Named and homed pools: each its own engine and connections, labeled in
SHOW SESSIONS; a homed pool's connections enter the cluster in its region."""

import os

import pytest
from sqlalchemy import text

from banko_ai.db import engine as E
from banko_ai.db.engine import get_engine, with_application_name, with_gateway_region

needs_db = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@needs_db
def test_a_named_pool_shows_its_application_name():
    url = with_application_name(os.environ["DATABASE_URL"], "banko-load-test")
    engine = get_engine(url, pool_size=3)
    with engine.connect() as c:
        assert c.execute(text("SHOW application_name")).scalar() == "banko-load-test"
    assert engine.pool.size() == 3 and get_engine(url) is engine
    with get_engine().connect() as c:
        assert c.execute(text("SHOW application_name")).scalar() == "banko"


def test_with_application_name_keeps_the_rest_of_the_url():
    url = with_application_name("postgresql://root@localhost:26258/defaultdb?sslmode=disable", "banko-load")
    assert "sslmode=disable" in url and "application_name=banko-load" in url


# A pool homed on a region keeps only connections whose gateway node is in
# that region, asking a load balancer over every node again until one lands
# there, and connects anywhere when the region has no live node.

class _Cursor:
    def __init__(self, region):
        self.region = region

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql):
        if self.region is None:
            raise RuntimeError("no region set on the locality flag on this node")

    def fetchone(self):
        return (self.region,)


class _Conn:
    def __init__(self, region, closed):
        self.region, self._closed_log, self.closed = region, closed, False

    def cursor(self):
        return _Cursor(self.region)

    def rollback(self):
        pass

    def close(self):
        self.closed = True
        self._closed_log.append(self.region)


def _balancer(regions):
    """Round robin over nodes in these regions; "down" refuses the connection."""
    closed, nodes = [], iter(regions)

    def connect():
        region = next(nodes)
        if region == "down":
            raise OSError("connection refused")
        return _Conn(region, closed)
    return connect, closed


def test_a_homed_connection_is_one_whose_gateway_is_in_the_region():
    connect, closed = _balancer(["eu-west-1", "us-west-2", "us-east-1", "eu-west-1"])
    conn, gateway = E.homed_connection(connect, "us-east-1")
    assert conn.region == gateway == "us-east-1" and closed == ["eu-west-1", "us-west-2"]


def test_a_region_without_a_live_node_connects_anywhere():
    connect, closed = _balancer(["down", "eu-west-1", "us-west-2"] * 10)
    conn, gateway = E.homed_connection(connect, "us-east-1", attempts=4)
    assert gateway in ("eu-west-1", "us-west-2") and conn.region == gateway and not conn.closed


def test_a_cluster_without_regions_keeps_its_first_connection():
    connect, closed = _balancer([None, None])
    conn, gateway = E.homed_connection(connect, "us-east-1")
    assert conn.region is None and gateway is None and closed == []


def test_a_connection_away_from_home_is_replaced_after_a_while():
    away = {"gateway": "eu-west-1", "opened": 100.0}
    assert E.should_rehome(away, "us-east-1", now=100.0 + E.REHOME_AFTER_S + 1)
    assert not E.should_rehome(away, "us-east-1", now=101.0)
    assert not E.should_rehome({"gateway": "us-east-1", "opened": 0.0}, "us-east-1", now=1e9)
    assert not E.should_rehome({"gateway": None, "opened": 0.0}, "us-east-1", now=1e9)


@needs_db
def test_a_gateway_region_pool_is_its_own_and_works_without_regions():
    url = with_gateway_region(with_application_name(os.environ["DATABASE_URL"], "banko-home-test"), "us-east-1")
    engine = get_engine(url, pool_size=2)
    assert engine is not get_engine(with_application_name(os.environ["DATABASE_URL"], "banko-home-test"))
    with engine.connect() as c:
        assert c.execute(text("SHOW application_name")).scalar() == "banko-home-test"


@pytest.mark.skipif(not os.getenv("BANKO_TEST_MR_URL"), reason="BANKO_TEST_MR_URL not set")
def test_a_homed_pool_connects_through_its_own_region():
    mr = os.environ["BANKO_TEST_MR_URL"]
    with get_engine(mr).connect() as c:
        regions = [str(r[0]) for r in c.execute(text("SELECT region FROM [SHOW REGIONS FROM CLUSTER]"))]
    assert len(regions) > 1
    for region in regions:
        engine = get_engine(with_gateway_region(mr, region), pool_size=2)
        for _ in range(3):
            with engine.connect() as c:
                assert c.execute(text("SELECT gateway_region()")).scalar() == region
