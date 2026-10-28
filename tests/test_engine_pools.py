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
        if self.region == "broken":
            raise RuntimeError("connection reset by peer")

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


def test_an_unreadable_gateway_is_asked_again_not_kept():
    """A gateway check that fails (a draining node, a reset during an outage)
    says nothing about regions: that connection is dropped and the balancer
    asked again, instead of keeping it as if the cluster had none."""
    connect, closed = _balancer(["broken", "us-east-1"])
    conn, gateway = E.homed_connection(connect, "us-east-1")
    assert gateway == "us-east-1" and closed == ["broken"]


def test_a_connection_whose_gateway_stays_unreadable_counts_as_away():
    connect, closed = _balancer(["broken"] * 30)
    conn, gateway = E.homed_connection(connect, "us-east-1", attempts=3)
    assert gateway == "unknown" and E.should_rehome({"gateway": gateway, "opened": 0.0}, "us-east-1", now=1e9)


def _counting(regions):
    """A balancer that counts connections."""
    connect, closed = _balancer(regions)
    count = {"n": 0}

    def counted():
        count["n"] += 1
        return connect()
    return counted, count


def test_a_home_this_url_never_reaches_is_not_shopped_for_on_every_connection():
    """A URL to one region (CockroachDB Cloud, one node, a sticky balancer)
    never reaches the others: after one miss, new connections come as they
    land, and the home is tried again only after UNREACHABLE_RETRY_S."""
    connect, count = _counting(["eu-west-1"] * 1000)
    home = E._Home("us-east-1")
    home.connect(connect, now=0.0)
    first = count["n"]
    assert first == E.HOME_ATTEMPTS + 1
    for t in (1.0, 60.0, E.UNREACHABLE_RETRY_S - 1):
        home.connect(connect, now=t)
    assert count["n"] == first + 3
    away = {"gateway": "eu-west-1", "opened": 0.0}
    assert not home.may_rehome(away, now=E.UNREACHABLE_RETRY_S - 1)
    assert home.may_rehome(away, now=E.UNREACHABLE_RETRY_S + 1)
    home.connect(connect, now=E.UNREACHABLE_RETRY_S + 1)
    assert count["n"] == first + 3 + E.HOME_ATTEMPTS + 1


def test_a_home_that_was_reached_is_tried_again_soon_after_a_miss():
    """Once a URL has reached the home region, a miss means the region is
    down: new connections stop shopping for REHOME_AFTER_S, then try home."""
    connect, count = _counting(["us-east-1"] + ["eu-west-1"] * 1000)
    home = E._Home("us-east-1")
    assert home.connect(connect, now=0.0)[1] == "us-east-1"
    home.connect(connect, now=10.0)                      # the region went down: a full miss
    before = count["n"]
    home.connect(connect, now=11.0)                      # cooling: one connection, as it lands
    assert count["n"] == before + 1
    away = {"gateway": "eu-west-1", "opened": 10.0}
    assert not home.may_rehome(away, now=10.0 + E.REHOME_AFTER_S - 1)
    assert home.may_rehome(away, now=10.0 + E.REHOME_AFTER_S + 1)


def test_the_home_listeners_bring_an_away_connection_back(monkeypatch):
    """The do_connect and checkout listeners together: a pooled connection
    that landed away is replaced on checkout once home may be tried again."""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import QueuePool

    gateways = iter(["eu-west-1"] * (E.HOME_ATTEMPTS + 1) + ["us-east-1"] * 5)
    monkeypatch.setattr(E, "_gateway_of", lambda conn: next(gateways))
    clock = {"t": 0.0}
    monkeypatch.setattr(E, "_now", lambda: clock["t"])
    engine = create_engine("sqlite://", poolclass=QueuePool, pool_size=1, max_overflow=0)
    E._home(engine, "us-east-1")
    with engine.connect() as c:
        assert c.connection.info["gateway"] == "eu-west-1"
    clock["t"] = E.UNREACHABLE_RETRY_S + 1
    with engine.connect() as c:
        assert c.connection.info["gateway"] == "us-east-1"


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


@needs_db
def test_an_operation_timeout_bounds_every_statement_on_its_pool():
    """A partition or brownout makes statements hang until TCP gives up;
    a pool with an operation timeout ends them after that many seconds."""
    url = E.with_operation_timeout(with_application_name(os.environ["DATABASE_URL"], "banko-timeout-test"), 7)
    with get_engine(url, pool_size=1).connect() as c:
        assert c.execute(text("SHOW statement_timeout")).scalar() in ("7000", "7s", "00:00:07")
        assert c.execute(text("SHOW application_name")).scalar() == "banko-timeout-test"
