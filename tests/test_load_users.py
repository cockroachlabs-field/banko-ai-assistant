"""Load users: simulated customers built from the seed profiles, homed
across the database's regions, removed with every row they own."""

import os
from decimal import Decimal

import pytest
from sqlalchemy import text

from banko_ai.config.catalog import get_catalog
from banko_ai.db.engine import get_engine
from banko_ai.services.integrity import check_integrity
from banko_ai.services.load_users import LOAD_PREFIX, clear, load_users, prepare

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")
MR_URL = os.getenv("BANKO_TEST_MR_URL")


def _ledger_rows(ids: list[str]) -> int:
    with get_engine().connect() as c:
        return c.execute(text("SELECT count(*) FROM ledger_entries WHERE user_id = ANY(CAST(:ids AS UUID[]))"),
                         {"ids": ids}).scalar()


def _balances(user) -> dict[str, Decimal]:
    with get_engine().connect() as c:
        by_id = {str(r[0]): r[1] for r in c.execute(text(
            "SELECT account_id, balance FROM accounts WHERE user_id = CAST(:u AS UUID)"), {"u": user.user_id})}
    return {key: by_id[acct] for key, acct in user.accounts.items()}


def test_prepare_creates_profile_accounts_and_tops_up():
    clear()
    first = prepare(4)
    assert len(first) == 4 and all(u.username.startswith(LOAD_PREFIX) for u in first)
    profiles = get_catalog().profiles
    assert len({u.style for u in first}) == min(4, len(profiles))
    profile = profiles[first[0].style]
    assert set(first[0].accounts) == {a.key for a in profile.accounts}
    assert _balances(first[0]) == {a.key: a.opening_balance for a in profile.accounts}
    assert len(prepare(2)) == 4 and len(prepare(6)) == 6
    users = load_users()
    assert [u.user_id for u in users[:4]] == [u.user_id for u in first]
    assert _ledger_rows([u.user_id for u in users]) == 0
    for u in users:
        assert check_integrity(follower_reads=False, user_id=u.user_id).ok
    assert clear() == 6 and load_users() == []


def test_a_single_region_database_homes_nobody():
    clear()
    try:
        assert all(u.region is None for u in prepare(2))
    finally:
        clear()


needs_mr = pytest.mark.skipif(not MR_URL, reason="BANKO_TEST_MR_URL not set")


@pytest.fixture(scope="module")
def mr_db():
    """A fresh multi-region database with six load users, dropped afterwards."""
    from sqlalchemy.engine import make_url

    from banko_ai.db.schema import ensure_schema
    from banko_ai.utils.db_retry import normalize_db_url

    admin = get_engine(MR_URL)
    with admin.connect() as c:
        c.execution_options(isolation_level="AUTOCOMMIT")
        c.execute(text("DROP DATABASE IF EXISTS banko_load_mr_test CASCADE"))
        c.execute(text("CREATE DATABASE banko_load_mr_test"))
    url = make_url(normalize_db_url(MR_URL)).set(database="banko_load_mr_test").render_as_string(hide_password=False)
    try:
        ensure_schema(url)
        yield url, prepare(6, url=url)
    finally:
        with admin.connect() as c:
            c.execution_options(isolation_level="AUTOCOMMIT")
            c.execute(text("DROP DATABASE IF EXISTS banko_load_mr_test CASCADE"))


@needs_mr
def test_load_users_rows_live_in_their_home_regions(mr_db):
    from banko_ai.db.regions import detect_database_regions

    url, users = mr_db
    regions = sorted(detect_database_regions(url))
    assert len(regions) > 1
    assert [u.region for u in users] == [regions[i % len(regions)] for i in range(6)]
    with get_engine(url).connect() as c:
        rows = c.execute(text("SELECT a.crdb_region::STRING, u.home_region FROM accounts a "
                              "JOIN users u ON u.user_id = a.user_id WHERE u.username LIKE 'load-%'")).fetchall()
    assert len(rows) == sum(len(u.accounts) for u in users) and all(r[0] == r[1] for r in rows)


@needs_mr
def test_reads_pinned_to_the_home_region_return_what_unpinned_reads_do(mr_db):
    """The simulator names each customer's home region in its reads, which
    is only right because every row a load customer owns lives there."""
    import random

    from banko_ai.db.engine import with_gateway_region
    from banko_ai.services import accounts, transactions
    from banko_ai.services.platform import sample_serving
    from banko_ai.workers.workload import OpContext, VectorCache, run_op

    url, users = mr_db
    ctx, rng = OpContext(url=url, vectors=VectorCache.build()), random.Random(9)
    for user in users[:3]:
        for op in ("purchase", "purchase", "deposit", "transfer"):
            run_op(op, user, ctx, rng)
        uid, region, card = user.user_id, user.region, user.accounts["card"]
        assert accounts.list_accounts(uid, url=url, region=region) == accounts.list_accounts(uid, url=url)
        assert accounts.get_account(uid, card, url=url, region=region) == accounts.get_account(uid, card, url=url)
        pinned = transactions.list_transactions(uid, limit=20, url=url, region=region)
        assert pinned == transactions.list_transactions(uid, limit=20, url=url) and len(pinned["items"]) >= 4
        assert (transactions.semantic_search(uid, "coffee", limit=10, url=url, region=region)
                == transactions.semantic_search(uid, "coffee", limit=10, url=url))
        assert region in (sample_serving(uid, url=with_gateway_region(url, region), region=region) or "")


@needs_mr
def test_clear_removes_every_load_user_on_a_regional_database(mr_db):
    url, _ = mr_db
    assert clear(url=url) == 6
