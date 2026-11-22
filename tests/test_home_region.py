"""An account's home region: where its rows live on a multi-region database.

The accounts the Google sign-in makes had none. Their rows then landed in
whichever region the load balancer sent each write to, purchases in one and
the ledger in another, and no read could name a region, so every statement
looked in all of them. On the hosted instance each answer said it was served
from three regions, the dashboard was slow, and the choice of a region that
the sign-up form gives was gone.

A sign-in account now starts in the database's primary region, and anyone can
move their own account: the rows go along."""

import os
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url

from banko_ai.db.engine import get_engine
from tests._app import app_client, stop_apps
from tests._money import drop_user, make_user

DB = os.getenv("DATABASE_URL")
MR_URL = os.getenv("BANKO_TEST_MR_URL")
pytestmark = pytest.mark.skipif(not DB, reason="DATABASE_URL not set")
needs_regions = pytest.mark.skipif(not MR_URL, reason="BANKO_TEST_MR_URL not set")
SCRATCH = "banko_home_region_test"


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    stop_apps()


# On one region there is nothing to choose.

def test_a_database_with_one_region_has_no_home_region_to_give():
    from banko_ai.services import home_region as home
    assert home.regions_of(DB) == []
    assert home.default_region(DB) is None


def test_choosing_a_region_needs_a_database_that_has_regions():
    from banko_ai.services import home_region as home
    uid = make_user("balanced")
    try:
        with pytest.raises(home.NotRegional):
            home.set_home_region(uid, "us-east-1", url=DB)
    finally:
        drop_user(uid)


def test_a_persona_keeps_the_region_its_file_gives_it():
    """Maya, Sam and Riley get their regions from the personas file at every
    boot: a move would be undone by the next restart."""
    from banko_ai.config.catalog import get_catalog
    from banko_ai.services import home_region as home
    persona = get_catalog().personas[0]
    with pytest.raises(home.NotTheirs):
        home.set_home_region(persona.user_id, "us-east-1", url=DB)


def test_the_api_lists_no_regions_and_refuses_a_move_on_one_region(monkeypatch):
    uid = make_user("balanced")
    try:
        client = app_client(monkeypatch)
        with client.session_transaction() as s:
            s["user_id"] = uid
        assert client.get("/api/v1/me").get_json()["regions"] == []
        refused = client.post("/api/v1/me/home-region", json={"region": "us-east-1"})
        assert refused.status_code == 409
        assert "one region" in refused.get_json()["error"]              # a sentence the page can show as it is
        assert client.post("/api/v1/me/home-region", json={}).status_code == 400
    finally:
        drop_user(uid)


# On a multi-region database (BANKO_TEST_MR_URL: a cluster of three regions or more).

@pytest.fixture(scope="module")
def regional():
    """Banko's schema in a scratch database on the multi-region cluster."""
    from banko_ai.db import regions
    from banko_ai.db.schema import ensure_schema
    from banko_ai.utils.db_retry import normalize_db_url
    admin = get_engine(MR_URL)
    with admin.connect() as c:
        c.execution_options(isolation_level="AUTOCOMMIT")
        c.execute(text(f"DROP DATABASE IF EXISTS {SCRATCH} CASCADE"))
        c.execute(text(f"CREATE DATABASE {SCRATCH}"))
    url = make_url(normalize_db_url(MR_URL)).set(database=SCRATCH).render_as_string(hide_password=False)
    ensure_schema(url)
    regions.clear_region_caches()
    yield url
    with admin.connect() as c:
        c.execution_options(isolation_level="AUTOCOMMIT")
        c.execute(text(f"DROP DATABASE IF EXISTS {SCRATCH} CASCADE"))
    regions.clear_region_caches()


def _rows(url: str, user_id: str) -> dict[str, dict[str, int]]:
    """This user's rows in every regional table that has any, counted by the region they live in."""
    from banko_ai.db import schema
    out = {}
    with get_engine(url).connect() as c:
        for table in schema.tables():
            if table.locality != "REGIONAL BY ROW":
                continue
            found = {str(region): int(n) for region, n in c.execute(text(
                f"SELECT crdb_region::STRING, count(*) FROM {table.name} WHERE user_id = CAST(:u AS UUID) GROUP BY 1"),
                {"u": user_id})}
            if found:
                out[table.name] = found
    return out


def _member(url: str, region: str | None) -> str:
    """An account with a month of history, living in this region (or nowhere in particular)."""
    from banko_ai.services import seeding
    uid = str(uuid.uuid4())
    with get_engine(url).begin() as c:
        c.execute(text("INSERT INTO users (user_id, username, spending_style, home_region, demo_user) "
                       "VALUES (CAST(:u AS UUID), :n, 'balanced', :r, true)"),
                  {"u": uid, "n": f"t-{uid[:8]}", "r": region})
    seeding.seed_user(uid, months=1, url=url)
    return uid


def _total(rows: dict[str, dict[str, int]]) -> dict[str, int]:
    return {table: sum(found.values()) for table, found in rows.items()}


@needs_regions
def test_an_account_the_sign_in_makes_starts_in_the_primary_region_and_its_rows_live_there(regional):
    from banko_ai.db import regions
    from banko_ai.services import home_region as home
    from banko_ai.services import seeding, staff
    primary = home.default_region(regional)
    assert primary in home.regions_of(regional) and len(home.regions_of(regional)) >= 3
    user, start = staff.ensure(f"accounts.google.com:{uuid.uuid4().int}", "pat.example@example.com", url=regional)
    assert start and user["home_region"] == primary
    seeding.seed_user(user["user_id"], months=1, url=regional)
    rows = _rows(regional, user["user_id"])
    assert {"accounts", "ledger_entries", "expenses", "user_budgets"} <= set(rows)
    assert all(set(found) == {primary} for found in rows.values()), rows      # every row of every table
    assert regions.read_region(user["user_id"], regional) == primary         # so a read can name its region


@needs_regions
def test_a_users_merchants_are_read_in_their_home_region(regional, monkeypatch):
    """The answer cache asks which names are a user's merchants. On REGIONAL
    BY ROW tables a lookup by user alone looks in every region; this one
    names the home region, for an account whose rows all live there. Then
    "<merchant> purchases" finds the answer kept for "<merchant>"."""
    from banko_ai.services import caches, ledger
    from banko_ai.services import home_region as home
    from banko_ai.services._vector import embed
    primary = home.default_region(regional)
    other = sorted(r for r in home.regions_of(regional) if r != primary)[0]
    uid = _member(regional, other)
    named, real = [], caches._merchant_names
    monkeypatch.setattr(caches, "_merchant_names",
                        lambda conn, user, region=None: named.append(region) or real(conn, user, region))
    caches.forget_merchants()
    with get_engine(regional).connect() as c:
        stored = {caches._as_asked(str(r[0])) for r in c.execute(text(
            "SELECT DISTINCT merchant FROM expenses WHERE user_id = CAST(:u AS UUID)"), {"u": uid})}
    version = ledger.data_version(uid, regional)
    is_merchant = caches.merchants_of(uid, version, regional)
    assert stored and all(is_merchant(name) for name in stored) and not is_merchant("no such place")
    assert named == [other]                                    # read once, inside the home region

    # Through the answer cache, which asks for the region before it takes its connection.
    name = next(n for n in sorted(stored)
                if caches.subject_ask(f"{n} purchases", is_merchant=stored.__contains__) == (f"@{n}", False)
                and caches.subject_ask(n, is_merchant=stored.__contains__) == (f"@{n}", True))
    caches.forget_merchants()
    caches.put_answer(uid, version=version, language="English", query=name, qvec=embed(name), answer="kept",
                      tokens=5, standalone=True, url=regional)
    hit = caches.get_answer(uid, version=version, language="English", query=f"{name} purchases",
                            qvec=embed(f"{name} purchases"), match="words", url=regional)
    assert hit and hit["match"] == "words" and hit["answer"] == "kept"
    assert named == [other, other]


@needs_regions
def test_moving_home_takes_every_row_along(regional):
    from banko_ai.db import regions
    from banko_ai.services import home_region as home
    from banko_ai.services import ledger
    from banko_ai.services.accounts import list_accounts
    primary = home.default_region(regional)
    other = sorted(r for r in home.regions_of(regional) if r != primary)[0]
    uid = _member(regional, primary)
    before = _rows(regional, uid)
    assert all(set(found) == {primary} for found in before.values()), before

    moved = home.set_home_region(uid, other, url=regional)

    after = _rows(regional, uid)
    assert _total(after) == _total(before)                                    # nothing lost, nothing doubled
    assert all(set(found) == {other} for found in after.values()), after
    assert moved == {"home_region": other, "rows": sum(_total(before).values())}
    with get_engine(regional).connect() as c:
        assert c.execute(text("SELECT home_region FROM users WHERE user_id = CAST(:u AS UUID)"), {"u": uid}).scalar() == other
    assert regions.read_region(uid, regional) == other                        # reads name the new region at once
    card = next(a for a in list_accounts(uid, url=regional)["accounts"] if a["kind"] == "credit")
    ledger.post_purchase(uid, card["account_id"], amount="4.50", merchant="Corner Cafe", category="Coffee", url=regional)
    born = _rows(regional, uid)
    assert all(set(found) == {other} for found in born.values()), born         # and what is bought next is born there
    assert home.set_home_region(uid, other, url=regional) == {"home_region": other, "rows": 0}   # asked again: nothing to do


@needs_regions
def test_rows_that_were_scattered_are_gathered(regional):
    """The account as the sign-in used to make it: no home region, rows in
    whichever region each write went through."""
    from banko_ai.db import regions
    from banko_ai.services import home_region as home
    uid = _member(regional, None)
    assert regions.read_region(uid, regional) is None
    home_to_be = sorted(home.regions_of(regional))[-1]
    home.set_home_region(uid, home_to_be, url=regional)
    rows = _rows(regional, uid)
    assert rows and all(set(found) == {home_to_be} for found in rows.values()), rows
    assert regions.read_region(uid, regional) == home_to_be


@needs_regions
def test_a_row_that_lands_in_the_old_region_during_the_move_is_fetched_too(regional, monkeypatch):
    """A purchase that began before the account was pointed at its new region
    commits in the old one, and can do so after its table was gone through.
    Left there, no read of this account could name a region again."""
    from banko_ai.db import regions
    from banko_ai.services import home_region as home
    primary = home.default_region(regional)
    other = sorted(r for r in home.regions_of(regional) if r != primary)[0]
    uid = _member(regional, primary)
    real, late = home._move, []

    def then_a_late_write(table, user_id, region, url):
        moved = real(table, user_id, region, url)
        if table == "goals" and not late:      # goals is done; a write that started earlier lands in the old region
            with get_engine(url).begin() as c:
                late.append(c.execute(text("UPDATE goals SET crdb_region = CAST(:r AS crdb_internal_region) "
                                           "WHERE user_id = CAST(:u AS UUID)"), {"r": primary, "u": user_id}).rowcount)
        return moved

    monkeypatch.setattr(home, "_move", then_a_late_write)
    home.set_home_region(uid, other, url=regional)
    assert late and late[0] >= 1                                             # the late write happened
    rows = _rows(regional, uid)
    assert all(set(found) == {other} for found in rows.values()), rows
    assert regions.read_region(uid, regional) == other


@needs_regions
def test_a_region_the_database_does_not_have_is_refused(regional):
    from banko_ai.services import home_region as home
    uid = _member(regional, home.default_region(regional))
    before = _rows(regional, uid)
    with pytest.raises(home.UnknownRegion):
        home.set_home_region(uid, "mars-north-1", url=regional)
    assert _rows(regional, uid) == before


@needs_regions
def test_the_api_moves_the_signed_in_account(regional, monkeypatch):
    from banko_ai.config.settings import reset_config
    from banko_ai.db import rls
    from banko_ai.services import home_region as home
    primary = home.default_region(regional)
    other = sorted(r for r in home.regions_of(regional) if r != primary)[0]
    uid = _member(regional, primary)
    suite = (rls._status, rls._app_url)        # one app URL per process: this app's must not outlive the test
    try:
        client = app_client(monkeypatch, DATABASE_URL=regional)
        # Let the boot finish: a warm-up step that reports after the test has ended writes to a closed log.
        assert client.application.extensions["banko_readiness"].wait_for("locality", 120)
        with client.session_transaction() as s:
            s["user_id"] = uid
        me = client.get("/api/v1/me").get_json()
        assert me["home_region"] == primary and sorted(me["regions"]) == sorted(home.regions_of(regional))
        done = client.post("/api/v1/me/home-region", json={"region": other})
        assert done.status_code == 200, done.get_data(as_text=True)
        assert done.get_json()["home_region"] == other and done.get_json()["rows"] > 0
        assert client.get("/api/v1/me").get_json()["home_region"] == other
        assert all(set(found) == {other} for found in _rows(regional, uid).values())
        assert client.post("/api/v1/me/home-region", json={"region": "mars-north-1"}).status_code == 400
    finally:
        stop_apps()
        reset_config()
        rls._status, rls._app_url = suite
