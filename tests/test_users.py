import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url

from banko_ai.services.seeding import ensure_personas
from banko_ai.web.auth import UserManager

DB = os.getenv("DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB, reason="DATABASE_URL not set")

# clear_demo_users is destructive by design, so these tests get their own
# database. Built with make_url so it is correct no matter what the real
# database in DATABASE_URL is called.
SCRATCH_NAME = "banko_users_test"


@pytest.fixture(scope="module")
def scratch_db():
    scratch_url = str(make_url(DB).set(database=SCRATCH_NAME))
    assert make_url(scratch_url).database == SCRATCH_NAME
    admin = create_engine(DB)
    with admin.connect() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {SCRATCH_NAME}"))
        conn.commit()
    # The dependent tables clear_demo_users walks, minimal shapes.
    eng = create_engine(scratch_url)
    with eng.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS expenses (
                expense_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL)"""))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS spending_signals (
                signal_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL)"""))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS coach_nudges (
                nudge_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                signal_id UUID,
                user_id UUID NOT NULL)"""))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_budgets (
                budget_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL)"""))
        conn.commit()
    eng.dispose()
    yield scratch_url
    with admin.connect() as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {SCRATCH_NAME} CASCADE"))
        conn.commit()
    admin.dispose()


@pytest.fixture
def um(scratch_db):
    m = UserManager(scratch_db)
    m.ensure_schema()
    yield m
    m.delete_by_username("spec-test-user")


def test_create_and_return(um):
    u = um.create("spec-test-user", "diner")
    assert u["username"] == "spec-test-user"
    again = um.get_by_username("spec-test-user")
    assert again["user_id"] == u["user_id"]
    assert again["spending_style"] == "diner"


def test_unknown_username_is_none(um):
    assert um.get_by_username("nobody-here") is None


def test_duplicate_create_raises(um):
    um.create("spec-test-user", "saver")
    with pytest.raises(Exception):
        um.create("spec-test-user", "diner")


def test_persona_backfill_idempotent(um, scratch_db):
    ensure_personas(scratch_db)
    ensure_personas(scratch_db)
    maya = um.get("00000000-0000-0000-0000-0000000000a1")
    assert maya is not None and maya["spending_style"] == "diner"


def test_clear_demo_users_removes_data(um, scratch_db):
    from banko_ai.web.auth import clear_demo_users
    um.create("spec-wipe-me", "saver", demo_user=True)
    counts = clear_demo_users(scratch_db)
    assert counts["users"] >= 1
    assert um.get_by_username("spec-wipe-me") is None


def test_clear_demo_users_preserves_personas(um, scratch_db):
    ensure_personas(scratch_db)
    um.create("spec-wipe-me-2", "subscriber", demo_user=True)
    from banko_ai.web.auth import clear_demo_users
    clear_demo_users(scratch_db)
    maya = um.get("00000000-0000-0000-0000-0000000000a1")
    sam = um.get("00000000-0000-0000-0000-0000000000a2")
    riley = um.get("00000000-0000-0000-0000-0000000000a3")
    assert maya is not None and maya["username"] == "maya"
    assert sam is not None and sam["username"] == "sam"
    assert riley is not None and riley["username"] == "riley"
    assert um.get_by_username("spec-wipe-me-2") is None
