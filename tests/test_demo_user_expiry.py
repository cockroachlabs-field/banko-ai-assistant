"""The demo user sweep removes everything an expired user owns, money rows
and seed state included, and never touches a persona."""

import os
from datetime import date

import pytest
from sqlalchemy import text

from banko_ai.db.engine import get_engine
from banko_ai.services import seeding as S
from banko_ai.utils.migration import DatabaseMigration
from tests._money import MONEY_TABLES, drop_user, make_user

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


def test_expired_demo_user_takes_its_money_rows_and_seed_state_along(monkeypatch):
    monkeypatch.setenv("DEMO_USER_TTL_DAYS", "1")
    personas = S.ensure_personas()
    uid = make_user("saver")
    try:
        S.seed_user(uid, months=1, today=date(2026, 10, 11))
        with get_engine().begin() as c:
            c.execute(text("UPDATE users SET created_at = now() - INTERVAL '3 days' "
                           "WHERE user_id = CAST(:u AS UUID)"), {"u": uid})
            c.execute(text("UPSERT INTO banko_meta (key, value, updated_at) VALUES (:k, '{}', now())"),
                      {"k": f"seed_lock:{uid}"})
        with get_engine().connect() as c:
            assert c.execute(text("SELECT count(*) FROM ledger_entries WHERE user_id = CAST(:u AS UUID)"),
                             {"u": uid}).scalar() > 0

        assert DatabaseMigration(os.environ["DATABASE_URL"]).expire_demo_users() >= 1

        with get_engine().connect() as c:
            for table in (*MONEY_TABLES, "users"):
                left = c.execute(text(f"SELECT count(*) FROM {table} WHERE user_id = CAST(:u AS UUID)"),
                                 {"u": uid}).scalar()
                assert left == 0, table
            meta = c.execute(text("SELECT count(*) FROM banko_meta WHERE key IN (:a, :b)"),
                             {"a": f"seed:{uid}", "b": f"seed_lock:{uid}"}).scalar()
            assert meta == 0
            kept = {r[0] for r in c.execute(text("SELECT username FROM users WHERE username = ANY(:n)"),
                                            {"n": personas})}
        assert kept == set(personas)
    finally:
        drop_user(uid)
