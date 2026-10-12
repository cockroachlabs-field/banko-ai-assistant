"""Checks from the official cockroachdb-skills audit round: TIMESTAMPTZ
everywhere on banko's tables, users GLOBAL on multi-region clusters, and
the follower-read spending summary actually executing."""

import os

import pytest
from sqlalchemy import create_engine, text

from banko_ai.db.regions import detect_regions
from banko_ai.utils.migration import DatabaseMigration

DB = os.getenv("DATABASE_URL")
pytestmark = pytest.mark.skipif(not DB, reason="DATABASE_URL not set")

BANKO_TABLES = ("expenses", "agent_state", "agent_memory", "agent_tasks",
                "agent_decisions", "documents", "query_cache",
                "embedding_cache", "vector_search_cache", "cache_stats")


def test_no_naive_timestamp_columns_after_migration():
    assert DatabaseMigration(DB).migrate_timestamptz() is True
    eng = create_engine(DB)
    with eng.connect() as conn:
        rows = conn.execute(text("""
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND data_type = 'timestamp without time zone'
        """)).fetchall()
    eng.dispose()
    leftovers = [(t, c) for t, c in rows if t in BANKO_TABLES]
    assert leftovers == []


def test_users_table_is_global_on_multi_region():
    if len(detect_regions(DB)) < 2:
        pytest.skip("single-region cluster")
    eng = create_engine(DB)
    with eng.connect() as conn:
        loc = conn.execute(text(
            "SELECT locality FROM [SHOW TABLES] WHERE table_name = 'users'"
        )).scalar()
    eng.dispose()
    assert loc == "GLOBAL"


def test_spending_summary_uses_follower_read():
    from banko_ai.vector_search.search import VectorSearchEngine
    eng = VectorSearchEngine(DB)
    # Any user id works; the point is the AS OF SYSTEM TIME statement
    # planning and executing, rows or not.
    summary = eng.get_user_spending_summary(
        "00000000-0000-0000-0000-0000000000a1", days=30)
    assert summary["period_days"] == 30
