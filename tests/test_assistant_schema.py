"""The assistant's tables exist with the shapes later code relies on."""

import os

import pytest
from sqlalchemy import text

from banko_ai.config.settings import get_config
from banko_ai.db import rls
from banko_ai.db.engine import get_engine
from banko_ai.db.schema import Topology, tables

NEW = {"receipts", "chat_threads", "memories", "answer_cache", "search_cache", "assistant_actions"}


def _by_name():
    return {t.name: t for t in tables()}


def test_registry_declares_the_assistant_tables():
    reg = _by_name()
    assert NEW <= set(reg)
    for name in NEW:
        assert reg[name].locality == "REGIONAL BY ROW", name


def test_vector_tables_carry_per_user_vector_indexes():
    reg = _by_name()
    want = {"receipts": "receipts_user_embedding", "memories": "memories_user_embedding",
            "answer_cache": "answer_cache_user_embedding"}
    for table, index in want.items():
        idx = {i.name: i for i in reg[table].indexes}[index]
        assert idx.vector and idx.columns.startswith("user_id,"), table


def test_caches_and_actions_expire_by_row_level_ttl():
    reg = _by_name()
    for name in ("answer_cache", "search_cache", "assistant_actions"):
        assert "ttl_expiration_expression = 'expires_at'" in reg[name].options, name


def test_rls_covers_chat_threads_and_never_a_vector_table():
    assert "chat_threads" in rls.RLS_TABLES
    assert not {"receipts", "memories", "answer_cache", "expenses"} & set(rls.RLS_TABLES)


def test_create_sql_is_inline_on_multi_region():
    sql = _by_name()["memories"].create_sql(Topology(regional=True, regions=("a", "b")))
    assert "VECTOR INDEX memories_user_embedding (user_id, embedding vector_cosine_ops)" in sql
    assert sql.endswith("LOCALITY REGIONAL BY ROW")


def test_assistant_settings_defaults():
    cfg = get_config()
    assert cfg.brand_name == "Banko"
    assert cfg.assistant_max_steps == 5
    assert cfg.assistant_history_window == 12
    assert cfg.assistant_implicit_memory is True
    assert cfg.search_cache_ttl_minutes == 10
    assert cfg.receipt_default_card_limit == 5000.0


@pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")
def test_tables_exist_in_the_database():
    with get_engine().connect() as c:
        names = {r[0] for r in c.execute(text("SELECT table_name FROM [SHOW TABLES]"))}
        assert NEW <= names
        cols = {r[0] for r in c.execute(text("SELECT column_name FROM [SHOW COLUMNS FROM receipts]"))}
    assert {"document_id", "status", "extracted", "expense_id", "thumbnail", "embedding"} <= cols
