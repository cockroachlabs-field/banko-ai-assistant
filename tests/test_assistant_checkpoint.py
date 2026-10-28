"""The assistant's checkpointer rides out a dropped connection. A canceled
session (a node drain, a load balancer, a chaos kill) costs a retry on a
fresh connection, not the chat turn."""

import os
import uuid

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from sqlalchemy import text

from banko_ai.assistant import checkpoint
from banko_ai.db.engine import get_engine

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


def _checkpointer_sessions() -> int:
    with get_engine().connect() as c:
        return c.execute(text("SELECT count(*) FROM [SHOW CLUSTER SESSIONS] WHERE application_name = :a"),
                         {"a": checkpoint.APPLICATION_NAME}).scalar()


def test_a_canceled_session_costs_a_retry_not_the_checkpoint():
    saver = checkpoint.get_saver()
    config = {"configurable": {"thread_id": str(uuid.uuid4()), "checkpoint_ns": ""}}
    written = saver.put(config, empty_checkpoint(), {"source": "input", "step": -1}, {})
    assert _checkpointer_sessions() > 0, "the checkpointer's connections carry their application name"
    with get_engine().connect() as c:
        c.execute(text("CANCEL SESSIONS (SELECT session_id FROM [SHOW CLUSTER SESSIONS] "
                       "WHERE application_name = :a)"), {"a": checkpoint.APPLICATION_NAME})
    found = saver.get_tuple(written)
    assert found is not None and found.config["configurable"]["thread_id"] == config["configurable"]["thread_id"]
    saver.delete_thread(config["configurable"]["thread_id"])


def test_the_pool_follows_checkpoint_pool_size_and_db_pool_timeout(monkeypatch):
    from banko_ai.config.settings import reset_config

    monkeypatch.setenv("CHECKPOINT_POOL_SIZE", "3")
    monkeypatch.setenv("DB_POOL_TIMEOUT", "7")
    reset_config()
    monkeypatch.setattr(checkpoint, "_saver", None)
    saver = checkpoint.get_saver()
    try:
        assert saver.conn.max_size == 3 and saver.conn.min_size == 1
        assert saver.conn.timeout == 7
    finally:
        saver.conn.close()
        reset_config()
