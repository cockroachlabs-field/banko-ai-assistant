"""
Database schema for agentic AI system.

Creates tables for:
- Agent state and coordination
- Agent memory (vector + transactional)
- Cross-agent communication
- Decision tracking and learning
- Conversation history
- Document processing
"""

import sys

from sqlalchemy.sql import quoted_name

# Whitelist of tables that drop_agent_schema is allowed to drop.
# Adding to this list is a deliberate code change reviewed in PR.
_ALLOWED_DROP_TABLES: frozenset[str] = frozenset({
    "documents",
    "agent_decisions",
    "agent_tasks",
    "agent_memory",
    "agent_state",
})


def _tables_to_drop() -> list[str]:
    """Return the ordered list of tables to drop. Indirection exists so tests
    can monkeypatch and verify the safety check refuses unknown names."""
    # Reverse FK order matters: documents -> agent_decisions -> tasks/memory -> state
    return [
        "documents",
        "agent_decisions",
        "agent_tasks",
        "agent_memory",
        "agent_state",
    ]


def drop_agent_schema(database_url: str, confirm: bool = False) -> bool:
    """Drop agent schema tables in reverse FK order. Requires explicit
    confirmation flag and refuses unknown table names.

    Args:
        database_url: CockroachDB connection string
        confirm: Must be True or the function exits early

    Returns:
        True if successful, False otherwise
    """
    if not confirm:
        print("⚠️  drop_agent_schema: confirm must be True to proceed")
        return False

    from sqlalchemy import text

    tables = _tables_to_drop()
    for name in tables:
        if name not in _ALLOWED_DROP_TABLES:
            raise ValueError(f"table '{name}' is not in the allowed drop set")

    try:
        from ..db.engine import get_engine
        engine = get_engine(database_url)

        with engine.connect() as conn:
            for table in tables:
                # table names come from the frozen allowlist, safe to interpolate
                conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
                print(f"   dropped {table}")
            conn.commit()

        print("✅ Agent schema tables dropped")
        return True

    except Exception as e:
        print(f"❌ Error dropping agent schema: {e}")
        import traceback
        traceback.print_exc()
        return False
