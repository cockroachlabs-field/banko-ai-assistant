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
    """
    Drop all agent tables (dangerous!).

    Args:
        database_url: CockroachDB connection string
        confirm: Must be True to actually drop tables

    Returns:
        True if successful
    """
    if not confirm:
        print("⚠️  Must pass confirm=True to drop tables")
        return False

    from sqlalchemy import text

    print("🗑️  Dropping agent schema...")

    # Defense-in-depth: validate table names before creating the engine.
    # ValueError propagates to caller (not caught here).
    tables = _tables_to_drop()
    for table in tables:
        if table not in _ALLOWED_DROP_TABLES:
            raise ValueError(
                f"Refusing to drop {table!r}: not in the allowed drop set "
                f"{sorted(_ALLOWED_DROP_TABLES)}"
            )

    try:
        from ..db.engine import get_engine
        engine = get_engine(database_url)

        with engine.connect() as conn:
            for table in tables:
                safe_name = quoted_name(table, quote=True)
                print(f"  Dropping table: {safe_name}")
                conn.execute(text(f"DROP TABLE IF EXISTS {safe_name} CASCADE"))

            conn.commit()

        print("✅ Agent schema dropped")
        return True

    except Exception as e:
        print(f"❌ Failed to drop schema: {e}")
        return False
