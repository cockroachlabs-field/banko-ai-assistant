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
from typing import Optional

from sqlalchemy import text
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


AGENT_SCHEMA_SQL = """
-- Agent state and coordination
CREATE TABLE IF NOT EXISTS agent_state (
    agent_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_type STRING NOT NULL,
    region STRING NOT NULL,
    status STRING NOT NULL DEFAULT 'idle',
    current_task JSONB,
    last_heartbeat TIMESTAMPTZ DEFAULT now(),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_agent_state_region_status 
    ON agent_state (region, status);
CREATE INDEX IF NOT EXISTS idx_agent_state_type 
    ON agent_state (agent_type, status);

-- Agent memory (vector + metadata)
CREATE TABLE IF NOT EXISTS agent_memory (
    memory_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES agent_state(agent_id) ON DELETE CASCADE,
    user_id STRING,
    memory_type STRING NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now(),
    accessed_at TIMESTAMPTZ DEFAULT now(),
    access_count INT DEFAULT 0
);

-- Vector index for semantic search
CREATE INDEX IF NOT EXISTS idx_agent_memory_embedding 
    ON agent_memory USING cspann (user_id, embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_agent_memory_agent 
    ON agent_memory (agent_id, memory_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_memory_user 
    ON agent_memory (user_id, memory_type, created_at DESC);

-- Cross-agent communication and task queue
CREATE TABLE IF NOT EXISTS agent_tasks (
    task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_agent_id UUID REFERENCES agent_state(agent_id),
    target_agent_id UUID REFERENCES agent_state(agent_id),
    task_type STRING NOT NULL,
    priority INT DEFAULT 5,
    payload JSONB,
    status STRING DEFAULT 'pending',
    region STRING,
    created_at TIMESTAMPTZ DEFAULT now(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    result JSONB,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_agent_tasks_status 
    ON agent_tasks (status, priority DESC, created_at);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_target 
    ON agent_tasks (target_agent_id, status);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_region 
    ON agent_tasks (region, status);

-- Agent decisions and audit log (for learning)
CREATE TABLE IF NOT EXISTS agent_decisions (
    decision_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES agent_state(agent_id),
    decision_type STRING NOT NULL,
    context JSONB NOT NULL,
    reasoning TEXT,
    action JSONB,
    confidence FLOAT,
    user_feedback STRING,
    created_at TIMESTAMPTZ DEFAULT now(),
    feedback_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_agent_decisions_agent 
    ON agent_decisions (agent_id, decision_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_feedback 
    ON agent_decisions (user_feedback, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_type 
    ON agent_decisions (decision_type, created_at DESC);

-- Documents and receipts
CREATE TABLE IF NOT EXISTS documents (
    document_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id STRING NOT NULL,
    document_type STRING NOT NULL,
    s3_key STRING NOT NULL,
    original_filename STRING,
    extracted_text TEXT,
    extracted_data JSONB,
    embedding VECTOR(384),
    processing_status STRING DEFAULT 'pending',
    processed_by_agent_id UUID REFERENCES agent_state(agent_id),
    created_at TIMESTAMPTZ DEFAULT now(),
    processed_at TIMESTAMPTZ,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_documents_user 
    ON documents (user_id, document_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_documents_status 
    ON documents (processing_status, created_at);
CREATE INDEX IF NOT EXISTS idx_documents_embedding 
    ON documents USING cspann (user_id, embedding vector_cosine_ops);
"""


def create_agent_schema(database_url: str, verbose: bool = True) -> bool:
    """
    Create agent-related database schema.
    
    Args:
        database_url: CockroachDB connection string
        verbose: Print progress messages
    
    Returns:
        True if successful, False otherwise
    """
    from ..db.engine import get_engine
    
    try:
        if verbose:
            print("🔧 Creating agent schema in CockroachDB...")
        
        engine = get_engine(self.database_url if hasattr(self, "database_url") else database_url)
        
        with engine.connect() as conn:
            # Execute schema creation
            statements = [s.strip() for s in AGENT_SCHEMA_SQL.split(';') if s.strip()]
            
            for i, statement in enumerate(statements, 1):
                if verbose:
                    # Extract table/index name for progress
                    if 'CREATE TABLE' in statement:
                        table_name = statement.split('CREATE TABLE IF NOT EXISTS')[1].split('(')[0].strip()
                        print(f"  [{i}/{len(statements)}] Creating table: {table_name}")
                    elif 'CREATE INDEX' in statement:
                        index_name = statement.split('CREATE INDEX IF NOT EXISTS')[1].split('ON')[0].strip()
                        print(f"  [{i}/{len(statements)}] Creating index: {index_name}")
                
                conn.execute(text(statement))
            
            conn.commit()
if verbose:
            print("✅ Agent schema created successfully")
            print("\nCreated tables:")
            print("  - agent_state: Agent registration and status")
            print("  - agent_memory: Vector memory + metadata")
            print("  - agent_tasks: Cross-agent communication")
            print("  - agent_decisions: Decision audit log")
            print("  - documents: Receipt/document storage")
        
        return True
    
    except Exception as e:
        if verbose:
            print(f"❌ Failed to create agent schema: {e}")
        return False


def verify_agent_schema(database_url: str) -> dict:
    """
    Verify that all agent tables exist.
    
    Args:
        database_url: CockroachDB connection string
    
    Returns:
        Dictionary with table existence status
    """
    
    required_tables = [
        'agent_state',
        'agent_memory', 
        'agent_tasks',
        'agent_decisions',
        'documents'
    ]
    
    results = {}
    
    try:
        engine = get_engine(self.database_url if hasattr(self, "database_url") else database_url)
        
        with engine.connect() as conn:
            for table in required_tables:
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = :table_name
                    )
                """), {'table_name': table})
                results[table] = result.scalar()
except Exception as e:
        print(f"Error verifying schema: {e}")
        for table in required_tables:
            results[table] = False
    
    return results


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
        engine = get_engine(self.database_url if hasattr(self, "database_url") else database_url)

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


if __name__ == "__main__":
    """
    Run this script directly to create agent schema:
    python -m banko_ai.utils.agent_schema
    """
    from .db_retry import get_database_url
    
    database_url = get_database_url()
    
    print(f"Database: {database_url.split('@')[1] if '@' in database_url else database_url}")
    print()
    
    # Create schema
    success = create_agent_schema(database_url, verbose=True)
    
    if success:
        print("\n" + "="*60)
        print("Verifying schema...")
        print("="*60)
        
        results = verify_agent_schema(database_url)
        all_exist = all(results.values())
        
        for table, exists in results.items():
            status = "✅" if exists else "❌"
            print(f"{status} {table}")
        
        if all_exist:
            print("\n🎉 All agent tables created successfully!")
        else:
            print("\n⚠️  Some tables missing")
            sys.exit(1)
    else:
        sys.exit(1)
