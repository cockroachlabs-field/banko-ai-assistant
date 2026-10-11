"""
Database management utilities.

This module provides database schema creation and management functionality.
"""


from .db_retry import create_resilient_engine, get_database_url


class DatabaseManager:
    """Database management utilities."""
    
    def __init__(self, database_url: str | None = None):
        """Initialize database manager."""
        self.database_url = get_database_url(database_url)
        self._engine = None
    
    @property
    def engine(self):
        """Get SQLAlchemy engine (lazy import)."""
        if self._engine is None:
            # Use official sqlalchemy-cockroachdb dialect (no conversion needed!)
            self._engine = create_resilient_engine(
                self.database_url,
                connect_args={
                    "options": "-c default_transaction_isolation=serializable"
                }
            )
        return self._engine
    
    def create_tables(self) -> bool:
        """Create all required tables including agent tables."""
        try:
            from sqlalchemy import text
            
            print("🔧 Creating database tables...")
            
            with self.engine.connect() as conn:
                # Create expenses table with vector support
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS expenses (
                        expense_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        user_id UUID NOT NULL,
                        expense_date DATE NOT NULL,
                        expense_amount DECIMAL(10,2) NOT NULL,
                        shopping_type STRING NOT NULL,
                        description STRING,
                        merchant STRING,
                        payment_method STRING NOT NULL,
                        recurring BOOL DEFAULT false,
                        tags STRING[],
                        embedding VECTOR(384),
                        created_at TIMESTAMPTZ DEFAULT now()
                    )
                """))
                
                # Create vector index for general search
                conn.execute(text("""
                    CREATE VECTOR INDEX IF NOT EXISTS idx_expenses_embedding 
                    ON expenses (embedding vector_cosine_ops)
                """))
                
                # Create user-specific vector index
                conn.execute(text("""
                    CREATE VECTOR INDEX IF NOT EXISTS idx_expenses_user_embedding 
                    ON expenses (user_id, embedding vector_cosine_ops)
                """))
                
                # Create additional indexes for common queries
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_expenses_user_date 
                    ON expenses (user_id, expense_date DESC)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_expenses_merchant 
                    ON expenses (merchant)
                """))
                
                conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_expenses_shopping_type 
                    ON expenses (shopping_type)
                """))
                
                # Create agent tables
                print("  📊 Creating agent tables...")
                
                # Agent state
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS agent_state (
                        agent_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        agent_type STRING NOT NULL,
                        region STRING NOT NULL,
                        status STRING NOT NULL DEFAULT 'idle',
                        current_task JSONB,
                        last_heartbeat TIMESTAMPTZ DEFAULT now(),
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT now()
                    )
                """))
                
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_agent_state_region_status ON agent_state (region, status)"))
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_agent_state_type ON agent_state (agent_type, status)"))
                
                # Agent memory
                conn.execute(text("""
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
                    )
                """))
                
                # Agent decisions
                conn.execute(text("""
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
                    )
                """))
                
                # Agent tasks
                conn.execute(text("""
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
                    )
                """))
                
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_agent_tasks_status ON agent_tasks (status, priority DESC, created_at)"))
                
                # Documents
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS documents (
                        document_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        user_id STRING NOT NULL,
                        document_type STRING NOT NULL,
                        s3_key STRING,
                        original_filename STRING,
                        extracted_text TEXT,
                        extracted_data JSONB,
                        embedding VECTOR(384),
                        processing_status STRING DEFAULT 'pending',
                        created_at TIMESTAMPTZ DEFAULT now(),
                        processed_at TIMESTAMPTZ
                    )
                """))
                
                conn.execute(text("CREATE INDEX IF NOT EXISTS idx_documents_user_type ON documents (user_id, document_type, created_at DESC)"))
                
                print("  ✅ Agent tables created")
                
                conn.commit()
                print("✅ All database tables created successfully")
                return True
                
        except Exception as e:
            print(f"❌ Error creating tables: {e}")
            return False
    
    def drop_tables(self) -> bool:
        """Drop all tables."""
        try:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                conn.execute(text("DROP TABLE IF EXISTS expenses CASCADE"))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error dropping tables: {e}")
            return False
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        try:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = :table_name
                    )
                """), {"table_name": table_name})
                return result.scalar()
        except Exception as e:
            print(f"Error checking table existence: {e}")
            return False
    
    def get_table_info(self, table_name: str) -> dict:
        """Get table information."""
        try:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                # Get column information
                columns_result = conn.execute(text("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_name = :table_name
                    ORDER BY ordinal_position
                """), {"table_name": table_name})
                
                columns = []
                for row in columns_result:
                    columns.append({
                        "name": row[0],
                        "type": row[1],
                        "nullable": row[2] == "YES",
                        "default": row[3]
                    })
                
                # Get index information
                indexes_result = conn.execute(text("""
                    SELECT indexname, indexdef
                    FROM pg_indexes
                    WHERE tablename = :table_name
                """), {"table_name": table_name})
                
                indexes = []
                for row in indexes_result:
                    indexes.append({
                        "name": row[0],
                        "definition": row[1]
                    })
                
                return {
                    "table_name": table_name,
                    "columns": columns,
                    "indexes": indexes
                }
                
        except Exception as e:
            print(f"Error getting table info: {e}")
            return {}
    
    def get_record_count(self, table_name: str) -> int:
        """Get record count for a table."""
        allowed_tables = {
            'expenses', 'documents', 'agent_state', 'agent_decisions',
            'agent_memory', 'agent_tasks', 'query_cache', 'embedding_cache',
            'vector_search_cache', 'cache_stats', 'accounts', 'items',
            'payments', 'payment_audit_log',
        }
        if table_name not in allowed_tables:
            print(f"⚠️  get_record_count: table '{table_name}' not in allowlist")
            return 0
        try:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                return result.scalar()
        except Exception as e:
            print(f"Error getting record count: {e}")
            return 0
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            from sqlalchemy import text
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False
