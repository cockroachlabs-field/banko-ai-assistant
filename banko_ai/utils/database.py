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
        """Apply the schema registry (a single SELECT when it is current)."""
        from ..db.schema import ensure_schema
        try:
            result = ensure_schema(self.database_url)
            print("✅ Database schema current" if not result.changed
                  else f"✅ Database schema applied ({result.statements} statements, {result.seconds:.1f}s)")
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
