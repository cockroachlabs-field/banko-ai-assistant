"""
Search tools for agents to find and retrieve expense data.

Provides:
- Vector search (semantic similarity)
- SQL search (exact matches, filters)
- Hybrid search (combined)
"""

import json
from typing import Any

from langchain_core.tools import Tool
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool


def create_search_tools(database_url: str, embedding_model) -> list[Tool]:
    """
    Create search tools for agents.
    
    Args:
        database_url: CockroachDB connection string
        embedding_model: Sentence transformer model for embeddings
    
    Returns:
        List of LangChain Tool objects
    """
    
    def vector_search_expenses(
        query: str,
        user_id: str | None = None,
        limit: int = 5
    ) -> str:
        """
        Search expenses using semantic similarity (vector search).
        
        Args:
            query: Natural language search query
            user_id: Optional user ID to filter by
            limit: Number of results to return
        
        Returns:
            JSON string with search results
        """
        try:
            # Generate embedding for query
            query_embedding = embedding_model.encode(query).tolist()
            
            # Format embedding as JSON array for CockroachDB v25.4.0+
            # No CAST needed with v25.4.0 GA
            import json
            embedding_json = json.dumps(query_embedding)
            
            engine = get_engine(self.database_url if hasattr(self, "database_url") else database_url)
            
            with engine.connect() as conn:
                # Build query - v25.4.0 no longer requires CAST
                sql = """
                    SELECT 
                        expense_id,
                        user_id,
                        description,
                        expense_amount,
                        merchant,
                        shopping_type,
                        expense_date,
                        payment_method,
                        embedding <=> :embedding as distance
                    FROM expenses
                """
                
                params = {'embedding': embedding_json, 'limit': limit}
                
                if user_id:
                    sql += " WHERE user_id = :user_id"
                    params['user_id'] = user_id
                
                # Order by expression (not alias) for index usage
                sql += " ORDER BY embedding <=> :embedding LIMIT :limit"
                
                result = conn.execute(text(sql), params)
                rows = result.fetchall()
                
                expenses = []
                for row in rows:
                    expenses.append({
                        'expense_id': str(row[0]),
                        'user_id': str(row[1]),
                        'description': row[2],
                        'amount': float(row[3]),
                        'merchant': row[4],
                        'category': row[5],
                        'date': row[6].isoformat() if row[6] else None,
                        'payment_method': row[7],
                        'similarity_score': 1.0 - float(row[8])  # Convert distance to similarity
                    })
return json.dumps({
                'success': True,
                'query': query,
                'results': expenses,
                'count': len(expenses)
            }, indent=2)
        
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e)
            })
    
    def sql_search_expenses(
        filters: dict[str, Any],
        limit: int = 10
    ) -> str:
        """
        Search expenses using SQL filters (exact matches).
        
        Args:
            filters: Dictionary of filters (e.g., {'category': 'food', 'min_amount': 100})
            limit: Number of results to return
        
        Returns:
            JSON string with search results
        """
        try:
            engine = get_engine(self.database_url if hasattr(self, "database_url") else database_url)
            
            with engine.connect() as conn:
                # Build WHERE clause from filters
                where_clauses = []
                params = {'limit': limit}
                
                if 'user_id' in filters:
                    where_clauses.append("user_id = :user_id")
                    params['user_id'] = filters['user_id']
                
                if 'category' in filters:
                    where_clauses.append("shopping_type = :category")
                    params['category'] = filters['category']
                
                if 'merchant' in filters:
                    where_clauses.append("merchant ILIKE :merchant")
                    params['merchant'] = f"%{filters['merchant']}%"
                
                if 'min_amount' in filters:
                    where_clauses.append("expense_amount >= :min_amount")
                    params['min_amount'] = filters['min_amount']
                
                if 'max_amount' in filters:
                    where_clauses.append("expense_amount <= :max_amount")
                    params['max_amount'] = filters['max_amount']
                
                if 'start_date' in filters:
                    where_clauses.append("expense_date >= :start_date")
                    params['start_date'] = filters['start_date']
                
                if 'end_date' in filters:
                    where_clauses.append("expense_date <= :end_date")
                    params['end_date'] = filters['end_date']
                
                where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
                
                sql = f"""
                    SELECT 
                        expense_id, user_id, description, expense_amount, merchant,
                        shopping_type, expense_date, payment_method
                    FROM expenses
                    WHERE {where_sql}
                    ORDER BY expense_date DESC
                    LIMIT :limit
                """
                
                result = conn.execute(text(sql), params)
                rows = result.fetchall()
                
                expenses = []
                for row in rows:
                    expenses.append({
                        'expense_id': str(row[0]),
                        'user_id': str(row[1]),
                        'description': row[2],
                        'amount': float(row[3]),
                        'merchant': row[4],
                        'category': row[5],
                        'date': row[6].isoformat() if row[6] else None,
                        'payment_method': row[7]
                    })
return json.dumps({
                'success': True,
                'filters': filters,
                'results': expenses,
                'count': len(expenses)
            }, indent=2)
        
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e)
            })
    
    def get_expense_by_id(expense_id: str) -> str:
        """
        Get a specific expense by ID.
        
        Args:
            expense_id: UUID of the expense
        
        Returns:
            JSON string with expense details
        """
        try:
            engine = get_engine(self.database_url if hasattr(self, "database_url") else database_url)
            
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        expense_id, user_id, description, expense_amount, merchant,
                        shopping_type, expense_date, payment_method
                    FROM expenses
                    WHERE expense_id = :expense_id
                """), {'expense_id': expense_id})
                
                row = result.fetchone()
                
                if not row:
                    return json.dumps({
                        'success': False,
                        'error': 'Expense not found'
                    })
                
                expense = {
                    'expense_id': str(row[0]),
                    'user_id': str(row[1]),
                    'description': row[2],
                    'expense_amount': float(row[3]),
                    'merchant': row[4],
                    'category': row[5],
                    'expense_date': row[6].isoformat() if row[6] else None,
                    'payment_method': row[7]
                }
return json.dumps({
                'success': True,
                'expense': expense
            }, indent=2)
        
        except Exception as e:
            return json.dumps({
                'success': False,
                'error': str(e)
            })
    
    # Create LangChain tools
    tools = [
        Tool(
            name="vector_search_expenses",
            description="""Search expenses using natural language semantic similarity.
            Use this when the user asks about expenses in conversational terms.
            Args: query (str), user_id (optional str), limit (optional int, default 5)
            Returns: JSON with matching expenses and similarity scores""",
            func=lambda query, user_id=None, limit=5: vector_search_expenses(query, user_id, limit)
        ),
        Tool(
            name="sql_search_expenses",
            description="""Search expenses using exact filters (category, amount, date, merchant).
            Use this for precise queries with specific criteria.
            Args: filters (dict with keys: user_id, category, merchant, min_amount, max_amount, start_date, end_date), limit (optional int, default 10)
            Returns: JSON with filtered expenses""",
            func=lambda filters, limit=10: sql_search_expenses(filters, limit)
        ),
        Tool(
            name="get_expense_by_id",
            description="""Get details of a specific expense by its ID.
            Args: expense_id (str UUID)
            Returns: JSON with complete expense details""",
            func=get_expense_by_id
        )
    ]
    
    return tools
