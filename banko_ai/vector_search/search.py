"""
Vector search engine for expense data.

This module provides vector similarity search functionality with user-specific
filtering and advanced indexing support.
"""

import json
import time
from typing import Any

from sqlalchemy import text

from banko_ai.utils.embeddings import load_embedding_model

from ..ai_providers.base import SearchResult
from ..utils.db_retry import db_retry, get_database_url
from ..utils.request_trace import plan_indexes, serving_regions, trace_step


class VectorSearchEngine:
    """Vector search engine for expense data with user-specific filtering."""

    def __init__(self, database_url: str | None = None, cache_manager=None):
        """Initialize the vector search engine."""
        self.database_url = get_database_url(database_url)
        self.cache_manager = cache_manager

        from ..db.engine import get_engine
        self.engine = get_engine(self.database_url)

    @property
    def embedding_model(self):
        """Loaded on first use, so constructing the engine at boot is free."""
        return load_embedding_model()

    @db_retry(max_attempts=3, initial_delay=0.5)
    def simple_search_expenses(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        Simple search function that matches the original implementation exactly.
        Returns list of dictionaries like the original search_expenses function.
        """
        print("\n🔍 SIMPLE VECTOR SEARCH:")
        print(f"1. Query: '{query}' | Limit: {limit}")
        
        # Generate embedding
        raw_embedding = self.embedding_model.encode(query)
        print(f"2. Generated embedding with {len(raw_embedding)} dimensions")
        
        # Convert to PostgreSQL vector format (matching original implementation)
        search_embedding = json.dumps(raw_embedding.flatten().tolist())
        
        # Use the exact same query as the original implementation
        search_query = text("""
            SELECT 
                description,
                merchant,
                shopping_type,
                expense_amount,
                embedding <=> :search_embedding as similarity_score
            FROM expenses
            ORDER BY embedding <=> :search_embedding
            LIMIT :limit
        """)
        
        with self.engine.connect() as conn:
            results = conn.execute(search_query, 
                                 {'search_embedding': search_embedding, 'limit': limit})
            search_results = [dict(row._mapping) for row in results]
            print(f"3. Database query returned {len(search_results)} expense records")
            
            return search_results
    
    @db_retry(max_attempts=3, initial_delay=0.5)
    def search_expenses(
        self,
        query: str,
        user_id: str | None = None,
        limit: int = 10,
        threshold: float = 0.7,
        use_user_index: bool = True,
        date_start=None,
        date_end=None,
        region: str | None = None,
        capture_explain: bool = False,
    ) -> list[SearchResult] | tuple[list[SearchResult], str]:
        """
        Search for expenses using vector similarity.
        
        Args:
            query: Search query text
            user_id: Optional user ID to filter results
            limit: Maximum number of results to return
            threshold: Minimum similarity score threshold
            use_user_index: Whether to use user-specific vector index
            
        Returns:
            List of SearchResult objects
        """
        print("\n🔍 VECTOR SEARCH (with caching):")
        print(f"1. Query: '{query}' | Limit: {limit}")

        scope_key = f"{user_id or ''}|{date_start}|{date_end}|{region}|{limit}"

        if self.cache_manager:
            start_time = time.time()
            raw_embedding = self.cache_manager._get_embedding_with_cache(query)
            embed_duration = (time.time() - start_time) * 1000
            print(f"   Embedding generated in {embed_duration:.1f}ms")

            cached_results = self.cache_manager.get_cached_vector_search(raw_embedding, limit, scope=scope_key)
            
            if cached_results:
                print(f"2. ✅ Vector search cache HIT! Found {len(cached_results)} cached results")
                trace_step("cache", "Vector search cache",
                           f"Hit: {len(cached_results[:limit])} rows reused, no vector query ran",
                           status="hit")
                search_results = []
                for result in cached_results[:limit]:
                    search_results.append(SearchResult(
                        expense_id=result['expense_id'],
                        user_id=result['user_id'],
                        description=result['description'],
                        merchant=result['merchant'],
                        amount=result['expense_amount'],
                        date=result['expense_date'],
                        similarity_score=result['similarity_score'],
                        metadata={
                            'shopping_type': result['shopping_type'],
                            'payment_method': result['payment_method'],
                            'recurring': result.get('recurring', False),
                            'tags': result.get('tags', [])
                        }
                    ))
                if capture_explain:
                    return (search_results, "(cached results, no fresh EXPLAIN)")
                return search_results
            print("2. ❌ Vector search cache MISS, querying database")
            trace_step("cache", "Vector search cache", "Miss: querying CockroachDB", status="miss")
        else:
            start_time = time.time()
            query_embedding = self.embedding_model.encode([query])[0]
            embed_duration = (time.time() - start_time) * 1000
            raw_embedding = query_embedding
            print(f"2. Generated fresh embedding in {embed_duration:.1f}ms (no cache available)")
        
        # Convert to PostgreSQL vector format (matching original implementation)
        search_embedding = json.dumps(raw_embedding.flatten().tolist())
        
        # Build SQL query based on whether we're using user-specific search
        filter_clause = ""
        if date_start is not None:
            filter_clause += " AND expense_date >= :date_start"
        if date_end is not None:
            filter_clause += " AND expense_date < :date_end"
        if region is not None:
            filter_clause += " AND crdb_region = :region"
        if user_id and use_user_index:
            sql = self._build_user_specific_query(filter_clause)
        else:
            sql = self._build_general_query(filter_clause)
        
        # Prepare parameters as a dictionary
        params = {
            'search_embedding': search_embedding,
            'limit': limit
        }

        if user_id:
            params['user_id'] = user_id
        if date_start is not None:
            params['date_start'] = date_start
        if date_end is not None:
            params['date_end'] = date_end
        if region is not None:
            params['region'] = region

        # Execute query
        start_time = time.time()
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params)
            rows = result.fetchall()
        query_duration = (time.time() - start_time) * 1000
        
        # Convert to SearchResult objects
        results = []
        for row in rows:
            results.append(SearchResult(
                expense_id=str(row[0]),
                user_id=str(row[1]),
                description=row[2] or "",
                merchant=row[3] or "",
                amount=float(row[4]),
                date=str(row[5]),
                similarity_score=float(row[6]),
                metadata={
                    "shopping_type": row[7] if len(row) > 7 else None,
                    "payment_method": row[8] if len(row) > 8 else None,
                    "recurring": row[9] if len(row) > 9 else None,
                    "tags": row[10] if len(row) > 10 else None
                }
            ))

        print(f"3. Database query returned {len(results)} expense records in {query_duration:.1f}ms")
        trace_step("vector", "Vector search",
                   f"{len(results)} rows by cosine distance (<=>)"
                   + (", filtered to this user" if user_id else "")
                   + (f", scoped to {region}" if region else ""),
                   ms=query_duration, status="success")

        # Cache the results for future use (convert back to dict format for caching)
        if self.cache_manager and results:
            # Convert SearchResult objects back to dict format for caching
            search_results_dict = []
            for result in results:
                search_results_dict.append({
                    'expense_id': result.expense_id,
                    'user_id': result.user_id,
                    'description': result.description,
                    'merchant': result.merchant,
                    'expense_amount': result.amount,
                    'expense_date': result.date,
                    'similarity_score': result.similarity_score,
                    'shopping_type': result.metadata.get('shopping_type'),
                    'payment_method': result.metadata.get('payment_method'),
                    'recurring': result.metadata.get('recurring'),
                    'tags': result.metadata.get('tags')
                })

            self.cache_manager.cache_vector_search_results(raw_embedding, search_results_dict, scope=scope_key)
            print("4. ✅ Cached vector search results for future queries")

        if capture_explain:
            explain_start = time.time()
            try:
                # CockroachDB rejects placeholders in EXPLAIN under server-side
                # binding, so bind client-side like explain_aggregation does.
                import psycopg

                from ..utils.aggregations import _to_pyformat
                with self.engine.connect() as conn:
                    raw_conn = conn.connection.driver_connection
                    with psycopg.ClientCursor(raw_conn) as cur:
                        cur.execute(f"EXPLAIN ANALYZE {_to_pyformat(sql)}", params)
                        explain_rows = cur.fetchall()
                explain_text = "\n".join(str(row[0]) for row in explain_rows)
                parts = []
                indexes = plan_indexes(explain_text)
                if indexes:
                    parts.append("index " + ", ".join(indexes))
                served = serving_regions(explain_text)
                if served:
                    parts.append(f"served by {served}")
                trace_step("explain", "EXPLAIN ANALYZE",
                           "Plan captured" + (": " + "; ".join(parts) if parts else ""),
                           ms=(time.time() - explain_start) * 1000)
            except Exception as e:
                explain_text = f"(explain unavailable: {str(e)})"
                trace_step("explain", "EXPLAIN ANALYZE", "Plan unavailable for this query",
                           ms=(time.time() - explain_start) * 1000, status="warning")
            return (results, explain_text)

        return results
    
    def _build_user_specific_query(self, filter_clause: str = "") -> str:
        """Build SQL query for user-specific vector search."""
        return """
        SELECT
            expense_id,
            user_id,
            description,
            merchant,
            expense_amount,
            expense_date,
            embedding <=> :search_embedding as similarity_score,
            shopping_type,
            payment_method,
            recurring,
            tags
        FROM expenses
        WHERE user_id = :user_id""" + filter_clause + """
        ORDER BY embedding <=> :search_embedding
        LIMIT :limit
        """

    def _build_general_query(self, filter_clause: str = "") -> str:
        """Build SQL query for general vector search."""
        return """
        SELECT
            expense_id,
            user_id,
            description,
            merchant,
            expense_amount,
            expense_date,
            embedding <=> :search_embedding as similarity_score,
            shopping_type,
            payment_method,
            recurring,
            tags
        FROM expenses
        """ + (("WHERE " + filter_clause.lstrip()[4:].lstrip())
               if filter_clause else "") + """
        ORDER BY embedding <=> :search_embedding
        LIMIT :limit
        """
    
    @db_retry(max_attempts=3, initial_delay=0.5)
    def get_user_spending_summary(
        self, 
        user_id: str, 
        days: int = 30
    ) -> dict[str, Any]:
        """Get spending summary for a specific user.

        Reads with follower_read_timestamp(): dashboard totals tolerate a
        few seconds of staleness, and the nearest replica can serve them
        without a trip to the leaseholder. Chat answers and the coach
        stay on fresh reads; their immediacy is demo-visible."""
        sql = """
        SELECT
            COUNT(*) as transaction_count,
            SUM(expense_amount) as total_amount,
            AVG(expense_amount) as average_amount,
            shopping_type,
            COUNT(*) as category_count
        FROM expenses
        AS OF SYSTEM TIME follower_read_timestamp()
        WHERE user_id = :user_id
        AND expense_date >= CURRENT_DATE - CAST(:days || ' days' AS INTERVAL)
        GROUP BY shopping_type
        ORDER BY total_amount DESC
        """
        
        # Autocommit: a statement-level AS OF SYSTEM TIME is rejected
        # inside the transaction SQLAlchemy auto-opens.
        with self.engine.connect().execution_options(
                isolation_level="AUTOCOMMIT") as conn:
            result = conn.execute(text(sql), {'user_id': user_id, 'days': days})
            rows = result.fetchall()
        
        summary = {
            "user_id": user_id,
            "period_days": days,
            "total_transactions": sum(row[0] for row in rows),
            "total_amount": sum(row[1] for row in rows),
            "average_transaction": sum(row[2] for row in rows) / len(rows) if rows else 0,
            "categories": []
        }
        
        for row in rows:
            summary["categories"].append({
                "category": row[3],
                "count": row[0],
                "total_amount": float(row[1]),
                "average_amount": float(row[2])
            })
        
        return summary
    

