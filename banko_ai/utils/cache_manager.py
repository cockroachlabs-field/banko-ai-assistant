"""
Banko AI Token Optimization & Caching System

This module implements a multi-layer caching strategy to reduce token usage:
1. Query Similarity Cache - Cache responses for semantically similar queries
2. Embedding Cache - Cache embeddings to avoid regeneration
3. Response Fragment Cache - Cache financial insights and recommendations
4. Vector Search Cache - Cache vector search results

Uses CockroachDB for persistent caching with TTL support.
"""

import decimal
import hashlib
import json
import time
import uuid
from datetime import datetime, timedelta

import numpy as np
from sqlalchemy import text

from banko_ai.utils.embeddings import ModelWarmingUp, load_embedding_model

from ..config.settings import get_config
from .db_retry import db_retry
from .request_scope import current_user_scope
from .request_trace import trace_step


def _engine():
    from ..db.engine import get_engine
    return get_engine()

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle Decimal and UUID objects"""
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        return super().default(obj)

def safe_json_dumps(obj, **kwargs):
    """Safe JSON dumps that handles Decimal and UUID objects"""
    return json.dumps(obj, cls=CustomJSONEncoder, **kwargs)

class BankoCacheManager:
    """
    Intelligent caching system for Banko AI to optimize token usage.
    
    Caching Strategy:
    - High confidence (≥0.90): Exact semantic match - always use cache
    - Medium confidence (0.70-0.89): Similar match - use cache if data matches
    - Low confidence (<0.70): Different query - generate fresh response
    
    Environment Variables:
    - CACHE_SIMILARITY_THRESHOLD: Similarity threshold (default: 0.75)
    - CACHE_TTL_HOURS: Cache time-to-live in hours (default: 24)
    - CACHE_STRICT_MODE: Require exact data match (default: true)
    """
    
    def __init__(self, similarity_threshold=None, cache_ttl_hours=None, strict_mode=None):
        """
        Initialize the cache manager.

        Args:
            similarity_threshold: Minimum similarity score (default: from env or 0.75)
            cache_ttl_hours: Time-to-live for cached responses (default: from env or 24)
            strict_mode: Require exact expense data match (default: from env or True)
        """
        # Load from environment variables with fallbacks
        cfg = get_config()
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None
            else cfg.cache_similarity_threshold
        )
        self.cache_ttl_hours = (
            cache_ttl_hours if cache_ttl_hours is not None
            else cfg.cache_ttl_hours
        )
        self.strict_mode = (
            strict_mode if strict_mode is not None
            else cfg.cache_strict_mode
        )
        
        print("📦 Cache Manager initialized:")
        print(f"   - Similarity threshold: {self.similarity_threshold}")
        print(f"   - TTL: {self.cache_ttl_hours} hours")
        print(f"   - Strict mode: {self.strict_mode}")

        self.model = None  # Lazy load the model
    
    def _get_model(self):
        """Lazy load the SentenceTransformer model."""
        if self.model is None:
            try:
                # Use configurable embedding model from environment or default
                embedding_model_name = get_config().embedding_model
                self.model = load_embedding_model(embedding_model_name)
            except ModelWarmingUp:
                raise  # the request gets a 503 warming_up, not a silent miss
            except Exception as e:
                print(f"Warning: Could not load SentenceTransformer model: {e}")
                print("Cache functionality will be limited.")
                return None
        return self.model
    
    def _generate_hash(self, content: str) -> str:
        """Generate a consistent hash for content."""
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _normalize_expense_data_for_cache(self, expense_data: list[dict]) -> str:
        """
        Normalize expense data for consistent cache key generation.
        
        Removes fields that vary but don't affect cache validity:
        - similarity_score: Varies with floating-point precision
        - Random ordering: Sorts by stable key
        
        This ensures the same expense data always produces the same hash,
        regardless of whether it came from a fresh DB query or cache.
        
        Args:
            expense_data: List of expense dictionaries
            
        Returns:
            JSON string of normalized expense data
        """
        normalized = []
        for expense in expense_data:
            # Convert date to string if it's a date object
            expense_date = expense.get('expense_date', '')
            if hasattr(expense_date, 'isoformat'):
                expense_date = expense_date.isoformat()
            elif expense_date:
                expense_date = str(expense_date)
            
            # Only include stable fields that define the expense
            normalized_expense = {
                'expense_id': expense.get('expense_id', ''),
                'user_id': expense.get('user_id', ''),
                'merchant': expense.get('merchant', ''),
                'expense_amount': round(float(expense.get('expense_amount', 0)), 2),  # Round to 2 decimals
                'description': expense.get('description', ''),
                'expense_date': expense_date,
                'shopping_type': expense.get('shopping_type', ''),
                # Explicitly exclude: similarity_score (varies with precision)
            }
            normalized.append(normalized_expense)
        
        # Sort by expense_id for consistent ordering
        # If expense_id is empty, fall back to merchant + amount
        normalized.sort(key=lambda x: (
            x.get('expense_id') or '',
            x.get('merchant') or '',
            x.get('expense_amount') or 0
        ))
        
        return safe_json_dumps(normalized, sort_keys=True)
    
    @db_retry(max_attempts=3, initial_delay=0.5)
    def _get_embedding_with_cache(self, input_text: str) -> np.ndarray:
        """Get embedding for text, using cache when possible."""
        text_hash = self._generate_hash(input_text)

        # Try to get from cache first
        model_name = get_config().embedding_model
        lookup_start = time.perf_counter()
        cache_query = text("""
            SELECT embedding, access_count
            FROM embedding_cache 
            WHERE text_hash = :text_hash AND model_name = :model_name
        """)
        with _engine().connect() as conn:
            result = conn.execute(cache_query, {'text_hash': text_hash, 'model_name': model_name})
            row = result.fetchone()
            
            if row:
                update_query = text("""
                    UPDATE embedding_cache 
                    SET access_count = access_count + 1 
                    WHERE text_hash = :text_hash AND model_name = :model_name
                """)
                conn.execute(update_query, {'text_hash': text_hash, 'model_name': model_name})
                conn.commit()
                
                self._log_cache_stat('embedding', 'hit', tokens_saved=10)
                trace_step("cache", "Embedding cache", "Hit: reused the stored embedding",
                           ms=(time.perf_counter() - lookup_start) * 1000, status="hit")
                return np.array(json.loads(row.embedding))
        
        # Cache miss - generate embedding and store
        model = self._get_model()
        if model is None:
            return None
        encode_start = time.perf_counter()
        embedding = model.encode(input_text)
        trace_step("cache", "Embedding cache",
                   f"Miss: embedded locally with {model_name} ({len(embedding)} dims)",
                   ms=(time.perf_counter() - encode_start) * 1000, status="miss")
        embedding_json = json.dumps(embedding.tolist())
        
        try:
            with _engine().connect() as conn:
                insert_query = text("""
                    INSERT INTO embedding_cache (text_hash, text_content, embedding, model_name, access_count)
                    VALUES (:text_hash, :text_content, :embedding, :model_name, 1)
                    ON CONFLICT (text_hash, model_name) DO UPDATE SET access_count = embedding_cache.access_count + 1
                """)
                conn.execute(insert_query, {
                    'text_hash': text_hash,
                    'text_content': input_text[:500],
                    'embedding': embedding_json,
                    'model_name': model_name,
                })
                conn.commit()
                self._log_cache_stat('embedding', 'miss')
                
        except Exception as e:
            print(f"⚠️ Error caching embedding: {e}")
        
        return embedding
    
    @db_retry(max_attempts=3, initial_delay=0.5)
    def get_cached_response(self, query: str, expense_data: list[dict], ai_service: str, language: str = "en") -> str | None:
        """
        Check if we have a cached response for a similar query.

        Args:
            query: User query text
            expense_data: Current expense data context
            ai_service: AI service being used (watsonx, bedrock)
            language: Response language code or name

        Returns:
            Cached response text if found, None otherwise
        """
        scope = current_user_scope()
        query_embedding = self._get_embedding_with_cache(query)
        lookup_start = time.perf_counter()
        # Use normalized expense data for consistent hashing
        expense_hash = self._generate_hash(self._normalize_expense_data_for_cache(expense_data))
        
        # Debug: Print what we're searching for
        print("   🔎 Searching query_cache:")
        print(f"      - ai_service: '{ai_service}'")
        print(f"      - language: '{language}'")
        print(f"      - expense_hash: {expense_hash}")
        print(f"      - query: '{query[:60]}...'")
        print(f"      - similarity_threshold: {self.similarity_threshold}")
        
        # First, check if ANY entries exist for this service
        count_query = text("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN expense_data_hash = :expense_hash THEN 1 ELSE 0 END) as hash_match,
                   SUM(CASE WHEN expires_at > now() THEN 1 ELSE 0 END) as not_expired
            FROM query_cache
            WHERE ai_service = :ai_service AND scope = :scope
        """)
        
        # Find similar cached queries
        # In strict mode: require exact expense_hash match
        # In lenient mode: check similarity first, then validate data overlap
        if self.strict_mode:
            similarity_query = text("""
                SELECT cache_id, query_text, response_text, response_tokens, prompt_tokens,
                       query_embedding <=> :query_embedding as similarity_distance,
                       hit_count, expires_at,
                       expense_data_hash,
                       expires_at > now() as not_expired,
                       now() as current_time
                FROM query_cache
                WHERE ai_service = :ai_service
                  AND language = :language
                  AND scope = :scope
                  AND scope != ''
                  AND expense_data_hash = :expense_hash
                  AND expires_at > now()
                  AND query_embedding IS NOT NULL
                ORDER BY query_embedding <=> :query_embedding
                LIMIT 5
            """)
        else:
            similarity_query = text("""
                SELECT cache_id, query_text, response_text, response_tokens, prompt_tokens,
                       query_embedding <=> :query_embedding as similarity_distance,
                       hit_count, expires_at,
                       expense_data_hash,
                       expires_at > now() as not_expired,
                       now() as current_time
                FROM query_cache
                WHERE ai_service = :ai_service
                  AND language = :language
                  AND scope = :scope
                  AND scope != ''
                  AND expires_at > now()
                  AND query_embedding IS NOT NULL
                ORDER BY query_embedding <=> :query_embedding
                LIMIT 5
            """)
        
        try:
            with _engine().connect() as conn:
                # First check counts for debugging
                count_result = conn.execute(count_query, {
                    'ai_service': ai_service,
                    'scope': scope,
                    'expense_hash': expense_hash
                })
                count_row = count_result.fetchone()
                print(f"      - Total entries for '{ai_service}': {count_row.total}")
                print(f"      - Entries with matching hash: {count_row.hash_match}")
                print(f"      - Non-expired entries: {count_row.not_expired}")

                result = conn.execute(similarity_query, {
                    'query_embedding': json.dumps(query_embedding.tolist()),
                    'ai_service': ai_service,
                    'language': language,
                    'scope': scope,
                    'expense_hash': expense_hash
                })
                
                rows = list(result)
                if rows:
                    print(f"   🔍 Found {len(rows)} candidate cache entries after filtering")
                    for i, row in enumerate(rows, 1):
                        similarity_distance = row.similarity_distance
                        similarity_score = 1 - similarity_distance if similarity_distance <= 1 else 0
                        print(f"      {i}. Query: '{row.query_text[:40]}...'")
                        print(f"         - Expense hash: {row.expense_data_hash[:20]}...")
                        print(f"         - Not expired: {row.not_expired}")
                        print(f"         - Similarity distance: {similarity_distance:.6f}")
                        print(f"         - Similarity score: {similarity_score:.6f} (threshold: {self.similarity_threshold})")
                        print(f"         - Current time: {row.current_time}")
                        print(f"         - Expires at: {row.expires_at}")
                else:
                    print(f"   🔍 No cache entries match all criteria (service={ai_service}, hash match, not expired, has embedding)")
                
                for row in rows:
                    similarity_distance = row.similarity_distance
                    # Convert distance to similarity score (0-1 range, higher is better)
                    # For cosine distance: similarity = 1 - distance
                    similarity_score = 1 - similarity_distance if similarity_distance <= 1 else 0
                    
                    # Determine confidence level
                    if similarity_score >= 0.90:
                        confidence = "HIGH"
                    elif similarity_score >= 0.70:
                        confidence = "MEDIUM"
                    else:
                        confidence = "LOW"
                    
                    if similarity_score >= self.similarity_threshold:
                        # In lenient mode with medium confidence, warn about potential mismatch
                        if not self.strict_mode and confidence == "MEDIUM" and row.expense_data_hash != expense_hash:
                            print(f"   ⚠️  MEDIUM confidence match with different data (similarity: {similarity_score:.3f})")
                            print("      Consider enabling strict mode for higher accuracy")
                        # Cache hit! Update statistics
                        update_query = text("""
                            UPDATE query_cache 
                            SET hit_count = hit_count + 1, last_accessed = now()
                            WHERE cache_id = :cache_id
                        """)
                        conn.execute(update_query, {'cache_id': row.cache_id})
                        conn.commit()
                        
                        tokens_saved = (row.response_tokens or 500) + (row.prompt_tokens or 400)
                        self._log_cache_stat('query', 'hit', tokens_saved=tokens_saved, details={
                            'original_query': row.query_text,
                            'new_query': query,
                            'similarity_score': similarity_score
                        })
                        
                        print(f"🎯 Cache HIT! Confidence: {confidence} | Similarity: {similarity_score:.3f} | Tokens saved: {tokens_saved}")
                        print(f"   Original: '{row.query_text[:50]}...'")
                        print(f"   Current:  '{query[:50]}...'")
                        print(f"   Mode: {'STRICT (exact data match)' if self.strict_mode else 'LENIENT (similarity only)'}")
                        trace_step("cache", "Response cache",
                                   f"Hit: similarity {similarity_score:.3f} ({confidence.lower()} confidence), "
                                   f"est. {tokens_saved} tokens saved",
                                   ms=(time.perf_counter() - lookup_start) * 1000, status="hit")
                        
                        return row.response_text
                
        except Exception as e:
            print(f"⚠️ Error checking query cache: {e}")
        
        self._log_cache_stat('query', 'miss')
        trace_step("cache", "Response cache", "Miss: no similar answer cached",
                   ms=(time.perf_counter() - lookup_start) * 1000, status="miss")
        return None
    
    def cache_response(self, query: str, response: str, expense_data: list[dict],
                      ai_service: str, prompt_tokens: int = 0, response_tokens: int = 0,
                      language: str = "en"):
        """
        Cache a query response for future use.

        Args:
            query: Original user query
            response: AI response
            expense_data: Expense data context
            ai_service: AI service used
            prompt_tokens: Number of prompt tokens used
            response_tokens: Number of response tokens generated
            language: Response language code or name
        """
        scope = current_user_scope()
        query_hash = self._generate_hash(f"{scope}|{query}")
        query_embedding = self._get_embedding_with_cache(query)
        # Use normalized expense data for consistent hashing
        expense_hash = self._generate_hash(self._normalize_expense_data_for_cache(expense_data))
        expires_at = datetime.utcnow() + timedelta(hours=self.cache_ttl_hours)
        write_start = time.perf_counter()
        
        try:
            with _engine().connect() as conn:
                insert_query = text("""
                    INSERT INTO query_cache (
                        query_hash, query_text, query_embedding, response_text,
                        response_tokens, prompt_tokens, ai_service, language,
                        expense_data_hash, scope, expires_at
                    ) VALUES (
                        :query_hash, :query_text, :query_embedding, :response_text,
                        :response_tokens, :prompt_tokens, :ai_service, :language,
                        :expense_hash, :scope, :expires_at
                    )
                    ON CONFLICT (query_hash, ai_service, language) DO UPDATE SET
                        response_text = EXCLUDED.response_text,
                        response_tokens = EXCLUDED.response_tokens,
                        prompt_tokens = EXCLUDED.prompt_tokens,
                        expense_data_hash = EXCLUDED.expense_data_hash,
                        query_embedding = EXCLUDED.query_embedding,
                        expires_at = EXCLUDED.expires_at,
                        hit_count = 0,
                        last_accessed = now()
                """)

                conn.execute(insert_query, {
                    'query_hash': query_hash,
                    'query_text': query,
                    'query_embedding': json.dumps(query_embedding.tolist()),
                    'response_text': response,
                    'response_tokens': response_tokens,
                    'prompt_tokens': prompt_tokens,
                    'ai_service': ai_service,
                    'language': language,
                    'expense_hash': expense_hash,
                    'scope': scope,
                    'expires_at': expires_at
                })
                conn.commit()
                print("   💾 Stored in query_cache:")
                trace_step("cache", "Response cache write",
                           f"Stored for {self.cache_ttl_hours}h",
                           ms=(time.perf_counter() - write_start) * 1000, status="success")
                print(f"      - query: '{query[:60]}...'")
                print(f"      - ai_service: '{ai_service}'")
                print(f"      - expense_hash: {expense_hash}")
                print(f"      - expires_at: {expires_at}")
                print(f"      - embedding dimensions: {len(query_embedding)}")
                
                # Verify the entry was actually stored with correct hash
                verify_query = text("""
                    SELECT cache_id, expense_data_hash, expires_at > now() as not_expired
                    FROM query_cache
                    WHERE query_hash = :query_hash AND ai_service = :ai_service
                """)
                verify_result = conn.execute(verify_query, {
                    'query_hash': query_hash,
                    'ai_service': ai_service
                })
                verify_row = verify_result.fetchone()
                if verify_row:
                    hash_matches = verify_row.expense_data_hash == expense_hash
                    hash_status = "✅" if hash_matches else "❌"
                    print(f"      ✅ Verified: Entry stored (cache_id: {str(verify_row.cache_id)[:8]}..., not_expired: {verify_row.not_expired})")
                    print(f"      {hash_status} Expense hash match: {hash_matches}")
                    if not hash_matches:
                        print(f"         Expected: {expense_hash}")
                        print(f"         Got:      {verify_row.expense_data_hash}")
                else:
                    print("      ⚠️  Warning: Could not verify entry was stored")
                
                self._log_cache_stat('query', 'write', details={
                    'query_length': len(query),
                    'response_length': len(response)
                })
                
        except Exception as e:
            print(f"⚠️ Error caching response: {e}")
            trace_step("cache", "Response cache write", "Not stored this time",
                       ms=(time.perf_counter() - write_start) * 1000, status="warning")
    
    @db_retry(max_attempts=3, initial_delay=0.5)
    def get_cached_vector_search(self, query_embedding: np.ndarray, limit: int = 5, scope: str | None = None) -> list[dict] | None:
        """Get cached vector search results."""
        scope_key = scope if scope is not None else current_user_scope()
        embedding_hash = self._generate_hash(f"{scope_key}|{json.dumps(query_embedding.tolist())}")
        
        cache_query = text("""
            SELECT search_results, access_count
            FROM vector_search_cache
            WHERE query_embedding_hash = :embedding_hash
              AND expires_at > now()
              AND result_count >= :limit
            ORDER BY created_at DESC
            LIMIT 1
        """)
        
        try:
            with _engine().connect() as conn:
                result = conn.execute(cache_query, {
                    'embedding_hash': embedding_hash,
                    'limit': limit
                })
                row = result.fetchone()
                
                if row:
                    # Update access count
                    update_query = text("""
                        UPDATE vector_search_cache 
                        SET access_count = access_count + 1 
                        WHERE query_embedding_hash = :embedding_hash
                    """)
                    conn.execute(update_query, {'embedding_hash': embedding_hash})
                    conn.commit()
                    
                    self._log_cache_stat('vector_search', 'hit', tokens_saved=50)
                    # search_results is already a list from JSONB, no need to parse
                    if isinstance(row.search_results, list):
                        return row.search_results[:limit]
                    else:
                        # Fallback: try to parse if it's a string
                        return json.loads(row.search_results)[:limit]
                    
        except Exception as e:
            print(f"⚠️ Error reading vector search cache: {e}")
        
        self._log_cache_stat('vector_search', 'miss')
        return None
    
    def cache_vector_search_results(self, query_embedding: np.ndarray, results: list[dict], scope: str | None = None):
        """Cache vector search results."""
        scope_key = scope if scope is not None else current_user_scope()
        embedding_hash = self._generate_hash(f"{scope_key}|{json.dumps(query_embedding.tolist())}")
        expires_at = datetime.utcnow() + timedelta(hours=self.cache_ttl_hours)
        
        try:
            with _engine().connect() as conn:
                insert_query = text("""
                    INSERT INTO vector_search_cache (
                        query_embedding_hash, search_results, result_count, 
                        similarity_threshold, expires_at
                    ) VALUES (
                        :embedding_hash, :results, :count, :threshold, :expires_at
                    )
                    ON CONFLICT (query_embedding_hash) DO UPDATE SET
                        search_results = EXCLUDED.search_results,
                        result_count = EXCLUDED.result_count,
                        expires_at = EXCLUDED.expires_at,
                        access_count = 0
                """)
                
                conn.execute(insert_query, {
                    'embedding_hash': embedding_hash,
                    'results': safe_json_dumps(results),
                    'count': len(results),
                    'threshold': self.similarity_threshold,
                    'expires_at': expires_at
                })
                conn.commit()
                
                self._log_cache_stat('vector_search', 'write')
                
        except Exception as e:
            print(f"⚠️ Error caching vector search: {e}")
    

    
    def _log_cache_stat(self, cache_type: str, operation: str, tokens_saved: int = 0, details: dict = None):
        """Log cache statistics for monitoring."""
        try:
            with _engine().connect() as conn:
                insert_query = text("""
                    INSERT INTO cache_stats (cache_type, operation, tokens_saved, details)
                    VALUES (:cache_type, :operation, :tokens_saved, :details)
                """)
                conn.execute(insert_query, {
                    'cache_type': cache_type,
                    'operation': operation,
                    'tokens_saved': tokens_saved,
                    'details': json.dumps(details) if details else None
                })
                conn.commit()
        except Exception as e:
            print(f"⚠️ Error logging cache stats: {e}")
    
    def get_cache_stats(self, hours: int = 24) -> dict:
        """Get cache performance statistics."""
        stats_query = text("""
            WITH cache_summary AS (
                SELECT 
                    cache_type,
                    operation,
                    COUNT(*) as count,
                    SUM(tokens_saved) as total_tokens_saved
                FROM cache_stats 
                WHERE timestamp >= now() - (:hours * INTERVAL '1 hour')
                GROUP BY cache_type, operation
            )
            SELECT 
                cache_type,
                SUM(CASE WHEN operation = 'hit' THEN count ELSE 0 END) as hits,
                SUM(CASE WHEN operation = 'miss' THEN count ELSE 0 END) as misses,
                SUM(CASE WHEN operation = 'write' THEN count ELSE 0 END) as writes,
                SUM(total_tokens_saved) as tokens_saved
            FROM cache_summary
            GROUP BY cache_type
            ORDER BY cache_type
        """)
        
        try:
            with _engine().connect() as conn:
                result = conn.execute(stats_query, {'hours': hours})
                stats = {}
                total_tokens_saved = 0
                
                for row in result:
                    hit_rate = row.hits / (row.hits + row.misses) if (row.hits + row.misses) > 0 else 0
                    stats[row.cache_type] = {
                        'hits': row.hits,
                        'misses': row.misses,
                        'writes': row.writes,
                        'hit_rate': hit_rate,
                        'tokens_saved': row.tokens_saved
                    }
                    total_tokens_saved += float(row.tokens_saved or 0)
                
                stats['total_tokens_saved'] = total_tokens_saved
                return stats
                
        except Exception as e:
            print(f"⚠️ Error getting cache stats: {e}")
            return {}
    
    def cleanup_expired_cache(self):
        """Remove expired cache entries."""
        cleanup_queries = [
            "DELETE FROM query_cache WHERE expires_at < now()",
            "DELETE FROM vector_search_cache WHERE expires_at < now()"
        ]
        
        try:
            with _engine().connect() as conn:
                for query in cleanup_queries:
                    result = conn.execute(text(query))
                    print(f"🧹 Cleaned up {result.rowcount} expired cache entries")
                conn.commit()
        except Exception as e:
            print(f"⚠️ Error cleaning up cache: {e}")
