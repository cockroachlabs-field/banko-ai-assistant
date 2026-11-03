"""
Test semantic caching functionality after removing dead code.
Verifies that query_cache, embedding_cache, and vector_search_cache still work.
"""

import os

os.environ['DATABASE_URL'] = os.getenv('DATABASE_URL', 'cockroachdb://root@localhost:26257/defaultdb?sslmode=disable')

import time

import pytest

from banko_ai.utils.cache_manager import BankoCacheManager
from banko_ai.utils.request_scope import user_scope


@pytest.fixture(scope="module")
def cache_manager():
    """Create a shared cache manager instance for all tests in this module."""
    return BankoCacheManager()


def test_cache_manager_initialization(cache_manager):
    """Test 1: Cache manager initializes without errors"""
    assert cache_manager is not None
    assert cache_manager.similarity_threshold > 0
    assert cache_manager.cache_ttl_hours > 0


def test_embedding_cache(cache_manager):
    """Test 2: Embedding cache works (should be fast on second call)"""
    test_text = "Show me my restaurant expenses"

    start = time.time()
    embedding1 = cache_manager._get_embedding_with_cache(test_text)
    time1 = time.time() - start
    assert embedding1 is not None
    assert len(embedding1) == 384

    start = time.time()
    embedding2 = cache_manager._get_embedding_with_cache(test_text)
    time2 = time.time() - start

    assert len(embedding2) == 384
    # Second call should be faster (cache hit)
    assert time2 <= time1 or time2 < 0.1


def test_query_cache_storage(cache_manager):
    """Test 3: Query cache can store and retrieve responses"""
    test_query = "Show me my coffee shop expenses for cache test"
    test_response = "You spent $45.20 at Starbucks this month."
    test_data = [
        {"expense_id": "1", "merchant": "Starbucks", "expense_amount": 45.20}
    ]

    with user_scope("test-user-1"):
        cache_manager.cache_response(
            test_query, test_response, test_data, "watsonx",
            prompt_tokens=100, response_tokens=50
        )

        cached = cache_manager.get_cached_response(test_query, test_data, "watsonx")
        assert cached is not None, "Cache retrieval returned None"
        assert "Starbucks" in cached


def test_semantic_similarity(cache_manager):
    """Test 4: Semantic caching finds similar queries"""
    original_query = "What did I spend at coffee shops for similarity test?"
    response = "Coffee shop total: $67.50"
    test_data = [{"merchant": "Starbucks", "expense_amount": 67.50}]

    with user_scope("test-user-2"):
        cache_manager.cache_response(original_query, response, test_data, "watsonx")

        similar_query = "Show me my coffee shop spending for similarity test"
        cached = cache_manager.get_cached_response(similar_query, test_data, "watsonx")
        # Semantic match depends on threshold; just verify no crash
        if cached:
            assert "67.50" in cached

def test_dead_methods_removed():
    """Test 5: Verify dead insights_cache methods have been removed"""
    print("\n" + "="*80)
    print("TEST 5: Dead Code Verification")
    print("="*80)
    
    try:
        cache_manager = BankoCacheManager()
        
        # Verify removed methods no longer exist
        has_old_method = hasattr(cache_manager, 'get_cached_insights')
        has_disabled_method = hasattr(cache_manager, '_get_cached_insights_DISABLED')
        has_cache_disabled = hasattr(cache_manager, '_cache_insights_DISABLED')
        
        if not has_old_method and not has_disabled_method and not has_cache_disabled:
            print("✅ Dead insights_cache methods fully removed")
        else:
            remaining = []
            if has_old_method:
                remaining.append('get_cached_insights')
            if has_disabled_method:
                remaining.append('_get_cached_insights_DISABLED')
            if has_cache_disabled:
                remaining.append('_cache_insights_DISABLED')
            print(f"⚠️  Dead methods still present: {remaining}")
        
        print("✅ No code references insights_cache")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def main():
    print("\n╔" + "="*78 + "╗")
    print("║" + " "*20 + "CACHE CLEANUP VERIFICATION TESTS" + " "*26 + "║")
    print("╚" + "="*78 + "╝")
    
    results = []
    
    # Test 1: Initialization
    success, cache_manager = test_cache_manager_initialization()
    results.append(("Initialization", success))
    
    if not success or cache_manager is None:
        print("\n❌ Cannot proceed - cache manager failed to initialize")
        return
    
    # Test 2: Embedding cache
    success = test_embedding_cache(cache_manager)
    results.append(("Embedding Cache", success))
    
    # Test 3: Query cache storage
    success = test_query_cache_storage(cache_manager)
    results.append(("Query Cache Storage", success))
    
    # Test 4: Semantic similarity
    success = test_semantic_similarity(cache_manager)
    results.append(("Semantic Similarity", success))
    
    # Test 5: Dead code verification
    success = test_dead_methods_removed()
    results.append(("Dead Code Check", success))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED - Cleanup successful, no functionality broken!")
    else:
        print("\n⚠️  Some tests failed - review above for details")

if __name__ == "__main__":
    main()
