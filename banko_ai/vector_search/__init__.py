"""Vector search functionality for Banko AI Assistant."""

def get_data_enricher():
    """Get DataEnricher (lazy import)."""
    from .enrichment import DataEnricher
    return DataEnricher

def get_vector_search_engine():
    """Get VectorSearchEngine (lazy import)."""
    from .search import VectorSearchEngine
    return VectorSearchEngine

__all__ = ["get_data_enricher", "get_vector_search_engine"]
