"""
Vector Store - Store and search vector embeddings.

Architecture based on GoF patterns:
- Strategy: Index, search, and persistence algorithms
- Facade: VectorStoreImpl simplifies subsystem interactions
- Factory: create_vector_store function

Example usage:
    from vector_store import VectorStoreImpl, create_vector_store
    from vector_store.core import VectorDocument, DistanceMetric
    
    # Quick start
    store = create_vector_store("memory")
    
    # Add documents
    docs = [
        VectorDocument(id="1", vector=[0.1, 0.2], content="Text 1"),
        VectorDocument(id="2", vector=[0.9, 0.8], content="Text 2"),
    ]
    store.add_documents(docs)
    
    # Search
    results = store.search(query_vector=[0.15, 0.25], top_k=2)
    
    # With MMR for diverse results
    from vector_store import MMRSearch
    store.set_search_strategy(MMRSearch(lambda_param=0.7))
"""

from .core import (
    DistanceMetric,
    VectorDocument,
    SearchResult,
    SearchQuery,
    VectorIndex,
    SearchStrategy,
    PersistenceStrategy,
    VectorStore,
)
from .store import VectorStoreImpl, create_vector_store
from .indexes import InMemoryIndex, FaissIndex
from .strategies import SimilaritySearch, MMRSearch
from .persistence import JsonPersistence, PicklePersistence

__all__ = [
    # Core types
    "DistanceMetric",
    "VectorDocument",
    "SearchResult",
    "SearchQuery",
    "VectorIndex",
    "SearchStrategy",
    "PersistenceStrategy",
    "VectorStore",
    
    # Implementations
    "VectorStoreImpl",
    "create_vector_store",
    "InMemoryIndex",
    "FaissIndex",
    "SimilaritySearch",
    "MMRSearch",
    "JsonPersistence",
    "PicklePersistence",
]
