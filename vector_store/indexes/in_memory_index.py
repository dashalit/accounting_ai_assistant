"""
In-memory vector index.
Simple implementation for small datasets.
"""
import math
from typing import List, Dict, Optional
from dataclasses import field, dataclass

from ..core import VectorIndex, VectorDocument, SearchResult, SearchQuery, DistanceMetric


@dataclass
class InMemoryIndex(VectorIndex):
    """
    In-memory vector index with brute-force search.
    Suitable for small datasets (< 10k vectors).
    """
    
    _documents: Dict[str, VectorDocument] = field(default_factory=dict)
    _metric: DistanceMetric = DistanceMetric.COSINE
    
    def __init__(
        self,
        documents: Optional[Dict[str, VectorDocument]] = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ):
        self._documents = documents or {}
        self._metric = metric
    
    def add(self, documents: List[VectorDocument]) -> None:
        """Add documents to index."""
        for doc in documents:
            self._documents[doc.id] = doc
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search for similar vectors using brute-force."""
        if not self._documents:
            return []
        
        query_vector = query.vector
        results = []
        
        for doc_id, doc in self._documents.items():
            # Apply filter if provided
            if query.filter_fn and not query.filter_fn(doc):
                continue
            
            # Calculate similarity
            score = self._calculate_similarity(query_vector, doc.vector)
            results.append(SearchResult(document=doc, score=score))
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        
        # Apply top_k
        results = results[:query.top_k]
        
        # Assign ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        # Optionally remove vectors from results
        if not query.include_vectors:
            for result in results:
                result.document.vector = []
        
        return results
    
    def remove(self, doc_ids: List[str]) -> None:
        """Remove documents by ID."""
        for doc_id in doc_ids:
            self._documents.pop(doc_id, None)
    
    def clear(self) -> None:
        """Clear all documents from index."""
        self._documents.clear()
    
    @property
    def size(self) -> int:
        """Number of documents in index."""
        return len(self._documents)
    
    @property
    def dimension(self) -> Optional[int]:
        """Vector dimension."""
        if not self._documents:
            return None
        first_doc = next(iter(self._documents.values()))
        return first_doc.dimension
    
    def _calculate_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate similarity based on metric."""
        if self._metric == DistanceMetric.COSINE:
            return self._cosine_similarity(v1, v2)
        elif self._metric == DistanceMetric.DOT_PRODUCT:
            return self._dot_product(v1, v2)
        elif self._metric == DistanceMetric.EUCLIDEAN:
            return self._euclidean_similarity(v1, v2)
        else:
            raise ValueError(f"Unknown metric: {self._metric}")
    
    @staticmethod
    def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity."""
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    @staticmethod
    def _dot_product(v1: List[float], v2: List[float]) -> float:
        """Calculate dot product."""
        return sum(a * b for a, b in zip(v1, v2))
    
    @staticmethod
    def _euclidean_similarity(v1: List[float], v2: List[float]) -> float:
        """Calculate euclidean distance converted to similarity."""
        squared_diff = sum((a - b) ** 2 for a, b in zip(v1, v2))
        distance = math.sqrt(squared_diff)
        # Convert distance to similarity (higher = more similar)
        return 1.0 / (1.0 + distance)
    
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get document by ID."""
        return self._documents.get(doc_id)
    
    def get_all_documents(self) -> List[VectorDocument]:
        """Get all documents."""
        return list(self._documents.values())
