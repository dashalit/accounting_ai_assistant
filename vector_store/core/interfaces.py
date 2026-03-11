"""
Core interfaces and domain models for vector store.
Implements GRASP Information Expert pattern.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class DistanceMetric(Enum):
    """Distance metrics for similarity search."""
    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


@dataclass
class VectorDocument:
    """
    Document with vector representation.
    GRASP Information Expert - holds vector and metadata.
    """
    id: str
    vector: List[float]
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def dimension(self) -> int:
        return len(self.vector)


@dataclass
class SearchResult:
    """Search result with score."""
    document: VectorDocument
    score: float
    rank: int = 0


@dataclass
class SearchQuery:
    """Search query configuration."""
    vector: List[float]
    top_k: int = 5
    filter_fn: Optional[callable] = None
    include_vectors: bool = False


class VectorIndex(ABC):
    """
    Strategy pattern for vector indexing.
    Different index implementations can be swapped at runtime.
    """
    
    @abstractmethod
    def add(self, documents: List[VectorDocument]) -> None:
        """Add documents to index."""
        pass
    
    @abstractmethod
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    def remove(self, doc_ids: List[str]) -> None:
        """Remove documents by ID."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from index."""
        pass
    
    @property
    @abstractmethod
    def size(self) -> int:
        """Number of documents in index."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> Optional[int]:
        """Vector dimension."""
        pass


class SearchStrategy(ABC):
    """
    Strategy pattern for search algorithms.
    Allows switching between similarity, MMR, etc.
    """
    
    @abstractmethod
    def search(
        self,
        index: VectorIndex,
        query: SearchQuery,
    ) -> List[SearchResult]:
        """Execute search strategy."""
        pass


class PersistenceStrategy(ABC):
    """
    Strategy pattern for persistence.
    Supports different storage backends.
    """
    
    @abstractmethod
    def save(self, index: VectorIndex, path: str) -> None:
        """Save index to storage."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> VectorIndex:
        """Load index from storage."""
        pass
    
    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check if index exists at path."""
        pass


class VectorStore(ABC):
    """
    Facade interface for vector store operations.
    GoF Facade pattern - simplifies complex subsystem.
    """
    
    @abstractmethod
    def add_documents(self, documents: List[VectorDocument]) -> List[str]:
        """Add documents and return IDs."""
        pass
    
    @abstractmethod
    def search(self, query_vector: List[float], top_k: int = 5) -> List[SearchResult]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def delete(self, doc_ids: List[str]) -> None:
        """Delete documents by ID."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Persist store to disk."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "VectorStore":
        """Load store from disk."""
        pass
