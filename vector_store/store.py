"""
Vector Store Facade.
GoF Facade pattern - provides simplified interface to vector store subsystem.
"""
import uuid
from typing import List, Optional, Callable, Union
from pathlib import Path

from .core import (
    VectorStore,
    VectorIndex,
    VectorDocument,
    SearchResult,
    SearchQuery,
    SearchStrategy,
    PersistenceStrategy,
    DistanceMetric,
)
from .indexes import InMemoryIndex, FaissIndex
from .strategies import SimilaritySearch, MMRSearch
from .persistence import JsonPersistence


class VectorStoreImpl(VectorStore):
    """
    Main vector store implementation.
    GoF Facade pattern - simplifies complex subsystem interactions.
    """
    
    def __init__(
        self,
        index: Optional[VectorIndex] = None,
        search_strategy: Optional[SearchStrategy] = None,
        persistence: Optional[PersistenceStrategy] = None,
        dimension: Optional[int] = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ):
        """
        Initialize vector store.
        
        Args:
            index: Vector index to use (creates InMemoryIndex if None).
            search_strategy: Search strategy (creates SimilaritySearch if None).
            persistence: Persistence strategy (creates JsonPersistence if None).
            dimension: Vector dimension (required for FaissIndex).
            metric: Distance metric for similarity calculation.
        """
        self._index = index or InMemoryIndex(metric=metric)
        self._search_strategy = search_strategy or SimilaritySearch()
        self._persistence = persistence or JsonPersistence()
        self._dimension = dimension
        self._metric = metric
    
    def add_documents(
        self,
        documents: List[VectorDocument],
    ) -> List[str]:
        """
        Add documents to store.
        
        Args:
            documents: Documents to add.
            
        Returns:
            List of document IDs.
        """
        # Assign IDs to documents without them
        for doc in documents:
            if not doc.id:
                doc.id = str(uuid.uuid4())
        
        self._index.add(documents)
        return [doc.id for doc in documents]
    
    def add_texts(
        self,
        texts: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add texts with vectors to store.
        
        Args:
            texts: Text contents.
            vectors: Corresponding vectors.
            metadatas: Optional metadata for each text.
            ids: Optional IDs for each text.
            
        Returns:
            List of document IDs.
        """
        documents = []
        for i, (text, vector) in enumerate(zip(texts, vectors)):
            doc = VectorDocument(
                id=ids[i] if ids and i < len(ids) else str(uuid.uuid4()),
                vector=vector,
                content=text,
                metadata=metadatas[i] if metadatas and i < len(metadatas) else {},
            )
            documents.append(doc)
        
        return self.add_documents(documents)
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filter_fn: Optional[Callable[[VectorDocument], bool]] = None,
        strategy: Optional[SearchStrategy] = None,
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query_vector: Query vector.
            top_k: Number of results to return.
            filter_fn: Optional filter function.
            strategy: Override search strategy for this query.
            
        Returns:
            List of search results.
        """
        query = SearchQuery(
            vector=query_vector,
            top_k=top_k,
            filter_fn=filter_fn,
        )
        
        # Use provided strategy or default
        search_strategy = strategy or self._search_strategy
        
        return search_strategy.search(self._index, query)
    
    def search_by_text(
        self,
        query_text: str,
        embedder,  # Embedding model from document_processor
        top_k: int = 5,
        filter_fn: Optional[Callable[[VectorDocument], bool]] = None,
    ) -> List[SearchResult]:
        """
        Search using text query (requires embedding model).
        
        Args:
            query_text: Query text.
            embedder: Embedding model to convert text to vector.
            top_k: Number of results.
            filter_fn: Optional filter function.
            
        Returns:
            List of search results.
        """
        from document_processor.core import Chunk
        
        # Generate embedding for query
        chunk = Chunk(content=query_text)
        vectors = embedder.embed([chunk])
        
        if not vectors:
            return []
        
        query_vector = vectors[0].values
        return self.search(query_vector, top_k, filter_fn)
    
    def delete(self, doc_ids: List[str]) -> None:
        """
        Delete documents by ID.
        
        Args:
            doc_ids: IDs of documents to delete.
        """
        self._index.remove(doc_ids)
    
    def get(self, doc_id: str) -> Optional[VectorDocument]:
        """
        Get document by ID.
        
        Args:
            doc_id: Document ID.
            
        Returns:
            Document or None.
        """
        return self._index.get_document(doc_id)
    
    def get_all(self) -> List[VectorDocument]:
        """Get all documents."""
        return self._index.get_all_documents()
    
    def clear(self) -> None:
        """Clear all documents."""
        self._index.clear()
    
    @property
    def size(self) -> int:
        """Number of documents in store."""
        return self._index.size
    
    @property
    def dimension(self) -> Optional[int]:
        """Vector dimension."""
        return self._index.dimension
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Persist store to disk.
        
        Args:
            path: Path to save file.
        """
        self._persistence.save(self._index, str(path))
    
    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        search_strategy: Optional[SearchStrategy] = None,
    ) -> "VectorStoreImpl":
        """
        Load store from disk.
        
        Args:
            path: Path to load file.
            search_strategy: Search strategy to use.
            
        Returns:
            Loaded VectorStore.
        """
        persistence = JsonPersistence()
        index = persistence.load(str(path))
        
        return cls(
            index=index,
            search_strategy=search_strategy,
            persistence=persistence,
        )
    
    @property
    def index(self) -> VectorIndex:
        """Get underlying index."""
        return self._index
    
    def set_search_strategy(self, strategy: SearchStrategy) -> None:
        """
        Change search strategy at runtime.
        Strategy pattern - allows dynamic algorithm swapping.
        
        Args:
            strategy: New search strategy.
        """
        self._search_strategy = strategy


# Factory function for common configurations
def create_vector_store(
    store_type: str = "memory",
    dimension: Optional[int] = None,
    metric: DistanceMetric = DistanceMetric.COSINE,
    persistence_path: Optional[str] = None,
) -> VectorStoreImpl:
    """
    Factory function to create vector store.
    
    Args:
        store_type: Type of store ("memory" or "faiss").
        dimension: Vector dimension (required for faiss).
        metric: Distance metric.
        persistence_path: Path for persistence.
        
    Returns:
        Configured VectorStore.
        
    Raises:
        ValueError: If invalid store_type or missing dimension for faiss.
    """
    if store_type == "memory":
        index = InMemoryIndex(metric=metric)
    elif store_type == "faiss":
        if dimension is None:
            raise ValueError("dimension is required for faiss store")
        index = FaissIndex(dimension=dimension, metric=metric)
    else:
        raise ValueError(f"Unknown store_type: {store_type}")
    
    persistence = JsonPersistence() if persistence_path else None
    
    return VectorStoreImpl(
        index=index,
        persistence=persistence,
        dimension=dimension,
        metric=metric,
    )
