"""
FAISS vector index.
High-performance similarity search using Facebook AI Similarity Search.
"""
import os
import tempfile
import numpy as np
from typing import List, Dict, Optional
from dataclasses import field, dataclass

from ..core import VectorIndex, VectorDocument, SearchResult, SearchQuery, DistanceMetric


@dataclass
class FaissIndex(VectorIndex):
    """
    FAISS-based vector index.
    Suitable for large datasets with fast search.
    
    Supports:
    - IndexFlatIP: Inner product (dot product)
    - IndexFlatL2: L2 distance (euclidean)
    - IndexIDMap: Map IDs to vectors
    """
    
    _index: Optional[object] = field(default=None, repr=False)
    _id_map: Dict[int, str] = field(default_factory=dict)
    _documents: Dict[str, VectorDocument] = field(default_factory=dict)
    _dimension: int = 0
    _metric: DistanceMetric = DistanceMetric.COSINE
    
    def __post_init__(self):
        """Initialize FAISS index after dataclass init."""
        if self._dimension > 0 and self._index is None:
            self._create_index()
    
    def _create_index(self) -> None:
        """Create FAISS index."""
        import faiss
        
        # FAISS uses inner product for similarity
        # For cosine similarity, vectors should be normalized
        if self._metric in (DistanceMetric.COSINE, DistanceMetric.DOT_PRODUCT):
            self._index = faiss.IndexFlatIP(self._dimension)
        else:
            self._index = faiss.IndexFlatL2(self._dimension)
    
    def add(self, documents: List[VectorDocument]) -> None:
        """Add documents to index."""
        if not documents:
            return
        
        import faiss
        
        # Set dimension from first document if not set
        if self._dimension == 0:
            self._dimension = documents[0].dimension
            self._create_index()
        
        # Prepare vectors
        vectors = []
        for doc in documents:
            if doc.dimension != self._dimension:
                raise ValueError(
                    f"Vector dimension mismatch: expected {self._dimension}, "
                    f"got {doc.dimension}"
                )
            
            # Normalize for cosine similarity
            if self._metric == DistanceMetric.COSINE:
                vec = np.array(doc.vector, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                vectors.append(vec)
            else:
                vectors.append(np.array(doc.vector, dtype=np.float32))
            
            # Store document and map ID
            self._documents[doc.id] = doc
            faiss_id = len(self._id_map)
            self._id_map[faiss_id] = doc.id
        
        # Add to FAISS index
        vector_matrix = np.vstack(vectors).astype(np.float32)
        self._index.add(vector_matrix)
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Search for similar vectors."""
        if self._index is None or self._index.ntotal == 0:
            return []
        
        import faiss
        
        # Prepare query vector
        query_vector = np.array(query.vector, dtype=np.float32).reshape(1, -1)
        
        # Normalize for cosine similarity
        if self._metric == DistanceMetric.COSINE:
            norm = np.linalg.norm(query_vector)
            if norm > 0:
                query_vector = query_vector / norm
        
        # Search
        k = min(query.top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vector, k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            
            doc_id = self._id_map.get(idx)
            if doc_id is None:
                continue
            
            doc = self._documents.get(doc_id)
            if doc is None:
                continue
            
            # Apply filter if provided
            if query.filter_fn and not query.filter_fn(doc):
                continue
            
            # Create result document (optionally without vector)
            result_doc = VectorDocument(
                id=doc.id,
                vector=[] if not query.include_vectors else doc.vector,
                content=doc.content,
                metadata=doc.metadata,
            )
            
            results.append(SearchResult(
                document=result_doc,
                score=float(score),
                rank=i + 1,
            ))
        
        return results
    
    def remove(self, doc_ids: List[str]) -> None:
        """Remove documents by ID."""
        # Note: FAISS doesn't support direct removal
        # We mark as removed in our map
        ids_to_remove = []
        for faiss_id, doc_id in self._id_map.items():
            if doc_id in doc_ids:
                ids_to_remove.append(faiss_id)
                self._documents.pop(doc_id, None)
        
        for faiss_id in ids_to_remove:
            self._id_map.pop(faiss_id, None)
    
    def clear(self) -> None:
        """Clear all documents from index."""
        if self._index is not None:
            self._index.reset()
        self._id_map.clear()
        self._documents.clear()
    
    @property
    def size(self) -> int:
        """Number of documents in index."""
        return len(self._documents)
    
    @property
    def dimension(self) -> Optional[int]:
        """Vector dimension."""
        return self._dimension if self._dimension > 0 else None
    
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get document by ID."""
        return self._documents.get(doc_id)
    
    def get_all_documents(self) -> List[VectorDocument]:
        """Get all documents."""
        return list(self._documents.values())
    
    def save(self, path: str) -> None:
        """Save index to disk."""
        import faiss
        
        # Create directory if needed
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self._index, os.path.join(path, "index.faiss"))
        
        # Save metadata
        import json
        metadata = {
            "id_map": {str(k): v for k, v in self._id_map.items()},
            "dimension": self._dimension,
            "metric": self._metric.value,
        }
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
        
        # Save documents
        docs_data = []
        for doc in self._documents.values():
            docs_data.append({
                "id": doc.id,
                "vector": doc.vector,
                "content": doc.content,
                "metadata": doc.metadata,
            })
        with open(os.path.join(path, "documents.json"), "w") as f:
            json.dump(docs_data, f)
    
    @classmethod
    def load(cls, path: str) -> "FaissIndex":
        """Load index from disk."""
        import faiss
        import json
        
        # Load FAISS index
        index = faiss.read_index(os.path.join(path, "index.faiss"))
        
        # Load metadata
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        # Load documents
        with open(os.path.join(path, "documents.json"), "r") as f:
            docs_data = json.load(f)
        
        documents = [
            VectorDocument(
                id=d["id"],
                vector=d["vector"],
                content=d["content"],
                metadata=d["metadata"],
            )
            for d in docs_data
        ]
        
        # Create instance
        instance = cls(
            dimension=metadata["dimension"],
            metric=DistanceMetric(metadata["metric"]),
        )
        instance._index = index
        instance._id_map = {int(k): v for k, v in metadata["id_map"].items()}
        instance._documents = {doc.id: doc for doc in documents}
        
        return instance
