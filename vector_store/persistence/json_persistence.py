"""
JSON persistence strategy.
Saves vector store to JSON files.
"""
import json
import os
from typing import List, Dict, Any

from ..core import PersistenceStrategy, VectorIndex, VectorDocument
from ..indexes.in_memory_index import InMemoryIndex


class JsonPersistence(PersistenceStrategy):
    """
    JSON-based persistence strategy.
    Stores vectors and metadata in JSON format.
    """
    
    def save(self, index: VectorIndex, path: str) -> None:
        """Save index to JSON file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Collect documents
        if isinstance(index, InMemoryIndex):
            documents = index.get_all_documents()
        else:
            # For other index types, try to get documents
            documents = getattr(index, 'get_all_documents', lambda: [])()
        
        # Prepare data
        data = {
            "documents": [
                {
                    "id": doc.id,
                    "vector": doc.vector,
                    "content": doc.content,
                    "metadata": doc.metadata,
                }
                for doc in documents
            ],
            "metadata": {
                "size": index.size,
                "dimension": index.dimension,
            }
        }
        
        # Write to file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, path: str) -> VectorIndex:
        """Load index from JSON file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Create index
        index = InMemoryIndex()
        
        # Load documents
        documents = []
        for doc_data in data.get("documents", []):
            doc = VectorDocument(
                id=doc_data["id"],
                vector=doc_data["vector"],
                content=doc_data["content"],
                metadata=doc_data.get("metadata", {}),
            )
            documents.append(doc)
        
        index.add(documents)
        
        return index
    
    def exists(self, path: str) -> bool:
        """Check if index file exists."""
        return os.path.isfile(path)


class PicklePersistence(PersistenceStrategy):
    """
    Pickle-based persistence strategy.
    Faster than JSON but not human-readable.
    """
    
    def save(self, index: VectorIndex, path: str) -> None:
        """Save index using pickle."""
        import pickle
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(index, f)
    
    def load(self, path: str) -> VectorIndex:
        """Load index using pickle."""
        import pickle
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file not found: {path}")
        
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def exists(self, path: str) -> bool:
        """Check if index file exists."""
        return os.path.isfile(path)
