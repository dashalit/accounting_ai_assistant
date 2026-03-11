"""
Mock embedding model for testing.
Returns deterministic pseudo-vectors.
"""
import hashlib
import math
from typing import List

from ..core import Chunk, Vector, EmbeddingModel


class MockEmbedder(EmbeddingModel):
    """
    Mock embedder for testing.
    Generates deterministic vectors based on text hash.
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize mock embedder.
        
        Args:
            dimension: Dimension of output vectors.
        """
        self._dimension = dimension
    
    def embed(self, chunks: List[Chunk]) -> List[Vector]:
        """
        Generate deterministic pseudo-embeddings.
        
        Args:
            chunks: List of chunks to embed.
            
        Returns:
            List of Vector objects.
        """
        vectors = []
        
        for chunk in chunks:
            # Create deterministic vector from text hash
            vector = self._text_to_vector(chunk.content)
            vectors.append(Vector(
                values=vector,
                chunk=chunk,
            ))
        
        return vectors
    
    def _text_to_vector(self, text: str) -> List[float]:
        """Convert text to deterministic vector."""
        # Hash the text
        hash_bytes = hashlib.sha256(text.encode()).digest()
        
        # Generate vector components from hash
        vector = []
        for i in range(self._dimension):
            # Use hash + index for variation
            seed = int.from_bytes(
                hash_bytes[i % len(hash_bytes):i % len(hash_bytes) + 4],
                byteorder='big'
            )
            # Normalize to [-1, 1]
            value = 2 * (seed / 0xFFFFFFFF) - 1
            # Add position-based variation
            value *= math.cos(i * 0.1)
            vector.append(value)
        
        # Normalize vector
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        
        return vector
    
    @property
    def dimension(self) -> int:
        return self._dimension
