"""
Sentence Transformers embedding model.
Local embedding generation using HuggingFace models.
"""
from typing import List, Optional
from functools import cached_property

from ..core import Chunk, Vector, EmbeddingModel


class SentenceTransformerEmbedder(EmbeddingModel):
    """
    Adapter for sentence-transformers library.
    Generates embeddings using local models.
    """
    
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        """
        Initialize embedder.
        
        Args:
            model_name: HuggingFace model name or path.
            device: Device to run model on (cuda/cpu/mps).
        """
        self._model_name = model_name
        self._device = device
        self._model = None
    
    @cached_property
    def _embedding_model(self):
        """Lazy load model."""
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self._model_name, device=self._device)
    
    def embed(self, chunks: List[Chunk]) -> List[Vector]:
        """
        Generate embeddings for chunks.
        
        Args:
            chunks: List of chunks to embed.
            
        Returns:
            List of Vector objects.
        """
        if not chunks:
            return []
        
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        embeddings = self._embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 10,
        )
        
        # Create Vector objects
        vectors = []
        for chunk, embedding in zip(chunks, embeddings):
            vectors.append(Vector(
                values=embedding.tolist(),
                chunk=chunk,
            ))
        
        return vectors
    
    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        # Load model to get dimension
        model = self._embedding_model
        return model.get_sentence_embedding_dimension()
    
    @property
    def model_name(self) -> str:
        return self._model_name
