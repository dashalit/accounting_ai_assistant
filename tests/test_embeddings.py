"""Tests for embedding models."""
import pytest
import math

from document_processor.embeddings import MockEmbedder, SentenceTransformerEmbedder
from document_processor.core import Chunk


class TestMockEmbedder:
    """Tests for MockEmbedder."""
    
    def test_empty_chunks(self):
        """Test embedding empty list."""
        embedder = MockEmbedder(dimension=384)
        vectors = embedder.embed([])
        assert vectors == []
    
    def test_single_chunk(self):
        """Test embedding single chunk."""
        embedder = MockEmbedder(dimension=384)
        chunk = Chunk(content="Test text")
        
        vectors = embedder.embed([chunk])
        
        assert len(vectors) == 1
        assert len(vectors[0].values) == 384
        assert vectors[0].chunk is chunk
    
    def test_multiple_chunks(self):
        """Test embedding multiple chunks."""
        embedder = MockEmbedder(dimension=128)
        chunks = [
            Chunk(content="First chunk"),
            Chunk(content="Second chunk"),
            Chunk(content="Third chunk"),
        ]
        
        vectors = embedder.embed(chunks)
        
        assert len(vectors) == 3
        assert all(len(v.values) == 128 for v in vectors)
    
    def test_deterministic_output(self):
        """Test same input produces same output."""
        embedder = MockEmbedder(dimension=64)
        chunk = Chunk(content="Deterministic test")
        
        vectors1 = embedder.embed([chunk])
        vectors2 = embedder.embed([chunk])
        
        assert vectors1[0].values == vectors2[0].values
    
    def test_different_texts_different_vectors(self):
        """Test different texts produce different vectors."""
        embedder = MockEmbedder(dimension=64)
        chunks = [
            Chunk(content="First text"),
            Chunk(content="Second text"),
        ]
        
        vectors = embedder.embed(chunks)
        
        assert vectors[0].values != vectors[1].values
    
    def test_vector_normalization(self):
        """Test vectors are approximately normalized."""
        embedder = MockEmbedder(dimension=128)
        chunk = Chunk(content="Test normalization")
        
        vectors = embedder.embed([chunk])
        norm = math.sqrt(sum(v * v for v in vectors[0].values))
        
        # Should be close to 1.0
        assert abs(norm - 1.0) < 0.01
    
    def test_dimension_property(self):
        """Test dimension property."""
        for dim in [64, 128, 384, 768]:
            embedder = MockEmbedder(dimension=dim)
            assert embedder.dimension == dim


class TestSentenceTransformerEmbedder:
    """Tests for SentenceTransformerEmbedder."""
    
    @pytest.mark.skip(reason="Requires model download")
    def test_embed_real_model(self):
        """Test with real sentence transformer model."""
        embedder = SentenceTransformerEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        chunk = Chunk(content="Test sentence")
        
        vectors = embedder.embed([chunk])
        
        assert len(vectors) == 1
        assert vectors[0].dimension == 384
    
    @pytest.mark.skip(reason="Requires model download")
    def test_dimension_property(self):
        """Test dimension property with real model."""
        embedder = SentenceTransformerEmbedder()
        assert embedder.dimension == 384
