"""Tests for in-memory vector index."""
import pytest
import math

from vector_store.indexes import InMemoryIndex
from vector_store.core import VectorDocument, SearchQuery, DistanceMetric


class TestInMemoryIndex:
    """Tests for InMemoryIndex."""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            VectorDocument(id="1", vector=[1.0, 0.0], content="Doc 1"),
            VectorDocument(id="2", vector=[0.0, 1.0], content="Doc 2"),
            VectorDocument(id="3", vector=[1.0, 1.0], content="Doc 3"),
        ]
    
    @pytest.fixture
    def index(self, sample_documents):
        """Create index with sample documents."""
        idx = InMemoryIndex()
        idx.add(sample_documents)
        return idx
    
    def test_add_documents(self, index):
        """Test adding documents."""
        assert index.size == 3
    
    def test_search_cosine_similarity(self, index):
        """Test search with cosine similarity."""
        query = SearchQuery(vector=[1.0, 0.0], top_k=2)
        results = index.search(query)
        
        assert len(results) == 2
        # First result should be most similar
        assert results[0].document.id == "1"
        assert results[0].rank == 1
    
    def test_search_top_k(self, index):
        """Test top_k parameter."""
        query = SearchQuery(vector=[0.5, 0.5], top_k=1)
        results = index.search(query)
        
        assert len(results) == 1
    
    def test_search_empty_index(self):
        """Test search on empty index."""
        index = InMemoryIndex()
        query = SearchQuery(vector=[1.0, 0.0], top_k=5)
        results = index.search(query)
        
        assert results == []
    
    def test_remove_documents(self, index):
        """Test removing documents."""
        index.remove(["1"])
        
        assert index.size == 2
        assert index.get_document("1") is None
    
    def test_clear_index(self, index):
        """Test clearing index."""
        index.clear()
        
        assert index.size == 0
    
    def test_dimension_property(self, index):
        """Test dimension property."""
        assert index.dimension == 2
    
    def test_get_document(self, index):
        """Test getting document by ID."""
        doc = index.get_document("2")
        
        assert doc is not None
        assert doc.id == "2"
        assert doc.content == "Doc 2"
    
    def test_get_all_documents(self, index):
        """Test getting all documents."""
        docs = index.get_all_documents()
        
        assert len(docs) == 3
    
    def test_cosine_similarity_calculation(self):
        """Test cosine similarity calculation."""
        index = InMemoryIndex(metric=DistanceMetric.COSINE)
        
        # Identical vectors should have similarity 1
        v1 = [1.0, 0.0]
        v2 = [1.0, 0.0]
        sim = index._cosine_similarity(v1, v2)
        assert abs(sim - 1.0) < 0.001
        
        # Orthogonal vectors should have similarity 0
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        sim = index._cosine_similarity(v1, v2)
        assert abs(sim) < 0.001
    
    def test_dot_product_metric(self):
        """Test dot product metric."""
        index = InMemoryIndex(metric=DistanceMetric.DOT_PRODUCT)
        
        v1 = [1.0, 2.0]
        v2 = [3.0, 4.0]
        sim = index._dot_product(v1, v2)
        
        # 1*3 + 2*4 = 11
        assert sim == 11.0
    
    def test_euclidean_similarity_metric(self):
        """Test euclidean similarity metric."""
        index = InMemoryIndex(metric=DistanceMetric.EUCLIDEAN)
        
        # Identical vectors
        v1 = [1.0, 2.0]
        v2 = [1.0, 2.0]
        sim = index._euclidean_similarity(v1, v2)
        
        # Distance is 0, similarity should be 1
        assert abs(sim - 1.0) < 0.001
        
        # Different vectors
        v1 = [0.0, 0.0]
        v2 = [3.0, 4.0]
        sim = index._euclidean_similarity(v1, v2)
        
        # Distance is 5, similarity is 1/(1+5) = 0.166...
        assert abs(sim - 1/6) < 0.001
    
    def test_filter_function(self, index):
        """Test filter function in search."""
        def filter_fn(doc):
            return doc.id != "1"
        
        query = SearchQuery(vector=[1.0, 0.0], top_k=5, filter_fn=filter_fn)
        results = index.search(query)
        
        # Should not include doc "1"
        assert all(r.document.id != "1" for r in results)
    
    def test_include_vectors(self, index):
        """Test include_vectors option."""
        # With vectors
        query = SearchQuery(vector=[1.0, 0.0], top_k=1, include_vectors=True)
        results = index.search(query)
        assert results[0].document.vector == [1.0, 0.0]
        
        # Without vectors
        query = SearchQuery(vector=[1.0, 0.0], top_k=1, include_vectors=False)
        results = index.search(query)
        assert results[0].document.vector == []
