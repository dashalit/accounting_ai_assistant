"""Tests for search strategies."""
import pytest

from vector_store.indexes import InMemoryIndex
from vector_store.strategies import SimilaritySearch, MMRSearch
from vector_store.core import VectorDocument, SearchQuery


class TestSimilaritySearch:
    """Tests for SimilaritySearch strategy."""
    
    @pytest.fixture
    def index(self):
        """Create index with sample documents."""
        idx = InMemoryIndex()
        idx.add([
            VectorDocument(id="1", vector=[1.0, 0.0], content="Doc 1"),
            VectorDocument(id="2", vector=[0.8, 0.2], content="Doc 2"),
            VectorDocument(id="3", vector=[0.0, 1.0], content="Doc 3"),
        ])
        return idx
    
    def test_basic_search(self, index):
        """Test basic similarity search."""
        strategy = SimilaritySearch()
        query = SearchQuery(vector=[1.0, 0.0], top_k=2)
        
        results = strategy.search(index, query)
        
        assert len(results) == 2
        assert results[0].document.id == "1"  # Most similar
    
    def test_normalize_scores(self, index):
        """Test score normalization."""
        strategy = SimilaritySearch(normalize_scores=True)
        query = SearchQuery(vector=[1.0, 0.0], top_k=3)
        
        results = strategy.search(index, query)
        
        # Scores should be normalized to [0, 1]
        assert all(0 <= r.score <= 1 for r in results)
        
        # Best result should have score 1.0
        assert abs(results[0].score - 1.0) < 0.001


class TestMMRSearch:
    """Tests for MMRSearch strategy."""
    
    @pytest.fixture
    def similar_documents(self):
        """Create index with similar documents."""
        idx = InMemoryIndex()
        # Add very similar documents
        idx.add([
            VectorDocument(id="1", vector=[1.0, 0.0, 0.0], content="Doc 1"),
            VectorDocument(id="2", vector=[0.98, 0.02, 0.0], content="Doc 2 - similar to 1"),
            VectorDocument(id="3", vector=[0.0, 1.0, 0.0], content="Doc 3 - different"),
            VectorDocument(id="4", vector=[0.0, 0.0, 1.0], content="Doc 4 - different"),
            VectorDocument(id="5", vector=[0.97, 0.03, 0.0], content="Doc 5 - similar to 1"),
        ])
        return idx
    
    def test_mmr_diversity(self, similar_documents):
        """Test that MMR provides diverse results."""
        strategy = MMRSearch(lambda_param=0.5)
        query = SearchQuery(vector=[1.0, 0.0, 0.0], top_k=3, include_vectors=True)
        
        results = strategy.search(similar_documents, query)
        
        # Should get diverse results, not just the 3 most similar
        result_ids = [r.document.id for r in results]
        
        # With MMR, we should get documents from different clusters
        assert "1" in result_ids  # Most similar
        # Should include at least one diverse document
        assert "3" in result_ids or "4" in result_ids
    
    def test_mmr_lambda_param(self, similar_documents):
        """Test lambda parameter effect."""
        # High lambda = more relevance focused
        high_lambda = MMRSearch(lambda_param=0.9)
        query = SearchQuery(vector=[1.0, 0.0, 0.0], top_k=2, include_vectors=True)
        results_high = high_lambda.search(similar_documents, query)
        
        # Low lambda = more diversity focused
        low_lambda = MMRSearch(lambda_param=0.1)
        results_low = low_lambda.search(similar_documents, query)
        
        # Results should be different
        ids_high = [r.document.id for r in results_high]
        ids_low = [r.document.id for r in results_low]
        
        # With high lambda, should get most similar docs
        assert ids_high[0] == "1"
    
    def test_invalid_lambda(self):
        """Test validation of lambda parameter."""
        with pytest.raises(ValueError):
            MMRSearch(lambda_param=1.5)
        
        with pytest.raises(ValueError):
            MMRSearch(lambda_param=-0.1)
    
    def test_mmr_empty_index(self):
        """Test MMR with empty index."""
        idx = InMemoryIndex()
        strategy = MMRSearch()
        query = SearchQuery(vector=[1.0, 0.0], top_k=5)
        
        results = strategy.search(idx, query)
        
        assert results == []
    
    def test_mmr_rank(self, similar_documents):
        """Test that results are properly ranked."""
        strategy = MMRSearch(lambda_param=0.5)
        query = SearchQuery(vector=[1.0, 0.0, 0.0], top_k=3)
        
        results = strategy.search(similar_documents, query)
        
        # Check ranks are sequential
        for i, result in enumerate(results):
            assert result.rank == i + 1
