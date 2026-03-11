"""Tests for VectorStore facade."""
import pytest
import os
import tempfile

from vector_store import (
    VectorStoreImpl,
    create_vector_store,
    VectorDocument,
    DistanceMetric,
    SimilaritySearch,
    MMRSearch,
)


class TestVectorStoreImpl:
    """Tests for VectorStoreImpl."""
    
    @pytest.fixture
    def store(self):
        """Create empty store."""
        return VectorStoreImpl()
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents."""
        return [
            VectorDocument(id="1", vector=[1.0, 0.0], content="First document"),
            VectorDocument(id="2", vector=[0.0, 1.0], content="Second document"),
            VectorDocument(id="3", vector=[1.0, 1.0], content="Third document"),
        ]
    
    def test_add_documents(self, store, sample_documents):
        """Test adding documents."""
        ids = store.add_documents(sample_documents)
        
        assert len(ids) == 3
        assert store.size == 3
    
    def test_add_documents_auto_id(self, store):
        """Test that IDs are generated for documents without them."""
        docs = [
            VectorDocument(id="", vector=[1.0, 0.0], content="No ID"),
        ]
        ids = store.add_documents(docs)
        
        assert len(ids) == 1
        assert ids[0]  # Should have generated ID
    
    def test_add_texts(self, store):
        """Test adding texts with vectors."""
        ids = store.add_texts(
            texts=["Text 1", "Text 2"],
            vectors=[[1.0, 0.0], [0.0, 1.0]],
            metadatas=[{"key": "value1"}, {"key": "value2"}],
        )
        
        assert len(ids) == 2
        assert store.size == 2
    
    def test_search(self, store, sample_documents):
        """Test searching."""
        store.add_documents(sample_documents)
        
        results = store.search(query_vector=[1.0, 0.0], top_k=2)
        
        assert len(results) == 2
        assert results[0].document.id == "1"
    
    def test_search_with_filter(self, store, sample_documents):
        """Test search with filter."""
        store.add_documents(sample_documents)
        
        def filter_fn(doc):
            return doc.id != "1"
        
        results = store.search(query_vector=[1.0, 0.0], top_k=5, filter_fn=filter_fn)
        
        assert all(r.document.id != "1" for r in results)
    
    def test_delete(self, store, sample_documents):
        """Test deleting documents."""
        store.add_documents(sample_documents)
        store.delete(["1", "2"])
        
        assert store.size == 1
        assert store.get("1") is None
    
    def test_get_document(self, store, sample_documents):
        """Test getting document by ID."""
        store.add_documents(sample_documents)
        
        doc = store.get("2")
        
        assert doc is not None
        assert doc.content == "Second document"
    
    def test_get_all(self, store, sample_documents):
        """Test getting all documents."""
        store.add_documents(sample_documents)
        
        docs = store.get_all()
        
        assert len(docs) == 3
    
    def test_clear(self, store, sample_documents):
        """Test clearing store."""
        store.add_documents(sample_documents)
        store.clear()
        
        assert store.size == 0
    
    def test_save_and_load(self, store, sample_documents):
        """Test saving and loading."""
        store.add_documents(sample_documents)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            store.save(temp_path)
            
            # Load into new store
            loaded_store = VectorStoreImpl.load(temp_path)
            
            assert loaded_store.size == 3
            assert loaded_store.get("1") is not None
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_set_search_strategy(self, store):
        """Test changing search strategy."""
        mmr = MMRSearch(lambda_param=0.7)
        store.set_search_strategy(mmr)
        
        assert store._search_strategy is mmr
    
    def test_dimension_property(self, store, sample_documents):
        """Test dimension property."""
        store.add_documents(sample_documents)
        
        assert store.dimension == 2


class TestCreateVectorStore:
    """Tests for create_vector_store factory."""
    
    def test_create_memory_store(self):
        """Test creating in-memory store."""
        store = create_vector_store("memory")
        
        assert store.size == 0
        assert isinstance(store.index, type(store.index))  # InMemoryIndex
    
    def test_create_faiss_store(self):
        """Test creating FAISS store."""
        pytest.importorskip("faiss")
        
        store = create_vector_store("faiss", dimension=128)
        
        assert store.dimension == 128
    
    def test_create_faiss_without_dimension(self):
        """Test that FAISS requires dimension."""
        with pytest.raises(ValueError):
            create_vector_store("faiss")
    
    def test_create_with_metric(self):
        """Test creating store with custom metric."""
        store = create_vector_store("memory", metric=DistanceMetric.DOT_PRODUCT)
        
        assert store._metric == DistanceMetric.DOT_PRODUCT
    
    def test_create_with_persistence(self):
        """Test creating store with persistence path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "store.json")
            store = create_vector_store("memory", persistence_path=path)
            
            assert store._persistence is not None


class TestVectorStoreIntegration:
    """Integration tests for vector store."""
    
    def test_full_workflow(self):
        """Test complete workflow: add, search, save, load."""
        # Create store
        store = create_vector_store("memory")
        
        # Add documents
        docs = [
            VectorDocument(id="1", vector=[0.1] * 10, content="Doc 1"),
            VectorDocument(id="2", vector=[0.9] * 10, content="Doc 2"),
            VectorDocument(id="3", vector=[0.5] * 10, content="Doc 3"),
        ]
        store.add_documents(docs)
        
        # Search
        results = store.search([0.15] * 10, top_k=2)
        assert len(results) == 2
        assert results[0].document.id == "1"
        
        # Save and load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            store.save(temp_path)
            loaded = VectorStoreImpl.load(temp_path)
            
            assert loaded.size == 3
            loaded_results = loaded.search([0.15] * 10, top_k=2)
            assert len(loaded_results) == 2
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_mmr_for_diverse_results(self):
        """Test MMR for getting diverse results."""
        store = VectorStoreImpl(search_strategy=MMRSearch(lambda_param=0.5))
        
        # Add clusters of similar documents
        docs = [
            # Cluster 1
            VectorDocument(id="1a", vector=[1.0, 0.0, 0.0], content="1a"),
            VectorDocument(id="1b", vector=[0.98, 0.02, 0.0], content="1b"),
            VectorDocument(id="1c", vector=[0.97, 0.03, 0.0], content="1c"),
            # Cluster 2
            VectorDocument(id="2a", vector=[0.0, 1.0, 0.0], content="2a"),
            VectorDocument(id="2b", vector=[0.02, 0.98, 0.0], content="2b"),
            # Cluster 3
            VectorDocument(id="3a", vector=[0.0, 0.0, 1.0], content="3a"),
        ]
        store.add_documents(docs)
        
        # Search with MMR
        results = store.search([1.0, 0.0, 0.0], top_k=3)
        
        # Should get diverse results from different clusters
        result_ids = [r.document.id for r in results]
        
        # With MMR, should not get all from cluster 1
        cluster1_count = sum(1 for id in result_ids if id.startswith("1"))
        assert cluster1_count < 3  # Should have some diversity
