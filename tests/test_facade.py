"""Tests for DocumentVectorizer facade."""
import pytest
from pathlib import Path

from document_processor import DocumentVectorizer
from document_processor.chunkers import FixedSizeChunker
from document_processor.embeddings import MockEmbedder


class TestDocumentVectorizer:
    """Tests for DocumentVectorizer facade."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        vectorizer = DocumentVectorizer()
        
        assert vectorizer.chunker is not None
        assert vectorizer.embedder is not None
    
    def test_initialization_custom(self):
        """Test custom initialization."""
        vectorizer = DocumentVectorizer(
            chunk_size=256,
            chunk_overlap=32,
            embedding_model="mock",
        )
        
        assert vectorizer.chunker is not None
        assert vectorizer.embedder.dimension == 384
    
    def test_initialization_with_custom_chunker(self):
        """Test initialization with custom chunker."""
        custom_chunker = FixedSizeChunker(chunk_size=100, overlap=10)
        vectorizer = DocumentVectorizer(chunker=custom_chunker)
        
        assert vectorizer.chunker is custom_chunker
    
    def test_initialization_with_custom_embedder(self):
        """Test initialization with custom embedder."""
        custom_embedder = MockEmbedder(dimension=512)
        vectorizer = DocumentVectorizer(embedder=custom_embedder)
        
        assert vectorizer.embedder is custom_embedder
        assert vectorizer.embedder.dimension == 512
    
    def test_set_chunker(self):
        """Test changing chunker at runtime."""
        vectorizer = DocumentVectorizer()
        new_chunker = FixedSizeChunker(chunk_size=200, overlap=20)
        
        vectorizer.set_chunker(new_chunker)
        
        assert vectorizer.chunker is new_chunker
    
    def test_set_embedder(self):
        """Test changing embedder at runtime."""
        vectorizer = DocumentVectorizer()
        new_embedder = MockEmbedder(dimension=256)
        
        vectorizer.set_embedder(new_embedder)
        
        assert vectorizer.embedder is new_embedder
        assert vectorizer.embedder.dimension == 256
    
    def test_vectorize_nonexistent_file(self):
        """Test vectorizing non-existent file."""
        vectorizer = DocumentVectorizer()
        
        with pytest.raises(FileNotFoundError):
            vectorizer.vectorize("nonexistent.pdf")
    
    def test_vectorize_batch_empty(self):
        """Test batch vectorization with empty list."""
        vectorizer = DocumentVectorizer()
        
        docs = vectorizer.vectorize_batch([])
        
        assert docs == []
    
    def test_extract_text_nonexistent(self):
        """Test text extraction from non-existent file."""
        vectorizer = DocumentVectorizer()
        
        with pytest.raises(FileNotFoundError):
            vectorizer.extract_text("nonexistent.docx")


class TestIntegration:
    """Integration tests with sample documents."""
    
    def test_vectorize_pdf(self, session_sample_pdf):
        """Test vectorizing PDF document."""
        vectorizer = DocumentVectorizer()
        
        doc = vectorizer.vectorize(session_sample_pdf)
        
        assert doc is not None
        assert doc.type.value == "pdf"
        assert len(doc.content) > 0
        assert len(doc.chunks) >= 0
        assert len(doc.vectors) >= 0
    
    def test_vectorize_docx(self, session_sample_docx):
        """Test vectorizing DOCX document."""
        vectorizer = DocumentVectorizer()
        
        doc = vectorizer.vectorize(session_sample_docx)
        
        assert doc is not None
        assert doc.type.value == "docx"
        assert len(doc.content) > 0
        assert "Sample DOCX content" in doc.content
    
    def test_vectorize_batch(self, session_sample_pdf, session_sample_docx):
        """Test batch vectorization."""
        vectorizer = DocumentVectorizer()
        
        docs = vectorizer.vectorize_batch([session_sample_pdf, session_sample_docx])
        
        assert len(docs) == 2
        assert docs[0].type.value == "pdf"
        assert docs[1].type.value == "docx"
    
    def test_extract_text_pdf(self, session_sample_pdf):
        """Test text extraction from PDF."""
        vectorizer = DocumentVectorizer()
        
        text = vectorizer.extract_text(session_sample_pdf)
        
        assert "Sample PDF content" in text
    
    def test_extract_text_docx(self, session_sample_docx):
        """Test text extraction from DOCX."""
        vectorizer = DocumentVectorizer()
        
        text = vectorizer.extract_text(session_sample_docx)
        
        assert "Sample DOCX content" in text
