"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Set environment variables before importing app
os.environ["EMBEDDING_MODEL"] = "mock"
os.environ["CHUNK_SIZE"] = "500"
os.environ["CHUNK_OVERLAP"] = "50"

from api.main import app, UPLOAD_DIR, STORE_PATH, vectorizer, vector_store

# Create test client with lifespan
@pytest.fixture(scope="module")
def client():
    """Create test client with initialized app."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup before and after tests."""
    # Cleanup before
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.iterdir():
            f.unlink()
    if STORE_PATH.exists():
        STORE_PATH.unlink()
    
    yield
    
    # Cleanup after
    if UPLOAD_DIR.exists():
        for f in UPLOAD_DIR.iterdir():
            f.unlink()
    if STORE_PATH.exists():
        STORE_PATH.unlink()


@pytest.fixture
def sample_pdf(tmp_path):
    """Create sample PDF for testing."""
    import fitz
    
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Sample PDF content for API testing.")
    doc.save(pdf_path)
    doc.close()
    
    return pdf_path


@pytest.fixture
def sample_docx(tmp_path):
    """Create sample DOCX for testing."""
    from docx import Document
    
    docx_path = tmp_path / "test.docx"
    doc = Document()
    doc.add_paragraph("Sample DOCX content for API testing.")
    doc.save(docx_path)
    
    return docx_path


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_health_check(self, client):
        """Test detailed health check."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "vectorizer" in data
        assert "vector_store" in data


class TestDocumentEndpoints:
    """Tests for document processing endpoints."""
    
    def test_process_pdf(self, client, sample_pdf):
        """Test processing PDF document."""
        with open(sample_pdf, "rb") as f:
            response = client.post(
                "/documents",
                files={"file": ("test.pdf", f, "application/pdf")},
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert data["filename"] == "test.pdf"
        assert data["chunks_count"] > 0
        assert data["vectors_count"] > 0
    
    def test_process_docx(self, client, sample_docx):
        """Test processing DOCX document."""
        with open(sample_docx, "rb") as f:
            response = client.post(
                "/documents",
                files={"file": ("test.docx", f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "document_id" in data
        assert data["filename"] == "test.docx"
    
    def test_process_unsupported_type(self, client, tmp_path):
        """Test processing unsupported file type."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("Content")
        
        with open(txt_path, "rb") as f:
            response = client.post(
                "/documents",
                files={"file": ("test.txt", f, "text/plain")},
            )
        
        assert response.status_code == 400
    
    def test_list_documents(self, client, sample_pdf):
        """Test listing documents."""
        # First add a document
        with open(sample_pdf, "rb") as f:
            client.post(
                "/documents",
                files={"file": ("test.pdf", f, "application/pdf")},
            )
        
        # Then list
        response = client.get("/documents")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
    
    def test_delete_document(self, client, sample_pdf):
        """Test deleting document."""
        # First add a document
        with open(sample_pdf, "rb") as f:
            result = client.post(
                "/documents",
                files={"file": ("test.pdf", f, "application/pdf")},
            )
        
        doc_id = result.json()["document_id"]
        
        # Then delete
        response = client.delete(f"/documents/{doc_id}")
        
        assert response.status_code == 200


class TestSearchEndpoints:
    """Tests for search endpoints."""
    
    def test_search_post(self, client, sample_pdf):
        """Test search with POST."""
        # Add document first
        with open(sample_pdf, "rb") as f:
            client.post(
                "/documents",
                files={"file": ("test.pdf", f, "application/pdf")},
            )
        
        # Search
        response = client.post(
            "/search",
            json={
                "query": "sample content",
                "top_k": 5,
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if data:  # If there are results
            assert "content" in data[0]
            assert "score" in data[0]
    
    def test_search_get(self, client, sample_pdf):
        """Test search with GET."""
        # Add document first
        with open(sample_pdf, "rb") as f:
            client.post(
                "/documents",
                files={"file": ("test.pdf", f, "application/pdf")},
            )
        
        # Search
        response = client.get("/search?q=sample&top_k=3")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_search_with_mmr(self, client, sample_pdf):
        """Test search with MMR."""
        # Add document first
        with open(sample_pdf, "rb") as f:
            client.post(
                "/documents",
                files={"file": ("test.pdf", f, "application/pdf")},
            )
        
        # Search with MMR
        response = client.post(
            "/search",
            json={
                "query": "sample content",
                "top_k": 5,
                "use_mmr": True,
                "mmr_lambda": 0.7,
            },
        )
        
        assert response.status_code == 200


class TestStatsEndpoint:
    """Tests for stats endpoint."""
    
    def test_get_stats(self, client):
        """Test getting statistics."""
        response = client.get("/stats")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_vectors" in data
        assert "total_documents" in data


class TestBatchEndpoint:
    """Tests for batch processing endpoint."""
    
    def test_process_batch(self, client, sample_pdf, sample_docx):
        """Test batch processing."""
        with open(sample_pdf, "rb") as f1, open(sample_docx, "rb") as f2:
            response = client.post(
                "/documents/batch",
                files=[
                    ("files", ("test1.pdf", f1, "application/pdf")),
                    ("files", ("test2.docx", f2, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")),
                ],
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "processed" in data
        assert "errors" in data
