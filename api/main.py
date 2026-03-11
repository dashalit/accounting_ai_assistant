"""
Document Vector Search API.
REST API for document processing and vector search.
"""
import os
import uuid
import shutil
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import from local modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from document_processor import DocumentVectorizer
from vector_store import (
    VectorStoreImpl,
    create_vector_store,
    VectorDocument,
    MMRSearch,
    SimilaritySearch,
)
from document_processor.core import Chunk


# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
STORE_PATH = DATA_DIR / "store" / "vectors.json"

# For Docker compatibility
if os.getenv("DOCKER_ENV", "false").lower() == "true":
    UPLOAD_DIR = Path("/app/data/uploads")
    STORE_PATH = Path("/app/data/store/vectors.json")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mock")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
USE_FAISS = os.getenv("USE_FAISS", "false").lower() == "true"
VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "384"))

# Validate embedding model availability
if EMBEDDING_MODEL != "mock":
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        EMBEDDING_MODEL = "mock"


# Global instances
vectorizer: Optional[DocumentVectorizer] = None
vector_store: Optional[VectorStoreImpl] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources."""
    global vectorizer, vector_store
    
    # Startup
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    vectorizer = DocumentVectorizer(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        embedding_model=EMBEDDING_MODEL,
    )
    
    store_type = "faiss" if USE_FAISS else "memory"
    vector_store = create_vector_store(
        store_type,
        dimension=VECTOR_DIMENSION if USE_FAISS else None,
    )
    
    # Load existing store if exists
    if STORE_PATH.exists():
        try:
            vector_store = VectorStoreImpl.load(str(STORE_PATH))
        except Exception:
            pass
    
    yield
    
    # Shutdown
    if vector_store:
        vector_store.save(str(STORE_PATH))


app = FastAPI(
    title="Document Vector Search API",
    description="API for processing documents and searching by vector similarity",
    version="1.0.0",
    lifespan=lifespan,
)


# Pydantic models
class SearchResult(BaseModel):
    id: str
    content: str
    score: float
    rank: int
    metadata: dict = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    id: str
    filename: str
    chunks_count: int
    status: str


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=5, ge=1, le=100)
    use_mmr: bool = Field(default=False, description="Use MMR for diverse results")
    mmr_lambda: float = Field(default=0.5, ge=0, le=1)


class ProcessResponse(BaseModel):
    document_id: str
    filename: str
    chunks_count: int
    vectors_count: int


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "store_size": vector_store.size if vector_store else 0,
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "vectorizer": "initialized" if vectorizer else "not initialized",
        "vector_store": {
            "size": vector_store.size if vector_store else 0,
            "dimension": vector_store.dimension if vector_store else None,
        },
    }


@app.post("/documents", response_model=ProcessResponse)
async def process_document(file: UploadFile = File(...)):
    """
    Process a document: extract text, chunk, and generate vectors.
    
    Supported formats: PDF, DOCX
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in [".pdf", ".docx"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Use PDF or DOCX."
        )
    
    # Save uploaded file
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}{ext}"
    
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process document
        doc = vectorizer.vectorize(str(file_path))
        
        # Add vectors to store
        store_docs = []
        for vector in doc.vectors:
            store_docs.append(VectorDocument(
                id=f"{file_id}_{vector.chunk.metadata.get('chunk_index', 0)}",
                vector=vector.values,
                content=vector.chunk.content,
                metadata={
                    **vector.chunk.metadata,
                    "filename": file.filename,
                    "file_id": file_id,
                    "source_path": doc.path,
                }
            ))
        
        vector_store.add_documents(store_docs)
        
        # Save store
        vector_store.save(str(STORE_PATH))
        
        return ProcessResponse(
            document_id=file_id,
            filename=file.filename,
            chunks_count=len(doc.chunks),
            vectors_count=len(doc.vectors),
        )
    
    except Exception as e:
        # Cleanup on error
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all processed documents."""
    # Group by file_id
    files = {}
    for doc in vector_store.get_all():
        file_id = doc.metadata.get("file_id", "unknown")
        if file_id not in files:
            files[file_id] = {
                "filename": doc.metadata.get("filename", "unknown"),
                "chunks": 0,
            }
        files[file_id]["chunks"] += 1
    
    return [
        DocumentInfo(
            id=file_id,
            filename=data["filename"],
            chunks_count=data["chunks"],
            status="indexed",
        )
        for file_id, data in files.items()
    ]


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its vectors."""
    # Find all vectors for this document
    docs_to_delete = [
        doc.id for doc in vector_store.get_all()
        if doc.metadata.get("file_id") == document_id
    ]
    
    if docs_to_delete:
        vector_store.delete(docs_to_delete)
        vector_store.save(str(STORE_PATH))
    
    # Delete uploaded file
    for ext in [".pdf", ".docx"]:
        file_path = UPLOAD_DIR / f"{document_id}{ext}"
        if file_path.exists():
            file_path.unlink()
    
    return {"deleted": len(docs_to_delete)}


@app.post("/search", response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    """
    Search for documents by text query.
    
    - **query**: Search query text
    - **top_k**: Number of results to return
    - **use_mmr**: Use MMR for diverse results
    - **mmr_lambda**: MMR lambda parameter (0=diverse, 1=relevant)
    """
    # Generate embedding for query
    chunk = Chunk(content=request.query)
    vectors = vectorizer.embedder.embed([chunk])
    
    if not vectors:
        raise HTTPException(status_code=500, detail="Failed to generate query embedding")
    
    query_vector = vectors[0].values
    
    # Set search strategy
    if request.use_mmr:
        vector_store.set_search_strategy(MMRSearch(lambda_param=request.mmr_lambda))
    else:
        vector_store.set_search_strategy(SimilaritySearch())
    
    # Search
    results = vector_store.search(query_vector, top_k=request.top_k)
    
    return [
        SearchResult(
            id=result.document.id,
            content=result.document.content,
            score=result.score,
            rank=result.rank,
            metadata=result.document.metadata,
        )
        for result in results
    ]


@app.get("/search")
async def search_documents_get(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(default=5, ge=1, le=100),
    use_mmr: bool = Query(default=False),
):
    """Simplified search endpoint using GET."""
    request = SearchRequest(query=q, top_k=top_k, use_mmr=use_mmr)
    return await search_documents(request)


@app.post("/documents/batch")
async def process_documents_batch(files: List[UploadFile] = File(...)):
    """Process multiple documents at once."""
    results = []
    errors = []
    
    for file in files:
        try:
            result = await process_document(file)
            results.append(result)
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})
    
    return {
        "processed": results,
        "errors": errors,
    }


@app.get("/stats")
async def get_stats():
    """Get statistics about the vector store."""
    docs = vector_store.get_all()
    
    # Count by file
    files_count = len(set(doc.metadata.get("file_id") for doc in docs))
    
    # Average chunk size
    avg_content_length = sum(len(doc.content) for doc in docs) / len(docs) if docs else 0
    
    return {
        "total_vectors": len(docs),
        "total_documents": files_count,
        "dimension": vector_store.dimension,
        "average_chunk_length": round(avg_content_length, 2),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
