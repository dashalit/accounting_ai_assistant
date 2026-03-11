"""
Document Processor - Convert PDF and DOCX documents to vectors.

Architecture based on GoF and GRASP patterns:
- Strategy: Chunking and embedding algorithms
- Factory Method: Document loader creation
- Facade: Simplified interface via DocumentVectorizer
- Adapter: Third-party library integration
- Controller: Document processing orchestration
- Information Expert: Domain logic in models

Example usage:
    from document_processor import DocumentVectorizer

    # Initialize with default settings
    vectorizer = DocumentVectorizer()

    # Process single document
    doc = vectorizer.vectorize("document.pdf")
    print(f"Generated {len(doc.vectors)} vectors")

    # Process batch
    docs = vectorizer.vectorize_batch(["file1.pdf", "file2.docx"])

    # Use real embedding model
    vectorizer = DocumentVectorizer(
        chunk_size=500,
        chunk_overlap=50,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )
"""

from .core import (
    Document,
    Chunk,
    Vector,
    DocumentType,
    ChunkingStrategy,
    EmbeddingModel,
    DocumentProcessor,
)
from .facade import DocumentVectorizer
from .chunkers import FixedSizeChunker, RecursiveChunker
from .embeddings import SentenceTransformerEmbedder, MockEmbedder
from .loaders import LoaderFactory, PDFLoader, DOCXLoader

__all__ = [
    # Main facade
    "DocumentVectorizer",
    
    # Domain models
    "Document",
    "Chunk",
    "Vector",
    "DocumentType",
    
    # Interfaces
    "ChunkingStrategy",
    "EmbeddingModel",
    "DocumentProcessor",
    
    # Chunkers
    "FixedSizeChunker",
    "RecursiveChunker",
    
    # Embedders
    "SentenceTransformerEmbedder",
    "MockEmbedder",
    
    # Loaders
    "LoaderFactory",
    "PDFLoader",
    "DOCXLoader",
]

__version__ = "0.1.0"
