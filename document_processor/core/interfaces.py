"""
Core interfaces and domain models for document processing.
Implements GRASP Information Expert pattern.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Protocol
from enum import Enum


class DocumentType(Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    content: str
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.content.strip():
            raise ValueError("Chunk content cannot be empty")


@dataclass
class Vector:
    """Represents an embedding vector."""
    values: List[float]
    chunk: Chunk
    
    @property
    def dimension(self) -> int:
        return len(self.values)


@dataclass
class Document:
    """Domain model for a processed document."""
    path: str
    type: DocumentType
    content: str
    metadata: dict = field(default_factory=dict)
    chunks: List[Chunk] = field(default_factory=list)
    vectors: List[Vector] = field(default_factory=list)
    
    def add_chunk(self, chunk: Chunk) -> None:
        """Add a chunk to the document."""
        self.chunks.append(chunk)
    
    def add_vector(self, vector: Vector) -> None:
        """Add a vector to the document."""
        self.vectors.append(vector)


class DocumentLoader(ABC):
    """
    Abstract factory for document loading.
    GoF Factory Method pattern.
    """
    
    @abstractmethod
    def load(self, path: str) -> Document:
        """Load document from path."""
        pass
    
    @abstractmethod
    def supports(self, document_type: DocumentType) -> bool:
        """Check if loader supports document type."""
        pass


class ChunkingStrategy(ABC):
    """
    Strategy pattern for text chunking.
    Allows switching chunking algorithms at runtime.
    """
    
    @abstractmethod
    def chunk(self, text: str, metadata: dict) -> List[Chunk]:
        """Split text into chunks."""
        pass


class EmbeddingModel(ABC):
    """
    Strategy pattern for embedding generation.
    Encapsulates different embedding providers.
    """
    
    @abstractmethod
    def embed(self, chunks: List[Chunk]) -> List[Vector]:
        """Generate embeddings for chunks."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        pass


class DocumentProcessor(ABC):
    """
    Controller facade for document processing pipeline.
    GRASP Controller pattern.
    """
    
    @abstractmethod
    def process(self, path: str) -> Document:
        """Process document: load -> chunk -> embed."""
        pass
    
    @abstractmethod
    def process_batch(self, paths: List[str]) -> List[Document]:
        """Process multiple documents."""
        pass
