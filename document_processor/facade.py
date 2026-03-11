"""
Document Processing Facade - GoF Facade pattern.
Provides simplified interface to the document processing subsystem.
"""
from typing import List, Optional, Union
from pathlib import Path

from .core import Document, Chunk, Vector
from .core.processor import DefaultDocumentProcessor
from .chunkers import RecursiveChunker, FixedSizeChunker, ChunkingStrategy
from .embeddings import SentenceTransformerEmbedder, MockEmbedder, EmbeddingModel


class DocumentVectorizer:
    """
    Facade for document processing.
    GoF Facade pattern - simplifies complex subsystem interactions.
    
    Example usage:
        vectorizer = DocumentVectorizer()
        document = vectorizer.vectorize("file.pdf")
        vectors = document.vectors
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_model: str = "mock",
        device: Optional[str] = None,
        chunker: Optional[ChunkingStrategy] = None,
        embedder: Optional[EmbeddingModel] = None,
    ):
        """
        Initialize vectorizer with default or custom components.
        
        Args:
            chunk_size: Size of text chunks in characters.
            chunk_overlap: Overlap between consecutive chunks.
            embedding_model: Model name or 'mock' for testing.
            device: Device for embedding model (cuda/cpu/mps).
            chunker: Custom chunking strategy (overrides chunk_size/overlap).
            embedder: Custom embedding model (overrides embedding_model).
        """
        # Initialize chunker
        if chunker:
            self._chunker = chunker
        else:
            self._chunker = RecursiveChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        
        # Initialize embedder
        if embedder:
            self._embedder = embedder
        elif embedding_model == "mock":
            self._embedder = MockEmbedder(dimension=384)
        else:
            self._embedder = SentenceTransformerEmbedder(
                model_name=embedding_model,
                device=device,
            )
        
        # Create processor
        self._processor = DefaultDocumentProcessor(
            chunker=self._chunker,
            embedder=self._embedder,
        )
    
    def vectorize(self, path: Union[str, Path]) -> Document:
        """
        Convert document to vectors.
        
        Args:
            path: Path to PDF or DOCX file.
            
        Returns:
            Document with chunks and vectors.
        """
        return self._processor.process(str(path))
    
    def vectorize_batch(self, paths: List[Union[str, Path]]) -> List[Document]:
        """
        Convert multiple documents to vectors.
        
        Args:
            paths: List of paths to PDF or DOCX files.
            
        Returns:
            List of Documents with chunks and vectors.
        """
        return self._processor.process_batch([str(p) for p in paths])
    
    def extract_text(self, path: Union[str, Path]) -> str:
        """
        Extract text from document without chunking/embedding.
        
        Args:
            path: Path to PDF or DOCX file.
            
        Returns:
            Extracted text content.
        """
        from .loaders import LoaderFactory
        
        loader = LoaderFactory.from_path(str(path))
        document = loader.load(str(path))
        return document.content
    
    @property
    def chunker(self) -> ChunkingStrategy:
        """Get current chunking strategy."""
        return self._chunker
    
    @property
    def embedder(self) -> EmbeddingModel:
        """Get current embedding model."""
        return self._embedder
    
    def set_chunker(self, chunker: ChunkingStrategy) -> None:
        """
        Change chunking strategy at runtime.
        Strategy pattern - allows dynamic algorithm swapping.
        
        Args:
            chunker: New chunking strategy.
        """
        self._chunker = chunker
        self._processor = DefaultDocumentProcessor(
            chunker=chunker,
            embedder=self._embedder,
        )
    
    def set_embedder(self, embedder: EmbeddingModel) -> None:
        """
        Change embedding model at runtime.
        Strategy pattern - allows dynamic algorithm swapping.
        
        Args:
            embedder: New embedding model.
        """
        self._embedder = embedder
        self._processor = DefaultDocumentProcessor(
            chunker=self._chunker,
            embedder=embedder,
        )
