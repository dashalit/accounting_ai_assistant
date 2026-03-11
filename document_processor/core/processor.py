"""
Document Processor - GRASP Controller pattern.
Orchestrates the document processing pipeline.
"""
from typing import List

from ..core.interfaces import (
    Document,
    DocumentProcessor,
    ChunkingStrategy,
    EmbeddingModel,
)
from ..loaders.factory import LoaderFactory


class DefaultDocumentProcessor(DocumentProcessor):
    """
    Main document processor implementing the pipeline.
    GRASP Controller pattern - controls document processing flow.
    
    Pipeline: Load -> Chunk -> Embed
    """
    
    def __init__(
        self,
        chunker: ChunkingStrategy,
        embedder: EmbeddingModel,
    ):
        """
        Initialize processor.
        
        Args:
            chunker: Strategy for text chunking.
            embedder: Model for embedding generation.
        """
        self._chunker = chunker
        self._embedder = embedder
    
    def process(self, path: str) -> Document:
        """
        Process single document.

        Args:
            path: Path to document file.

        Returns:
            Processed Document with chunks and vectors.
        """
        # Load document
        loader = LoaderFactory.from_path(path)
        document = loader.load(path)
        
        if document is None:
            raise ValueError(f"Failed to load document: {path}")

        # Chunk document
        chunks = self._chunker.chunk(
            document.content,
            metadata={"source": document.path, "type": document.type.value}
        )

        for chunk in chunks:
            document.add_chunk(chunk)

        # Generate embeddings
        vectors = self._embedder.embed(chunks)

        for vector in vectors:
            document.add_vector(vector)

        return document
    
    def process_batch(self, paths: List[str]) -> List[Document]:
        """
        Process multiple documents.
        
        Args:
            paths: List of document paths.
            
        Returns:
            List of processed Documents.
        """
        return [self.process(path) for path in paths]
    
    @property
    def chunker(self) -> ChunkingStrategy:
        return self._chunker
    
    @property
    def embedder(self) -> EmbeddingModel:
        return self._embedder
