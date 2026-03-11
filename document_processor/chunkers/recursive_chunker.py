"""
Recursive chunking strategy.
Splits text hierarchically: paragraphs -> sentences -> words.
Best for preserving semantic boundaries.
"""
from typing import List, Optional
import re

from ..core import Chunk, ChunkingStrategy


class RecursiveChunker(ChunkingStrategy):
    """
    Recursive character text splitter.
    Tries to split at semantic boundaries in order:
    1. Paragraphs (\n\n)
    2. Sentences (. ! ?)
    3. Words (spaces)
    4. Characters (fallback)
    """
    
    DEFAULT_SEPARATORS = [
        "\n\n",      # Paragraph breaks
        "\n",        # Line breaks
        ". ",        # Sentence ends
        "! ",
        "? ",
        "; ",
        ", ",
        " ",         # Word breaks
        "",          # Character level (fallback)
    ]
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ):
        """
        Initialize recursive chunker.
        
        Args:
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between consecutive chunks.
            separators: List of separators in priority order.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = separators or self.DEFAULT_SEPARATORS
    
    def chunk(self, text: str, metadata: dict = None) -> List[Chunk]:
        """
        Split text using recursive strategy.
        
        Args:
            text: Text to split.
            metadata: Optional metadata for chunks.
            
        Returns:
            List of Chunk objects.
        """
        if not text.strip():
            return []
        
        metadata = metadata or {}
        chunks = self._split_text(text, separator_index=0)
        
        # Add metadata to chunks
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                **metadata,
                "chunk_index": i,
                "chunk_size": len(chunk.content),
            })
        
        return chunks
    
    def _split_text(self, text: str, separator_index: int) -> List[Chunk]:
        """Recursively split text."""
        # Base case: text fits in chunk
        if len(text) <= self._chunk_size:
            return [Chunk(content=text)] if text.strip() else []
        
        # Base case: no more separators
        if separator_index >= len(self._separators):
            return [Chunk(content=text[:self._chunk_size])] if text.strip() else []
        
        separator = self._separators[separator_index]
        
        # Split by current separator
        if separator:
            parts = text.split(separator)
        else:
            # Character-level split
            parts = list(text)
        
        # Try to merge parts into chunks
        chunks = []
        current_chunk = ""
        
        for part in parts:
            # Add separator back
            part_with_sep = part + separator if separator else part
            
            if len(current_chunk) + len(part_with_sep) <= self._chunk_size:
                current_chunk += part_with_sep
            else:
                if current_chunk.strip():
                    chunks.extend(self._finalize_chunk(current_chunk, separator_index))
                
                # Start new chunk with overlap if needed
                if self._chunk_overlap > 0 and current_chunk:
                    current_chunk = self._get_overlap(current_chunk)
                else:
                    current_chunk = ""
                
                current_chunk += part_with_sep
        
        # Handle remaining text
        if current_chunk.strip():
            chunks.extend(self._finalize_chunk(current_chunk, separator_index))
        
        return chunks
    
    def _finalize_chunk(self, text: str, separator_index: int) -> List[Chunk]:
        """Finalize chunk or split further if needed."""
        if len(text) <= self._chunk_size:
            return [Chunk(content=text.strip())] if text.strip() else []
        
        # Recursively split with next separator
        return self._split_text(text, separator_index + 1)
    
    def _get_overlap(self, text: str) -> str:
        """Get overlap portion from end of text."""
        if len(text) <= self._chunk_overlap:
            return text
        
        overlap_text = text[-self._chunk_overlap:]
        
        # Try to break at word boundary
        match = re.search(r'\b\w+\s*$', overlap_text)
        if match:
            return overlap_text[match.start():]
        
        return overlap_text
