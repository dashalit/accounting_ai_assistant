"""
Fixed-size chunking strategy.
Splits text into chunks of approximately equal size.
"""
from typing import List
import re

from ..core import Chunk, ChunkingStrategy


class FixedSizeChunker(ChunkingStrategy):
    """
    Chunk by fixed number of characters with overlap.
    Simple strategy for uniform chunk sizes.
    """
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Maximum characters per chunk.
            overlap: Number of overlapping characters between chunks.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        
        self._chunk_size = chunk_size
        self._overlap = overlap
    
    def chunk(self, text: str, metadata: dict = None) -> List[Chunk]:
        """
        Split text into fixed-size chunks.
        
        Args:
            text: Text to split.
            metadata: Optional metadata to include in each chunk.
            
        Returns:
            List of Chunk objects.
        """
        if not text.strip():
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # Try to break at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self._chunk_size:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk.strip():
                    chunks.append(Chunk(
                        content=current_chunk.strip(),
                        metadata={
                            **metadata,
                            "chunk_index": chunk_index,
                            "chunk_size": len(current_chunk),
                        }
                    ))
                    chunk_index += 1
                
                # Handle overlap
                if self._overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self._overlap:]
                    # Find word boundary
                    match = re.search(r'\b\w+\s*$', overlap_text)
                    if match:
                        current_chunk = overlap_text[match.start():]
                    else:
                        current_chunk = overlap_text
                else:
                    current_chunk = ""
                
                current_chunk += (" " if current_chunk else "") + sentence
        
        # Add remaining text
        if current_chunk.strip():
            chunks.append(Chunk(
                content=current_chunk.strip(),
                metadata={
                    **metadata,
                    "chunk_index": chunk_index,
                    "chunk_size": len(current_chunk),
                }
            ))
        
        return chunks
    
    @property
    def chunk_size(self) -> int:
        return self._chunk_size
    
    @property
    def overlap(self) -> int:
        return self._overlap
