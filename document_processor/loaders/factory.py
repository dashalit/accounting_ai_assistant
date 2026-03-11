"""
Loader Factory - GoF Factory Method pattern.
Provides unified access to document loaders.
"""
from typing import Dict, Type, Optional
from pathlib import Path

from ..core import DocumentLoader, DocumentType
from .pdf_loader import PDFLoader
from .docx_loader import DOCXLoader


class LoaderFactory:
    """
    Factory for creating document loaders.
    Implements GoF Factory Method pattern.
    """
    
    _loaders: Dict[DocumentType, Type[DocumentLoader]] = {
        DocumentType.PDF: PDFLoader,
        DocumentType.DOCX: DOCXLoader,
    }
    
    @classmethod
    def get_loader(cls, document_type: DocumentType) -> DocumentLoader:
        """
        Get loader instance for document type.
        
        Args:
            document_type: Type of document to load.
            
        Returns:
            DocumentLoader instance.
            
        Raises:
            ValueError: If no loader registered for type.
        """
        loader_class = cls._loaders.get(document_type)
        if not loader_class:
            raise ValueError(f"No loader registered for {document_type}")
        return loader_class()
    
    @classmethod
    def from_path(cls, path: str) -> DocumentLoader:
        """
        Get appropriate loader based on file extension.
        
        Args:
            path: Path to document.
            
        Returns:
            DocumentLoader instance.
            
        Raises:
            ValueError: If file extension not supported.
        """
        ext = Path(path).suffix.lower()
        
        extension_map = {
            ".pdf": DocumentType.PDF,
            ".docx": DocumentType.DOCX,
        }
        
        if ext not in extension_map:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        return cls.get_loader(extension_map[ext])
    
    @classmethod
    def register_loader(cls, document_type: DocumentType, loader_class: Type[DocumentLoader]) -> None:
        """
        Register custom loader.
        
        Args:
            document_type: Type to register loader for.
            loader_class: Loader class to register.
        """
        cls._loaders[document_type] = loader_class
    
    @classmethod
    def supported_types(cls) -> list[DocumentType]:
        """Return list of supported document types."""
        return list(cls._loaders.keys())
