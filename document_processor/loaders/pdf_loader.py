"""
PDF Document Loader using PyMuPDF (fitz).
Implements Adapter pattern for third-party library.
"""
import fitz  # PyMuPDF
from pathlib import Path

from ..core import Document, DocumentLoader, DocumentType


class PDFLoader(DocumentLoader):
    """
    Adapter for PyMuPDF library.
    Loads PDF documents and extracts text content.
    """
    
    def __init__(self):
        self._supported_types = [DocumentType.PDF]
    
    def load(self, path: str) -> Document:
        """
        Load PDF and extract text.
        
        Args:
            path: Path to PDF file.
            
        Returns:
            Document with extracted text.
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file is not a valid PDF.
        """
        file_path = Path(path).absolute()
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {path} (absolute: {file_path})")
        
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected .pdf file, got: {file_path.suffix}")
        
        doc = fitz.open(file_path)
        
        try:
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            
            content = "\n".join(text_parts)
            
            return Document(
                path=str(file_path),
                type=DocumentType.PDF,
                content=content,
                metadata={
                    "pages": len(doc),
                    "filename": file_path.name,
                }
            )
        finally:
            doc.close()
    
    def supports(self, document_type: DocumentType) -> bool:
        """Check if loader supports PDF type."""
        return document_type in self._supported_types
