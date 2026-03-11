"""Document loaders package."""
from .pdf_loader import PDFLoader
from .docx_loader import DOCXLoader
from .factory import LoaderFactory

__all__ = [
    "PDFLoader",
    "DOCXLoader",
    "LoaderFactory",
]
