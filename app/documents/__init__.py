"""Replaceable PDF processing boundary and the V1 PyMuPDF adapter."""

from app.documents.pymupdf import ArtifactReader, PyMuPDFDocumentProcessor
from app.ports import DocumentProcessor

__all__ = ["ArtifactReader", "DocumentProcessor", "PyMuPDFDocumentProcessor"]
