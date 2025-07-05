"""
Core Interfaces for ArXiv RAG System

This module defines the protocol interfaces that all components must implement
to ensure modularity and swappability.
"""

from .document_processor import DocumentProcessorProtocol, BaseDocumentProcessor

__all__ = [
    "DocumentProcessorProtocol",
    "BaseDocumentProcessor"
] 