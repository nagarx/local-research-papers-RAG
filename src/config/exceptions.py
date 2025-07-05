"""
Exception Hierarchy for ArXiv RAG System

This module defines custom exceptions for better error handling
and debugging across the RAG system components.
"""

from typing import Optional, Any, Dict


class RAGSystemError(Exception):
    """Base exception for all RAG system errors"""
    
    def __init__(self, message: str, component: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.component = component
        self.details = details or {}
        
    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.component:
            base_msg = f"[{self.component}] {base_msg}"
        return base_msg


class ConfigurationError(RAGSystemError):
    """Raised when there's a configuration error"""
    pass


class ModelLoadError(RAGSystemError):
    """Raised when model loading fails"""
    pass


class DocumentProcessingError(RAGSystemError):
    """Raised when document processing fails"""
    pass


class EmbeddingError(RAGSystemError):
    """Raised when embedding generation fails"""
    pass


class VectorStoreError(RAGSystemError):
    """Raised when vector store operations fail"""
    pass


class OllamaError(RAGSystemError):
    """Raised when Ollama operations fail"""
    pass


class QueryProcessingError(RAGSystemError):
    """Raised when query processing fails"""
    pass


class SourceTrackingError(RAGSystemError):
    """Raised when source tracking fails"""
    pass


# Specific error types for common scenarios
class FileNotFoundError(DocumentProcessingError):
    """Raised when a required file is not found"""
    pass


class UnsupportedFileTypeError(DocumentProcessingError):
    """Raised when trying to process unsupported file type"""
    pass


class ModelNotFoundError(ModelLoadError):
    """Raised when a required model is not found"""
    pass


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Raised when embedding dimensions don't match"""
    pass


class VectorIndexError(VectorStoreError):
    """Raised when vector index operations fail"""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when Ollama connection fails"""
    pass


class OllamaModelError(OllamaError):
    """Raised when Ollama model operations fail"""
    pass


class InvalidQueryError(QueryProcessingError):
    """Raised when query is invalid or malformed"""
    pass


class NoResultsError(QueryProcessingError):
    """Raised when no results are found for a query"""
    pass


class SourceNotFoundError(SourceTrackingError):
    """Raised when source information is not found"""
    pass 