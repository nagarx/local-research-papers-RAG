"""
Document Ingestion Module

This module handles document processing, parsing, and chunking using the Marker tool.
Now with modular architecture for better maintainability and testing.
"""

# Main interface (modular implementation)
from .document_processor import DocumentProcessor, get_global_marker_models

# Specialized components for advanced usage
from .marker_integration import MarkerProcessor
from .text_chunking import TextChunker
from .document_cache import DocumentCache
from .document_io import DocumentIO

__all__ = [
    'DocumentProcessor', 
    'get_global_marker_models',
    'MarkerProcessor',
    'TextChunker', 
    'DocumentCache',
    'DocumentIO'
]
