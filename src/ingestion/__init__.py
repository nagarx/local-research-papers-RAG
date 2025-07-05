"""
Document Ingestion Module

This module handles document processing, parsing, and chunking using the Marker tool.
"""

from .document_processor import DocumentProcessor, get_global_marker_models

__all__ = ['DocumentProcessor', 'get_global_marker_models']
