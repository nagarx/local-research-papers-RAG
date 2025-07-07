"""
Configuration Module

This module handles all configuration management, base classes, and exceptions
for the ArXiv RAG system.
"""

from .config import get_config, Config, reload_config, get_logger
from .base import BaseStats
from .exceptions import (
    RAGSystemError, ConfigurationError, ModelLoadError, DocumentProcessingError,
    EmbeddingError, VectorStoreError, OllamaError, QueryProcessingError, 
    SourceTrackingError
)

__all__ = [
    # Configuration
    'get_config', 'Config', 'reload_config', 'get_logger',
    # Base Components
    'BaseStats',
    # Exceptions
    'RAGSystemError', 'ConfigurationError', 'ModelLoadError', 'DocumentProcessingError',
    'EmbeddingError', 'VectorStoreError', 'OllamaError', 'QueryProcessingError',
    'SourceTrackingError'
]
