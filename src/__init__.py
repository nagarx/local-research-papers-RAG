"""
ArXiv Paper RAG Assistant

A comprehensive RAG system for processing and querying academic papers.
"""

from .config import (
    get_config,
    get_logger,
    OllamaConfig,
    EmbeddingConfig,
    VectorStorageConfig,
    DocumentProcessingConfig,
    MarkerConfig,
    StoragePathsConfig,
    StreamlitConfig,
    PerformanceConfig,
    UIConfig,
    Config,
    reload_config
)

from .config.exceptions import (
    RAGSystemError,
    ConfigurationError,
    DocumentProcessingError,
    EmbeddingError,
    VectorStoreError,
    OllamaError,
    QueryProcessingError,
    SourceTrackingError
)

# Core components
from .ingestion import DocumentProcessor
from .storage import ChromaVectorStore
from .llm import OllamaClient
from .chat import ChatEngine
from .tracking import SourceTracker

# Utilities
from .utils import (
    setup_torch_device,
    get_device_info,
    estimate_memory_usage,
    monitor_memory_usage,
    optimize_torch_settings,
    cleanup_torch_cache,
    get_torch_device,
    set_torch_device,
    cleanup_utils
)

# Version info
__version__ = "1.0.0"
__author__ = "ArXiv RAG Assistant Team"

# Main exports
__all__ = [
    # Configuration
    "get_config",
    "get_logger",
    "OllamaConfig",
    "EmbeddingConfig", 
    "VectorStorageConfig",
    "DocumentProcessingConfig",
    "MarkerConfig",
    "StoragePathsConfig",
    "StreamlitConfig",
    "PerformanceConfig",
    "UIConfig",
    "Config",
    "reload_config",
    
    # Core components
    "DocumentProcessor",
    "ChromaVectorStore",
    "OllamaClient",
    "ChatEngine",
    "SourceTracker",
    
    # Utilities
    "setup_torch_device",
    "get_device_info",
    "estimate_memory_usage",
    "monitor_memory_usage",
    "optimize_torch_settings",
    "cleanup_torch_cache",
    "get_torch_device",
    "set_torch_device",
    "cleanup_utils",
    
    # Exception classes
    "RAGSystemError",
    "ConfigurationError",
    "DocumentProcessingError",
    "EmbeddingError", "VectorStoreError", "OllamaError", "QueryProcessingError", "SourceTrackingError"
] 