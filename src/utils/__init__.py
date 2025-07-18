"""
Utils Module

This module contains utility functions and helpers for the RAG system.
"""

# Import only what's needed for console scripts
from .check_documents import main as check_documents_main
from .document_status import DocumentStatusChecker
from .enhanced_logging import (
    get_enhanced_logger, startup_banner, startup_complete, suppress_noisy_loggers,
    EnhancedLogger, ProgressTracker, PerformanceMonitor, configure_marker_logging,
    enable_all_logging
)

# Thread-safe progress tracking (lazy import to avoid circular dependencies)
# from .thread_safe_progress import ThreadSafeProgressTracker, StreamlitProgressRenderer, ProgressState

# Other imports available on-demand to avoid circular imports
__all__ = [
    # Document status utilities
    'check_documents_main',
    'DocumentStatusChecker',
    # Enhanced logging
    'get_enhanced_logger', 'startup_banner', 'startup_complete', 'suppress_noisy_loggers',
    'EnhancedLogger', 'ProgressTracker', 'PerformanceMonitor', 'configure_marker_logging',
    'enable_all_logging',
    # Thread-safe progress tracking (imported on-demand)
    # 'ThreadSafeProgressTracker', 'StreamlitProgressRenderer', 'ProgressState'
]
