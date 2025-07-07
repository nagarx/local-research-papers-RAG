"""
Base Classes and Common Patterns

This module provides base classes and common patterns to eliminate
code duplication across the RAG system components.
"""

import time
from datetime import datetime
from typing import Dict, Any

# Local imports
from .config import get_config, get_logger


class BaseStats:
    """Base class for statistics tracking"""
    
    def __init__(self):
        self._stats = {
            "created_at": datetime.utcnow().isoformat(),
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_time": 0.0,
            "last_operation_at": None
        }
    
    def record_operation(self, success: bool, duration: float = 0.0):
        """Record an operation with timing"""
        self._stats["total_operations"] += 1
        self._stats["total_time"] += duration
        self._stats["last_operation_at"] = datetime.utcnow().isoformat()
        
        if success:
            self._stats["successful_operations"] += 1
        else:
            self._stats["failed_operations"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        stats = self._stats.copy()
        
        # Calculate derived metrics
        if stats["total_operations"] > 0:
            stats["success_rate"] = stats["successful_operations"] / stats["total_operations"]
            stats["average_operation_time"] = stats["total_time"] / stats["total_operations"]
        else:
            stats["success_rate"] = 0.0
            stats["average_operation_time"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset all statistics"""
        self._stats = {
            "created_at": datetime.utcnow().isoformat(),
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "total_time": 0.0,
            "last_operation_at": None
        } 