"""
Base Classes and Common Patterns

This module provides base classes and common patterns to eliminate
code duplication across the RAG system components.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, TypeVar, Generic
from pathlib import Path
import json

# Local imports
from .config import get_config, get_logger

T = TypeVar('T')


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


class BaseComponent(ABC):
    """Base class for all RAG system components"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = get_config() if config is None else config
        self.logger = get_logger(f"{__name__}.{name}")
        self.stats = BaseStats()
        self._initialized = False
        
        # Initialize component
        self._initialize()
    
    def _initialize(self):
        """Initialize the component"""
        try:
            self.logger.info(f"Initializing {self.name}...")
            start_time = time.time()
            
            self.setup()
            
            duration = time.time() - start_time
            self.stats.record_operation(success=True, duration=duration)
            self._initialized = True
            
            self.logger.info(f"{self.name} initialized successfully in {duration:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.name}: {e}")
            self.stats.record_operation(success=False)
            raise
    
    @abstractmethod
    def setup(self):
        """Setup method to be implemented by subclasses"""
        pass
    
    def is_initialized(self) -> bool:
        """Check if component is properly initialized"""
        return self._initialized
    
    def get_status(self) -> Dict[str, Any]:
        """Get component status"""
        return {
            "name": self.name,
            "initialized": self._initialized,
            "stats": self.stats.get_stats(),
            "config": self._get_config_summary()
        }
    
    def _get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary (override in subclasses)"""
        return {"type": self.__class__.__name__}


class AsyncComponentMixin:
    """Mixin for async operation support"""
    
    async def run_async(self, operation: callable, *args, **kwargs) -> Any:
        """Run a synchronous operation asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Run in thread pool for CPU-bound operations
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, operation, *args, **kwargs)
    
    async def batch_process_async(
        self, 
        items: List[T], 
        processor: callable,
        batch_size: int = 5,
        progress_callback: Optional[callable] = None
    ) -> List[Any]:
        """Process items in batches asynchronously"""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(items) + batch_size - 1) // batch_size
            
            if progress_callback:
                progress_callback(batch_num, total_batches)
            
            # Process batch
            batch_results = []
            for item in batch:
                try:
                    result = await self.run_async(processor, item)
                    batch_results.append(result)
                except Exception as e:
                    self.logger.error(f"Error processing item in batch {batch_num}: {e}")
                    batch_results.append(None)
            
            results.extend(batch_results)
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        return results


class CacheManager:
    """Base class for cache management"""
    
    def __init__(self, cache_dir: Path, max_size: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(f"{__name__}.CacheManager")
    
    def _get_cache_key(self, key: str) -> str:
        """Generate cache key"""
        import hashlib
        return hashlib.md5(key.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        cache_key = self._get_cache_key(key)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
                cache_path.unlink(missing_ok=True)
        
        return None
    
    def set(self, key: str, value: Any):
        """Set item in cache"""
        cache_key = self._get_cache_key(key)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(value, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")
    
    def clear(self):
        """Clear all cache"""
        cache_files = list(self.cache_dir.glob("*.json"))
        for cache_file in cache_files:
            try:
                cache_file.unlink()
            except Exception as e:
                self.logger.error(f"Error deleting cache file: {e}")
        
        self.logger.info(f"Cleared {len(cache_files)} cache files")


class ErrorHandler:
    """Centralized error handling"""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.logger = get_logger(f"{__name__}.{component_name}")
    
    def handle_error(self, error: Exception, context: str = "") -> None:
        """Handle and log errors consistently"""
        error_msg = f"Error in {self.component_name}"
        if context:
            error_msg += f" ({context})"
        error_msg += f": {error}"
        
        self.logger.error(error_msg)
        
        # Log stack trace for debugging
        import traceback
        self.logger.debug(traceback.format_exc())
    
    def safe_execute(self, operation: callable, *args, **kwargs) -> Optional[Any]:
        """Execute operation with error handling"""
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            self.handle_error(e, f"executing {operation.__name__}")
            return None
    
    async def safe_execute_async(self, operation: callable, *args, **kwargs) -> Optional[Any]:
        """Execute async operation with error handling"""
        try:
            return await operation(*args, **kwargs)
        except Exception as e:
            self.handle_error(e, f"executing async {operation.__name__}")
            return None


class ValidationMixin:
    """Mixin for common validation patterns"""
    
    def validate_file_path(self, file_path: Union[str, Path], extensions: List[str] = None) -> Path:
        """Validate file path and extension"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        if extensions:
            if file_path.suffix.lower() not in [ext.lower() for ext in extensions]:
                raise ValueError(f"Invalid file extension. Expected: {extensions}, got: {file_path.suffix}")
        
        return file_path
    
    def validate_non_empty_string(self, value: str, field_name: str) -> str:
        """Validate non-empty string"""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
        return value.strip()
    
    def validate_positive_number(self, value: Union[int, float], field_name: str) -> Union[int, float]:
        """Validate positive number"""
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"{field_name} must be a positive number")
        return value
    
    def validate_list_not_empty(self, value: List[Any], field_name: str) -> List[Any]:
        """Validate non-empty list"""
        if not isinstance(value, list) or len(value) == 0:
            raise ValueError(f"{field_name} must be a non-empty list")
        return value 