"""
Enhanced Logging System

Provides better progress tracking, cleaner formatting, and more informative
messages for the RAG pipeline with performance metrics and system status.
"""

import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager
import psutil
import torch


class ProgressTracker:
    """Tracks progress for long-running operations"""
    
    def __init__(self, operation_name: str, total_steps: int, logger: logging.Logger):
        self.operation_name = operation_name
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        self.logger = logger
        self.last_update = 0
        self.update_interval = 2.0  # Update every 2 seconds
        
    def update(self, step: int, message: str = ""):
        """Update progress"""
        self.current_step = step
        current_time = time.time()
        
        # Only log if enough time has passed or it's the final step
        if (current_time - self.last_update >= self.update_interval or 
            step >= self.total_steps):
            
            percentage = (step / self.total_steps) * 100
            elapsed = current_time - self.start_time
            
            if step > 0:
                eta = (elapsed / step) * (self.total_steps - step)
                eta_str = f"ETA: {eta:.1f}s"
            else:
                eta_str = "ETA: calculating..."
            
            progress_bar = self._create_progress_bar(percentage)
            
            self.logger.info(
                f"📊 {self.operation_name}: {progress_bar} "
                f"{step}/{self.total_steps} ({percentage:.1f}%) | "
                f"Elapsed: {elapsed:.1f}s | {eta_str}"
                f"{' | ' + message if message else ''}"
            )
            
            self.last_update = current_time
    
    def _create_progress_bar(self, percentage: float, width: int = 20) -> str:
        """Create a visual progress bar"""
        filled = int(width * percentage / 100)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}]"
    
    def finish(self, message: str = ""):
        """Mark operation as finished"""
        elapsed = time.time() - self.start_time
        self.logger.info(
            f"✅ {self.operation_name} completed in {elapsed:.2f}s"
            f"{' | ' + message if message else ''}"
        )


class PerformanceMonitor:
    """Monitors system performance metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        process = psutil.Process()
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        stats = {
            "elapsed_time": time.time() - self.start_time,
            "memory_mb": current_memory,
            "memory_delta_mb": current_memory - self.start_memory,
            "cpu_percent": process.cpu_percent(),
        }
        
        # Add GPU stats if available
        if torch.cuda.is_available():
            stats.update({
                "gpu_memory_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "gpu_utilization": "N/A"  # Would need nvidia-ml-py for this
            })
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            stats["gpu_type"] = "Apple MPS"
            # MPS doesn't expose memory stats easily
        
        return stats


class EnhancedLogger:
    """Enhanced logger with better formatting and progress tracking"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.performance_monitor = PerformanceMonitor()
        self._active_trackers = {}
        
    def startup_stage(self, stage_name: str, details: str = ""):
        """Log startup stage with clear formatting"""
        self.logger.info(f"🚀 STARTUP: {stage_name}{' - ' + details if details else ''}")
    
    def system_ready(self, component: str, details: str = ""):
        """Log when a system component is ready"""
        self.logger.info(f"✅ READY: {component}{' - ' + details if details else ''}")
    
    def processing_start(self, operation: str, details: str = ""):
        """Log start of processing operation"""
        self.logger.info(f"⚙️  PROCESSING: {operation}{' - ' + details if details else ''}")
    
    def processing_complete(self, operation: str, duration: float, details: str = ""):
        """Log completion of processing operation"""
        self.logger.info(
            f"✅ COMPLETED: {operation} in {duration:.2f}s"
            f"{' - ' + details if details else ''}"
        )
    
    def query_start(self, query: str, query_id: str = ""):
        """Log start of query processing"""
        query_preview = query[:50] + "..." if len(query) > 50 else query
        self.logger.info(f"🔍 QUERY: {query_preview}{' (' + query_id + ')' if query_id else ''}")
    
    def query_step(self, step: str, details: str = ""):
        """Log query processing step"""
        self.logger.info(f"   ├─ {step}{': ' + details if details else ''}")
    
    def query_complete(self, duration: float, sources_found: int, details: str = ""):
        """Log query completion"""
        self.logger.info(
            f"✅ QUERY COMPLETE: {duration:.2f}s | {sources_found} sources"
            f"{' | ' + details if details else ''}"
        )
    
    def document_start(self, filename: str, operation: str = "processing"):
        """Log start of document operation"""
        self.logger.info(f"📄 DOCUMENT: {operation} '{filename}'")
    
    def document_complete(self, filename: str, duration: float, chunks: int = 0, operation: str = "processed"):
        """Log document operation completion"""
        details = f"{chunks} chunks" if chunks > 0 else ""
        self.logger.info(
            f"✅ DOCUMENT: '{filename}' {operation} in {duration:.2f}s"
            f"{' - ' + details if details else ''}"
        )
    
    def create_progress_tracker(self, operation_name: str, total_steps: int) -> ProgressTracker:
        """Create a progress tracker for long operations"""
        tracker = ProgressTracker(operation_name, total_steps, self.logger)
        self._active_trackers[operation_name] = tracker
        return tracker
    
    def performance_summary(self, operation: str = ""):
        """Log performance summary"""
        stats = self.performance_monitor.get_current_stats()
        
        self.logger.info(
            f"📊 PERFORMANCE{' (' + operation + ')' if operation else ''}: "
            f"Memory: {stats['memory_mb']:.1f}MB "
            f"({stats['memory_delta_mb']:+.1f}MB), "
            f"CPU: {stats['cpu_percent']:.1f}%, "
            f"Time: {stats['elapsed_time']:.1f}s"
        )
        
        if 'gpu_memory_mb' in stats:
            self.logger.info(
                f"📊 GPU: {stats['gpu_memory_mb']:.1f}MB allocated, "
                f"{stats['gpu_memory_reserved_mb']:.1f}MB reserved"
            )
    
    def warning_clean(self, message: str):
        """Log warning with clean formatting"""
        self.logger.warning(f"⚠️  {message}")
    
    def error_clean(self, message: str, exception: Exception = None):
        """Log error with clean formatting"""
        error_msg = f"❌ ERROR: {message}"
        if exception:
            error_msg += f" | {type(exception).__name__}: {str(exception)}"
        self.logger.error(error_msg)
    
    def system_status(self, status: Dict[str, Any]):
        """Log system status summary"""
        self.logger.info("📋 SYSTEM STATUS:")
        for component, details in status.items():
            if isinstance(details, dict):
                status_icon = "✅" if details.get('status') == 'healthy' else "❌"
                self.logger.info(f"   ├─ {status_icon} {component}: {details.get('status', 'unknown')}")
            else:
                self.logger.info(f"   ├─ {component}: {details}")
    
    @contextmanager
    def operation_timer(self, operation_name: str):
        """Context manager for timing operations"""
        start_time = time.time()
        self.processing_start(operation_name)
        
        try:
            yield
            duration = time.time() - start_time
            self.processing_complete(operation_name, duration)
        except Exception as e:
            duration = time.time() - start_time
            self.error_clean(f"{operation_name} failed after {duration:.2f}s", e)
            raise


class LoggingManager:
    """Central logging manager for the RAG system"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.loggers = {}
            self.startup_time = time.time()
            self.initialized = True
    
    def get_logger(self, name: str) -> EnhancedLogger:
        """Get or create an enhanced logger"""
        if name not in self.loggers:
            self.loggers[name] = EnhancedLogger(name)
        return self.loggers[name]
    
    def startup_banner(self, app_name: str, version: str, environment: str):
        """Display startup banner"""
        logger = self.get_logger('system')
        banner_lines = [
            "=" * 70,
            f"🚀 {app_name} v{version}",
            f"Environment: {environment} | Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70
        ]
        
        for line in banner_lines:
            logger.logger.info(line)
            print(line)  # Also print to console for visibility
    
    def startup_complete(self, total_time: float):
        """Log startup completion"""
        logger = self.get_logger('system')
        completion_lines = [
            "=" * 70,
            f"✅ STARTUP COMPLETE: System ready in {total_time:.2f}s",
            "=" * 70
        ]
        
        for line in completion_lines:
            logger.logger.info(line)
            print(line)  # Also print to console for visibility
    
    def suppress_noisy_loggers(self):
        """Suppress noisy third-party loggers"""
        noisy_loggers = [
            'torch', 'transformers', 'sentence_transformers', 
            'urllib3', 'requests', 'httpx'
        ]
        
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.ERROR)


# Global instance
_logging_manager = LoggingManager()

def get_enhanced_logger(name: str) -> EnhancedLogger:
    """Get an enhanced logger instance"""
    return _logging_manager.get_logger(name)

def startup_banner(app_name: str, version: str, environment: str):
    """Display startup banner"""
    _logging_manager.startup_banner(app_name, version, environment)

def startup_complete(total_time: float):
    """Log startup completion"""
    _logging_manager.startup_complete(total_time)

def suppress_noisy_loggers():
    """Suppress noisy third-party loggers"""
    _logging_manager.suppress_noisy_loggers() 