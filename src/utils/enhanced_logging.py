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

    def marker_model_loading(self, model_type: str, details: str = ""):
        """Log marker model loading"""
        self.logger.info(f"🤖 MARKER MODEL: Loading {model_type}{' - ' + details if details else ''}")
    
    def marker_model_ready(self, model_type: str, load_time: float, details: str = ""):
        """Log marker model ready"""
        self.logger.info(
            f"✅ MARKER MODEL: {model_type} ready in {load_time:.2f}s"
            f"{' - ' + details if details else ''}"
        )
    
    def marker_processing_start(self, filename: str, pages: int = 0):
        """Log marker processing start"""
        pages_info = f" ({pages} pages)" if pages > 0 else ""
        self.logger.info(f"📄 MARKER: Processing '{filename}'{pages_info}")
    
    def marker_processing_stage(self, stage: str, filename: str, details: str = ""):
        """Log marker processing stage"""
        self.logger.info(f"   ├─ {stage}: {filename}{' - ' + details if details else ''}")
    
    def marker_processing_complete(self, filename: str, duration: float, pages: int = 0, chunks: int = 0):
        """Log marker processing completion"""
        pages_info = f"{pages} pages, " if pages > 0 else ""
        chunks_info = f"{chunks} chunks" if chunks > 0 else ""
        self.logger.info(
            f"✅ MARKER: '{filename}' processed in {duration:.2f}s - {pages_info}{chunks_info}"
        )
    
    def marker_ocr_start(self, filename: str, pages: int = 0):
        """Log marker OCR start"""
        pages_info = f" ({pages} pages)" if pages > 0 else ""
        self.logger.info(f"👁️  MARKER OCR: Processing '{filename}'{pages_info}")
    
    def marker_ocr_complete(self, filename: str, duration: float, pages_processed: int = 0):
        """Log marker OCR completion"""
        pages_info = f"{pages_processed} pages " if pages_processed > 0 else ""
        self.logger.info(f"✅ MARKER OCR: '{filename}' - {pages_info}processed in {duration:.2f}s")
    
    def marker_layout_detection(self, filename: str, pages: int = 0):
        """Log marker layout detection"""
        pages_info = f" ({pages} pages)" if pages > 0 else ""
        self.logger.info(f"🔍 MARKER LAYOUT: Detecting layout for '{filename}'{pages_info}")
    
    def marker_equation_processing(self, filename: str, equations: int = 0):
        """Log marker equation processing"""
        eq_info = f" ({equations} equations)" if equations > 0 else ""
        self.logger.info(f"📐 MARKER EQUATIONS: Processing '{filename}'{eq_info}")
    
    def marker_image_extraction(self, filename: str, images: int = 0):
        """Log marker image extraction"""
        img_info = f" ({images} images)" if images > 0 else ""
        self.logger.info(f"🖼️  MARKER IMAGES: Extracting from '{filename}'{img_info}")
    
    def marker_cleanup(self, operation: str, details: str = ""):
        """Log marker cleanup operations"""
        self.logger.info(f"🧹 MARKER CLEANUP: {operation}{' - ' + details if details else ''}")
    
    def marker_warning(self, message: str, filename: str = ""):
        """Log marker-specific warning"""
        file_info = f" [{filename}]" if filename else ""
        self.logger.warning(f"⚠️  MARKER{file_info}: {message}")
    
    def marker_error(self, message: str, filename: str = "", exception: Exception = None):
        """Log marker-specific error"""
        file_info = f" [{filename}]" if filename else ""
        error_msg = f"❌ MARKER{file_info}: {message}"
        if exception:
            error_msg += f" | {type(exception).__name__}: {str(exception)}"
        self.logger.error(error_msg)
    
    def marker_performance(self, operation: str, filename: str = "", **metrics):
        """Log marker performance metrics"""
        file_info = f" [{filename}]" if filename else ""
        metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        self.logger.info(f"📊 MARKER PERF{file_info}: {operation} - {metrics_str}")
    
    def marker_resource_usage(self, operation: str, memory_mb: float, gpu_memory_mb: float = 0):
        """Log marker resource usage"""
        gpu_info = f", GPU: {gpu_memory_mb:.1f}MB" if gpu_memory_mb > 0 else ""
        self.logger.info(f"💾 MARKER RESOURCES: {operation} - Memory: {memory_mb:.1f}MB{gpu_info}")
    
    def marker_cache_info(self, operation: str, cache_hit: bool, details: str = ""):
        """Log marker cache operations"""
        cache_status = "HIT" if cache_hit else "MISS"
        self.logger.info(f"🗄️  MARKER CACHE: {operation} - {cache_status}{' - ' + details if details else ''}")
    
    def marker_batch_progress(self, current: int, total: int, operation: str = "processing"):
        """Log marker batch processing progress"""
        percentage = (current / total) * 100
        self.logger.info(f"📊 MARKER BATCH: {operation} {current}/{total} ({percentage:.1f}%)")
    
    def marker_model_stats(self, model_type: str, **stats):
        """Log marker model statistics"""
        stats_str = ", ".join([f"{k}: {v}" for k, v in stats.items()])
        self.logger.info(f"📈 MARKER MODEL STATS: {model_type} - {stats_str}")


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
    
    def enable_all_logging(self):
        """ENABLE ALL LOGGING - No suppression, show everything from all libraries"""
        # Import config to check debug mode
        try:
            from ..config import get_config
            config = get_config()
            debug_mode = config.debug
        except ImportError:
            debug_mode = False
        
        # ALL loggers - enable EVERYTHING at DEBUG level
        all_loggers = [
            # Core Python libraries
            'torch', 'transformers', 'sentence_transformers', 
            'urllib3', 'requests', 'httpx',
            
            # Marker-specific loggers
            'marker', 'marker.models', 'marker.converters', 'marker.providers',
            'marker.builders', 'marker.processors', 'marker.renderers', 'marker.services',
            'surya', 'surya.ocr', 'surya.layout', 'surya.model',
            'texify', 'texify.inference', 'pdfium', 'pypdfium2', 'pdf_postprocessor',
            
            # Image processing
            'PIL', 'PIL.PngImagePlugin', 'PIL.JpegImagePlugin',
            
            # ML/AI libraries
            'transformers.tokenization_utils', 'transformers.modeling_utils',
            'accelerate', 'torch.nn.parallel', 'torch.distributed',
            'huggingface_hub', 'huggingface_hub.file_download',
            
            # Network libraries
            'urllib3.connectionpool', 'requests.packages.urllib3.connectionpool',
            
            # ChromaDB
            'chromadb', 'chromadb.db', 'chromadb.api',
            
            # Streamlit
            'streamlit', 'streamlit.runtime',
            
            # General Python
            'asyncio', 'multiprocessing', 'threading',
        ]
        
        # ENABLE ALL LOGGING - Set everything to DEBUG for maximum visibility
        for logger_name in all_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)
            logger.disabled = False
            print(f"🔍 ENABLED ALL LOGGING: {logger_name} -> DEBUG level")
        
        # Set root logger to DEBUG to catch everything
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        print(f"🔍 ENABLED ROOT LOGGER: -> DEBUG level")
        
        # Also enable tqdm for progress bars
        logging.getLogger('tqdm').setLevel(logging.DEBUG)
        print(f"🔍 ENABLED TQDM LOGGING: -> DEBUG level")
        
        print("🎯 ALL LOGGING ENABLED - You will see EVERYTHING from ALL libraries!")
        print("📋 No logs are suppressed - complete pipeline visibility!")
    
    # Keep old function name for backwards compatibility
    def suppress_noisy_loggers(self):
        """Backwards compatibility - now enables all logging instead of suppressing"""
        self.enable_all_logging()


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
    """Backwards compatibility - now enables all logging instead of suppressing"""
    _logging_manager.enable_all_logging()

def enable_all_logging():
    """Enable all logging from all libraries - complete pipeline visibility"""
    _logging_manager.enable_all_logging()


def configure_marker_logging(enable_debug: bool = False, log_level: str = "ERROR"):
    """Configure ALL logging levels dynamically - ENABLE EVERYTHING, NO SUPPRESSION"""
    import logging
    
    # ALL loggers - enable EVERYTHING
    all_loggers = [
        # Core Python libraries
        'torch', 'transformers', 'sentence_transformers', 
        'urllib3', 'requests', 'httpx',
        
        # Marker-specific loggers
        'marker', 'marker.models', 'marker.converters', 'marker.providers',
        'marker.builders', 'marker.processors', 'marker.renderers', 'marker.services',
        'surya', 'surya.ocr', 'surya.layout', 'surya.model',
        'texify', 'texify.inference', 'pdfium', 'pypdfium2', 'pdf_postprocessor',
        
        # Image processing
        'PIL', 'PIL.PngImagePlugin', 'PIL.JpegImagePlugin',
        
        # ML/AI libraries
        'transformers.tokenization_utils', 'transformers.modeling_utils',
        'accelerate', 'torch.nn.parallel', 'torch.distributed',
        'huggingface_hub', 'huggingface_hub.file_download',
        
        # Network libraries
        'urllib3.connectionpool', 'requests.packages.urllib3.connectionpool',
        
        # ChromaDB
        'chromadb', 'chromadb.db', 'chromadb.api',
        
        # Streamlit
        'streamlit', 'streamlit.runtime',
        
        # General Python
        'asyncio', 'multiprocessing', 'threading', 'tqdm'
    ]
    
    # ENABLE ALL logging at DEBUG level for maximum visibility
    for logger_name in all_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.disabled = False
        print(f"🔍 ALL LOGGING ENABLED: {logger_name} -> DEBUG")
    
    # Set root logger to DEBUG to catch everything
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    print(f"🔍 ROOT LOGGER ENABLED: -> DEBUG")
    
    print("🎯 ALL LOGGERS ENABLED - Complete pipeline visibility!")
    print("📋 NO suppression - you will see EVERYTHING from ALL libraries!")


def get_marker_logging_stats():
    """Get current marker logging configuration stats"""
    import logging
    
    marker_loggers = [
        'marker', 'surya', 'texify', 'pdfium', 'pypdfium2', 'pdf_postprocessor'
    ]
    
    stats = {}
    for logger_name in marker_loggers:
        logger = logging.getLogger(logger_name)
        stats[logger_name] = {
            'level': logging.getLevelName(logger.level),
            'effective_level': logging.getLevelName(logger.getEffectiveLevel()),
            'handlers': len(logger.handlers),
            'disabled': logger.disabled
        }
    
    return stats 