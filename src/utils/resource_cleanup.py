"""
Resource Cleanup Utilities

This module provides utilities for cleaning up system resources and preventing leaks.
"""

import os
import gc
import sys
import threading
import multiprocessing
import platform
import warnings
import logging
from typing import Dict, Any, Optional
from pathlib import Path


def cleanup_multiprocessing_resources():
    """Clean up multiprocessing resources to prevent semaphore leaks"""
    try:
        # Force garbage collection
        gc.collect()
        
        # On macOS, try to clean up any remaining semaphore objects
        if platform.system() == 'Darwin':
            try:
                # Reset multiprocessing context if needed
                ctx = multiprocessing.get_context()
                if hasattr(ctx, '_semaphore_tracker'):
                    ctx._semaphore_tracker._stop()
            except Exception:
                pass
        
        return True
    except Exception as e:
        logging.warning(f"Failed to cleanup multiprocessing resources: {e}")
        return False


def cleanup_thread_resources():
    """Clean up threading resources"""
    try:
        # Get current thread count
        active_threads = threading.active_count()
        
        # Force garbage collection to clean up thread objects
        gc.collect()
        
        # Join any daemon threads that can be joined
        for thread in threading.enumerate():
            if thread != threading.current_thread() and thread.daemon:
                try:
                    thread.join(timeout=0.1)
                except Exception:
                    pass
        
        return True
    except Exception as e:
        logging.warning(f"Failed to cleanup thread resources: {e}")
        return False


def cleanup_gpu_memory():
    """Clean up GPU memory if available"""
    try:
        import torch
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Clear MPS cache if available (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        return True
    except ImportError:
        return False
    except Exception as e:
        logging.warning(f"Failed to cleanup GPU memory: {e}")
        return False


def cleanup_temporary_files(temp_dirs: Optional[list] = None):
    """Clean up temporary files"""
    if temp_dirs is None:
        temp_dirs = ["temp_uploads", "data/temp", "/tmp/marker_*"]
    
    cleaned_count = 0
    errors = []
    
    for temp_dir in temp_dirs:
        try:
            temp_path = Path(temp_dir)
            if temp_path.exists() and temp_path.is_dir():
                for file_path in temp_path.iterdir():
                    try:
                        if file_path.is_file():
                            file_path.unlink()
                            cleaned_count += 1
                        elif file_path.is_dir():
                            import shutil
                            shutil.rmtree(file_path)
                            cleaned_count += 1
                    except Exception as e:
                        errors.append(f"Error cleaning {file_path}: {e}")
        except Exception as e:
            errors.append(f"Error cleaning directory {temp_dir}: {e}")
    
    return {
        'cleaned_count': cleaned_count,
        'errors': errors
    }


def suppress_resource_warnings():
    """Suppress common resource warnings"""
    # Suppress multiprocessing resource tracker warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')
    
    # Suppress threading warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='threading')
    
    # Suppress PyTorch warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch')
    
    # Set environment variables to reduce noise
    os.environ.setdefault('PYTHONWARNINGS', 'ignore::UserWarning:multiprocessing')


def perform_comprehensive_cleanup():
    """Perform comprehensive cleanup of all resources"""
    results = {
        'multiprocessing': cleanup_multiprocessing_resources(),
        'threading': cleanup_thread_resources(),
        'gpu_memory': cleanup_gpu_memory(),
        'temporary_files': cleanup_temporary_files()
    }
    
    # Force final garbage collection
    gc.collect()
    
    return results


def setup_resource_monitoring():
    """Setup resource monitoring and cleanup hooks"""
    import atexit
    
    # Register cleanup function to run on exit
    atexit.register(perform_comprehensive_cleanup)
    
    # Suppress warnings
    suppress_resource_warnings()
    
    return True


class ResourceManager:
    """Context manager for handling resources safely"""
    
    def __init__(self, cleanup_on_exit=True):
        self.cleanup_on_exit = cleanup_on_exit
        self.initial_thread_count = threading.active_count()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup_on_exit:
            try:
                cleanup_multiprocessing_resources()
                cleanup_thread_resources()
                cleanup_gpu_memory()
            except Exception as e:
                logging.warning(f"Error during resource cleanup: {e}")
    
    def cleanup_now(self):
        """Force cleanup immediately"""
        return perform_comprehensive_cleanup()


# Auto-setup when module is imported
setup_resource_monitoring() 