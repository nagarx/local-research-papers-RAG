"""
PyTorch Utilities

Utilities to handle PyTorch configuration and optimization.
"""

import logging
import os
import warnings

# Global flag to prevent redundant configuration
_torch_configured = False

def configure_torch_for_production():
    """Configure PyTorch settings for production use"""
    global _torch_configured
    
    if _torch_configured:
        return  # Already configured, skip
    
    # Import torch here to avoid issues if not available
    try:
        import torch
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.warning("PyTorch not available, skipping configuration")
        _torch_configured = True
        return
    
    # Suppress PyTorch warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.*')
    warnings.filterwarnings('ignore', message='.*torch.classes.*')
    warnings.filterwarnings('ignore', message='.*Tried to instantiate class.*')
    
    # Set environment variables to suppress warnings at C++ level
    os.environ.setdefault('TORCH_CPP_LOG_LEVEL', 'ERROR')
    os.environ.setdefault('PYTHONWARNINGS', 'ignore::UserWarning:torch')
    
    # Set PyTorch logging level to reduce noise
    torch_logger = logging.getLogger('torch')
    torch_logger.setLevel(logging.ERROR)
    
    # Set number of threads for optimal performance
    num_threads = min(4, torch.get_num_threads())
    torch.set_num_threads(num_threads)
    
    # Disable gradient computation globally (we're not training)
    torch.set_grad_enabled(False)
    
    logger = logging.getLogger(__name__)
    logger.info(f"PyTorch configured for production (threads: {num_threads})")
    _torch_configured = True

def is_torch_configured():
    """Check if PyTorch has been configured"""
    return _torch_configured

def clear_gpu_cache():
    """Clear GPU cache if available"""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger = logging.getLogger(__name__)
            logger.debug("GPU cache cleared")
    except ImportError:
        logger = logging.getLogger(__name__)
        logger.debug("PyTorch not available, cannot clear GPU cache")
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to clear GPU cache: {e}")

def get_gpu_memory_info():
    """Get GPU memory information"""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated_memory = torch.cuda.memory_allocated(device)
        cached_memory = torch.cuda.memory_reserved(device)
        
        return {
            "available": True,
            "device": device,
            "total_memory": total_memory,
            "allocated_memory": allocated_memory,
            "cached_memory": cached_memory,
            "free_memory": total_memory - allocated_memory
        }
    except ImportError:
        return {"available": False, "error": "PyTorch not available"}
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to get GPU memory info: {e}")
        return {"available": False, "error": str(e)} 