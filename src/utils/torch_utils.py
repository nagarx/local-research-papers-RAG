"""
PyTorch Utilities

This module contains utility functions for PyTorch optimization and configuration.
"""

import logging
import os
import warnings
import sys
import platform
from typing import Dict, Any, Optional

# Global flag to prevent redundant configuration
_torch_configured = False

def configure_matplotlib_for_headless():
    """Configure matplotlib to run in headless mode to prevent GUI issues"""
    try:
        # Set matplotlib backend before importing matplotlib
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        
        # Additional configuration for headless operation
        os.environ['MPLBACKEND'] = 'Agg'
        
        # Suppress matplotlib warnings
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        
        return True
    except ImportError:
        # matplotlib not installed, that's fine
        return False
    except Exception as e:
        logging.warning(f"Failed to configure matplotlib for headless operation: {e}")
        return False


def configure_macos_multiprocessing():
    """Configure macOS-specific multiprocessing settings"""
    if platform.system() == 'Darwin':  # macOS
        try:
            # Set multiprocessing start method to 'spawn' to avoid GUI issues
            import multiprocessing
            if multiprocessing.get_start_method(allow_none=True) != 'spawn':
                multiprocessing.set_start_method('spawn', force=True)
            
            # Set environment variables to prevent GUI library issues
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
            
            return True
        except Exception as e:
            logging.warning(f"Failed to configure macOS multiprocessing: {e}")
            return False
    return False


def configure_torch_for_production():
    """Configure PyTorch for production environment"""
    try:
        import torch
        import warnings
        
        # Suppress PyTorch warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='torch')
        warnings.filterwarnings('ignore', category=FutureWarning, module='torch')
        
        # Suppress specific PyTorch class path warnings
        warnings.filterwarnings('ignore', message='.*torch.classes.*')
        warnings.filterwarnings('ignore', message='.*Tried to instantiate class.*')
        warnings.filterwarnings('ignore', message='.*Examining the path of torch.classes.*')
        
        # Set environment variables to suppress warnings at C++ level
        os.environ.setdefault('TORCH_CPP_LOG_LEVEL', 'ERROR')
        os.environ.setdefault('PYTHONWARNINGS', 'ignore::UserWarning:torch')
        
        # Set PyTorch to use deterministic algorithms where possible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Configure for production (no autograd if not needed)
        torch.set_grad_enabled(False)
        
        # Set threading settings for better performance
        torch.set_num_threads(min(4, torch.get_num_threads()))
        
        # Configure MPS (Metal Performance Shaders) for Apple Silicon
        if torch.backends.mps.is_available():
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            
        return True
    except ImportError:
        return False
    except Exception as e:
        logging.warning(f"Failed to configure PyTorch for production: {e}")
        return False


def configure_for_headless_operation():
    """Configure all libraries for headless operation"""
    results = {
        'matplotlib': configure_matplotlib_for_headless(),
        'macos_multiprocessing': configure_macos_multiprocessing(),
        'torch_production': configure_torch_for_production()
    }
    
    # Set additional environment variables for headless operation
    os.environ['DISPLAY'] = ''  # Disable X11 display
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Qt headless
    
    return results


def get_device_info() -> Dict[str, Any]:
    """Get device information for PyTorch"""
    device_info = {
        'platform': platform.system(),
        'architecture': platform.machine(),
        'cpu_count': os.cpu_count(),
        'cuda_available': False,
        'mps_available': False,
        'recommended_device': 'cpu'
    }
    
    try:
        import torch
        device_info['cuda_available'] = torch.cuda.is_available()
        device_info['mps_available'] = torch.backends.mps.is_available()
        
        if device_info['cuda_available']:
            device_info['recommended_device'] = 'cuda'
            device_info['cuda_device_count'] = torch.cuda.device_count()
        elif device_info['mps_available']:
            device_info['recommended_device'] = 'mps'
    except ImportError:
        pass
    
    return device_info


def optimize_for_inference():
    """Optimize PyTorch for inference only"""
    try:
        import torch
        
        # Disable gradient computation globally
        torch.set_grad_enabled(False)
        
        # Enable inference mode optimizations
        torch.inference_mode()
        
        # Set threading for inference
        torch.set_num_threads(min(4, torch.get_num_threads()))
        
        return True
    except ImportError:
        return False
    except Exception as e:
        logging.warning(f"Failed to optimize PyTorch for inference: {e}")
        return False


# Apply headless configuration when module is imported
configure_for_headless_operation()

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