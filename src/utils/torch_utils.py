"""
PyTorch Utilities

Utilities to handle PyTorch warnings and optimize logging.
"""

import warnings
import logging
import os

def suppress_torch_warnings():
    """Suppress common PyTorch warnings that don't affect functionality"""
    
    # Suppress specific PyTorch warnings (more comprehensive)
    warnings.filterwarnings('ignore', message='.*torch.classes.*')
    warnings.filterwarnings('ignore', message='.*Tried to instantiate class.*')
    warnings.filterwarnings('ignore', message='.*torch::class_.*')
    warnings.filterwarnings('ignore', message='.*__path__._path.*')
    warnings.filterwarnings('ignore', message='.*Examining the path of torch.classes.*')
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.*')
    
    # Set environment variables to suppress warnings at C++ level
    os.environ.setdefault('TORCH_CPP_LOG_LEVEL', 'ERROR')
    os.environ.setdefault('PYTHONWARNINGS', 'ignore::UserWarning:torch')
    
    # Set PyTorch logging level to reduce noise
    torch_logger = logging.getLogger('torch')
    torch_logger.setLevel(logging.ERROR)
    
    # Also suppress warnings from common ML libraries
    for logger_name in ['torch', 'transformers', 'sentence_transformers']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
    
    print("🔇 PyTorch warnings suppressed for cleaner output")

# Global flag to prevent redundant configuration
_torch_configured = False

def configure_torch_for_production():
    """Configure PyTorch settings for production use"""
    global _torch_configured
    
    if _torch_configured:
        return  # Already configured, skip
    
    suppress_torch_warnings()
    
    # Set number of threads for optimal performance
    try:
        import torch
        
        # Set number of threads based on CPU cores
        num_threads = min(4, torch.get_num_threads())
        torch.set_num_threads(num_threads)
        
        # Disable gradient computation globally (we're not training)
        torch.set_grad_enabled(False)
        
        print(f"🚀 PyTorch configured for production (threads: {num_threads})")
        _torch_configured = True
        
    except ImportError:
        print("⚠️  PyTorch not available, skipping configuration")
        _torch_configured = True  # Mark as configured even if PyTorch not available

# Call configuration on import only once
if __name__ != "__main__" and not _torch_configured:
    configure_torch_for_production() 