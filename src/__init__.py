"""
ArXiv Paper RAG Assistant

An intelligent RAG (Retrieval-Augmented Generation) system designed specifically 
for academic research papers. Upload multiple PDFs, process them with state-of-the-art 
document understanding, and chat with your documents using local LLMs.

Version: 1.0.0
Author: ArXiv RAG Assistant Team
License: MIT
"""

# Early warning suppression - do this before any imports
import warnings
import os

# Suppress PyTorch warnings early
warnings.filterwarnings('ignore', message='.*torch.classes.*')
warnings.filterwarnings('ignore', message='.*Tried to instantiate class.*')
warnings.filterwarnings('ignore', message='.*torch::class_.*')
warnings.filterwarnings('ignore', message='.*__path__._path.*')
warnings.filterwarnings('ignore', message='.*Examining the path of torch.classes.*')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.*')

os.environ.setdefault('TORCH_CPP_LOG_LEVEL', 'ERROR')
os.environ.setdefault('PYTHONWARNINGS', 'ignore::UserWarning:torch')

__version__ = "1.0.0"
__author__ = "ArXiv RAG Assistant Team"
__email__ = "contact@arxivrag.com"
__license__ = "MIT"

# Note: Imports are now handled individually by modules to avoid circular dependencies
# This allows each module to be imported independently without loading the entire package

# Version info
VERSION_INFO = (1, 0, 0)

def get_version():
    """Get the current version string"""
    return __version__

def get_system_info():
    """Get system information"""
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__,
        "python_version": f"{VERSION_INFO[0]}.{VERSION_INFO[1]}.{VERSION_INFO[2]}"
    }

# Default configuration
__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "__license__",
    "get_version",
    "get_system_info"
] 