"""
Common Utilities

This module provides shared utilities and helper functions used across
the RAG system components to eliminate code duplication.
"""

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import logging

# ML/System dependencies
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Local imports
from ..config import get_logger


class FileUtils:
    """File and path utilities"""
    
    @staticmethod
    def validate_pdf_file(file_path: Union[str, Path]) -> Path:
        """Validate PDF file path"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        if file_path.suffix.lower() != '.pdf':
            raise ValueError(f"Only PDF files are supported: {file_path}")
        
        return file_path
    
    @staticmethod
    def ensure_directory(directory: Union[str, Path]) -> Path:
        """Ensure directory exists"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    
    @staticmethod
    def get_file_hash(file_path: Union[str, Path]) -> str:
        """Get file hash for caching"""
        file_path = Path(file_path)
        
        # Use file size and modification time for fast hashing
        stat = file_path.stat()
        content = f"{file_path.name}_{stat.st_size}_{stat.st_mtime}"
        
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    @staticmethod
    def safe_read_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Safely read JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"Failed to read JSON file {file_path}: {e}")
            return None
    
    @staticmethod
    def safe_write_json(file_path: Union[str, Path], data: Dict[str, Any]) -> bool:
        """Safely write JSON file"""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"Failed to write JSON file {file_path}: {e}")
            return False


class GPUUtils:
    """GPU memory management utilities"""
    
    @staticmethod
    def clear_gpu_cache():
        """Clear GPU cache if available"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger = get_logger(__name__)
                logger.debug("GPU cache cleared")
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"Failed to clear GPU cache: {e}")
    
    @staticmethod
    def get_gpu_memory_info() -> Dict[str, Any]:
        """Get GPU memory information"""
        if not torch.cuda.is_available():
            return {"available": False}
        
        try:
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
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"Failed to get GPU memory info: {e}")
            return {"available": False, "error": str(e)}


# Convenience functions for common operations
def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists"""
    return FileUtils.ensure_directory(path)


def clear_gpu_cache():
    """Clear GPU cache"""
    GPUUtils.clear_gpu_cache()


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Basic cleanup
    text = text.strip()
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    return text


def get_text_hash(text: str) -> str:
    """Get hash for text content"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


async def run_in_thread(operation: callable, *args, **kwargs) -> Any:
    """Run synchronous operation in thread pool"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        return await loop.run_in_executor(executor, operation, *args, **kwargs) 