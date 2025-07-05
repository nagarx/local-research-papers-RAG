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


class TextUtils:
    """Text processing utilities"""
    
    @staticmethod
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
    
    @staticmethod
    def truncate_text(text: str, max_length: int) -> str:
        """Truncate text to maximum length"""
        if len(text) <= max_length:
            return text
        
        # Try to truncate at word boundary
        if max_length > 10:
            truncated = text[:max_length - 3]
            last_space = truncated.rfind(' ')
            if last_space > max_length // 2:
                return truncated[:last_space] + "..."
        
        return text[:max_length - 3] + "..."
    
    @staticmethod
    def split_text_into_chunks(
        text: str, 
        chunk_size: int = 1000, 
        overlap: int = 100
    ) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence end
            if end < len(text) and chunk_size > 100:
                # Look for sentence endings
                sentence_ends = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
                best_break = -1
                
                for sentence_end in sentence_ends:
                    pos = chunk.rfind(sentence_end)
                    if pos > chunk_size // 2:  # Don't break too early
                        best_break = max(best_break, pos + len(sentence_end))
                
                if best_break > 0:
                    chunk = chunk[:best_break]
                    end = start + best_break
            
            chunks.append(chunk.strip())
            
            if end >= len(text):
                break
            
            start = max(start + chunk_size - overlap, start + 1)
        
        return [chunk for chunk in chunks if chunk]


class AsyncUtils:
    """Async operation utilities"""
    
    @staticmethod
    async def run_in_thread(operation: callable, *args, **kwargs) -> Any:
        """Run synchronous operation in thread pool"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await loop.run_in_executor(executor, operation, *args, **kwargs)
    
    @staticmethod
    async def gather_with_concurrency(
        operations: List[callable], 
        max_concurrency: int = 5
    ) -> List[Any]:
        """Run operations with limited concurrency"""
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def run_with_semaphore(operation):
            async with semaphore:
                return await operation()
        
        tasks = [run_with_semaphore(op) for op in operations]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    @staticmethod
    async def retry_async(
        operation: callable, 
        max_retries: int = 3, 
        delay: float = 1.0,
        backoff: float = 2.0
    ) -> Any:
        """Retry async operation with exponential backoff"""
        logger = get_logger(__name__)
        
        for attempt in range(max_retries + 1):
            try:
                return await operation()
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"Operation failed after {max_retries + 1} attempts: {e}")
                    raise
                
                wait_time = delay * (backoff ** attempt)
                logger.warning(f"Operation failed (attempt {attempt + 1}), retrying in {wait_time:.1f}s: {e}")
                await asyncio.sleep(wait_time)


class EmbeddingUtils:
    """Embedding-related utilities"""
    
    @staticmethod
    def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding vector"""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    @staticmethod
    def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        try:
            # Normalize embeddings
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(emb1, emb2) / (norm1 * norm2)
            
            # Ensure result is in valid range
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    @staticmethod
    def validate_embedding(embedding: np.ndarray, expected_dim: int) -> bool:
        """Validate embedding array"""
        if embedding is None:
            return False
        
        if not isinstance(embedding, np.ndarray):
            return False
        
        if len(embedding.shape) != 1:
            return False
        
        if embedding.shape[0] != expected_dim:
            return False
        
        # Check for NaN or infinite values
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            return False
        
        return True


class HashUtils:
    """Hashing utilities"""
    
    @staticmethod
    def get_text_hash(text: str) -> str:
        """Get hash for text content"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    @staticmethod
    def get_dict_hash(data: Dict[str, Any]) -> str:
        """Get hash for dictionary content"""
        # Create a sorted, deterministic string representation
        sorted_items = sorted(data.items())
        content = json.dumps(sorted_items, sort_keys=True)
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    @staticmethod
    def get_file_content_hash(file_path: Union[str, Path]) -> str:
        """Get hash of file content"""
        file_path = Path(file_path)
        
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()


class PerformanceUtils:
    """Performance monitoring utilities"""
    
    @staticmethod
    def measure_time(operation: callable, *args, **kwargs) -> Tuple[Any, float]:
        """Measure operation execution time"""
        start_time = time.time()
        result = operation(*args, **kwargs)
        duration = time.time() - start_time
        return result, duration
    
    @staticmethod
    async def measure_time_async(operation: callable, *args, **kwargs) -> Tuple[Any, float]:
        """Measure async operation execution time"""
        start_time = time.time()
        result = await operation(*args, **kwargs)
        duration = time.time() - start_time
        return result, duration
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration for display"""
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            remaining_seconds = seconds % 60
            return f"{minutes:.0f}m {remaining_seconds:.0f}s"
        else:
            hours = seconds // 3600
            remaining_minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {remaining_minutes:.0f}m"
    
    @staticmethod
    def format_size(bytes_count: int) -> str:
        """Format byte count for display"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_count < 1024.0:
                return f"{bytes_count:.1f}{unit}"
            bytes_count /= 1024.0
        return f"{bytes_count:.1f}PB"


class LoggerUtils:
    """Logger utilities"""
    
    @staticmethod
    def log_operation_start(logger: logging.Logger, operation: str, **kwargs):
        """Log operation start with context"""
        context = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(f"Starting {operation}" + (f" ({context})" if context else ""))
    
    @staticmethod
    def log_operation_end(logger: logging.Logger, operation: str, duration: float, **kwargs):
        """Log operation completion with timing"""
        context = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        duration_str = PerformanceUtils.format_duration(duration)
        logger.info(f"Completed {operation} in {duration_str}" + (f" ({context})" if context else ""))
    
    @staticmethod
    def log_operation_error(logger: logging.Logger, operation: str, error: Exception, **kwargs):
        """Log operation error with context"""
        context = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.error(f"Failed {operation}: {error}" + (f" ({context})" if context else ""))


# Convenience functions for common operations
def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists"""
    return FileUtils.ensure_directory(path)


def clear_gpu_cache():
    """Clear GPU cache"""
    GPUUtils.clear_gpu_cache()


def clean_text(text: str) -> str:
    """Clean text"""
    return TextUtils.clean_text(text)


def get_text_hash(text: str) -> str:
    """Get text hash"""
    return HashUtils.get_text_hash(text)


async def run_in_thread(operation: callable, *args, **kwargs) -> Any:
    """Run operation in thread"""
    return await AsyncUtils.run_in_thread(operation, *args, **kwargs) 