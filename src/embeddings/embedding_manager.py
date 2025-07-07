"""
Embedding Manager - Text Embedding Generation and Management

This module handles text embedding generation using SentenceTransformers
and provides efficient batching and caching capabilities.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os

# ML dependencies
from sentence_transformers import SentenceTransformer

# Local imports
from ..config import get_config, get_logger


class EmbeddingManager:
    """
    Manages text embedding generation and caching
    
    Features:
    - Efficient batch processing
    - GPU/CPU auto-detection with torch_utils integration
    - Secure embedding caching with SHA-256
    - Progress tracking for large batches
    - Improved error handling and fallbacks
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the embedding manager"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Initialize embedding model
        self.model = None
        self.device = None
        self._setup_model()
        
        # Performance tracking
        self._stats = {
            "total_texts_embedded": 0,
            "total_batches_processed": 0,
            "total_embedding_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors_recovered": 0
        }
        
        self.logger.info("EmbeddingManager initialized successfully")
    
    def _setup_model(self):
        """Setup the sentence transformer model with torch_utils integration"""
        self.logger.info("Loading embedding model...")
        
        model_name = self.config.embedding.model
        device = self.config.embedding.device
        
        # Integrate with torch_utils for device selection
        try:
            from ..utils.torch_utils import configure_torch_for_production, is_torch_configured
            
            # Configure PyTorch if not already done
            if not is_torch_configured():
                configure_torch_for_production()
                
        except ImportError:
            self.logger.warning("torch_utils not available, using fallback device selection")
        
        # Device selection with improved logic
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                    self.logger.info("Using GPU for embeddings")
                else:
                    self.device = "cpu"
                    self.logger.info("Using CPU for embeddings")
            except ImportError:
                self.device = "cpu"
                self.logger.info("PyTorch not available, using CPU for embeddings")
        else:
            self.device = device
            self.logger.info(f"Using specified device: {device}")
        
        try:
            start_time = time.time()
            self.model = SentenceTransformer(model_name, device=self.device)
            load_time = time.time() - start_time
            
            # Get model info
            embedding_dim = self.model.get_sentence_embedding_dimension()
            max_seq_length = self.model.max_seq_length
            
            self.logger.info(
                f"Loaded model '{model_name}' in {load_time:.2f}s "
                f"(dimension: {embedding_dim}, max_length: {max_seq_length})"
            )
            
            # Update config with actual dimension
            self.config.embedding.dimension = embedding_dim
            
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _get_cache_key(self, text: str) -> str:
        """Generate secure cache key for text using SHA-256"""
        # Use SHA-256 instead of MD5 for better collision resistance
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path for embedding"""
        return self.config.storage_paths.embeddings_dir / f"{cache_key}.npy"
    
    def _load_from_cache(self, text: str) -> Optional[np.ndarray]:
        """Load embedding from cache with validation"""
        cache_key = self._get_cache_key(text)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                embedding = np.load(cache_path)
                
                # Validate cached embedding dimensions
                if embedding.shape[0] != self.config.embedding.dimension:
                    self.logger.warning(
                        f"Cached embedding dimension mismatch: expected {self.config.embedding.dimension}, "
                        f"got {embedding.shape[0]}. Removing invalid cache."
                    )
                    cache_path.unlink(missing_ok=True)
                    self._stats["cache_misses"] += 1
                    return None
                
                self._stats["cache_hits"] += 1
                return embedding
                
            except Exception as e:
                self.logger.warning(f"Failed to load cached embedding: {e}")
                # Remove corrupted cache file
                cache_path.unlink(missing_ok=True)
        
        self._stats["cache_misses"] += 1
        return None
    
    def _save_to_cache(self, text: str, embedding: np.ndarray):
        """Save embedding to cache with error handling"""
        cache_key = self._get_cache_key(text)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            # Ensure directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Validate embedding before saving
            if not self._validate_embedding_array(embedding):
                self.logger.warning("Attempted to cache invalid embedding, skipping")
                return
            
            # Save embedding atomically (write to temp file first)
            # Note: np.save() automatically adds .npy extension, so we need to account for that
            temp_path = cache_path.with_suffix('.tmp.npy')
            np.save(temp_path.with_suffix(''), embedding)  # Remove .npy since np.save adds it
            temp_path.rename(cache_path)
            
        except Exception as e:
            self.logger.warning(f"Failed to cache embedding: {e}")
            # Clean up temp file if it exists
            temp_path = cache_path.with_suffix('.tmp.npy')
            temp_path.unlink(missing_ok=True)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding with improved handling"""
        if not text or not text.strip():
            return ""
        
        # Basic cleanup
        text = text.strip()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (more conservative estimate)
        max_chars = self.model.max_seq_length * 3  # More conservative character estimate
        if len(text) > max_chars:
            self.logger.debug(f"Truncating text from {len(text)} to {max_chars} characters")
            text = text[:max_chars]
        
        return text
    
    def _validate_embedding_array(self, embedding: np.ndarray) -> bool:
        """Validate a single embedding array"""
        if embedding is None:
            return False
        
        if not isinstance(embedding, np.ndarray):
            return False
        
        if len(embedding.shape) != 1:
            self.logger.error(f"Embedding should be 1D, got shape {embedding.shape}")
            return False
        
        if embedding.shape[0] != self.config.embedding.dimension:
            self.logger.error(
                f"Embedding dimension mismatch: expected {self.config.embedding.dimension}, "
                f"got {embedding.shape[0]}"
            )
            return False
        
        # Check for NaN or infinite values
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            self.logger.error("Embedding contains NaN or infinite values")
            return False
        
        return True
    
    def _validate_embeddings(self, embeddings: np.ndarray) -> bool:
        """Validate generated embeddings batch"""
        if embeddings is None:
            return False
        
        if not isinstance(embeddings, np.ndarray):
            return False
        
        if len(embeddings.shape) != 2:
            self.logger.error(f"Embeddings should be 2D, got shape {embeddings.shape}")
            return False
        
        if embeddings.shape[1] != self.config.embedding.dimension:
            self.logger.error(
                f"Embedding dimension mismatch: expected {self.config.embedding.dimension}, "
                f"got {embeddings.shape[1]}"
            )
            return False
        
        # Check for NaN or infinite values
        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
            self.logger.error("Embeddings contain NaN or infinite values")
            return False
        
        return True
    
    async def embed_text_async(self, text: str) -> np.ndarray:
        """Asynchronously embed a single text with improved error handling"""
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.config.embedding.dimension, dtype=np.float32)
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        if not processed_text:
            return np.zeros(self.config.embedding.dimension, dtype=np.float32)
        
        # Check cache first
        cached_embedding = self._load_from_cache(processed_text)
        if cached_embedding is not None:
            return cached_embedding
        
        # Generate embedding in thread pool with better resource management
        max_workers = min(2, os.cpu_count() or 1)  # Better thread management
        loop = asyncio.get_event_loop()
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                embedding = await loop.run_in_executor(
                    executor, 
                    self._generate_embedding, 
                    processed_text
                )
        except Exception as e:
            self.logger.error(f"Failed to generate embedding in thread pool: {e}")
            self._stats["errors_recovered"] += 1
            return np.zeros(self.config.embedding.dimension, dtype=np.float32)
        
        # Validate and cache the result
        if self._validate_embedding_array(embedding):
            self._save_to_cache(processed_text, embedding)
        else:
            self.logger.error("Generated invalid embedding, returning zero vector")
            self._stats["errors_recovered"] += 1
            return np.zeros(self.config.embedding.dimension, dtype=np.float32)
        
        return embedding
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text (synchronous) with better error handling"""
        try:
            start_time = time.time()
            
            # Generate embedding
            embedding = self.model.encode(
                text, 
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=1,
                normalize_embeddings=True  # Normalize for better similarity computation
            )
            
            # Convert to float32 for memory efficiency
            embedding = embedding.astype(np.float32)
            
            # Update stats
            self._stats["total_texts_embedded"] += 1
            self._stats["total_embedding_time"] += time.time() - start_time
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding for text: {e}")
            self._stats["errors_recovered"] += 1
            # Return zero vector as fallback
            return np.zeros(self.config.embedding.dimension, dtype=np.float32)
    
    async def embed_texts_batch_async(
        self, 
        texts: List[str],
        batch_size: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> List[np.ndarray]:
        """Asynchronously embed multiple texts with improved batching and error handling"""
        
        if not texts:
            return []
        
        if batch_size is None:
            batch_size = self.config.embedding.batch_size
        
        self.logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")
        
        # Preprocess all texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Check cache for all texts
        embeddings = []
        texts_to_process = []
        indices_to_process = []
        
        for i, text in enumerate(processed_texts):
            if not text:
                # Empty text
                embeddings.append(np.zeros(self.config.embedding.dimension, dtype=np.float32))
            else:
                cached_embedding = self._load_from_cache(text)
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                else:
                    embeddings.append(None)  # Placeholder
                    texts_to_process.append(text)
                    indices_to_process.append(i)
        
        # Process remaining texts in batches
        if texts_to_process:
            # Create batches
            total_batches = (len(texts_to_process) + batch_size - 1) // batch_size
            processed_embeddings = []
            
            # Better thread management for batch processing
            max_workers = min(1, os.cpu_count() or 1)  # Conservative for batch processing
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(texts_to_process))
                batch_texts = texts_to_process[start_idx:end_idx]
                
                # Process batch in thread pool
                loop = asyncio.get_event_loop()
                try:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        batch_embeddings = await loop.run_in_executor(
                            executor,
                            self._generate_batch_embeddings,
                            batch_texts
                        )
                    
                    processed_embeddings.extend(batch_embeddings)
                    
                    # Cache individual embeddings (only valid ones)
                    for text, embedding in zip(batch_texts, batch_embeddings):
                        if self._validate_embedding_array(embedding):
                            self._save_to_cache(text, embedding)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process batch {batch_idx + 1}: {e}")
                    # Create fallback embeddings for this batch
                    fallback_embeddings = [
                        np.zeros(self.config.embedding.dimension, dtype=np.float32)
                        for _ in batch_texts
                    ]
                    processed_embeddings.extend(fallback_embeddings)
                    self._stats["errors_recovered"] += len(batch_texts)
                
                # Progress callback
                if progress_callback:
                    progress_callback(batch_idx + 1, total_batches)
                
                # Update stats
                self._stats["total_batches_processed"] += 1
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)
            
            # Fill in processed embeddings
            for i, embedding in enumerate(processed_embeddings):
                original_idx = indices_to_process[i]
                embeddings[original_idx] = embedding
        
        self.logger.info(f"Completed embedding {len(texts)} texts")
        return embeddings
    
    def _generate_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for a batch of texts with improved error handling"""
        try:
            start_time = time.time()
            
            # Generate batch embeddings
            batch_embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=len(texts),
                normalize_embeddings=True  # Normalize for better similarity computation
            )
            
            # Validate batch embeddings
            if not self._validate_embeddings(batch_embeddings):
                self.logger.error("Generated invalid batch embeddings, using fallback")
                return [
                    np.zeros(self.config.embedding.dimension, dtype=np.float32)
                    for _ in texts
                ]
            
            # Convert to float32 and split into individual embeddings
            embeddings = []
            for i in range(len(texts)):
                embedding = batch_embeddings[i].astype(np.float32)
                embeddings.append(embedding)
            
            # Update stats
            self._stats["total_texts_embedded"] += len(texts)
            self._stats["total_embedding_time"] += time.time() - start_time
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to generate batch embeddings: {e}")
            self._stats["errors_recovered"] += len(texts)
            # Return zero vectors as fallback
            return [
                np.zeros(self.config.embedding.dimension, dtype=np.float32)
                for _ in texts
            ]
    
    def compute_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings with improved handling"""
        try:
            # Validate inputs
            if not self._validate_embedding_array(embedding1) or not self._validate_embedding_array(embedding2):
                self.logger.warning("Invalid embeddings provided for similarity computation")
                return 0.0
            
            # Normalize embeddings (they should already be normalized, but ensure it)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Compute cosine similarity
            similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
            
            # Ensure result is in valid range
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def find_most_similar(
        self, 
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """Find most similar embeddings to query with validation"""
        
        if not candidate_embeddings:
            return []
        
        if not self._validate_embedding_array(query_embedding):
            self.logger.error("Invalid query embedding provided")
            return []
        
        similarities = []
        
        for i, candidate_embedding in enumerate(candidate_embeddings):
            if self._validate_embedding_array(candidate_embedding):
                similarity = self.compute_similarity(query_embedding, candidate_embedding)
                similarities.append((i, similarity))
            else:
                self.logger.warning(f"Invalid candidate embedding at index {i}, skipping")
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return similarities[:top_k]
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics"""
        stats = self._stats.copy()
        
        # Calculate derived metrics
        if stats["total_texts_embedded"] > 0:
            stats["average_embedding_time"] = stats["total_embedding_time"] / stats["total_texts_embedded"]
        else:
            stats["average_embedding_time"] = 0.0
        
        if stats["cache_hits"] + stats["cache_misses"] > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / (stats["cache_hits"] + stats["cache_misses"])
        else:
            stats["cache_hit_rate"] = 0.0
        
        # Model info
        stats["model_info"] = {
            "model_name": self.config.embedding.model,
            "embedding_dimension": self.config.embedding.dimension,
            "device": self.device,
            "max_sequence_length": self.model.max_seq_length if self.model else 0
        }
        
        return stats
    
    def clear_cache(self):
        """Clear embedding cache with improved handling"""
        cache_dir = self.config.storage_paths.embeddings_dir
        
        if cache_dir.exists():
            cache_files = list(cache_dir.glob("*.npy"))
            removed_count = 0
            
            for cache_file in cache_files:
                try:
                    cache_file.unlink()
                    removed_count += 1
                except Exception as e:
                    self.logger.error(f"Error deleting cache file {cache_file}: {e}")
            
            self.logger.info(f"Cleared {removed_count}/{len(cache_files)} cached embeddings")
        
        # Reset cache statistics
        self._stats["cache_hits"] = 0
        self._stats["cache_misses"] = 0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self.model:
            return {"error": "No model loaded"}
        
        return {
            "model_name": self.config.embedding.model,
            "embedding_dimension": self.config.embedding.dimension,
            "max_sequence_length": self.model.max_seq_length,
            "device": self.device,
            "model_size": self.model.get_max_seq_length(),
            "tokenizer_info": {
                "vocab_size": getattr(self.model.tokenizer, 'vocab_size', 'unknown'),
                "model_max_length": getattr(self.model.tokenizer, 'model_max_length', 'unknown')
            }
        } 