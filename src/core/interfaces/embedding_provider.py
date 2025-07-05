"""
Embedding Provider Interface

Defines the protocol for embedding generation components.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, List, Optional
import numpy as np


class EmbeddingProviderProtocol(Protocol):
    """Protocol for embedding generation components"""
    
    @abstractmethod
    async def embed_texts(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Optional batch size for processing
            
        Returns:
            List of embedding vectors as numpy arrays
        """
        ...
    
    @abstractmethod
    async def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector as numpy array
        """
        ...
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the dimensionality of the embeddings.
        
        Returns:
            Embedding dimension
        """
        ...
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary with model name, version, capabilities, etc.
        """
        ...
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """
        Check if the embedding provider is properly initialized.
        
        Returns:
            True if initialized, False otherwise
        """
        ...
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get embedding generation statistics.
        
        Returns:
            Dictionary with statistics
        """
        ...


class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers implementing the protocol"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
        self._stats = {
            "texts_embedded": 0,
            "total_embedding_time": 0.0,
            "errors": 0
        }
    
    @abstractmethod
    async def _embed_texts_impl(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None
    ) -> List[np.ndarray]:
        """Implementation-specific embedding generation logic"""
        ...
    
    @abstractmethod
    def _get_dimension_impl(self) -> int:
        """Implementation-specific dimension retrieval"""
        ...
    
    @abstractmethod
    def _get_model_info_impl(self) -> Dict[str, Any]:
        """Implementation-specific model information"""
        ...
    
    @abstractmethod
    async def _initialize_impl(self) -> bool:
        """Implementation-specific initialization logic"""
        ...
    
    async def embed_texts(
        self, 
        texts: List[str], 
        batch_size: Optional[int] = None
    ) -> List[np.ndarray]:
        """Generate embeddings for a list of texts"""
        if not self._initialized:
            await self._initialize()
        
        import time
        start_time = time.time()
        
        try:
            embeddings = await self._embed_texts_impl(texts, batch_size)
            
            # Update stats
            self._stats["texts_embedded"] += len(texts)
            self._stats["total_embedding_time"] += time.time() - start_time
            
            return embeddings
            
        except Exception as e:
            self._stats["errors"] += 1
            raise
    
    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        embeddings = await self.embed_texts([text])
        return embeddings[0]
    
    def get_dimension(self) -> int:
        """Get the dimensionality of the embeddings"""
        return self._get_dimension_impl()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the embedding model"""
        info = self._get_model_info_impl()
        info.update({
            "initialized": self._initialized,
            "config": self.config,
            "dimension": self.get_dimension() if self._initialized else None
        })
        return info
    
    def is_initialized(self) -> bool:
        """Check if the embedding provider is properly initialized"""
        return self._initialized
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding generation statistics"""
        return self._stats.copy()
    
    async def _initialize(self) -> bool:
        """Initialize the embedding provider"""
        if not self._initialized:
            self._initialized = await self._initialize_impl()
        return self._initialized 