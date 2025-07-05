"""
Vector Store Interface

Defines the protocol for vector storage components.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, List, Optional, Tuple
import numpy as np


class SearchResult:
    """Result from vector similarity search"""
    
    def __init__(
        self, 
        chunk_id: str, 
        similarity: float, 
        metadata: Dict[str, Any]
    ):
        self.chunk_id = chunk_id
        self.similarity = similarity
        self.metadata = metadata
    
    def __repr__(self):
        return f"SearchResult(chunk_id={self.chunk_id}, similarity={self.similarity:.3f})"


class VectorStoreProtocol(Protocol):
    """Protocol for vector storage components"""
    
    @abstractmethod
    async def add_document(
        self, 
        document_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[np.ndarray],
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Add a document with its chunks and embeddings to the store.
        
        Args:
            document_id: Unique identifier for the document
            chunks: List of document chunks with text and metadata
            embeddings: List of embedding vectors for the chunks
            metadata: Document-level metadata
            
        Returns:
            True if successful, False otherwise
        """
        ...
    
    @abstractmethod
    async def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            filters: Optional filters for search results
            
        Returns:
            List of SearchResult objects
        """
        ...
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks from the store.
        
        Args:
            document_id: Unique identifier for the document
            
        Returns:
            True if successful, False otherwise
        """
        ...
    
    @abstractmethod
    async def list_documents(self) -> List[Dict[str, Any]]:
        """
        List all documents in the store.
        
        Returns:
            List of document metadata dictionaries
        """
        ...
    
    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific document.
        
        Args:
            document_id: Unique identifier for the document
            
        Returns:
            Document metadata dictionary or None if not found
        """
        ...
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Dictionary with statistics
        """
        ...
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """
        Check if the vector store is properly initialized.
        
        Returns:
            True if initialized, False otherwise
        """
        ...


class BaseVectorStore(ABC):
    """Base class for vector stores implementing the protocol"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
        self._stats = {
            "documents_added": 0,
            "total_chunks": 0,
            "total_searches": 0,
            "total_search_time": 0.0,
            "errors": 0
        }
    
    @abstractmethod
    async def _add_document_impl(
        self, 
        document_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[np.ndarray],
        metadata: Dict[str, Any]
    ) -> bool:
        """Implementation-specific document addition logic"""
        ...
    
    @abstractmethod
    async def _search_impl(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Implementation-specific search logic"""
        ...
    
    @abstractmethod
    async def _delete_document_impl(self, document_id: str) -> bool:
        """Implementation-specific document deletion logic"""
        ...
    
    @abstractmethod
    async def _list_documents_impl(self) -> List[Dict[str, Any]]:
        """Implementation-specific document listing logic"""
        ...
    
    @abstractmethod
    async def _get_document_impl(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Implementation-specific document retrieval logic"""
        ...
    
    @abstractmethod
    async def _initialize_impl(self) -> bool:
        """Implementation-specific initialization logic"""
        ...
    
    async def add_document(
        self, 
        document_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[np.ndarray],
        metadata: Dict[str, Any]
    ) -> bool:
        """Add a document with its chunks and embeddings to the store"""
        if not self._initialized:
            await self._initialize()
        
        try:
            result = await self._add_document_impl(document_id, chunks, embeddings, metadata)
            
            if result:
                self._stats["documents_added"] += 1
                self._stats["total_chunks"] += len(chunks)
            
            return result
            
        except Exception as e:
            self._stats["errors"] += 1
            raise
    
    async def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar chunks using vector similarity"""
        if not self._initialized:
            await self._initialize()
        
        import time
        start_time = time.time()
        
        try:
            results = await self._search_impl(query_embedding, top_k, similarity_threshold, filters)
            
            # Update stats
            self._stats["total_searches"] += 1
            self._stats["total_search_time"] += time.time() - start_time
            
            return results
            
        except Exception as e:
            self._stats["errors"] += 1
            raise
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks from the store"""
        if not self._initialized:
            await self._initialize()
        
        try:
            return await self._delete_document_impl(document_id)
        except Exception as e:
            self._stats["errors"] += 1
            raise
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the store"""
        if not self._initialized:
            await self._initialize()
        
        return await self._list_documents_impl()
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific document"""
        if not self._initialized:
            await self._initialize()
        
        return await self._get_document_impl(document_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return self._stats.copy()
    
    def is_initialized(self) -> bool:
        """Check if the vector store is properly initialized"""
        return self._initialized
    
    async def _initialize(self) -> bool:
        """Initialize the vector store"""
        if not self._initialized:
            self._initialized = await self._initialize_impl()
        return self._initialized 