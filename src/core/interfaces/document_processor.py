"""
Document Processor Interface

Defines the protocol for document processing components.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, List, Optional, Tuple
from pathlib import Path
import asyncio


class DocumentProcessorProtocol(Protocol):
    """Protocol for document processing components"""
    
    @abstractmethod
    async def process_document(
        self, 
        file_path: Path, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Process a document and return chunks with metadata.
        
        Args:
            file_path: Path to the document file
            metadata: Optional metadata for the document
            
        Returns:
            Tuple of (chunks, processing_metadata)
            - chunks: List of document chunks with text and metadata
            - processing_metadata: Information about the processing (stats, extracted images, etc.)
        """
        ...
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions (e.g., ['.pdf', '.docx'])
        """
        ...
    
    @abstractmethod
    def get_processor_info(self) -> Dict[str, Any]:
        """
        Get information about the processor.
        
        Returns:
            Dictionary with processor name, version, capabilities, etc.
        """
        ...
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """
        Check if the processor is properly initialized.
        
        Returns:
            True if initialized, False otherwise
        """
        ...
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        ...


class BaseDocumentProcessor(ABC):
    """Base class for document processors implementing the protocol"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
        self._stats = {
            "documents_processed": 0,
            "total_chunks": 0,
            "total_processing_time": 0.0,
            "errors": 0
        }
    
    @abstractmethod
    async def _process_document_impl(
        self, 
        file_path: Path, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Implementation-specific document processing logic"""
        ...
    
    @abstractmethod
    def _get_supported_formats_impl(self) -> List[str]:
        """Implementation-specific supported formats"""
        ...
    
    @abstractmethod
    async def _initialize_impl(self) -> bool:
        """Implementation-specific initialization logic"""
        ...
    
    async def process_document(
        self, 
        file_path: Path, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Process a document and return chunks with metadata"""
        if not self._initialized:
            await self._initialize()
        
        import time
        start_time = time.time()
        
        try:
            chunks, processing_metadata = await self._process_document_impl(file_path, metadata)
            
            # Update stats
            self._stats["documents_processed"] += 1
            self._stats["total_chunks"] += len(chunks)
            self._stats["total_processing_time"] += time.time() - start_time
            
            return chunks, processing_metadata
            
        except Exception as e:
            self._stats["errors"] += 1
            raise
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return self._get_supported_formats_impl()
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get information about the processor"""
        return {
            "name": self.__class__.__name__,
            "version": "1.0.0",
            "supported_formats": self.get_supported_formats(),
            "initialized": self._initialized,
            "config": self.config
        }
    
    def is_initialized(self) -> bool:
        """Check if the processor is properly initialized"""
        return self._initialized
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self._stats.copy()
    
    async def _initialize(self) -> bool:
        """Initialize the processor"""
        if not self._initialized:
            self._initialized = await self._initialize_impl()
        return self._initialized 