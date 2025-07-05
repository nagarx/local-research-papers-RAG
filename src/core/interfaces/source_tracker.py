"""
Source Tracker Interface

Defines the protocol for source tracking components.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, List, Optional


class SourceTrackerProtocol(Protocol):
    """Protocol for source tracking components"""
    
    @abstractmethod
    def track_source(
        self, 
        document_id: str, 
        filename: str, 
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Track a source document.
        
        Args:
            document_id: Unique identifier for the document
            filename: Original filename
            metadata: Document metadata
            
        Returns:
            True if successful, False otherwise
        """
        ...
    
    @abstractmethod
    def get_source_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get source information for a document.
        
        Args:
            document_id: Unique identifier for the document
            
        Returns:
            Source information dictionary or None if not found
        """
        ...
    
    @abstractmethod
    def list_sources(self) -> List[Dict[str, Any]]:
        """
        List all tracked sources.
        
        Returns:
            List of source information dictionaries
        """
        ...
    
    @abstractmethod
    def remove_source(self, document_id: str) -> bool:
        """
        Remove a tracked source.
        
        Args:
            document_id: Unique identifier for the document
            
        Returns:
            True if successful, False otherwise
        """
        ... 