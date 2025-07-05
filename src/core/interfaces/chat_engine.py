"""
Chat Engine Interface

Defines the protocol for chat engine components.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, List, Optional, AsyncIterator
from .llm_provider import Message, LLMResponse


class ChatEngineProtocol(Protocol):
    """Protocol for chat engine components"""
    
    @abstractmethod
    async def process_query(
        self, 
        query: str,
        chat_history: Optional[List[Message]] = None,
        context_filter: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """
        Process a query and generate a response using RAG.
        
        Args:
            query: User query
            chat_history: Optional chat history
            context_filter: Optional filters for context retrieval
            
        Returns:
            LLMResponse object
        """
        ...
    
    @abstractmethod
    async def process_query_stream(
        self, 
        query: str,
        chat_history: Optional[List[Message]] = None,
        context_filter: Optional[Dict[str, Any]] = None
    ) -> AsyncIterator[str]:
        """
        Process a query and generate a streaming response using RAG.
        
        Args:
            query: User query
            chat_history: Optional chat history
            context_filter: Optional filters for context retrieval
            
        Yields:
            String chunks of the response
        """
        ...
    
    @abstractmethod
    async def add_documents(
        self, 
        file_paths: List[str],
        callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Add documents to the knowledge base.
        
        Args:
            file_paths: List of file paths to add
            callback: Optional progress callback
            
        Returns:
            Dictionary with processing results
        """
        ...
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get chat engine statistics.
        
        Returns:
            Dictionary with statistics
        """
        ...
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """
        Check if the chat engine is properly initialized.
        
        Returns:
            True if initialized, False otherwise
        """
        ... 