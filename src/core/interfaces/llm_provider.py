"""
LLM Provider Interface

Defines the protocol for LLM provider components.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, List, Optional, AsyncIterator
from dataclasses import dataclass


@dataclass
class Message:
    """Message for LLM conversation"""
    role: str  # "user", "assistant", "system"
    content: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LLMResponse:
    """Response from LLM provider"""
    content: str
    metadata: Dict[str, Any]
    usage: Optional[Dict[str, int]] = None


class LLMProviderProtocol(Protocol):
    """Protocol for LLM provider components"""
    
    @abstractmethod
    async def chat(
        self, 
        messages: List[Message],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate a chat response from the LLM.
        
        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Returns:
            LLMResponse object
        """
        ...
    
    @abstractmethod
    async def chat_stream(
        self, 
        messages: List[Message],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[str]:
        """
        Generate a streaming chat response from the LLM.
        
        Args:
            messages: List of conversation messages
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            
        Yields:
            String chunks of the response
        """
        ...
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the LLM model.
        
        Returns:
            Dictionary with model name, version, capabilities, etc.
        """
        ...
    
    @abstractmethod
    def is_initialized(self) -> bool:
        """
        Check if the LLM provider is properly initialized.
        
        Returns:
            True if initialized, False otherwise
        """
        ...
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get LLM usage statistics.
        
        Returns:
            Dictionary with statistics
        """
        ...


class BaseLLMProvider(ABC):
    """Base class for LLM providers implementing the protocol"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
        self._stats = {
            "requests_sent": 0,
            "total_tokens": 0,
            "total_response_time": 0.0,
            "errors": 0
        }
    
    @abstractmethod
    async def _chat_impl(
        self, 
        messages: List[Message],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """Implementation-specific chat logic"""
        ...
    
    @abstractmethod
    async def _chat_stream_impl(
        self, 
        messages: List[Message],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[str]:
        """Implementation-specific streaming chat logic"""
        ...
    
    @abstractmethod
    def _get_model_info_impl(self) -> Dict[str, Any]:
        """Implementation-specific model information"""
        ...
    
    @abstractmethod
    async def _initialize_impl(self) -> bool:
        """Implementation-specific initialization logic"""
        ...
    
    async def chat(
        self, 
        messages: List[Message],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """Generate a chat response from the LLM"""
        if not self._initialized:
            await self._initialize()
        
        import time
        start_time = time.time()
        
        try:
            response = await self._chat_impl(messages, system_prompt, temperature, max_tokens)
            
            # Update stats
            self._stats["requests_sent"] += 1
            self._stats["total_response_time"] += time.time() - start_time
            if response.usage:
                self._stats["total_tokens"] += response.usage.get("total_tokens", 0)
            
            return response
            
        except Exception as e:
            self._stats["errors"] += 1
            raise
    
    async def chat_stream(
        self, 
        messages: List[Message],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncIterator[str]:
        """Generate a streaming chat response from the LLM"""
        if not self._initialized:
            await self._initialize()
        
        self._stats["requests_sent"] += 1
        
        try:
            async for chunk in self._chat_stream_impl(messages, system_prompt, temperature, max_tokens):
                yield chunk
        except Exception as e:
            self._stats["errors"] += 1
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the LLM model"""
        info = self._get_model_info_impl()
        info.update({
            "initialized": self._initialized,
            "config": self.config
        })
        return info
    
    def is_initialized(self) -> bool:
        """Check if the LLM provider is properly initialized"""
        return self._initialized
    
    def get_stats(self) -> Dict[str, Any]:
        """Get LLM usage statistics"""
        return self._stats.copy()
    
    async def _initialize(self) -> bool:
        """Initialize the LLM provider"""
        if not self._initialized:
            self._initialized = await self._initialize_impl()
        return self._initialized 