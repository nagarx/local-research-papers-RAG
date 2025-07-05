"""
Ollama Client - Local LLM Integration
"""

import json
import time
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

import requests
import aiohttp

from ..config import get_config, get_logger


class OllamaClient:
    """Client for interacting with Ollama local LLM server"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Ollama client"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Connection settings
        self.base_url = self.config.ollama.base_url
        self.model = self.config.ollama.model
        self.timeout = self.config.ollama.timeout
        
        # Default generation parameters
        self.default_params = {
            "temperature": self.config.ollama.temperature,
            "max_tokens": self.config.ollama.max_tokens,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
        }
        
        # Validate connection
        self._validate_connection()
        
        self.logger.info(f"OllamaClient initialized for model: {self.model}")
    
    def _validate_connection(self):
        """Validate connection to Ollama server"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m["name"] for m in models]
                
                if self.model in model_names:
                    self.logger.info(f"✅ Connected to Ollama, model {self.model} available")
                else:
                    self.logger.warning(f"⚠️  Model {self.model} not found. Available: {model_names}")
            else:
                self.logger.error(f"❌ Ollama server error: {response.status_code}")
                
        except requests.RequestException as e:
            self.logger.error(f"❌ Cannot connect to Ollama server: {e}")
            self.logger.error(f"Make sure Ollama is running at {self.base_url}")
    
    def _create_system_prompt(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Create system prompt with context for RAG"""
        
        context_text = ""
        for i, chunk in enumerate(context_chunks):
            filename = chunk.get("document_filename", "Unknown")
            page_num = chunk.get("page_number", "?")
            text = chunk.get("text", "")
            
            context_text += f"\n[Source {i+1}: {filename}, page {page_num}]\n{text}\n"
        
        system_prompt = f"""You are an intelligent research assistant helping users understand academic papers. You have access to the following document excerpts to answer questions:

{context_text}

Instructions:
1. Answer questions based ONLY on the provided context
2. If information is not in the context, clearly state "I don't have this information in the provided documents"
3. Always cite your sources using the format [Source X] where X is the source number
4. Be precise and concise in your answers
5. When referencing specific information, include the document name and page number
6. If multiple sources support your answer, cite all relevant sources

Remember: Your role is to help users understand and extract insights from their research papers with accurate citations."""

        return system_prompt
    
    def _format_chat_messages(
        self, 
        user_query: str, 
        context_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Format messages for chat completion"""
        
        messages = []
        
        # System message with context
        system_prompt = self._create_system_prompt(context_chunks)
        messages.append({
            "role": "system",
            "content": system_prompt
        })
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current user query
        messages.append({
            "role": "user", 
            "content": user_query
        })
        
        return messages
    
    async def generate_response_async(
        self,
        user_query: str,
        context_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        generation_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response asynchronously"""
        
        start_time = time.time()
        
        try:
            # Prepare request parameters
            params = self.default_params.copy()
            if generation_params:
                params.update(generation_params)
            
            # Format messages
            messages = self._format_chat_messages(
                user_query, 
                context_chunks, 
                conversation_history
            )
            
            # Prepare request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": params
            }
            
            # Make async request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Extract response
                        assistant_message = result.get("message", {}).get("content", "")
                        
                        # Calculate metrics
                        response_time = time.time() - start_time
                        
                        # Prepare response with metadata
                        response_data = {
                            "response": assistant_message,
                            "model": self.model,
                            "context_chunks": context_chunks,
                            "generation_params": params,
                            "metadata": {
                                "response_time": response_time,
                                "timestamp": datetime.utcnow().isoformat(),
                                "sources_used": len(context_chunks)
                            }
                        }
                        
                        self.logger.debug(f"Generated response in {response_time:.2f}s")
                        return response_data
                    
                    else:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error {response.status}: {error_text}")
        
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            
            # Return error response
            return {
                "response": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "error": True,
                "error_message": str(e),
                "model": self.model,
                "context_chunks": context_chunks,
                "metadata": {
                    "response_time": time.time() - start_time,
                    "timestamp": datetime.utcnow().isoformat(),
                    "sources_used": len(context_chunks)
                }
            }
    
    def generate_response_sync(
        self,
        user_query: str,
        context_chunks: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        generation_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate response synchronously"""
        
        start_time = time.time()
        
        try:
            # Prepare request parameters
            params = self.default_params.copy()
            if generation_params:
                params.update(generation_params)
            
            # Format messages
            messages = self._format_chat_messages(
                user_query, 
                context_chunks, 
                conversation_history
            )
            
            # Prepare request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": params
            }
            
            # Make request
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract response
                assistant_message = result.get("message", {}).get("content", "")
                
                # Calculate metrics
                response_time = time.time() - start_time
                
                return {
                    "response": assistant_message,
                    "model": self.model,
                    "context_chunks": context_chunks,
                    "generation_params": params,
                    "metadata": {
                        "response_time": response_time,
                        "timestamp": datetime.utcnow().isoformat(),
                        "sources_used": len(context_chunks)
                    }
                }
            
            else:
                raise Exception(f"Ollama API error {response.status_code}: {response.text}")
        
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "error": True,
                "error_message": str(e),
                "model": self.model,
                "context_chunks": context_chunks,
                "metadata": {
                    "response_time": time.time() - start_time,
                    "timestamp": datetime.utcnow().isoformat(),
                    "sources_used": len(context_chunks)
                }
            }
    
    def test_connection(self) -> bool:
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
