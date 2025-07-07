"""
Ollama Client - Local LLM Integration
"""

import json
import time
import asyncio
import re
from typing import Dict, Any, List, Optional, Set
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
        available_sources = []
        
        for i, chunk in enumerate(context_chunks):
            filename = chunk.get("document_filename", "Unknown")
            page_num = chunk.get("page_number", "?")
            text = chunk.get("text", "")
            
            context_text += f"\n[Source {i+1}: {filename}, page {page_num}]\n{text}\n"
            available_sources.append(f"Source {i+1}: {filename}, page {page_num}")
        
        available_sources_list = "\n".join([f"- {source}" for source in available_sources])
        
        system_prompt = f"""You are an intelligent research assistant helping users understand academic papers. You have access to the following document excerpts to answer questions:

{context_text}

AVAILABLE SOURCES (COMPLETE LIST):
{available_sources_list}

CRITICAL CITATION RULES:
1. You can ONLY cite sources that are explicitly listed above
2. You can ONLY reference page numbers that appear in the available sources
3. NEVER invent, guess, or extrapolate page numbers or document names
4. If information is not in the provided context, clearly state "I don't have this information in the provided documents"
5. Always cite your sources using the format [Source X] where X matches the source number above
6. When referencing specific information, include the exact document name and page number as shown in the available sources
7. If multiple sources support your answer, cite all relevant sources

FORBIDDEN ACTIONS:
- Do NOT cite pages that are not explicitly provided in the available sources
- Do NOT reference sections of documents not included in the context
- Do NOT combine or extrapolate information from multiple sources to create false citations

Remember: Accuracy and truthfulness in citations is paramount. Only use the sources and page numbers explicitly provided above."""

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
                        
                        # Validate and sanitize citations
                        validation_result = self._validate_response_citations(assistant_message, context_chunks)
                        sanitized_message = self._sanitize_response(assistant_message, validation_result)
                        
                        # Calculate metrics
                        response_time = time.time() - start_time
                        
                        # Prepare response with metadata
                        response_data = {
                            "response": sanitized_message,
                            "original_response": assistant_message if not validation_result["is_valid"] else None,
                            "citation_validation": validation_result,
                            "model": self.model,
                            "context_chunks": context_chunks,
                            "generation_params": params,
                            "metadata": {
                                "response_time": response_time,
                                "timestamp": datetime.utcnow().isoformat(),
                                "sources_used": len(context_chunks),
                                "citations_valid": validation_result["is_valid"]
                            }
                        }
                        
                        if not validation_result["is_valid"]:
                            self.logger.warning("Response contained invalid citations and was sanitized")
                        
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
                
                # Validate and sanitize citations
                validation_result = self._validate_response_citations(assistant_message, context_chunks)
                sanitized_message = self._sanitize_response(assistant_message, validation_result)
                
                # Calculate metrics
                response_time = time.time() - start_time
                
                return {
                    "response": sanitized_message,
                    "original_response": assistant_message if not validation_result["is_valid"] else None,
                    "citation_validation": validation_result,
                    "model": self.model,
                    "context_chunks": context_chunks,
                    "generation_params": params,
                    "metadata": {
                        "response_time": response_time,
                        "timestamp": datetime.utcnow().isoformat(),
                        "sources_used": len(context_chunks),
                        "citations_valid": validation_result["is_valid"]
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
    
    def _validate_response_citations(
        self, 
        response_text: str, 
        context_chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate that response citations match available sources"""
        
        # Extract available page numbers and document names from context
        available_pages = set()
        available_docs = set()
        
        for chunk in context_chunks:
            filename = chunk.get("document_filename", "Unknown")
            page_num = chunk.get("page_number")
            
            if filename != "Unknown":
                available_docs.add(filename)
            if page_num and isinstance(page_num, int):
                available_pages.add(page_num)
        
        # Find page number references in response
        page_references = re.findall(r'page\s+(\d+)', response_text, re.IGNORECASE)
        page_numbers = [int(p) for p in page_references]
        
        # Find document name references (simplified check)
        doc_references = []
        for doc in available_docs:
            if doc.replace('.pdf', '') in response_text:
                doc_references.append(doc)
        
        # Check for invalid citations
        invalid_pages = [p for p in page_numbers if p not in available_pages]
        
        validation_result = {
            "is_valid": len(invalid_pages) == 0,
            "available_pages": sorted(list(available_pages)),
            "referenced_pages": page_numbers,
            "invalid_pages": invalid_pages,
            "available_docs": list(available_docs),
            "referenced_docs": doc_references
        }
        
        if invalid_pages:
            self.logger.warning(
                f"Response contains invalid page references: {invalid_pages}. "
                f"Available pages: {sorted(list(available_pages))}"
            )
        
        return validation_result
    
    def _sanitize_response(
        self, 
        response_text: str, 
        validation_result: Dict[str, Any]
    ) -> str:
        """Remove or correct invalid citations from response"""
        
        if validation_result["is_valid"]:
            return response_text
        
        sanitized_response = response_text
        
        # Remove references to invalid pages
        for invalid_page in validation_result["invalid_pages"]:
            # Pattern to match "page X" where X is invalid
            pattern = rf'\b(?:page\s+{invalid_page}|p\.\s*{invalid_page})\b'
            
            # Replace with warning about unavailable page
            replacement = f"[Page {invalid_page} not available in provided sources]"
            sanitized_response = re.sub(pattern, replacement, sanitized_response, flags=re.IGNORECASE)
        
        # Add validation warning at the end
        if validation_result["invalid_pages"]:
            available_pages_str = ", ".join(map(str, validation_result["available_pages"]))
            warning_text = (
                f"\n\n**Note**: This response originally referenced pages that are not available "
                f"in the provided document excerpts. Available pages: {available_pages_str}"
            )
            sanitized_response += warning_text
        
        return sanitized_response
