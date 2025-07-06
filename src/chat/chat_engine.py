"""
Chat Engine - Handles RAG-based conversations with documents

This module provides a conversational interface for querying processed documents.
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..config import get_config, get_logger
from ..storage import ChromaVectorStore
from ..llm import OllamaClient
from ..core.interfaces.embedding_provider import EmbeddingProvider
from ..core.interfaces.chat_engine import ChatEngineProtocol


class ChatEngine(ChatEngineProtocol):
    """Main chat engine for RAG conversations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the chat engine"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.vector_store = ChromaVectorStore()
        self.llm_client = OllamaClient()
        self.embedding_provider = None  # Will be set when needed
        
        # Chat state
        self.conversation_history = []
        self.current_context = []
        
        self.logger.info("ChatEngine initialized successfully")
    
    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize vector store
            await self.vector_store.initialize()
            
            # Initialize LLM client
            await self.llm_client.initialize()
            
            # Initialize embedding provider if needed
            if self.embedding_provider:
                await self.embedding_provider.initialize()
            
            self.logger.info("ChatEngine components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChatEngine: {e}")
            raise
    
    async def process_query(
        self, 
        query: str, 
        top_k: int = 5,
        similarity_threshold: float = None,
        document_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a user query and return response with sources"""
        try:
            # Generate query embedding
            if not self.embedding_provider:
                # Load embedding provider on demand
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(self.config.embedding.model)
                query_embedding = model.encode([query])[0]
            else:
                query_embedding = await self.embedding_provider.embed_query(query)
            
            # Search vector store
            search_results = await self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                filters=document_filters
            )
            
            # Prepare context for LLM
            context_chunks = []
            sources = []
            
            for result in search_results:
                context_chunks.append(result.metadata.get("text", ""))
                sources.append({
                    "document_id": result.metadata.get("document_id", ""),
                    "filename": result.metadata.get("filename", ""),
                    "page_number": result.metadata.get("page_number", 1),
                    "chunk_id": result.chunk_id,
                    "similarity": result.similarity
                })
            
            # Generate response using LLM
            if context_chunks:
                context = "\n\n".join(context_chunks)
                response = await self.llm_client.generate_response(
                    query=query,
                    context=context,
                    conversation_history=self.conversation_history[-5:]  # Last 5 exchanges
                )
            else:
                response = "I couldn't find relevant information in the uploaded documents to answer your question."
            
            # Update conversation history
            self.conversation_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "response": response,
                "sources": sources
            })
            
            return {
                "response": response,
                "sources": sources,
                "context_used": len(context_chunks),
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                "response": f"I encountered an error while processing your query: {str(e)}",
                "sources": [],
                "context_used": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    async def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history"""
        return self.conversation_history.copy()
    
    async def clear_conversation_history(self):
        """Clear the conversation history"""
        self.conversation_history.clear()
        self.current_context.clear()
        self.logger.info("Conversation history cleared")
    
    async def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about available documents"""
        try:
            return self.vector_store.get_stats()
        except Exception as e:
            self.logger.error(f"Error getting document stats: {e}")
            return {"error": str(e)}
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all available documents"""
        try:
            return await self.vector_store.list_documents()
        except Exception as e:
            self.logger.error(f"Error listing documents: {e}")
            return []
    
    async def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific document"""
        try:
            return await self.vector_store.get_document(document_id)
        except Exception as e:
            self.logger.error(f"Error getting document info: {e}")
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on all components"""
        try:
            health_status = {
                "vector_store": "unknown",
                "llm_client": "unknown",
                "embedding_provider": "unknown",
                "overall": "unknown",
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check vector store
            try:
                stats = self.vector_store.get_stats()
                health_status["vector_store"] = "healthy" if stats else "error"
            except Exception as e:
                health_status["vector_store"] = f"error: {str(e)}"
            
            # Check LLM client
            try:
                llm_status = await self.llm_client.health_check()
                health_status["llm_client"] = "healthy" if llm_status.get("status") == "ok" else "error"
            except Exception as e:
                health_status["llm_client"] = f"error: {str(e)}"
            
            # Check embedding provider
            if self.embedding_provider:
                try:
                    embedding_status = await self.embedding_provider.health_check()
                    health_status["embedding_provider"] = "healthy" if embedding_status else "error"
                except Exception as e:
                    health_status["embedding_provider"] = f"error: {str(e)}"
            else:
                health_status["embedding_provider"] = "not_initialized"
            
            # Overall status
            component_statuses = [
                health_status["vector_store"],
                health_status["llm_client"]
            ]
            
            if all(status == "healthy" for status in component_statuses):
                health_status["overall"] = "healthy"
            elif any("error" in status for status in component_statuses):
                health_status["overall"] = "error"
            else:
                health_status["overall"] = "partial"
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Error during health check: {e}")
            return {
                "overall": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
