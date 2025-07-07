"""
Chat Engine - RAG System Orchestrator
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..config import get_config, get_logger
from ..ingestion import DocumentProcessor
from ..embeddings import EmbeddingManager
from ..storage import ChromaVectorStore
from ..llm import OllamaClient
from ..tracking import SourceTracker, SourceReference


class ChatEngine:
    """Complete RAG system orchestrating document processing, embedding, and chat"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the chat engine"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Initialize core components
        self.document_processor = DocumentProcessor()
        self.embedding_manager = EmbeddingManager()
        self.vector_store = ChromaVectorStore()
        self.ollama_client = OllamaClient()
        self.source_tracker = SourceTracker()
        
        # Re-register existing documents with source tracker
        try:
            registered_count = self.vector_store.re_register_existing_documents(self.source_tracker)
            if registered_count > 0:
                self.logger.info(f"Re-registered {registered_count} existing documents")
        except Exception as e:
            self.logger.warning(f"Failed to re-register existing documents: {e}")
        
        # Conversation state
        self.conversation_history = []
        self.max_history_length = 10
        
        self.logger.info("ChatEngine initialized successfully")
    
    async def add_documents_async(
        self, 
        file_paths: List[str],
        progress_callback: Optional[callable] = None,
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """Add multiple documents to the RAG system with duplicate detection"""
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing {len(file_paths)} documents...")
            
            # Process documents with Marker
            if progress_callback:
                progress_callback("Checking for existing documents...", 0, len(file_paths))
            
            processed_docs = []
            skipped_docs = []
            
            for i, file_path in enumerate(file_paths):
                try:
                    if progress_callback:
                        progress_callback(f"Processing {file_path}...", i + 1, len(file_paths))
                    
                    processed_doc = await self.document_processor.process_document_async(
                        file_path, 
                        force_reprocess=force_reprocess
                    )
                    
                    # Check if document was already processed
                    if processed_doc.get("status") == "already_processed":
                        skipped_docs.append(processed_doc)
                        self.logger.info(f"Skipped already processed document: {processed_doc['filename']}")
                    else:
                        processed_docs.append(processed_doc)
                        self.logger.info(f"Newly processed document: {processed_doc['filename']}")
                        
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {e}")
                    continue
            
            if not processed_docs and not skipped_docs:
                return {
                    "success": False,
                    "error": "No documents were successfully processed",
                    "total_documents": 0,
                    "processing_time": time.time() - start_time
                }
            
            total_chunks = 0
            successful_docs = 0
            failed_indexing = []
            
            # Process each newly processed document for indexing
            for doc_idx, processed_doc in enumerate(processed_docs):
                try:
                    if progress_callback:
                        progress_callback(
                            f"Indexing document {doc_idx + 1}/{len(processed_docs)}...", 
                            len(skipped_docs) + doc_idx, 
                            len(file_paths)
                        )
                    
                    # Check if document is already indexed
                    existing_indexed = self.vector_store.get_document_info(processed_doc["id"])
                    if existing_indexed and not force_reprocess:
                        self.logger.info(f"Document already indexed: {processed_doc['filename']}")
                        successful_docs += 1
                        total_chunks += existing_indexed.get("total_chunks", 0)
                        continue
                    
                    # Extract text chunks
                    chunks = processed_doc["content"]["blocks"]
                    chunk_texts = [chunk["text"] for chunk in chunks if chunk["text"].strip()]
                    
                    if not chunk_texts:
                        self.logger.warning(f"No text chunks found in {processed_doc['filename']}")
                        failed_indexing.append({
                            "filename": processed_doc['filename'],
                            "reason": "No text chunks found"
                        })
                        continue
                    
                    # Generate embeddings for chunks
                    embeddings = await self.embedding_manager.embed_texts_batch_async(chunk_texts)
                    
                    # Add to vector store
                    success = self.vector_store.add_document(
                        document_id=processed_doc["id"],
                        chunks=chunks,
                        embeddings=embeddings,
                        metadata=processed_doc
                    )
                    
                    if success:
                        # Register with source tracker
                        self.source_tracker.register_document(
                            processed_doc["id"],
                            processed_doc["source_path"],
                            processed_doc
                        )
                        
                        successful_docs += 1
                        total_chunks += len(chunks)
                        
                        self.logger.info(f"Successfully indexed {processed_doc['filename']} with {len(chunks)} chunks")
                    else:
                        self.logger.error(f"Failed to add {processed_doc['filename']} to vector store")
                        failed_indexing.append({
                            "filename": processed_doc['filename'],
                            "reason": "Vector store insertion failed"
                        })
                
                except Exception as e:
                    self.logger.error(f"Error indexing document {processed_doc.get('filename', 'unknown')}: {e}")
                    failed_indexing.append({
                        "filename": processed_doc.get('filename', 'unknown'),
                        "reason": str(e)
                    })
                    continue
            
            # Handle already processed documents
            for processed_doc in skipped_docs:
                try:
                    # Check if already indexed
                    existing_indexed = self.vector_store.get_document_info(processed_doc["id"])
                    if existing_indexed:
                        successful_docs += 1
                        total_chunks += existing_indexed.get("total_chunks", 0)
                        self.logger.info(f"Document already indexed: {processed_doc['filename']}")
                    else:
                        # Document was processed but not indexed - attempt to index it
                        self.logger.info(f"Document processed but not indexed, attempting to index: {processed_doc['filename']}")
                        
                        # Check if we have regenerated chunks from DocumentProcessor
                        chunks = processed_doc.get("content", {}).get("blocks", [])
                        if chunks:
                            # We have chunks - attempt to index them
                            chunk_texts = [chunk["text"] for chunk in chunks if chunk["text"].strip()]
                            
                            if chunk_texts:
                                try:
                                    # Generate embeddings for chunks
                                    embeddings = await self.embedding_manager.embed_texts_batch_async(chunk_texts)
                                    
                                    # Add to vector store
                                    success = self.vector_store.add_document(
                                        document_id=processed_doc["id"],
                                        chunks=chunks,
                                        embeddings=embeddings,
                                        metadata=processed_doc
                                    )
                                    
                                    if success:
                                        # Register with source tracker
                                        self.source_tracker.register_document(
                                            processed_doc["id"],
                                            processed_doc["source_path"],
                                            processed_doc
                                        )
                                        
                                        successful_docs += 1
                                        total_chunks += len(chunks)
                                        
                                        self.logger.info(f"Successfully indexed orphaned document {processed_doc['filename']} with {len(chunks)} chunks")
                                    else:
                                        self.logger.error(f"Failed to add orphaned document {processed_doc['filename']} to vector store")
                                        failed_indexing.append({
                                            "filename": processed_doc['filename'],
                                            "reason": "Vector store insertion failed for orphaned document"
                                        })
                                
                                except Exception as e:
                                    self.logger.error(f"Error indexing orphaned document {processed_doc['filename']}: {e}")
                                    failed_indexing.append({
                                        "filename": processed_doc['filename'],
                                        "reason": f"Indexing error for orphaned document: {str(e)}"
                                    })
                            else:
                                self.logger.warning(f"No valid text chunks found for orphaned document: {processed_doc['filename']}")
                                failed_indexing.append({
                                    "filename": processed_doc['filename'],
                                    "reason": "No valid text chunks found for orphaned document"
                                })
                        else:
                            # No chunks available - this is the old behavior
                            self.logger.warning(f"Document processed but not indexed (no chunks available): {processed_doc['filename']}")
                            failed_indexing.append({
                                "filename": processed_doc['filename'],
                                "reason": "Processed but not indexed (no chunks available)"
                            })
                
                except Exception as e:
                    self.logger.error(f"Error checking indexed status for {processed_doc.get('filename', 'unknown')}: {e}")
                    continue
            
            processing_time = time.time() - start_time
            
            result = {
                "success": True,
                "total_documents": successful_docs,
                "total_chunks": total_chunks,
                "processing_time": processing_time,
                "newly_processed": len(processed_docs),
                "already_processed": len(skipped_docs),
                "failed_indexing": len(failed_indexing),
                "failed_indexing_details": failed_indexing,
                "message": f"Successfully processed {successful_docs} documents with {total_chunks} chunks"
            }
            
            if progress_callback:
                progress_callback("Processing complete!", len(file_paths), len(file_paths))
            
            self.logger.info(f"Document processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in add_documents_async: {e}")
            return {
                "success": False,
                "error": str(e),
                "total_documents": 0,
                "processing_time": time.time() - start_time
            }
    
    async def query_async(
        self, 
        user_query: str,
        top_k: int = None,
        similarity_threshold: float = None,
        include_conversation_history: bool = True
    ) -> Dict[str, Any]:
        """Process a user query and generate a response"""
        
        start_time = time.time()
        
        try:
            if top_k is None:
                top_k = self.config.ui.default_query_limit
            
            if similarity_threshold is None:
                similarity_threshold = self.config.vector_storage.similarity_threshold
            
            self.logger.info(f"Processing query: {user_query[:100]}...")
            
            # Generate query embedding
            query_embedding = await self.embedding_manager.embed_text_async(user_query)
            
            # Search for relevant chunks
            search_results = self.vector_store.search(
                query_embedding,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            if not search_results:
                return {
                    "response": "I couldn't find any relevant information in the uploaded documents to answer your question.",
                    "sources": [],
                    "query": user_query,
                    "metadata": {
                        "total_sources": 0,
                        "response_time": time.time() - start_time,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
            
            # Prepare context chunks for LLM
            context_chunks = []
            source_references = []
            
            for chunk_id, similarity, chunk_metadata in search_results:
                context_chunks.append(chunk_metadata)
                
                # Create source reference
                source_ref = self.source_tracker.create_source_reference(
                    document_id=chunk_metadata["document_id"],
                    page_number=chunk_metadata["page_number"],
                    block_index=chunk_metadata["chunk_index"],
                    block_type=chunk_metadata["block_type"],
                    text_snippet=chunk_metadata["text"][:200],
                    confidence_score=similarity
                )
                source_references.append(source_ref)
            
            # Prepare conversation history
            history = None
            if include_conversation_history and self.conversation_history:
                history = self.conversation_history[-self.max_history_length:]
            
            # Generate LLM response
            llm_response = await self.ollama_client.generate_response_async(
                user_query=user_query,
                context_chunks=context_chunks,
                conversation_history=history
            )
            
            # Format sources
            formatted_sources = self._format_sources(source_references)
            
            # Update conversation history
            if include_conversation_history:
                self.conversation_history.append({
                    "role": "user",
                    "content": user_query
                })
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": llm_response["response"]
                })
                
                # Trim history if too long
                if len(self.conversation_history) > self.max_history_length * 2:
                    self.conversation_history = self.conversation_history[-self.max_history_length * 2:]
            
            # Calculate total response time
            total_response_time = time.time() - start_time
            
            # Prepare final response
            response_data = {
                "response": llm_response["response"],
                "sources": formatted_sources,
                "query": user_query,
                "context_chunks": context_chunks,
                "metadata": {
                    "total_sources": len(search_results),
                    "response_time": total_response_time,
                    "llm_response_time": llm_response["metadata"]["response_time"],
                    "similarity_scores": [score for _, score, _ in search_results],
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_used": llm_response["model"]
                }
            }
            
            # Add error information if present
            if llm_response.get("error"):
                response_data["error"] = llm_response["error"]
                response_data["error_message"] = llm_response.get("error_message")
            
            self.logger.info(f"Query processed successfully in {total_response_time:.2f}s")
            return response_data
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            
            return {
                "response": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "query": user_query,
                "error": True,
                "error_message": str(e),
                "metadata": {
                    "total_sources": 0,
                    "response_time": time.time() - start_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
    
    def _format_sources(self, source_references: List[SourceReference]) -> List[Dict[str, Any]]:
        """Format source references for display"""
        
        formatted_sources = []
        
        for source_ref in source_references:
            formatted_source = {
                "document_name": source_ref.document_name,
                "page_number": source_ref.page_number,
                "block_type": source_ref.block_type,
                "confidence_score": round(source_ref.confidence_score, 3),
                "text_snippet": source_ref.text_snippet,
                "citation": self.source_tracker.format_citation(source_ref, style="simple")
            }
            formatted_sources.append(formatted_source)
        
        return formatted_sources
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get current conversation history"""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.logger.info("Conversation history cleared")
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the system"""
        return self.vector_store.list_documents()
    
    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific document"""
        return self.vector_store.get_document_info(document_id)
    
    async def test_system_health(self) -> Dict[str, Any]:
        """Test system health and component connectivity"""
        
        health_status = {
            "overall_status": "healthy",
            "components": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Test Ollama connection
        ollama_connected = self.ollama_client.test_connection()
        health_status["components"]["ollama"] = {
            "status": "healthy" if ollama_connected else "unhealthy",
            "connected": ollama_connected,
            "model": self.config.ollama.model,
            "base_url": self.config.ollama.base_url
        }
        
        # Test embedding model
        try:
            test_embedding = await self.embedding_manager.embed_text_async("test")
            embedding_healthy = test_embedding is not None and len(test_embedding) > 0
        except Exception as e:
            embedding_healthy = False
            health_status["components"]["embedding_error"] = str(e)
        
        health_status["components"]["embedding_manager"] = {
            "status": "healthy" if embedding_healthy else "unhealthy",
            "model": self.config.embedding.model,
            "device": self.embedding_manager.device
        }
        
        # Test vector store
        vector_store_healthy = True
        try:
            stats = self.vector_store.get_stats()
            vector_store_healthy = isinstance(stats, dict)
        except Exception:
            vector_store_healthy = False
        
        health_status["components"]["vector_store"] = {
            "status": "healthy" if vector_store_healthy else "unhealthy",
            "total_documents": len(self.vector_store.list_documents()) if vector_store_healthy else 0
        }
        
        # Overall health
        component_statuses = [
            comp["status"] for comp in health_status["components"].values()
            if isinstance(comp, dict) and "status" in comp
        ]
        
        if all(status == "healthy" for status in component_statuses):
            health_status["overall_status"] = "healthy"
        elif any(status == "healthy" for status in component_statuses):
            health_status["overall_status"] = "degraded"
        else:
            health_status["overall_status"] = "unhealthy"
        
        return health_status
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "chat_engine": {
                "total_queries": len([msg for msg in self.conversation_history if msg["role"] == "user"]),
                "successful_queries": len([msg for msg in self.conversation_history if msg["role"] == "assistant"]),
                "conversation_length": len(self.conversation_history)
            },
            "document_processor": self.document_processor.get_processing_stats(),
            "embedding_manager": self.embedding_manager.get_embedding_stats(),
            "vector_store": self.vector_store.get_stats(),
            "ollama_client": {
                "model": self.config.ollama.model,
                "base_url": self.config.ollama.base_url
            }
        }
        
        return stats
