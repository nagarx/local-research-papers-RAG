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
from ..storage import ChromaVectorStore, SessionManager
from ..llm import OllamaClient
from ..tracking import SourceTracker, SourceReference


class ChatEngine:
    """Complete RAG system orchestrating document processing, embedding, and chat"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the chat engine"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Try to get enhanced logger
        try:
            from ..utils.enhanced_logging import get_enhanced_logger
            self.enhanced_logger = get_enhanced_logger(__name__)
            startup_time = time.time()
        except ImportError:
            self.enhanced_logger = None
            startup_time = time.time()
        
        # Initialize core components with progress tracking
        if self.enhanced_logger:
            self.enhanced_logger.startup_stage("Initializing DocumentProcessor")
        self.document_processor = DocumentProcessor()
        
        if self.enhanced_logger:
            self.enhanced_logger.startup_stage("Initializing EmbeddingManager")
        self.embedding_manager = EmbeddingManager()
        
        if self.enhanced_logger:
            self.enhanced_logger.startup_stage("Initializing ChromaVectorStore")
        self.vector_store = ChromaVectorStore()
        
        if self.enhanced_logger:
            self.enhanced_logger.startup_stage("Initializing OllamaClient")
        self.ollama_client = OllamaClient()
        
        if self.enhanced_logger:
            self.enhanced_logger.startup_stage("Initializing SourceTracker")
        self.source_tracker = SourceTracker()
        
        if self.enhanced_logger:
            self.enhanced_logger.startup_stage("Initializing SessionManager")
        self.session_manager = SessionManager()
        
        # Re-register existing documents with source tracker
        try:
            if self.enhanced_logger:
                self.enhanced_logger.startup_stage("Re-registering existing documents")
            registered_count = self.vector_store.re_register_existing_documents(self.source_tracker)
            if registered_count > 0:
                if self.enhanced_logger:
                    self.enhanced_logger.system_ready("Document re-registration", f"{registered_count} documents")
                else:
                    self.logger.info(f"Re-registered {registered_count} existing documents")
        except Exception as e:
            if self.enhanced_logger:
                self.enhanced_logger.warning_clean(f"Failed to re-register existing documents: {e}")
            else:
                self.logger.warning(f"Failed to re-register existing documents: {e}")
        
        # Conversation state
        self.conversation_history = []
        self.max_history_length = 10
        
        total_startup_time = time.time() - startup_time
        
        if self.enhanced_logger:
            from ..utils.enhanced_logging import startup_complete
            startup_complete(total_startup_time)
            self.enhanced_logger.system_ready("ChatEngine", "All components initialized and ready")
        else:
            self.logger.info("ChatEngine initialized successfully")
    
    def start_session(self) -> str:
        """Start a new document session"""
        return self.session_manager.start_session()
    
    def end_session(self) -> Dict[str, Any]:
        """End the current session and clean up temporary documents"""
        return self.session_manager.end_session(self.vector_store)
    
    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current session"""
        return self.session_manager.get_session_info()
    
    def get_permanent_documents(self) -> List[Dict[str, Any]]:
        """Get list of all permanent documents"""
        return self.session_manager.get_permanent_documents()
    
    def remove_permanent_document(self, document_id: str) -> bool:
        """Remove a document from permanent storage"""
        return self.session_manager.remove_permanent_document(document_id, self.vector_store)
    
    async def add_documents_async(
        self, 
        file_paths: List[str],
        storage_types: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None,
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """
        Add multiple documents to the RAG system with duplicate detection
        
        Args:
            file_paths: List of file paths to process
            storage_types: List of storage types ('temporary' or 'permanent') for each file.
                          If None, defaults to 'temporary' for all files.
            progress_callback: Optional callback for progress updates
            force_reprocess: Force reprocessing even if document exists
        """
        
        start_time = time.time()
        
        try:
            # Validate storage types
            if storage_types is None:
                storage_types = ['temporary'] * len(file_paths)
            elif len(storage_types) != len(file_paths):
                raise ValueError("storage_types must have the same length as file_paths")
            
            # Validate storage type values
            for storage_type in storage_types:
                if storage_type not in ['temporary', 'permanent']:
                    raise ValueError(f"Invalid storage type: {storage_type}. Must be 'temporary' or 'permanent'")
            
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
                        
                        # Add to session with appropriate storage type
                        storage_type = storage_types[file_paths.index(processed_doc["source_path"])]
                        self.session_manager.add_document_to_session(
                            processed_doc["id"],
                            storage_type,
                            processed_doc
                        )
                        
                        successful_docs += 1
                        total_chunks += len(chunks)
                        
                        self.logger.info(f"Successfully indexed {processed_doc['filename']} with {len(chunks)} chunks as {storage_type}")
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
        include_conversation_history: bool = True,
        document_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Process a user query and generate a response"""
        
        start_time = time.time()
        
        try:
            if top_k is None:
                top_k = self.config.ui.default_query_limit
            
            if similarity_threshold is None:
                similarity_threshold = self.config.vector_storage.similarity_threshold
            
            # Enhanced logging for query processing
            if self.enhanced_logger:
                self.enhanced_logger.query_start(user_query)
                self.enhanced_logger.query_step("Generating query embedding")
            else:
                self.logger.info(f"Processing query: {user_query[:100]}...")
                
            # Log document filtering if applied
            if document_ids:
                self.logger.info(f"Filtering query to {len(document_ids)} specific documents")
            
            # Generate query embedding
            query_embedding = await self.embedding_manager.embed_text_async(user_query)
            
            # Prepare filters for document-specific search
            search_filters = None
            if document_ids:
                # For multiple document IDs, we need to search each one separately
                # and combine results since ChromaDB doesn't support OR operations directly
                all_search_results = []
                
                for doc_id in document_ids:
                    doc_filters = {"document_id": doc_id}
                    doc_results = self.vector_store.search(
                        query_embedding,
                        top_k=top_k,
                        similarity_threshold=similarity_threshold,
                        filters=doc_filters
                    )
                    all_search_results.extend(doc_results)
                
                # Sort combined results by similarity and take top_k
                all_search_results.sort(key=lambda x: x[1], reverse=True)
                search_results = all_search_results[:top_k]
                
                # Log filtered search info
                if self.enhanced_logger:
                    self.enhanced_logger.query_step("Searching vector database (filtered)", 
                                                  f"documents={len(document_ids)}, top_k={top_k}, threshold={similarity_threshold}")
                else:
                    self.logger.info(f"Searching {len(document_ids)} documents with top_k={top_k}")
            else:
                # Search all documents
                if self.enhanced_logger:
                    self.enhanced_logger.query_step("Searching vector database", f"top_k={top_k}, threshold={similarity_threshold}")
                
                search_results = self.vector_store.search(
                    query_embedding,
                    top_k=top_k,
                    similarity_threshold=similarity_threshold
                )
            
            if not search_results:
                filter_msg = f" in the selected {len(document_ids)} document(s)" if document_ids else ""
                return {
                    "response": f"I couldn't find any relevant information{filter_msg} to answer your question.",
                    "sources": [],
                    "query": user_query,
                    "filtered_documents": document_ids,
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
            
            # Get unique document names for display
            document_names = list(set(source["document_name"] for source in formatted_sources))
            
            # Prepare final response
            response_data = {
                "response": llm_response["response"],
                "sources": formatted_sources,
                "query": user_query,
                "context_chunks": context_chunks,
                "filtered_documents": document_ids,
                "source_documents": document_names,
                "metadata": {
                    "total_sources": len(search_results),
                    "response_time": total_response_time,
                    "llm_response_time": llm_response["metadata"]["response_time"],
                    "similarity_scores": [score for _, score, _ in search_results],
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_used": llm_response["model"],
                    "documents_filtered": len(document_ids) if document_ids else 0
                }
            }
            
            # Add error information if present
            if llm_response.get("error"):
                response_data["error"] = llm_response["error"]
                response_data["error_message"] = llm_response.get("error_message")
            
            # Enhanced logging for query completion
            if self.enhanced_logger:
                self.enhanced_logger.query_complete(total_response_time, len(search_results), 
                                                  f"LLM: {llm_response['model']}")
            else:
                self.logger.info(f"Query processed successfully in {total_response_time:.2f}s")
            return response_data
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            
            return {
                "response": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "sources": [],
                "query": user_query,
                "filtered_documents": document_ids,
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
