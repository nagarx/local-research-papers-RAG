"""
ChromaDB Vector Store Implementation

This module provides a ChromaDB-based vector storage solution for the RAG system.
"""

import os
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

try:
    import numpy as np
    import chromadb
    from chromadb.config import Settings
    from chromadb.utils import embedding_functions
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    # Fallback for type hints
    chromadb = None
    Settings = None
    embedding_functions = None

from ..config import get_config, get_logger
from ..core.interfaces.vector_store import BaseVectorStore, SearchResult


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-based vector storage implementation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the ChromaDB vector store"""
        super().__init__(config)
        
        if not HAS_CHROMADB:
            raise ImportError("ChromaDB is not installed. Please install it with: pip install chromadb")
        
        self.app_config = get_config()
        self.logger = get_logger(__name__)
        
        # ChromaDB configuration
        self.collection_name = self.app_config.vector_storage.collection_name
        self.persist_directory = Path(self.app_config.vector_storage.persist_directory)
        self.distance_function = self.app_config.vector_storage.distance_function
        
        # Initialize ChromaDB client and collection
        self._client = None
        self._collection = None
        
        # Document metadata storage
        self._document_metadata = {}
        self._metadata_file = self.persist_directory / "document_metadata.json"
        
        self.logger.info("ChromaVectorStore initialized successfully")
    
    async def _initialize_impl(self) -> bool:
        """Initialize ChromaDB client and collection"""
        try:
            # Ensure persist directory exists
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self._client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Create or get collection
            try:
                self._collection = self._client.get_collection(
                    name=self.collection_name,
                    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=self.app_config.embedding.model
                    )
                )
                self.logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self._collection = self._client.create_collection(
                    name=self.collection_name,
                    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                        model_name=self.app_config.embedding.model
                    ),
                    metadata={"hnsw:space": self.distance_function}
                )
                self.logger.info(f"Created new collection: {self.collection_name}")
            
            # Load document metadata
            self._load_document_metadata()
            
            self._initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            return False
    
    def _load_document_metadata(self):
        """Load document metadata from disk"""
        try:
            if self._metadata_file.exists():
                with open(self._metadata_file, 'r', encoding='utf-8') as f:
                    self._document_metadata = json.load(f)
                self.logger.info(f"Loaded metadata for {len(self._document_metadata)} documents")
        except Exception as e:
            self.logger.error(f"Error loading document metadata: {e}")
            self._document_metadata = {}
    
    def _save_document_metadata(self):
        """Save document metadata to disk"""
        try:
            with open(self._metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self._document_metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving document metadata: {e}")
    
    async def _add_document_impl(
        self, 
        document_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[np.ndarray],
        metadata: Dict[str, Any]
    ) -> bool:
        """Add document chunks to ChromaDB collection"""
        try:
            if not self._collection:
                self.logger.error("Collection not initialized")
                return False
                
            if len(chunks) != len(embeddings):
                self.logger.error(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")
                return False
            
            # Prepare data for ChromaDB
            ids = []
            documents = []
            chunk_embeddings = []
            metadatas = []
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{document_id}_chunk_{i}"
                ids.append(chunk_id)
                documents.append(chunk.get("text", ""))
                chunk_embeddings.append(embedding.tolist())
                
                # Prepare metadata for ChromaDB
                chunk_metadata = {
                    "document_id": document_id,
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "page_number": chunk.get("source_info", {}).get("page_number", 1),
                    "block_type": chunk.get("block_type", "Text"),
                    "filename": metadata.get("filename", "unknown"),
                    "source_path": metadata.get("source_path", ""),
                    "processed_at": metadata.get("processed_at", datetime.utcnow().isoformat())
                }
                metadatas.append(chunk_metadata)
            
            # Add to ChromaDB collection
            self._collection.add(
                embeddings=chunk_embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            # Store document metadata
            self._document_metadata[document_id] = {
                "document_id": document_id,
                "filename": metadata.get("filename", "unknown"),
                "source_path": metadata.get("source_path", ""),
                "processed_at": metadata.get("processed_at", datetime.utcnow().isoformat()),
                "total_chunks": len(chunks),
                "chunk_ids": ids
            }
            
            # Save document metadata
            self._save_document_metadata()
            
            self.logger.info(f"Added document {document_id} with {len(chunks)} chunks to ChromaDB")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding document {document_id}: {e}")
            return False
    
    async def _search_impl(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        similarity_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search for similar chunks in ChromaDB"""
        try:
            if not self._collection:
                self.logger.error("Collection not initialized")
                return []
                
            if similarity_threshold is None:
                similarity_threshold = self.app_config.vector_storage.similarity_threshold
            
            # Prepare query embedding
            query_embeddings = [query_embedding.tolist()]
            
            # Prepare filters for ChromaDB
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    if key == "document_id":
                        where_clause["document_id"] = value
                    elif key == "page_number":
                        where_clause["page_number"] = value
            
            # Search in ChromaDB
            results = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=min(top_k * 2, 100),  # Get more results to filter
                where=where_clause,
                include=['metadatas', 'documents', 'distances']
            )
            
            # Process results
            search_results = []
            if results['ids'] and results['ids'][0]:
                for i, (chunk_id, distance, metadata, document) in enumerate(zip(
                    results['ids'][0],
                    results['distances'][0],
                    results['metadatas'][0],
                    results['documents'][0]
                )):
                    # Convert distance to similarity score
                    if self.distance_function == 'cosine':
                        similarity = 1.0 - distance
                    elif self.distance_function == 'l2':
                        similarity = 1.0 / (1.0 + distance)
                    else:  # ip (inner product)
                        similarity = distance
                    
                    # Apply similarity threshold
                    if similarity >= similarity_threshold:
                        # Add document text to metadata
                        metadata["text"] = document
                        metadata["similarity"] = similarity
                        
                        search_results.append(SearchResult(
                            chunk_id=chunk_id,
                            similarity=similarity,
                            metadata=metadata
                        ))
                    
                    if len(search_results) >= top_k:
                        break
            
            # Sort by similarity (descending)
            search_results.sort(key=lambda x: x.similarity, reverse=True)
            
            # Debug logging
            if results['distances'] and results['distances'][0]:
                max_similarity = max(1.0 - d for d in results['distances'][0]) if self.distance_function == 'cosine' else max(results['distances'][0])
                self.logger.info(f"Search found {len(results['ids'][0])} candidates, max similarity: {max_similarity:.3f}, threshold: {similarity_threshold:.3f}, results: {len(search_results)}")
            
            return search_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error during search: {e}")
            return []
    
    async def _delete_document_impl(self, document_id: str) -> bool:
        """Delete a document and all its chunks from ChromaDB"""
        try:
            if not self._collection:
                self.logger.error("Collection not initialized")
                return False
                
            if document_id not in self._document_metadata:
                self.logger.warning(f"Document {document_id} not found in metadata")
                return False
            
            # Get chunk IDs to delete
            chunk_ids = self._document_metadata[document_id]["chunk_ids"]
            
            # Delete from ChromaDB collection
            self._collection.delete(ids=chunk_ids)
            
            # Remove from document metadata
            del self._document_metadata[document_id]
            
            # Save updated metadata
            self._save_document_metadata()
            
            self.logger.info(f"Deleted document {document_id} with {len(chunk_ids)} chunks from ChromaDB")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    async def _list_documents_impl(self) -> List[Dict[str, Any]]:
        """List all documents in the store"""
        try:
            return [
                {
                    "document_id": doc_id,
                    "filename": metadata["filename"],
                    "total_chunks": metadata["total_chunks"],
                    "processed_at": metadata["processed_at"],
                    "source_path": metadata.get("source_path", "")
                }
                for doc_id, metadata in self._document_metadata.items()
            ]
        except Exception as e:
            self.logger.error(f"Error listing documents: {e}")
            return []
    
    async def _get_document_impl(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific document"""
        try:
            if not self._collection:
                self.logger.error("Collection not initialized")
                return None
                
            if document_id not in self._document_metadata:
                return None
            
            metadata = self._document_metadata[document_id].copy()
            
            # Get chunks from ChromaDB
            chunk_ids = metadata["chunk_ids"]
            results = self._collection.get(
                ids=chunk_ids,
                include=['metadatas', 'documents']
            )
            
            # Process chunk details
            chunks = {}
            if results['ids']:
                for chunk_id, chunk_metadata, document in zip(
                    results['ids'],
                    results['metadatas'],
                    results['documents']
                ):
                    chunks[chunk_id] = {
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_metadata.get("chunk_index", 0),
                        "text": document,
                        "page_number": chunk_metadata.get("page_number", 1),
                        "block_type": chunk_metadata.get("block_type", "Text"),
                    }
            
            metadata["chunks"] = chunks
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error getting document {document_id}: {e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            total_vectors = 0
            if self._collection:
                total_vectors = self._collection.count()
            
            return {
                "total_documents": len(self._document_metadata),
                "total_vectors": total_vectors,
                "total_chunks": sum(
                    metadata.get("total_chunks", 0) 
                    for metadata in self._document_metadata.values()
                ),
                "collection_name": self.collection_name,
                "distance_function": self.distance_function,
                "persist_directory": str(self.persist_directory),
                "last_updated": datetime.utcnow().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {
                "total_documents": 0,
                "total_vectors": 0,
                "total_chunks": 0,
                "error": str(e)
            }
    
    def get_document_metadata(self, document_id: str) -> Dict[str, Any]:
        """Get document metadata with page count estimation"""
        try:
            if document_id not in self._document_metadata:
                return {"error": f"Document {document_id} not found"}
            
            metadata = self._document_metadata[document_id]
            
            # Get page count from chunks
            if self._collection:
                chunk_ids = metadata["chunk_ids"]
                results = self._collection.get(
                    ids=chunk_ids,
                    include=['metadatas']
                )
                
                page_numbers = set()
                if results['metadatas']:
                    for chunk_metadata in results['metadatas']:
                        page_num = chunk_metadata.get("page_number", 1)
                        if page_num > 0:
                            page_numbers.add(page_num)
                
                estimated_pages = max(page_numbers) if page_numbers else 1
            else:
                estimated_pages = 1
            
            return {
                "document_id": document_id,
                "filename": metadata["filename"],
                "total_chunks": metadata["total_chunks"],
                "estimated_pages": estimated_pages,
                "processed_at": metadata["processed_at"]
            }
            
        except Exception as e:
            self.logger.error(f"Error getting document metadata for {document_id}: {e}")
            return {"error": str(e)}
    
    def cleanup_old_documents(self, days_old: int = 7) -> Dict[str, Any]:
        """Remove documents older than specified days"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            documents_to_remove = []
            
            for doc_id, metadata in self._document_metadata.items():
                processed_at_str = metadata.get("processed_at", "")
                if processed_at_str:
                    try:
                        processed_at = datetime.fromisoformat(processed_at_str.replace('Z', '+00:00'))
                        if processed_at < cutoff_date:
                            documents_to_remove.append(doc_id)
                    except ValueError:
                        self.logger.warning(f"Invalid date format for document {doc_id}: {processed_at_str}")
            
            # Remove old documents
            removed_count = 0
            for doc_id in documents_to_remove:
                if self.delete_document(doc_id):
                    removed_count += 1
            
            result = {
                "removed_documents": removed_count,
                "remaining_documents": len(self._document_metadata),
                "cutoff_date": cutoff_date.isoformat(),
                "days_old": days_old
            }
            
            self.logger.info(f"Cleanup completed: removed {removed_count} documents older than {days_old} days")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return {
                "removed_documents": 0,
                "remaining_documents": len(self._document_metadata),
                "error": str(e)
            }
    
    async def migrate_from_faiss_data(self, faiss_data_path: Path) -> Dict[str, Any]:
        """Migrate data from FAISS-based storage to ChromaDB"""
        try:
            # Import sentence transformers only when needed
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                return {"error": "sentence-transformers is not installed"}
            
            migration_stats = {
                "documents_migrated": 0,
                "chunks_migrated": 0,
                "errors": []
            }
            
            # Load FAISS metadata
            metadata_file = faiss_data_path / "metadata.json"
            if not metadata_file.exists():
                return {"error": "FAISS metadata file not found"}
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                faiss_data = json.load(f)
            
            faiss_metadata = faiss_data.get("metadata", {})
            
            # Process each document
            for doc_id, doc_metadata in faiss_metadata.items():
                try:
                    # Extract chunks and prepare for ChromaDB
                    chunks = []
                    chunk_texts = []
                    
                    for chunk_id, chunk_data in doc_metadata.get("chunks", {}).items():
                        chunk_info = {
                            "text": chunk_data.get("text", ""),
                            "source_info": {
                                "page_number": chunk_data.get("page_number", 1)
                            },
                            "block_type": chunk_data.get("block_type", "Text")
                        }
                        chunks.append(chunk_info)
                        chunk_texts.append(chunk_info["text"])
                    
                    if chunks:
                        # Generate embeddings for chunks
                        embedding_model = SentenceTransformer(self.app_config.embedding.model)
                        embeddings = embedding_model.encode(chunk_texts)
                        
                        # Add document to ChromaDB
                        success = await self._add_document_impl(
                            document_id=doc_id,
                            chunks=chunks,
                            embeddings=embeddings,
                            metadata={
                                "filename": doc_metadata.get("filename", "unknown"),
                                "source_path": doc_metadata.get("source_path", ""),
                                "processed_at": doc_metadata.get("processed_at", datetime.utcnow().isoformat())
                            }
                        )
                        
                        if success:
                            migration_stats["documents_migrated"] += 1
                            migration_stats["chunks_migrated"] += len(chunks)
                        else:
                            migration_stats["errors"].append(f"Failed to migrate document {doc_id}")
                    
                except Exception as e:
                    migration_stats["errors"].append(f"Error migrating document {doc_id}: {str(e)}")
            
            self.logger.info(f"Migration completed: {migration_stats['documents_migrated']} documents, {migration_stats['chunks_migrated']} chunks")
            return migration_stats
            
        except Exception as e:
            self.logger.error(f"Error during migration: {e}")
            return {"error": str(e)}