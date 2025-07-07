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


class SearchResult:
    """Result from vector similarity search"""
    
    def __init__(self, chunk_id: str, similarity: float, metadata: Dict[str, Any]):
        self.chunk_id = chunk_id
        self.similarity = similarity
        self.metadata = metadata
    
    def __repr__(self):
        return f"SearchResult(chunk_id={self.chunk_id}, similarity={self.similarity:.3f})"


class ChromaVectorStore:
    """ChromaDB-based vector storage implementation"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the ChromaDB vector store"""
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
        self._initialized = False
        
        # Document metadata storage
        self._document_metadata = {}
        self._metadata_file = self.persist_directory / "document_metadata.json"
        
        # Initialize the store
        self._initialize()
        
        self.logger.info("ChromaVectorStore initialized successfully")
    
    def _initialize(self):
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
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            self._initialized = False
    
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
    
    def add_document(
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
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        similarity_threshold: float = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
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
                        
                        search_results.append((chunk_id, similarity, metadata))
                    
                    if len(search_results) >= top_k:
                        break
            
            # Sort by similarity (descending)
            search_results.sort(key=lambda x: x[1], reverse=True)
            
            # Debug logging
            if results['distances'] and results['distances'][0]:
                max_similarity = max(1.0 - d for d in results['distances'][0]) if self.distance_function == 'cosine' else max(results['distances'][0])
                self.logger.info(f"Search found {len(results['ids'][0])} candidates, max similarity: {max_similarity:.3f}, threshold: {similarity_threshold:.3f}, results: {len(search_results)}")
            
            return search_results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error during search: {e}")
            return []
    
    def remove_document(self, document_id: str) -> bool:
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
    
    def list_documents(self) -> List[Dict[str, Any]]:
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
    
    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
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
    
    def re_register_existing_documents(self, source_tracker):
        """Re-register all existing documents with the source tracker"""
        try:
            registered_count = 0
            for document_id, metadata in self._document_metadata.items():
                # Create mock document metadata for registration
                mock_metadata = {
                    "id": document_id,
                    "filename": metadata.get("filename", "unknown"),
                    "source_path": metadata.get("source_path", ""),
                    "processed_at": metadata.get("processed_at", ""),
                    "content": {
                        "page_count": self._estimate_page_count(document_id)
                    }
                }
                
                source_tracker.register_document(
                    document_id,
                    metadata.get("source_path", ""),
                    mock_metadata
                )
                registered_count += 1
            
            if registered_count > 0:
                self.logger.info(f"Re-registered {registered_count} existing documents with source tracker")
            
            return registered_count
            
        except Exception as e:
            self.logger.error(f"Error re-registering documents: {e}")
            return 0
    
    def _estimate_page_count(self, document_id: str) -> int:
        """Estimate page count for a document from its chunks"""
        if document_id not in self._document_metadata:
            return 1
        
        try:
            chunk_ids = self._document_metadata[document_id]["chunk_ids"]
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
            
            return max(page_numbers) if page_numbers else 1
        except Exception:
            return 1