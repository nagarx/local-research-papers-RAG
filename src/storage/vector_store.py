"""
Vector Store - JSON-based Vector Storage with FAISS Indexing
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

import numpy as np
import faiss

from ..config import get_config, get_logger


class VectorStore:
    """JSON-based vector storage with FAISS indexing"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the vector store"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Storage paths
        self.index_dir = self.config.storage_paths.index_dir
        self.metadata_file = self.index_dir / "metadata.json"
        self.index_file = self.index_dir / "vector_index.faiss"
        
        # Initialize storage
        self._ensure_storage_directory()
        
        # Vector storage
        self._metadata = {}
        self._faiss_index = None
        self._id_to_faiss_index = {}
        self._faiss_index_to_id = {}
        
        # Load existing data
        self._load_storage()
        
        self.logger.info("VectorStore initialized successfully")
    
    def _ensure_storage_directory(self):
        """Ensure storage directory exists"""
        self.index_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_storage(self):
        """Load existing metadata and FAISS index"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    stored_data = json.load(f)
                    
                self._metadata = stored_data.get("metadata", {})
                self._id_to_faiss_index = stored_data.get("id_to_faiss_index", {})
                self._faiss_index_to_id = {v: k for k, v in self._id_to_faiss_index.items()}
                
                self.logger.info(f"Loaded {len(self._metadata)} document records")
            
            if self.index_file.exists():
                self._faiss_index = faiss.read_index(str(self.index_file))
                self.logger.info(f"Loaded FAISS index with {self._faiss_index.ntotal} vectors")
            
        except Exception as e:
            self.logger.error(f"Error loading storage: {e}")
            self._metadata = {}
            self._faiss_index = None
            self._id_to_faiss_index = {}
            self._faiss_index_to_id = {}
    
    def _save_storage(self):
        """Save metadata and FAISS index to disk"""
        try:
            storage_data = {
                "metadata": self._metadata,
                "id_to_faiss_index": self._id_to_faiss_index,
                "last_saved": datetime.utcnow().isoformat()
            }
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(storage_data, f, indent=2, ensure_ascii=False)
            
            if self._faiss_index is not None:
                faiss.write_index(self._faiss_index, str(self.index_file))
            
        except Exception as e:
            self.logger.error(f"Error saving storage: {e}")
    
    def _initialize_faiss_index(self, dimension: int):
        """Initialize FAISS index for given dimension"""
        self._faiss_index = faiss.IndexFlatIP(dimension)
        self.logger.info(f"Initialized FAISS index (dimension: {dimension})")
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector for cosine similarity"""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        else:
            return vector
    
    def add_document(
        self, 
        document_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[np.ndarray],
        metadata: Dict[str, Any]
    ) -> bool:
        """Add a document with its chunks and embeddings to the store"""
        
        if len(chunks) != len(embeddings):
            self.logger.error(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")
            return False
        
        try:
            if self._faiss_index is None and embeddings:
                dimension = embeddings[0].shape[0]
                self._initialize_faiss_index(dimension)
            
            document_metadata = {
                "document_id": document_id,
                "filename": metadata.get("filename", "unknown"),
                "source_path": metadata.get("source_path", ""),
                "processed_at": metadata.get("processed_at", datetime.utcnow().isoformat()),
                "total_chunks": len(chunks),
                "chunks": {}
            }
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_id = f"{document_id}_chunk_{i}"
                
                normalized_embedding = self._normalize_vector(embedding)
                
                faiss_index = self._faiss_index.ntotal
                self._faiss_index.add(normalized_embedding.reshape(1, -1))
                
                self._id_to_faiss_index[chunk_id] = faiss_index
                self._faiss_index_to_id[faiss_index] = chunk_id
                
                chunk_metadata = {
                    "chunk_id": chunk_id,
                    "chunk_index": i,
                    "text": chunk.get("text", ""),
                    "page_number": chunk.get("source_info", {}).get("page_number", 0),
                    "block_type": chunk.get("block_type", "Text"),
                    "source_info": chunk.get("source_info", {}),
                }
                
                document_metadata["chunks"][chunk_id] = chunk_metadata
            
            self._metadata[document_id] = document_metadata
            self._save_storage()
            
            self.logger.info(f"Added document {document_id} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding document {document_id}: {e}")
            return False
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        top_k: int = 5,
        similarity_threshold: float = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar chunks"""
        
        if self._faiss_index is None or self._faiss_index.ntotal == 0:
            return []
        
        if similarity_threshold is None:
            similarity_threshold = self.config.vector_storage.similarity_threshold
        
        try:
            normalized_query = self._normalize_vector(query_embedding)
            
            scores, indices = self._faiss_index.search(
                normalized_query.reshape(1, -1), 
                min(top_k * 2, self._faiss_index.ntotal)
            )
            
            results = []
            all_scores = []  # Debug: Track all scores
            
            for score, faiss_index in zip(scores[0], indices[0]):
                all_scores.append(float(score))  # Debug: Record all scores
                
                if faiss_index in self._faiss_index_to_id:
                    chunk_id = self._faiss_index_to_id[faiss_index]
                    document_id = chunk_id.split("_chunk_")[0]
                    
                    if document_id in self._metadata:
                        document_metadata = self._metadata[document_id]
                        
                        if chunk_id in document_metadata["chunks"]:
                            chunk_metadata = document_metadata["chunks"][chunk_id]
                            
                            similarity = float(score)
                            
                            if similarity >= similarity_threshold:
                                combined_metadata = {
                                    "document_id": document_id,
                                    "document_filename": document_metadata["filename"],
                                    "chunk_id": chunk_id,
                                    "chunk_index": chunk_metadata["chunk_index"],
                                    "text": chunk_metadata["text"],
                                    "page_number": chunk_metadata["page_number"],
                                    "block_type": chunk_metadata["block_type"],
                                    "source_info": chunk_metadata["source_info"],
                                    "similarity": similarity
                                }
                                
                                results.append((chunk_id, similarity, combined_metadata))
                
                if len(results) >= top_k:
                    break
            
            # Debug logging
            if all_scores:
                max_score = max(all_scores)
                self.logger.info(f"Search found {len(all_scores)} candidates, max similarity: {max_score:.3f}, threshold: {similarity_threshold:.3f}, results: {len(results)}")
            else:
                self.logger.warning("No search candidates found")
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error during search: {e}")
            return []
    
    def remove_document(self, document_id: str) -> bool:
        """Remove a document and all its chunks from the store"""
        try:
            if document_id not in self._metadata:
                self.logger.warning(f"Document {document_id} not found in metadata")
                return False
            
            # Get chunks to remove
            chunks_to_remove = list(self._metadata[document_id]["chunks"].keys())
            
            # Remove from metadata
            del self._metadata[document_id]
            
            # Remove chunk mappings
            faiss_indices_to_remove = []
            for chunk_id in chunks_to_remove:
                if chunk_id in self._id_to_faiss_index:
                    faiss_index = self._id_to_faiss_index[chunk_id]
                    faiss_indices_to_remove.append(faiss_index)
                    del self._id_to_faiss_index[chunk_id]
                    if faiss_index in self._faiss_index_to_id:
                        del self._faiss_index_to_id[faiss_index]
            
            # Rebuild FAISS index without removed chunks
            if faiss_indices_to_remove:
                self._rebuild_faiss_index_without_chunks(faiss_indices_to_remove)
            
            # Save changes
            self._save_storage()
            
            self.logger.info(f"Removed document {document_id} with {len(chunks_to_remove)} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Error removing document {document_id}: {e}")
            return False
    
    def _rebuild_faiss_index_without_chunks(self, faiss_indices_to_remove: List[int]):
        """Rebuild FAISS index excluding specified indices"""
        try:
            if self._faiss_index is None or self._faiss_index.ntotal == 0:
                return
            
            # Get all vectors
            all_vectors = []
            new_id_mapping = {}
            new_reverse_mapping = {}
            new_faiss_index = 0
            
            for old_faiss_index in range(self._faiss_index.ntotal):
                if old_faiss_index not in faiss_indices_to_remove:
                    # Get vector
                    vector = self._faiss_index.reconstruct(old_faiss_index)
                    all_vectors.append(vector)
                    
                    # Update mappings
                    if old_faiss_index in self._faiss_index_to_id:
                        chunk_id = self._faiss_index_to_id[old_faiss_index]
                        new_id_mapping[chunk_id] = new_faiss_index
                        new_reverse_mapping[new_faiss_index] = chunk_id
                        new_faiss_index += 1
            
            # Create new FAISS index
            if all_vectors:
                dimension = all_vectors[0].shape[0]
                new_index = faiss.IndexFlatIP(dimension)
                
                # Add all vectors to new index
                vectors_array = np.vstack(all_vectors)
                new_index.add(vectors_array)
                
                self._faiss_index = new_index
                self._id_to_faiss_index = new_id_mapping
                self._faiss_index_to_id = new_reverse_mapping
                
                self.logger.info(f"Rebuilt FAISS index: {len(all_vectors)} vectors remaining")
            else:
                # No vectors remaining
                self._faiss_index = None
                self._id_to_faiss_index = {}
                self._faiss_index_to_id = {}
                self.logger.info("No vectors remaining after cleanup")
                
        except Exception as e:
            self.logger.error(f"Error rebuilding FAISS index: {e}")
    
    def cleanup_old_documents(self, days_old: int = 7) -> Dict[str, Any]:
        """Remove documents older than specified days"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            documents_to_remove = []
            
            for doc_id, metadata in self._metadata.items():
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
                if self.remove_document(doc_id):
                    removed_count += 1
            
            result = {
                "removed_documents": removed_count,
                "remaining_documents": len(self._metadata),
                "cutoff_date": cutoff_date.isoformat(),
                "days_old": days_old
            }
            
            self.logger.info(f"Cleanup completed: removed {removed_count} documents older than {days_old} days")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return {
                "removed_documents": 0,
                "remaining_documents": len(self._metadata),
                "error": str(e)
            }
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get enhanced vector store statistics"""
        try:
            total_chunks = sum(
                len(metadata.get("chunks", {})) 
                for metadata in self._metadata.values()
            )
            
            # Calculate storage size
            storage_size = 0
            if self.metadata_file.exists():
                storage_size += self.metadata_file.stat().st_size
            if self.index_file.exists():
                storage_size += self.index_file.stat().st_size
            
            storage_size_mb = storage_size / (1024 * 1024)
            
            # Document type breakdown
            doc_types = {}
            for metadata in self._metadata.values():
                filename = metadata.get("filename", "unknown")
                if filename.endswith(".pdf"):
                    doc_type = "PDF"
                else:
                    doc_type = "Other"
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            return {
                "total_documents": len(self._metadata),
                "total_vectors": self._faiss_index.ntotal if self._faiss_index else 0,
                "total_chunks": total_chunks,
                "storage_size_mb": round(storage_size_mb, 2),
                "document_types": doc_types,
                "index_dimension": self._faiss_index.d if self._faiss_index else 0,
                "last_updated": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting enhanced stats: {e}")
            return self.get_stats()
    
    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific document"""
        if document_id not in self._metadata:
            return None
        
        metadata = self._metadata[document_id].copy()
        
        # Add chunk count and sample chunks
        chunks = metadata.get("chunks", {})
        metadata["chunk_details"] = {
            "total_chunks": len(chunks),
            "sample_chunks": list(chunks.keys())[:3] if chunks else []
        }
        
        return metadata
    
    def get_document_metadata(self, document_id: str) -> Dict[str, Any]:
        """Get document metadata with page count estimation"""
        if document_id not in self._metadata:
            return {"error": f"Document {document_id} not found"}
        
        metadata = self._metadata[document_id]
        
        # Estimate page count from chunks
        page_numbers = set()
        for chunk_data in metadata.get("chunks", {}).values():
            page_num = chunk_data.get("page_number", 1)
            if page_num > 0:
                page_numbers.add(page_num)
        
        estimated_pages = max(page_numbers) if page_numbers else 1
        
        return {
            "document_id": document_id,
            "filename": metadata.get("filename", "unknown"),
            "total_chunks": metadata.get("total_chunks", 0),
            "estimated_pages": estimated_pages,
            "processed_at": metadata.get("processed_at", "")
        }
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents in the store"""
        return [
            {
                "document_id": doc_id,
                "filename": metadata["filename"],
                "total_chunks": metadata["total_chunks"],
                "processed_at": metadata["processed_at"]
            }
            for doc_id, metadata in self._metadata.items()
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        return {
            "total_documents": len(self._metadata),
            "total_vectors": self._faiss_index.ntotal if self._faiss_index else 0,
        }
    
    def re_register_existing_documents(self, source_tracker):
        """Re-register all existing documents with the source tracker"""
        try:
            registered_count = 0
            for document_id, metadata in self._metadata.items():
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
        if document_id not in self._metadata:
            return 1
        
        page_numbers = set()
        for chunk_data in self._metadata[document_id].get("chunks", {}).values():
            page_num = chunk_data.get("page_number", 1)
            if page_num > 0:
                page_numbers.add(page_num)
        
        return max(page_numbers) if page_numbers else 1
