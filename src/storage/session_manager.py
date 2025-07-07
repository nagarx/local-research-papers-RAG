"""
Session Manager - Handle temporary and permanent document storage

This module manages document sessions, allowing users to choose between
temporary (session-only) and permanent storage of processed documents.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timedelta
import logging

from ..config import get_config, get_logger


class SessionManager:
    """Manages document sessions with temporary and permanent storage options"""
    
    def __init__(self):
        """Initialize the session manager"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Session storage paths
        self.sessions_dir = Path("data/sessions")
        self.permanent_docs_file = Path("data/permanent_documents.json")
        
        # Current session state
        self.current_session_id = None
        self.session_documents = {}  # document_id -> storage_type mapping
        self.permanent_documents = {}  # document_id -> metadata mapping
        
        # Initialize storage
        self._initialize_storage()
        self._load_permanent_documents()
        
        self.logger.info("SessionManager initialized successfully")
    
    def _initialize_storage(self):
        """Initialize session storage directories"""
        try:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
            self.permanent_docs_file.parent.mkdir(parents=True, exist_ok=True)
            
            if not self.permanent_docs_file.exists():
                self._save_permanent_documents()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize session storage: {e}")
    
    def _load_permanent_documents(self):
        """Load permanent documents registry"""
        try:
            if self.permanent_docs_file.exists():
                with open(self.permanent_docs_file, 'r', encoding='utf-8') as f:
                    self.permanent_documents = json.load(f)
                self.logger.info(f"Loaded {len(self.permanent_documents)} permanent documents")
        except Exception as e:
            self.logger.error(f"Failed to load permanent documents: {e}")
            self.permanent_documents = {}
    
    def _save_permanent_documents(self):
        """Save permanent documents registry"""
        try:
            with open(self.permanent_docs_file, 'w', encoding='utf-8') as f:
                json.dump(self.permanent_documents, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save permanent documents: {e}")
    
    def start_session(self) -> str:
        """Start a new session and return session ID"""
        try:
            self.current_session_id = str(uuid.uuid4())
            self.session_documents = {}
            
            # Create session directory
            session_dir = self.sessions_dir / self.current_session_id
            session_dir.mkdir(exist_ok=True)
            
            # Save session metadata
            session_metadata = {
                "session_id": self.current_session_id,
                "created_at": datetime.utcnow().isoformat(),
                "documents": {},
                "status": "active"
            }
            
            session_file = session_dir / "session.json"
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Started new session: {self.current_session_id}")
            return self.current_session_id
            
        except Exception as e:
            self.logger.error(f"Failed to start session: {e}")
            return None
    
    def add_document_to_session(
        self, 
        document_id: str, 
        storage_type: str, 
        document_metadata: Dict[str, Any]
    ) -> bool:
        """
        Add a document to the current session
        
        Args:
            document_id: Unique document identifier
            storage_type: 'temporary' or 'permanent'
            document_metadata: Document metadata from processing
        """
        try:
            if not self.current_session_id:
                self.logger.error("No active session")
                return False
            
            if storage_type not in ['temporary', 'permanent']:
                self.logger.error(f"Invalid storage type: {storage_type}")
                return False
            
            # Add to session documents
            self.session_documents[document_id] = storage_type
            
            # If permanent, add to permanent registry
            if storage_type == 'permanent':
                # Create a clean copy of metadata without non-serializable objects
                clean_metadata = {}
                for key, value in document_metadata.items():
                    if key == 'content' and isinstance(value, dict):
                        # Handle content dict - exclude images and other non-serializable objects
                        clean_content = {}
                        for content_key, content_value in value.items():
                            if content_key == 'images':
                                # Store only image count, not the actual images
                                clean_content['image_count'] = len(content_value) if content_value else 0
                            elif content_key in ['blocks', 'full_text']:
                                # Keep these as they are serializable
                                clean_content[content_key] = content_value
                            else:
                                # Try to serialize other values, skip if they fail
                                try:
                                    import json
                                    json.dumps(content_value)
                                    clean_content[content_key] = content_value
                                except (TypeError, ValueError):
                                    self.logger.warning(f"Skipping non-serializable content key: {content_key}")
                        clean_metadata[key] = clean_content
                    else:
                        # Try to serialize the value, skip if it fails
                        try:
                            import json
                            json.dumps(value)
                            clean_metadata[key] = value
                        except (TypeError, ValueError):
                            self.logger.warning(f"Skipping non-serializable metadata key: {key}")
                
                self.permanent_documents[document_id] = {
                    **clean_metadata,
                    "added_to_permanent": datetime.utcnow().isoformat(),
                    "session_id": self.current_session_id
                }
                self._save_permanent_documents()
            
            # Update session file
            self._update_session_file()
            
            self.logger.info(f"Added document {document_id} to session as {storage_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add document to session: {e}")
            return False
    
    def _update_session_file(self):
        """Update the current session file"""
        try:
            if not self.current_session_id:
                return
            
            session_dir = self.sessions_dir / self.current_session_id
            session_file = session_dir / "session.json"
            
            if session_file.exists():
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
            else:
                session_data = {
                    "session_id": self.current_session_id,
                    "created_at": datetime.utcnow().isoformat(),
                    "status": "active"
                }
            
            session_data["documents"] = self.session_documents
            session_data["last_updated"] = datetime.utcnow().isoformat()
            
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Failed to update session file: {e}")
    
    def end_session(self, vector_store) -> Dict[str, Any]:
        """
        End the current session and clean up temporary documents
        
        Args:
            vector_store: Vector store instance to clean temporary documents
        
        Returns:
            Summary of cleanup actions
        """
        try:
            if not self.current_session_id:
                return {"error": "No active session"}
            
            cleanup_summary = {
                "session_id": self.current_session_id,
                "temporary_docs_removed": 0,
                "permanent_docs_kept": 0,
                "errors": []
            }
            
            # Process each document in the session
            for document_id, storage_type in self.session_documents.items():
                try:
                    if storage_type == 'temporary':
                        # Remove temporary document from vector store
                        success = vector_store.remove_document(document_id)
                        if success:
                            cleanup_summary["temporary_docs_removed"] += 1
                        else:
                            cleanup_summary["errors"].append(f"Failed to remove temporary document: {document_id}")
                    
                    elif storage_type == 'permanent':
                        cleanup_summary["permanent_docs_kept"] += 1
                
                except Exception as e:
                    cleanup_summary["errors"].append(f"Error processing document {document_id}: {str(e)}")
            
            # Mark session as ended
            session_dir = self.sessions_dir / self.current_session_id
            session_file = session_dir / "session.json"
            
            if session_file.exists():
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                session_data["status"] = "ended"
                session_data["ended_at"] = datetime.utcnow().isoformat()
                session_data["cleanup_summary"] = cleanup_summary
                
                with open(session_file, 'w', encoding='utf-8') as f:
                    json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            # Clear current session
            old_session_id = self.current_session_id
            self.current_session_id = None
            self.session_documents = {}
            
            self.logger.info(f"Ended session {old_session_id}: {cleanup_summary}")
            return cleanup_summary
            
        except Exception as e:
            self.logger.error(f"Failed to end session: {e}")
            return {"error": str(e)}
    
    def get_session_documents(self) -> Dict[str, str]:
        """Get documents in the current session"""
        return self.session_documents.copy()
    
    def get_permanent_documents(self) -> List[Dict[str, Any]]:
        """Get list of all permanent documents"""
        try:
            documents = []
            for doc_id, metadata in self.permanent_documents.items():
                documents.append({
                    "document_id": doc_id,
                    "filename": metadata.get("filename", "unknown"),
                    "processed_at": metadata.get("processed_at", ""),
                    "added_to_permanent": metadata.get("added_to_permanent", ""),
                    "total_chunks": metadata.get("total_chunks", 0),
                    "session_id": metadata.get("session_id", "unknown")
                })
            
            # Sort by most recently added
            documents.sort(key=lambda x: x["added_to_permanent"], reverse=True)
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to get permanent documents: {e}")
            return []
    
    def remove_permanent_document(self, document_id: str, vector_store) -> bool:
        """Remove a document completely from all storage locations"""
        try:
            if document_id not in self.permanent_documents:
                self.logger.warning(f"Document {document_id} not in permanent storage")
                # Still try to remove from other locations in case it exists there
            
            success_count = 0
            total_operations = 0
            errors = []
            
            # 1. Remove from vector store (ChromaDB)
            total_operations += 1
            try:
                success = vector_store.remove_document(document_id)
                if success:
                    success_count += 1
                    self.logger.info(f"Removed document {document_id} from vector store")
                else:
                    errors.append("Failed to remove from vector store")
            except Exception as e:
                errors.append(f"Error removing from vector store: {str(e)}")
            
            # 2. Remove from permanent registry
            total_operations += 1
            try:
                if document_id in self.permanent_documents:
                    del self.permanent_documents[document_id]
                    self._save_permanent_documents()
                    success_count += 1
                    self.logger.info(f"Removed document {document_id} from permanent registry")
                else:
                    # Still count as success if it wasn't there
                    success_count += 1
            except Exception as e:
                errors.append(f"Error removing from permanent registry: {str(e)}")
            
            # 3. Remove from processed documents directory
            total_operations += 1
            try:
                processed_dir = Path("data/processed")
                removed_files = []
                
                if processed_dir.exists():
                    # Look for files related to this document
                    for file_path in processed_dir.iterdir():
                        if file_path.is_file():
                            # Check if filename contains the document ID
                            if document_id in file_path.name:
                                file_path.unlink()
                                removed_files.append(file_path.name)
                                self.logger.info(f"Removed processed file: {file_path.name}")
                            else:
                                # Check if the file contains metadata for this document
                                if file_path.name.endswith('_metadata.json'):
                                    try:
                                        with open(file_path, 'r', encoding='utf-8') as f:
                                            metadata = json.load(f)
                                        if metadata.get('document_id') == document_id:
                                            # Also remove the corresponding raw file
                                            raw_file = file_path.with_name(file_path.name.replace('_metadata.json', '_raw.md'))
                                            if raw_file.exists():
                                                raw_file.unlink()
                                                removed_files.append(raw_file.name)
                                            # Remove metadata file
                                            file_path.unlink()
                                            removed_files.append(file_path.name)
                                            self.logger.info(f"Removed processed files for document {document_id}")
                                    except Exception as file_error:
                                        self.logger.warning(f"Could not check metadata file {file_path.name}: {file_error}")
                
                if removed_files:
                    success_count += 1
                    self.logger.info(f"Removed {len(removed_files)} processed files: {removed_files}")
                else:
                    success_count += 1  # No files to remove is still success
                    
            except Exception as e:
                errors.append(f"Error removing processed files: {str(e)}")
            
            # 4. Remove from any embeddings directory
            total_operations += 1
            try:
                embeddings_dir = Path("data/embeddings")
                removed_embedding_files = []
                
                if embeddings_dir.exists():
                    for file_path in embeddings_dir.iterdir():
                        if file_path.is_file() and document_id in file_path.name:
                            file_path.unlink()
                            removed_embedding_files.append(file_path.name)
                            self.logger.info(f"Removed embedding file: {file_path.name}")
                
                if removed_embedding_files:
                    self.logger.info(f"Removed {len(removed_embedding_files)} embedding files")
                success_count += 1
                
            except Exception as e:
                errors.append(f"Error removing embedding files: {str(e)}")
            
            # 5. Remove from current session if present
            if self.current_session_id and document_id in self.session_documents:
                try:
                    del self.session_documents[document_id]
                    self._update_session_file()
                    self.logger.info(f"Removed document {document_id} from current session")
                except Exception as e:
                    errors.append(f"Error removing from current session: {str(e)}")
            
            # Determine overall success
            if success_count >= 3:  # At least vector store, permanent registry, and processed files
                self.logger.info(f"Successfully removed document {document_id} from {success_count}/{total_operations} locations")
                if errors:
                    self.logger.warning(f"Some errors occurred during removal: {errors}")
                return True
            else:
                self.logger.error(f"Failed to remove document {document_id} - only {success_count}/{total_operations} operations succeeded. Errors: {errors}")
                return False
            
        except Exception as e:
            self.logger.error(f"Critical error removing permanent document {document_id}: {e}")
            return False
    
    def cleanup_old_sessions(self, days_old: int = 7) -> Dict[str, Any]:
        """Clean up old session files"""
        try:
            cleanup_summary = {
                "sessions_cleaned": 0,
                "errors": []
            }
            
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            for session_dir in self.sessions_dir.iterdir():
                if not session_dir.is_dir():
                    continue
                
                session_file = session_dir / "session.json"
                if not session_file.exists():
                    continue
                
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    
                    created_at = datetime.fromisoformat(session_data.get("created_at", ""))
                    
                    if created_at < cutoff_date:
                        # Remove old session directory
                        import shutil
                        shutil.rmtree(session_dir)
                        cleanup_summary["sessions_cleaned"] += 1
                
                except Exception as e:
                    cleanup_summary["errors"].append(f"Error cleaning session {session_dir.name}: {str(e)}")
            
            self.logger.info(f"Cleaned up old sessions: {cleanup_summary}")
            return cleanup_summary
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old sessions: {e}")
            return {"error": str(e)}
    
    def get_session_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current session"""
        if not self.current_session_id:
            return None
        
        return {
            "session_id": self.current_session_id,
            "documents": self.session_documents,
            "document_count": len(self.session_documents),
            "temporary_count": sum(1 for storage_type in self.session_documents.values() if storage_type == 'temporary'),
            "permanent_count": sum(1 for storage_type in self.session_documents.values() if storage_type == 'permanent')
        } 