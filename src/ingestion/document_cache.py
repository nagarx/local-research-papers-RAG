"""
Document Cache Module

This module handles document caching, duplicate detection, and content hashing
for efficient document processing.
"""

import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

from ..config import get_config, get_logger


class DocumentCache:
    """
    Handles document caching and duplicate detection through content hashing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the document cache"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Cache directory
        self.processed_dir = self.config.storage_paths.processed_dir
        
        self.logger.info("DocumentCache initialized successfully")
    
    def calculate_file_hash(self, file_path: Union[str, Path]) -> Optional[str]:
        """Calculate SHA-256 hash of file content for duplicate detection"""
        file_path = Path(file_path)
        sha256_hash = hashlib.sha256()
        
        try:
            with open(file_path, "rb") as f:
                # Read file in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return None
    
    def find_existing_document_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Find existing document by content hash in processed files"""
        try:
            if not self.processed_dir.exists():
                return None
            
            # Search through metadata files for matching content hash
            metadata_files = list(self.processed_dir.glob("*_metadata.json"))
            
            for metadata_file in metadata_files:
                try:
                    import json
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    if metadata.get("content_hash") == content_hash:
                        # Found existing document with same content
                        self.logger.info(f"Found existing document with hash {content_hash[:8]}...")
                        return metadata
                        
                except Exception as e:
                    self.logger.warning(f"Failed to read metadata file {metadata_file}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error searching for existing document: {e}")
            return None
    
    def find_existing_document_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Find existing document by exact filename match"""
        try:
            if not self.processed_dir.exists():
                return None
            
            # Search through metadata files for matching filename
            metadata_files = list(self.processed_dir.glob("*_metadata.json"))
            
            for metadata_file in metadata_files:
                try:
                    import json
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    if metadata.get("original_filename") == filename:
                        # Found existing document with same filename
                        self.logger.info(f"Found existing document with filename: {filename}")
                        return metadata
                        
                except Exception as e:
                    self.logger.warning(f"Failed to read metadata file {metadata_file}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error searching for existing document by filename: {e}")
            return None
    
    def generate_content_based_document_id(self, content_hash: str, filename: str) -> str:
        """Generate consistent document ID based on content hash"""
        # Use first 12 characters of content hash + filename hash for consistency
        filename_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        return f"doc_{content_hash[:12]}_{filename_hash}"
    
    def is_document_processed(self, file_path: Union[str, Path], force_reprocess: bool = False) -> tuple:
        """
        Check if document is already processed
        
        Returns:
            tuple: (is_processed, existing_doc_metadata, content_hash)
        """
        if force_reprocess:
            return False, None, None
        
        # Calculate content hash for duplicate detection
        content_hash = self.calculate_file_hash(file_path)
        if not content_hash:
            return False, None, None
        
        # Check for existing document by content hash (most reliable)
        existing_doc = self.find_existing_document_by_hash(content_hash)
        if existing_doc:
            return True, existing_doc, content_hash
        
        # Fallback: check by filename
        filename = Path(file_path).name
        existing_doc = self.find_existing_document_by_filename(filename)
        if existing_doc:
            self.logger.warning(f"Found document with same filename but different content: {filename}")
            return False, existing_doc, content_hash  # Different content, need to reprocess
        
        return False, None, content_hash
    
    def list_cached_documents(self) -> List[Dict[str, Any]]:
        """List all cached documents"""
        try:
            if not self.processed_dir.exists():
                return []
            
            # Find all metadata files
            metadata_files = list(self.processed_dir.glob("*_metadata.json"))
            
            cached_docs = []
            for metadata_file in metadata_files:
                try:
                    import json
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Extract basic info
                    doc_info = {
                        "document_id": metadata.get("document_id", "unknown"),
                        "filename": metadata.get("original_filename", "unknown"),
                        "processed_at": metadata.get("extracted_at", "unknown"),
                        "text_length": metadata.get("text_length", 0),
                        "content_hash": metadata.get("content_hash", ""),
                        "metadata_file": str(metadata_file)
                    }
                    
                    cached_docs.append(doc_info)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process metadata file {metadata_file}: {e}")
                    continue
            
            return cached_docs
            
        except Exception as e:
            self.logger.error(f"Failed to list cached documents: {e}")
            return []
    
    def clear_cache(self, document_id: str = None) -> Dict[str, Any]:
        """Clear cache for specific document or all documents"""
        try:
            if not self.processed_dir.exists():
                return {"removed_files": 0, "errors": []}
            
            removed_files = 0
            errors = []
            
            if document_id:
                # Remove specific document files
                pattern = f"{document_id}_*"
                files_to_remove = list(self.processed_dir.glob(pattern))
            else:
                # Remove all processed files
                files_to_remove = [
                    f for f in self.processed_dir.iterdir() 
                    if f.is_file() and (f.name.endswith('.json') or f.name.endswith('.md') or f.name.endswith('.txt'))
                ]
            
            for file_path in files_to_remove:
                try:
                    file_path.unlink()
                    removed_files += 1
                except Exception as e:
                    errors.append(f"Failed to remove {file_path.name}: {str(e)}")
            
            self.logger.info(f"Cache cleanup: removed {removed_files} files, {len(errors)} errors")
            
            return {
                "removed_files": removed_files,
                "errors": errors
            }
            
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            return {"removed_files": 0, "errors": [str(e)]}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            if not self.processed_dir.exists():
                return {
                    "total_documents": 0,
                    "total_files": 0,
                    "cache_size_mb": 0,
                    "oldest_document": None,
                    "newest_document": None
                }
            
            # Count files and calculate sizes
            total_files = 0
            total_size = 0
            document_dates = []
            
            for file_path in self.processed_dir.iterdir():
                if file_path.is_file():
                    total_files += 1
                    total_size += file_path.stat().st_size
                    
                    # Extract dates from metadata files
                    if file_path.name.endswith('_metadata.json'):
                        try:
                            import json
                            with open(file_path, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            extracted_at = metadata.get("extracted_at")
                            if extracted_at:
                                document_dates.append(extracted_at)
                        except Exception:
                            pass
            
            # Count unique documents (metadata files)
            total_documents = len(list(self.processed_dir.glob("*_metadata.json")))
            
            # Find oldest and newest
            oldest_document = min(document_dates) if document_dates else None
            newest_document = max(document_dates) if document_dates else None
            
            return {
                "total_documents": total_documents,
                "total_files": total_files,
                "cache_size_mb": round(total_size / (1024 * 1024), 2),
                "oldest_document": oldest_document,
                "newest_document": newest_document,
                "cache_directory": str(self.processed_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}