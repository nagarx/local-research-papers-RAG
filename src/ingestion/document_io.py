"""
Document I/O Module

This module handles file I/O operations for document processing including
saving and loading raw extracted text and metadata.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from ..config import get_config, get_logger


class DocumentIO:
    """
    Handles document I/O operations for saving and loading processed documents
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the document I/O manager"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Storage directory
        self.processed_dir = self.config.storage_paths.processed_dir
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("DocumentIO initialized successfully")
    
    def save_raw_extracted_text(
        self, 
        text: str, 
        ext: str, 
        images: Dict[str, Any], 
        document_id: str, 
        filename: str, 
        content_hash: str
    ) -> Optional[Path]:
        """Save raw extracted text from Marker to disk with metadata"""
        try:
            # Create sanitized filename (remove special characters)
            safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
            base_name = Path(safe_filename).stem  # Remove extension
            
            # Create file paths
            text_file = self.processed_dir / f"{document_id}_{base_name}_raw.{ext}"
            metadata_file = self.processed_dir / f"{document_id}_{base_name}_metadata.json"
            
            # Save raw extracted text
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Save extraction metadata
            extraction_metadata = {
                "document_id": document_id,
                "original_filename": filename,
                "extracted_at": datetime.utcnow().isoformat(),
                "extraction_format": ext,
                "text_length": len(text),
                "image_count": len(images),
                "text_file": str(text_file.name),
                "images": list(images.keys()) if images else [],
                "extraction_stats": {
                    "lines": len(text.split('\n')),
                    "paragraphs": len([p for p in text.split('\n\n') if p.strip()]),
                    "characters": len(text),
                    "words": len(text.split())
                },
                "content_hash": content_hash
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_metadata, f, indent=2, ensure_ascii=False)
            
            # Save images if available
            if images:
                images_dir = self.processed_dir / f"{document_id}_{base_name}_images"
                images_dir.mkdir(exist_ok=True)
                
                for img_name, img_data in images.items():
                    if isinstance(img_data, bytes):
                        img_file = images_dir / img_name
                        with open(img_file, 'wb') as f:
                            f.write(img_data)
            
            self.logger.info(
                f"Saved raw extracted text: {text_file.name} "
                f"({len(text)} chars, {len(images)} images)"
            )
            
            return text_file
            
        except Exception as e:
            self.logger.error(f"Failed to save raw extracted text for {filename}: {e}")
            return None
    
    def load_saved_raw_text(self, document_id: str, filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve saved raw text and metadata for a document"""
        try:
            if filename:
                # Create sanitized filename like in save method
                safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
                base_name = Path(safe_filename).stem
                text_pattern = f"{document_id}_{base_name}_raw.*"
                metadata_pattern = f"{document_id}_{base_name}_metadata.json"
            else:
                # Search by document_id only
                text_pattern = f"{document_id}_*_raw.*"
                metadata_pattern = f"{document_id}_*_metadata.json"
            
            # Find text file
            text_files = list(self.processed_dir.glob(text_pattern))
            metadata_files = list(self.processed_dir.glob(metadata_pattern))
            
            if not text_files:
                self.logger.warning(f"No raw text file found for document_id: {document_id}")
                return None
            
            text_file = text_files[0]  # Take first match
            
            # Read raw text
            with open(text_file, 'r', encoding='utf-8') as f:
                raw_text = f.read()
            
            # Read metadata if available
            metadata = {}
            if metadata_files:
                with open(metadata_files[0], 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            return {
                "document_id": document_id,
                "raw_text": raw_text,
                "text_file_path": str(text_file),
                "metadata": metadata,
                "file_size": text_file.stat().st_size,
                "last_modified": datetime.fromtimestamp(text_file.stat().st_mtime).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve raw text for document_id {document_id}: {e}")
            return None
    
    def list_saved_raw_texts(self) -> List[Dict[str, Any]]:
        """List all saved raw text files"""
        try:
            if not self.processed_dir.exists():
                return []
            
            # Find all metadata files
            metadata_files = list(self.processed_dir.glob("*_metadata.json"))
            
            saved_texts = []
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    # Find corresponding text file
                    doc_id = metadata.get("document_id", "")
                    original_name = metadata.get("original_filename", "")
                    
                    if doc_id and original_name:
                        safe_filename = "".join(c for c in original_name if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
                        base_name = Path(safe_filename).stem
                        text_pattern = f"{doc_id}_{base_name}_raw.*"
                        text_files = list(self.processed_dir.glob(text_pattern))
                        
                        if text_files:
                            text_file = text_files[0]
                            saved_texts.append({
                                "document_id": doc_id,
                                "original_filename": original_name,
                                "text_file_path": str(text_file),
                                "metadata_file_path": str(metadata_file),
                                "extracted_at": metadata.get("extracted_at", ""),
                                "text_length": metadata.get("text_length", 0),
                                "file_size": text_file.stat().st_size
                            })
                
                except Exception as e:
                    self.logger.warning(f"Failed to process metadata file {metadata_file}: {e}")
                    continue
            
            return saved_texts
            
        except Exception as e:
            self.logger.error(f"Failed to list saved raw texts: {e}")
            return []
    
    def validate_file(self, file_path: Union[str, Path]) -> Path:
        """Validate file exists and is a PDF"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        if file_path.suffix.lower() != '.pdf':
            raise ValueError(f"Only PDF files are supported: {file_path}")
        
        return file_path
    
    def clean_filename(self, filename: str) -> str:
        """Create a sanitized filename"""
        return "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
    
    def get_io_stats(self) -> Dict[str, Any]:
        """Get I/O statistics"""
        try:
            if not self.processed_dir.exists():
                return {
                    "processed_directory": str(self.processed_dir),
                    "total_files": 0,
                    "total_size_mb": 0,
                    "file_types": {}
                }
            
            total_files = 0
            total_size = 0
            file_types = {}
            
            for file_path in self.processed_dir.iterdir():
                if file_path.is_file():
                    total_files += 1
                    total_size += file_path.stat().st_size
                    
                    # Count by file extension
                    ext = file_path.suffix.lower()
                    file_types[ext] = file_types.get(ext, 0) + 1
            
            return {
                "processed_directory": str(self.processed_dir),
                "total_files": total_files,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_types": file_types
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get I/O stats: {e}")
            return {"error": str(e)}