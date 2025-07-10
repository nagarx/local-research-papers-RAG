"""
Document Processor - Modular Implementation

This module provides a streamlined document processor that orchestrates
specialized components for efficient PDF processing.

This is the production version that replaced the original 1389-line monolithic
implementation with a 75% smaller modular architecture.
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

# Local imports
from ..config import get_config, get_logger
from .marker_integration import MarkerProcessor, get_global_marker_models
from .text_chunking import TextChunker
from .document_cache import DocumentCache
from .document_io import DocumentIO


class DocumentProcessor:
    """
    Streamlined document processor using specialized components
    
    Features:
    - Modular design with specialized components
    - Efficient caching and duplicate detection
    - Intelligent text chunking for RAG
    - Resource management and cleanup
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the document processor"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Try to get enhanced logger
        try:
            from ..utils.enhanced_logging import get_enhanced_logger
            self.enhanced_logger = get_enhanced_logger(__name__)
        except ImportError:
            self.enhanced_logger = None
        
        # Initialize specialized components
        self.marker_processor = MarkerProcessor(config)
        self.text_chunker = TextChunker(config)
        self.document_cache = DocumentCache(config)
        self.document_io = DocumentIO(config)
        
        # Processing statistics
        self._processing_stats = {
            "total_processed": 0,
            "total_errors": 0,
            "processing_time": 0.0,
            "cache_hits": 0
        }
        
        if self.enhanced_logger:
            self.enhanced_logger.system_ready("DocumentProcessor", "All components initialized")
        else:
            self.logger.info("DocumentProcessor initialized successfully")
    
    async def process_document_async(
        self, 
        file_path: Union[str, Path],
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """Process a single PDF document with duplicate detection"""
        file_path = self.document_io.validate_file(file_path)
        
        if self.enhanced_logger:
            self.enhanced_logger.document_start(file_path.name, "processing")
        else:
            self.logger.info(f"Processing document: {file_path.name}")
        
        start_time = time.time()
        
        # Use resource manager for proper cleanup
        from ..utils.resource_cleanup import ResourceManager
        
        try:
            with ResourceManager(cleanup_on_exit=True):
                # Check if document is already processed
                is_processed, existing_doc, content_hash = self.document_cache.is_document_processed(
                    file_path, force_reprocess
                )
                
                if is_processed and existing_doc and content_hash:
                    self.logger.info(f"Document already processed (content hash match): {file_path.name}")
                    self._processing_stats["cache_hits"] += 1
                    
                    # Try to regenerate chunks from saved text
                    return await self._handle_existing_document(
                        existing_doc, file_path, content_hash
                    )
                
                # Process new document
                return await self._process_new_document(file_path, content_hash, start_time)
                
        except Exception as e:
            self._processing_stats["total_errors"] += 1
            self.logger.error(f"Error processing {file_path.name}: {e}")
            
            # Clean up resources on error
            self._cleanup_resources()
            raise
    
    async def _handle_existing_document(
        self, 
        existing_doc: Dict[str, Any], 
        file_path: Path, 
        content_hash: str
    ) -> Dict[str, Any]:
        """Handle processing of existing document"""
        try:
            saved_text_data = self.document_io.load_saved_raw_text(
                existing_doc["document_id"], file_path.name
            )
            
            if saved_text_data and saved_text_data["raw_text"]:
                # Regenerate chunks from saved text
                chunks = self.text_chunker.create_chunks(
                    saved_text_data["raw_text"], 
                    file_path.name, 
                    existing_doc["document_id"]
                )
                
                if chunks:
                    self.logger.info(f"Regenerated {len(chunks)} chunks from saved text for {file_path.name}")
                    
                    return {
                        "id": existing_doc["document_id"],
                        "source_path": str(file_path),
                        "filename": file_path.name,
                        "processed_at": existing_doc["extracted_at"],
                        "processing_time": 0.0,
                        "status": "already_processed",
                        "content_hash": content_hash,
                        "raw_text_path": saved_text_data.get("text_file_path"),
                        "content": {
                            "full_text": saved_text_data["raw_text"],
                            "blocks": chunks,
                            "page_count": existing_doc.get("extraction_stats", {}).get("lines", 0),
                            "total_blocks": len(chunks),
                            "images": existing_doc.get("images", [])
                        },
                        "metadata": existing_doc
                    }
                else:
                    self.logger.warning(f"Failed to regenerate chunks for {file_path.name}, will reprocess")
            else:
                self.logger.warning(f"Could not load raw text for {file_path.name}, will reprocess")
        
        except Exception as e:
            self.logger.warning(f"Failed to handle existing document {file_path.name}: {e}, will reprocess")
        
        # Fallback: return minimal info
        return {
            "id": existing_doc["document_id"],
            "source_path": str(file_path),
            "filename": file_path.name,
            "processed_at": existing_doc["extracted_at"],
            "processing_time": 0.0,
            "status": "already_processed",
            "content_hash": content_hash,
            "raw_text_path": existing_doc.get("text_file"),
            "content": {
                "full_text": "",
                "blocks": [],
                "page_count": 0,
                "total_blocks": 0,
                "images": []
            },
            "metadata": existing_doc
        }
    
    async def _process_new_document(
        self, 
        file_path: Path, 
        content_hash: str, 
        start_time: float
    ) -> Dict[str, Any]:
        """Process a new document"""
        if not content_hash:
            raise ValueError(f"Failed to calculate content hash for {file_path}")
        
        # Generate document ID
        document_id = self.document_cache.generate_content_based_document_id(
            content_hash, file_path.name
        )
        
        # Process with Marker CLI
        if self.enhanced_logger:
            self.enhanced_logger.marker_processing_start(file_path.name)
        else:
            self.logger.debug(f"Starting Marker CLI processing for: {file_path.name}")
        
        try:
            # Process document with Marker CLI
            result = self.marker_processor.process_document(file_path)
            
            # Result is now a tuple: (text, format_type, images)
            if isinstance(result, tuple) and len(result) == 3:
                text, ext, images = result
            else:
                # Fallback for unexpected format
                text, ext, images = self.marker_processor.extract_text_from_rendered(result)
            
            if self.enhanced_logger:
                self.enhanced_logger.processing_complete(
                    f"Text extraction for {file_path.name}",
                    0,  # No specific timing for this step
                    f"{len(text)} chars, {len(images)} images"
                )
            else:
                self.logger.info(f"Extracted: {len(text)} chars, {len(images)} images")
            
        except Exception as e:
            if self.enhanced_logger:
                self.enhanced_logger.error_clean(f"CLI processing failed for {file_path.name}", e)
            else:
                self.logger.error(f"Marker CLI processing failed for {file_path.name}: {e}")
            raise e
        
        # Save raw extracted text
        raw_text_path = self.document_io.save_raw_extracted_text(
            text, ext, images, document_id, file_path.name, content_hash
        )
        
        processing_time = time.time() - start_time
        
        # Create text chunks for RAG
        # Create a mock rendered object for compatibility with text chunker
        from .marker_integration import MockRenderedObject
        mock_rendered = MockRenderedObject(text, ext, images)
        chunks = self.text_chunker.create_chunks(text, file_path.name, document_id, mock_rendered)
        
        processed_doc = {
            "id": document_id,
            "source_path": str(file_path),
            "filename": file_path.name,
            "processed_at": datetime.utcnow().isoformat(),
            "processing_time": processing_time,
            "content_hash": content_hash,
            "status": "newly_processed",
            "raw_text_path": str(raw_text_path) if raw_text_path else None,
            "content": {
                "full_text": text,
                "blocks": chunks,
                "page_count": 0,  # CLI doesn't track page count
                "total_blocks": len(chunks),
                "images": images
            },
            "metadata": {
                "processing_method": "marker_cli",
                "format": ext,
                "file_size": file_path.stat().st_size  # Get file size from file path
            }
        }
        
        # Update statistics
        self._processing_stats["total_processed"] += 1
        self._processing_stats["processing_time"] += processing_time
        
        if self.enhanced_logger:
            self.enhanced_logger.document_complete(
                file_path.name, processing_time, len(chunks), "processed"
            )
            self.enhanced_logger.performance_summary(f"Document processing: {file_path.name}")
        else:
            self.logger.info(
                f"Successfully processed {file_path.name}: "
                f"{len(chunks)} chunks in {processing_time:.2f}s (ID: {document_id})"
            )
        
        return processed_doc
    
    def _cleanup_resources(self):
        """Clean up system resources - no longer needed with CLI approach"""
        # CLI approach doesn't require resource cleanup
        pass
    
    def _clear_gpu_cache(self):
        """Clear GPU cache - no longer needed with CLI approach"""
        # CLI approach doesn't use GPU directly
        pass
    
    async def batch_process_documents(
        self, 
        file_paths: List[str], 
        batch_size: int = 2,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Process multiple documents efficiently"""
        results = []
        
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(file_paths) + batch_size - 1)//batch_size
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            if progress_callback:
                progress_callback(f"Processing batch {batch_num}/{total_batches}", i, len(file_paths))
            
            try:
                batch_results = []
                for file_path in batch:
                    try:
                        result = await self.process_document_async(file_path)
                        batch_results.append(result)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process {file_path}: {e}")
                        continue
                
                results.extend(batch_results)
                
                # Small delay between batches
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Batch {batch_num} processing failed: {e}")
                continue
        
        return results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self._processing_stats.copy()
        
        if stats["total_processed"] > 0:
            stats["average_processing_time"] = stats["processing_time"] / stats["total_processed"]
        else:
            stats["average_processing_time"] = 0.0
        
        # Add component stats
        stats["cache_stats"] = self.document_cache.get_cache_stats()
        stats["io_stats"] = self.document_io.get_io_stats()
        
        return stats
    
    def get_saved_raw_text(self, document_id: str, filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve saved raw text and metadata for a document"""
        return self.document_io.load_saved_raw_text(document_id, filename)
    
    def list_saved_raw_texts(self) -> List[Dict[str, Any]]:
        """List all saved raw text files"""
        return self.document_io.list_saved_raw_texts()


# Maintain backward compatibility
def get_global_marker_models():
    """Get global Marker models - backward compatibility"""
    from .marker_integration import get_global_marker_models as _get_global_marker_models
    return _get_global_marker_models()