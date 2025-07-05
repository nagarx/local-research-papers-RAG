"""
Document Processor - Marker Integration

This module handles PDF document processing using the Marker library,
optimized for academic research papers with precise source attribution.

FOLLOWS MARKER DOCUMENTATION PATTERNS EXACTLY for maximum performance.
"""

import asyncio
import hashlib
import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import logging

# Core dependencies
from bs4 import BeautifulSoup
import torch

# Marker imports
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered

# Local imports
from ..config import get_config, get_logger

# GLOBAL MODEL CACHE - Initialize once, reuse everywhere (per documentation)
_GLOBAL_MARKER_MODELS = None
_GLOBAL_MODEL_LOAD_TIME = None

def get_global_marker_models():
    """Get global Marker models - initialize once, reuse everywhere"""
    global _GLOBAL_MARKER_MODELS, _GLOBAL_MODEL_LOAD_TIME
    
    if _GLOBAL_MARKER_MODELS is None:
        logger = get_logger(__name__)
        logger.info("Loading Marker models globally (one-time setup)...")
        start_time = time.time()
        _GLOBAL_MARKER_MODELS = create_model_dict()
        _GLOBAL_MODEL_LOAD_TIME = time.time() - start_time
        logger.info(f"Global Marker models loaded in {_GLOBAL_MODEL_LOAD_TIME:.2f}s")
    
    return _GLOBAL_MARKER_MODELS


class DocumentProcessor:
    """
    Streamlined document processor using Marker following documentation patterns exactly
    
    Key optimizations:
    - Global model reuse (load once, use everywhere)
    - Simplified processing pipeline  
    - Direct text_from_rendered usage
    - GPU memory management
    - Minimal abstraction layers
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the document processor"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Use global models (following documentation pattern)
        self.marker_models = get_global_marker_models()
        
        # Setup converter (following documentation pattern)
        self._setup_converter()
        
        # Processing state
        self._processing_stats = {
            "total_processed": 0,
            "total_errors": 0,
            "processing_time": 0.0,
            "cache_hits": 0
        }
        
        self.logger.info("DocumentProcessor initialized successfully")
    
    def _setup_converter(self):
        """Setup converter following documentation pattern exactly"""
        self.logger.info("Setting up Marker converter...")
        
        # Configuration following documentation best practices
        config = {
            "output_format": "markdown",   # Standard format
            "paginate_output": True,      # Page attribution for RAG
            "format_lines": True,         # Better math formatting  
            "extract_images": True,       # Extract images
            "use_llm": False,            # No LLM for speed
            "force_ocr": False,          # Don't force OCR unless needed
            "disable_tqdm": True,        # No progress bars
        }
        
        # Only enable LLM if configured and API key available
        if self.config.marker.use_llm and hasattr(self.config.marker, 'gemini_api_key') and self.config.marker.gemini_api_key:
            config.update({
                "use_llm": True,
                "redo_inline_math": True,
                "llm_service": "marker.services.gemini.GoogleGeminiService",
            })
            self.logger.info("LLM features enabled for Marker")
        else:
            self.logger.info("LLM features disabled for Marker (no API key or not requested)")
        
        # Create converter (exactly as in documentation)
        config_parser = ConfigParser(config)
        self.converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=self.marker_models,  # Reuse global models
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service()
        )
        
        self.logger.info("Marker converter initialized")
    
    async def process_document_async(
        self, 
        file_path: Union[str, Path],
        force_reprocess: bool = False
    ) -> Dict[str, Any]:
        """Process a single PDF document following documentation pattern"""
        file_path = Path(file_path)
        
        # Validate file
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() != '.pdf':
            raise ValueError(f"Only PDF files are supported: {file_path}")
        
        self.logger.info(f"Processing document: {file_path.name}")
        start_time = time.time()
        
        try:
            # Process with Marker in thread pool (async wrapper)
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                rendered = await loop.run_in_executor(
                    executor,
                    self._process_with_marker_sync,
                    str(file_path)
                )
            
            # Extract text using proper Marker API (following documentation)
            try:
                text, ext, images = text_from_rendered(rendered)
                self.logger.info(f"Extracted: {len(text)} chars, {len(images)} images")
            except Exception as e:
                self.logger.warning(f"text_from_rendered failed, using fallback: {e}")
                text, ext, images = self._fallback_extraction(rendered)
            
            # Create streamlined document structure (minimal processing)
            document_id = f"doc_{int(time.time())}_{hash(str(file_path)) % 10000}"
            processing_time = time.time() - start_time
            
            # Split text into chunks for RAG with proper page attribution
            chunks = self._create_text_chunks(text, file_path.name, document_id, rendered)
            
            processed_doc = {
                "id": document_id,
                "source_path": str(file_path),
                "filename": file_path.name,
                "processed_at": datetime.utcnow().isoformat(),
                "processing_time": processing_time,
                "content": {
                    "full_text": text,
                    "blocks": chunks,  # Simple chunks for RAG
                    "page_count": getattr(rendered, 'page_count', 0) if hasattr(rendered, 'page_count') else 0,
                    "total_blocks": len(chunks),
                    "images": images
                },
                "metadata": getattr(rendered, 'metadata', {})
            }
            
            # Update statistics
            self._processing_stats["total_processed"] += 1
            self._processing_stats["processing_time"] += processing_time
            
            # Clear GPU cache (following documentation)
            self._clear_gpu_cache()
            
            self.logger.info(
                f"Successfully processed {file_path.name}: "
                f"{len(chunks)} chunks in {processing_time:.2f}s"
            )
            
            return processed_doc
            
        except Exception as e:
            self._processing_stats["total_errors"] += 1
            self.logger.error(f"Error processing {file_path.name}: {e}")
            
            # Clear GPU cache on error
            self._clear_gpu_cache()
            raise
    
    def _process_with_marker_sync(self, file_path: str):
        """Process document with Marker (synchronous, following documentation)"""
        return self.converter(file_path)
    
    def _fallback_extraction(self, rendered):
        """Fallback if text_from_rendered fails"""
        if hasattr(rendered, 'markdown'):
            return rendered.markdown, "md", getattr(rendered, 'images', {})
        elif hasattr(rendered, 'html'):
            return rendered.html, "html", getattr(rendered, 'images', {})
        else:
            return str(rendered), "txt", {}
    
    def _create_text_chunks(self, text: str, filename: str, document_id: str, rendered=None) -> List[Dict[str, Any]]:
        """Create text chunks for RAG with proper page attribution"""
        if not text:
            return []
        
        # Extract page breaks from Markdown if available
        page_markers = self._extract_page_markers(text)
        
        # Simple chunking by paragraphs/sections
        sections = text.split('\n\n')
        chunks = []
        current_chunk = ""
        chunk_size = self.config.document_processing.max_chunk_size
        current_position = 0
        
        for i, section in enumerate(sections):
            if len(current_chunk) + len(section) < chunk_size:
                current_chunk += section + "\n\n"
                current_position += len(section) + 2  # +2 for \n\n
            else:
                if current_chunk.strip():
                    # Determine page number based on position in text
                    page_number = self._get_page_number_for_position(current_position, page_markers)
                    
                    chunk = {
                        "id": f"{document_id}_chunk_{len(chunks)}",
                        "text": current_chunk.strip(),
                        "chunk_index": len(chunks),
                        "source_info": {
                            "document_id": document_id,
                            "document_name": filename,
                            "page_number": page_number,
                            "block_index": len(chunks),
                            "block_type": "text"
                        }
                    }
                    chunks.append(chunk)
                current_chunk = section + "\n\n"
                current_position += len(section) + 2
        
        # Add final chunk
        if current_chunk.strip():
            page_number = self._get_page_number_for_position(current_position, page_markers)
            chunk = {
                "id": f"{document_id}_chunk_{len(chunks)}",
                "text": current_chunk.strip(),
                "chunk_index": len(chunks),
                "source_info": {
                    "document_id": document_id,
                    "document_name": filename,
                    "page_number": page_number,
                    "block_index": len(chunks),
                    "block_type": "text"
                }
            }
            chunks.append(chunk)
        
        return chunks
    
    def _extract_page_markers(self, text: str) -> List[int]:
        """Extract page break positions from Marker's paginated output"""
        page_positions = [0]  # Start of document is page 1
        
        # Look for common page break markers that Marker uses
        lines = text.split('\n')
        position = 0
        
        for line in lines:
            position += len(line) + 1  # +1 for \n
            
            # Common page break patterns in Marker output
            if (line.strip().startswith('---') or  # Horizontal rule
                line.strip().startswith('# ') or   # New major section
                'Page ' in line or                 # Explicit page reference
                len(line.strip()) == 0):           # Empty lines often indicate breaks
                
                # Only add if it's a significant position change
                if position > page_positions[-1] + 1000:  # At least 1000 chars apart
                    page_positions.append(position)
        
        return page_positions
    
    def _get_page_number_for_position(self, position: int, page_markers: List[int]) -> int:
        """Determine page number based on character position in text"""
        if not page_markers:
            return 1
        
        page = 1
        for marker_pos in page_markers:
            if position >= marker_pos:
                page += 1
            else:
                break
        
        # Ensure reasonable page numbers (1-based, max 100 for safety)
        return max(1, min(page - 1, 100))
    
    def _clear_gpu_cache(self):
        """Clear GPU cache (following documentation)"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.debug("GPU cache cleared")
        except Exception as e:
            self.logger.warning(f"Failed to clear GPU cache: {e}")
    
    async def batch_process_documents(
        self, 
        file_paths: List[str], 
        batch_size: int = 2,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """Process multiple documents efficiently (following documentation patterns)"""
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
                
                # Clear GPU cache between batches (documentation recommendation)
                self._clear_gpu_cache()
                
                # Small delay between batches
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Batch {batch_num} processing failed: {e}")
                self._clear_gpu_cache()
                continue
        
        return results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self._processing_stats.copy()
        
        if stats["total_processed"] > 0:
            stats["average_processing_time"] = stats["processing_time"] / stats["total_processed"]
        else:
            stats["average_processing_time"] = 0.0
        
        return stats
 