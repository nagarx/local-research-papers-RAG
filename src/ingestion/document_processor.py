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
        
        # Try to get enhanced logger
        try:
            from ..utils.enhanced_logging import get_enhanced_logger
            self.enhanced_logger = get_enhanced_logger(__name__)
        except ImportError:
            self.enhanced_logger = None
        
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
        
        if self.enhanced_logger:
            self.enhanced_logger.system_ready("DocumentProcessor", "Marker models loaded and converter ready")
        else:
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
        """Process a single PDF document with duplicate detection"""
        file_path = Path(file_path)
        
        # Validate file
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() != '.pdf':
            raise ValueError(f"Only PDF files are supported: {file_path}")
        
        if self.enhanced_logger:
            self.enhanced_logger.document_start(file_path.name, "processing")
        else:
            self.logger.info(f"Processing document: {file_path.name}")
        start_time = time.time()
        
        try:
            # Calculate content hash for duplicate detection
            content_hash = self._calculate_file_hash(file_path)
            if not content_hash:
                raise ValueError(f"Failed to calculate content hash for {file_path}")
            
            # Check for existing document by content hash (most reliable)
            if not force_reprocess:
                existing_doc = self._find_existing_document_by_hash(content_hash)
                if existing_doc:
                    self.logger.info(f"Document already processed (content hash match): {file_path.name}")
                    
                    # Load the saved raw text and regenerate chunks for indexing
                    try:
                        saved_text_data = self.get_saved_raw_text(existing_doc["document_id"], file_path.name)
                        if saved_text_data and saved_text_data["raw_text"]:
                            # Regenerate chunks from saved text
                            chunks = self._create_text_chunks(
                                saved_text_data["raw_text"], 
                                file_path.name, 
                                existing_doc["document_id"]
                            )
                            
                            self.logger.info(f"Regenerated {len(chunks)} chunks from saved text for {file_path.name}")
                            
                            return {
                                "id": existing_doc["document_id"],
                                "source_path": str(file_path),
                                "filename": file_path.name,
                                "processed_at": existing_doc["extracted_at"],
                                "processing_time": 0.0,  # No processing time for existing document
                                "status": "already_processed",
                                "content_hash": content_hash,
                                "raw_text_path": saved_text_data.get("text_file_path"),
                                "content": {
                                    "full_text": saved_text_data["raw_text"],
                                    "blocks": chunks,  # Regenerated chunks for indexing
                                    "page_count": existing_doc.get("extraction_stats", {}).get("lines", 0),
                                    "total_blocks": len(chunks),
                                    "images": existing_doc.get("images", [])
                                },
                                "metadata": existing_doc
                            }
                        else:
                            self.logger.warning(f"Could not load raw text for {file_path.name}, will reprocess")
                    except Exception as e:
                        self.logger.warning(f"Failed to regenerate chunks for {file_path.name}: {e}, will reprocess")
                    
                    # Fallback: return minimal info if chunk regeneration fails
                    return {
                        "id": existing_doc["document_id"],
                        "source_path": str(file_path),
                        "filename": file_path.name,
                        "processed_at": existing_doc["extracted_at"],
                        "processing_time": 0.0,  # No processing time for existing document
                        "status": "already_processed",
                        "content_hash": content_hash,
                        "raw_text_path": existing_doc.get("text_file"),
                        "content": {
                            "full_text": "",  # Can be loaded on demand
                            "blocks": [],     # Would need to be regenerated if needed
                            "page_count": 0,
                            "total_blocks": 0,
                            "images": []
                        },
                        "metadata": existing_doc
                    }
                
                # Fallback: check by filename
                existing_doc = self._find_existing_document_by_filename(file_path.name)
                if existing_doc:
                    self.logger.warning(f"Found document with same filename but different content: {file_path.name}")
            
            # Generate content-based document ID
            document_id = self._generate_content_based_document_id(content_hash, file_path.name)
            
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
            
            # Save raw extracted text before processing (with content hash)
            raw_text_path = self._save_raw_extracted_text_enhanced(
                text, ext, images, document_id, file_path.name, content_hash
            )
            
            processing_time = time.time() - start_time
            
            # Split text into chunks for RAG with proper page attribution
            chunks = self._create_text_chunks(text, file_path.name, document_id, rendered)
            
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
            
        except Exception as e:
            self._processing_stats["total_errors"] += 1
            self.logger.error(f"Error processing {file_path.name}: {e}")
            
            # Clear GPU cache on error
            self._clear_gpu_cache()
            raise
    
    def _save_raw_extracted_text(
        self, 
        text: str, 
        ext: str, 
        images: Dict[str, Any], 
        document_id: str, 
        filename: str
    ) -> Optional[Path]:
        """Save raw extracted text from Marker to disk before processing"""
        try:
            # Ensure processed directory exists
            processed_dir = self.config.storage_paths.processed_dir
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Create sanitized filename (remove special characters)
            safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
            base_name = Path(safe_filename).stem  # Remove extension
            
            # Create file paths
            text_file = processed_dir / f"{document_id}_{base_name}_raw.{ext}"
            metadata_file = processed_dir / f"{document_id}_{base_name}_metadata.json"
            
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
                }
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_metadata, f, indent=2, ensure_ascii=False)
            
            # Save images if available
            if images:
                images_dir = processed_dir / f"{document_id}_{base_name}_images"
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
    
    def _save_raw_extracted_text_enhanced(
        self, 
        text: str, 
        ext: str, 
        images: Dict[str, Any], 
        document_id: str, 
        filename: str, 
        content_hash: str
    ) -> Optional[Path]:
        """Save raw extracted text from Marker to disk before processing with content hash"""
        try:
            # Ensure processed directory exists
            processed_dir = self.config.storage_paths.processed_dir
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Create sanitized filename (remove special characters)
            safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
            base_name = Path(safe_filename).stem  # Remove extension
            
            # Create file paths
            text_file = processed_dir / f"{document_id}_{base_name}_raw.{ext}"
            metadata_file = processed_dir / f"{document_id}_{base_name}_metadata.json"
            
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
                images_dir = processed_dir / f"{document_id}_{base_name}_images"
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
        """Create optimized text chunks for RAG with semantic coherence and proper page attribution"""
        if not text:
            return []
        
        # Extract page breaks from Markdown if available
        page_markers = self._extract_page_markers(text)
        
        # Configuration
        chunk_size = self.config.document_processing.max_chunk_size
        overlap_size = self.config.document_processing.chunk_overlap
        
        # Split text into semantic units (sentences/paragraphs)
        semantic_units = self._split_into_semantic_units(text)
        
        chunks = []
        current_chunk = ""
        current_chunk_units = []
        current_position = 0
        
        for unit in semantic_units:
            unit_length = len(unit["text"])
            
            # Check if adding this unit would exceed chunk size
            if (len(current_chunk) + unit_length > chunk_size and 
                current_chunk.strip() and 
                len(current_chunk_units) > 0):
                
                # Finalize current chunk
                chunk_info = self._finalize_chunk(
                    current_chunk_units, current_chunk, document_id, filename, 
                    chunks, page_markers, current_position - len(current_chunk)
                )
                chunks.append(chunk_info)
                
                # Start new chunk with overlap
                overlap_units, overlap_text = self._create_overlap(current_chunk_units, overlap_size)
                current_chunk_units = overlap_units
                current_chunk = overlap_text
            
            # Add current unit to chunk
            current_chunk_units.append(unit)
            current_chunk += unit["text"]
            current_position += unit_length
        
        # Add final chunk if there's content
        if current_chunk.strip():
            chunk_info = self._finalize_chunk(
                current_chunk_units, current_chunk, document_id, filename,
                chunks, page_markers, current_position - len(current_chunk)
            )
            chunks.append(chunk_info)
        
        # Post-process chunks for quality
        chunks = self._post_process_chunks(chunks)
        
        return chunks
    
    def _split_into_semantic_units(self, text: str) -> List[Dict[str, Any]]:
        """Split text into semantic units (sentences, paragraphs, sections)"""
        units = []
        position = 0
        
        # Split by paragraphs first (double newlines)
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if this is a heading or special section
            is_heading = self._is_heading(para)
            is_code_block = self._is_code_block(para)
            is_table = self._is_table(para)
            is_list = self._is_list(para)
            
            if is_heading or is_code_block or is_table or len(para) <= 300:
                # Keep small paragraphs, headings, code blocks, and tables as single units
                units.append({
                    "text": para + "\n\n",
                    "position": position,
                    "type": ("heading" if is_heading else 
                            "code" if is_code_block else 
                            "table" if is_table else 
                            "list" if is_list else "paragraph"),
                    "is_structural": is_heading or is_code_block or is_table
                })
            else:
                # Split long paragraphs into sentences
                sentences = self._split_into_sentences(para)
                for sentence in sentences:
                    if sentence.strip():
                        units.append({
                            "text": sentence + " ",
                            "position": position,
                            "type": "sentence",
                            "is_structural": False
                        })
                        position += len(sentence) + 1
                
                # Add paragraph break
                units.append({
                    "text": "\n\n",
                    "position": position,
                    "type": "paragraph_break",
                    "is_structural": False
                })
            
            position += len(para) + 2  # +2 for \n\n
        
        return units
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved handling of academic text"""
        import re
        
        # Fixed-width lookbehind pattern that handles academic citations and abbreviations
        # Split the complex pattern into multiple simpler patterns
        
        # First, protect common abbreviations by temporarily replacing them
        protected_text = text
        abbreviations = {
            'Dr.': '__DR_DOT__',
            'Mr.': '__MR_DOT__', 
            'Mrs.': '__MRS_DOT__',
            'Ms.': '__MS_DOT__',
            'Prof.': '__PROF_DOT__',
            'Fig.': '__FIG_DOT__',
            'vs.': '__VS_DOT__',
            'etc.': '__ETC_DOT__',
            'al.': '__AL_DOT__',
            'cf.': '__CF_DOT__',
            'e.g.': '__EG_DOT__',
            'i.e.': '__IE_DOT__',
            'p.': '__P_DOT__',
            'pp.': '__PP_DOT__'
        }
        
        # Replace abbreviations with placeholders
        for abbrev, placeholder in abbreviations.items():
            protected_text = protected_text.replace(abbrev, placeholder)
        
        # Simple sentence splitting pattern (fixed-width lookbehind)
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        
        sentences = re.split(sentence_pattern, protected_text)
        
        # Restore abbreviations
        for abbrev, placeholder in abbreviations.items():
            sentences = [s.replace(placeholder, abbrev) for s in sentences]
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter out very short fragments
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _is_heading(self, text: str) -> bool:
        """Check if text is likely a heading"""
        text = text.strip()
        
        # Markdown headings
        if text.startswith('#'):
            return True
        
        # Short lines that are all caps or title case
        if len(text) < 100 and len(text.split()) <= 10:
            if (text.isupper() or text.istitle() or 
                any(char in text for char in ['=', '-', '_']) or
                text.endswith(':') or
                any(word in text.lower() for word in ['chapter', 'section', 'abstract', 'introduction', 'conclusion', 'references'])):
                return True
        
        return False
    
    def _is_code_block(self, text: str) -> bool:
        """Check if text is a code block"""
        text = text.strip()
        
        # Markdown code blocks
        if text.startswith('```') or text.startswith('    '):
            return True
        
        # Mathematical equations
        if text.startswith('$$') or '\\begin{' in text or '\\end{' in text:
            return True
        
        return False
    
    def _is_table(self, text: str) -> bool:
        """Check if text is likely a table"""
        lines = text.strip().split('\n')
        
        # Check for markdown table patterns
        if len(lines) > 1:
            pipe_count = sum(1 for line in lines if '|' in line)
            if pipe_count >= len(lines) * 0.5:  # At least half the lines have pipes
                return True
        
        return False
    
    def _is_list(self, text: str) -> bool:
        """Check if text is a list"""
        text = text.strip()
        lines = text.split('\n')
        
        # Check for markdown list patterns
        list_indicators = ['-', '*', '+']
        numbered_pattern = r'^\d+\.'
        
        import re
        list_lines = 0
        for line in lines:
            line = line.strip()
            if (any(line.startswith(indicator + ' ') for indicator in list_indicators) or
                re.match(numbered_pattern, line)):
                list_lines += 1
        
        return list_lines >= len(lines) * 0.5  # At least half the lines are list items
    
    def _create_overlap(self, units: List[Dict[str, Any]], overlap_size: int) -> Tuple[List[Dict[str, Any]], str]:
        """Create overlap from the end of previous chunk"""
        if not units or overlap_size <= 0:
            return [], ""
        
        # Take units from the end that fit within overlap size
        overlap_units = []
        overlap_text = ""
        
        # Start from the end and work backwards
        for unit in reversed(units):
            if len(overlap_text) + len(unit["text"]) <= overlap_size:
                overlap_units.insert(0, unit)
                overlap_text = unit["text"] + overlap_text
            else:
                break
        
        return overlap_units, overlap_text
    
    def _finalize_chunk(
        self, 
        units: List[Dict[str, Any]], 
        text: str, 
        document_id: str, 
        filename: str,
        existing_chunks: List[Dict[str, Any]], 
        page_markers: List[int], 
        chunk_start_position: int
    ) -> Dict[str, Any]:
        """Finalize a chunk with metadata"""
        
        # Determine page number
        page_number = self._get_page_number_for_position(chunk_start_position, page_markers)
        
        # Determine chunk type based on constituent units
        chunk_types = [unit["type"] for unit in units]
        has_structural = any(unit.get("is_structural", False) for unit in units)
        
        if "heading" in chunk_types:
            chunk_type = "heading_section"
        elif "code" in chunk_types:
            chunk_type = "code_block"
        elif "table" in chunk_types:
            chunk_type = "table"
        elif has_structural:
            chunk_type = "mixed_content"
        else:
            chunk_type = "text"
        
        # Create chunk info
        chunk = {
            "id": f"{document_id}_chunk_{len(existing_chunks)}",
            "text": text.strip(),
            "chunk_index": len(existing_chunks),
            "chunk_type": chunk_type,
            "page_number": page_number,
            "block_type": "Text",  # Keep for compatibility
            "semantic_units": len(units),
            "has_structural_elements": has_structural,
            "source_info": {
                "document_id": document_id,
                "document_name": filename,
                "page_number": page_number,
                "block_index": len(existing_chunks),
                "block_type": "text",
                "chunk_type": chunk_type,
                "start_position": chunk_start_position,
                "end_position": chunk_start_position + len(text)
            }
        }
        
        return chunk
    
    def _post_process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process chunks to improve quality"""
        
        if not chunks:
            return chunks
        
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Skip very short chunks (less than 50 characters) unless they're structural
            text = chunk["text"].strip()
            if len(text) < 50 and not chunk.get("has_structural_elements", False):
                # Try to merge with previous chunk if possible
                if processed_chunks:
                    prev_chunk = processed_chunks[-1]
                    if len(prev_chunk["text"]) + len(text) < self.config.document_processing.max_chunk_size:
                        prev_chunk["text"] += "\n\n" + text
                        prev_chunk["semantic_units"] += chunk.get("semantic_units", 1)
                        continue
            
            # Clean up text
            chunk["text"] = self._clean_chunk_text(text)
            
            # Update indices after any merging
            chunk["chunk_index"] = len(processed_chunks)
            chunk["id"] = f"{chunk['source_info']['document_id']}_chunk_{len(processed_chunks)}"
            chunk["source_info"]["block_index"] = len(processed_chunks)
            
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _clean_chunk_text(self, text: str) -> str:
        """Clean up chunk text for better processing"""
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\n{3,}', '\n\n', text)  # Limit to max 2 consecutive newlines
        text = re.sub(r' {2,}', ' ', text)      # Limit to single spaces
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
        
        # Clean up common formatting issues
        text = text.strip()
        
        return text
    
    def _extract_page_markers(self, text: str) -> List[int]:
        """Extract page break positions from Marker's paginated output"""
        page_positions = [0]  # Start of document is page 1
        
        # Look for Marker's actual page break markers
        lines = text.split('\n')
        position = 0
        
        import re
        
        for line in lines:
            position += len(line) + 1  # +1 for \n
            
            # Look for Marker's page break patterns:
            # 1. {N}------------------------------------------------ (primary page markers)
            line_stripped = line.strip()
            
            # Primary pattern: {N}------------------------------------------------
            page_marker_match = re.match(r'^\{(\d+)\}[-]+$', line_stripped)
            if page_marker_match:
                page_positions.append(position)
                continue
            
            # NOTE: Removed secondary pattern detection to avoid double-counting
            # The {N}------------------------------------------------ markers are sufficient and accurate
        
        return page_positions
    
    def _get_page_number_for_position(self, position: int, page_markers: List[int]) -> int:
        """Determine page number based on character position in text"""
        if not page_markers:
            return 1
        
        # page_markers structure:
        # page_markers[0] = 0 (start of document)
        # page_markers[1] = position of {0} marker (indicates page 1)
        # page_markers[2] = position of {1} marker (indicates page 2)
        # page_markers[3] = position of {2} marker (indicates page 3)
        # etc.
        
        # Default to page 1 for content before any markers
        page = 1
        
        # Check each marker position (skip index 0 which is just document start)
        for i in range(1, len(page_markers)):
            if position >= page_markers[i]:
                # The marker at page_markers[i] indicates page number (i-1)+1 = i
                # But since Marker uses 0-based indexing for page numbers:
                # {0} = page 1, {1} = page 2, {2} = page 3, etc.
                # So marker at index i corresponds to page i
                page = i
            else:
                break
        
        # Ensure page is at least 1 and reasonable
        return max(1, min(page, 50))
    
    def _clear_gpu_cache(self):
        """Clear GPU cache (following documentation)"""
        try:
            from ..utils.torch_utils import clear_gpu_cache
            clear_gpu_cache()
        except ImportError:
            self.logger.warning("Cannot clear GPU cache - torch_utils not available")
    
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
    
    def get_saved_raw_text(self, document_id: str, filename: str = None) -> Optional[Dict[str, Any]]:
        """Retrieve saved raw text and metadata for a document"""
        try:
            processed_dir = self.config.storage_paths.processed_dir
            
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
            text_files = list(processed_dir.glob(text_pattern))
            metadata_files = list(processed_dir.glob(metadata_pattern))
            
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
            processed_dir = self.config.storage_paths.processed_dir
            
            if not processed_dir.exists():
                return []
            
            # Find all metadata files
            metadata_files = list(processed_dir.glob("*_metadata.json"))
            
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
                        text_files = list(processed_dir.glob(text_pattern))
                        
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
    
    def _calculate_file_hash(self, file_path: Union[str, Path]) -> str:
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
    
    def _find_existing_document_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Find existing document by content hash in processed files"""
        try:
            processed_dir = self.config.storage_paths.processed_dir
            
            if not processed_dir.exists():
                return None
            
            # Search through metadata files for matching content hash
            metadata_files = list(processed_dir.glob("*_metadata.json"))
            
            for metadata_file in metadata_files:
                try:
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
    
    def _find_existing_document_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """Find existing document by exact filename match"""
        try:
            processed_dir = self.config.storage_paths.processed_dir
            
            if not processed_dir.exists():
                return None
            
            # Search through metadata files for matching filename
            metadata_files = list(processed_dir.glob("*_metadata.json"))
            
            for metadata_file in metadata_files:
                try:
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
    
    def _generate_content_based_document_id(self, content_hash: str, filename: str) -> str:
        """Generate consistent document ID based on content hash"""
        # Use first 8 characters of content hash + filename hash for consistency
        filename_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        return f"doc_{content_hash[:12]}_{filename_hash}"
 