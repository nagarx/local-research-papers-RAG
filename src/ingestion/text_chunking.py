"""
Text Chunking Module

This module handles intelligent text chunking for RAG with semantic coherence
and proper page attribution.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from ..config import get_config, get_logger


class TextChunker:
    """
    Handles intelligent text chunking for RAG with semantic coherence
    and proper page attribution.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the text chunker"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Configuration
        self.chunk_size = self.config.document_processing.max_chunk_size
        self.overlap_size = self.config.document_processing.chunk_overlap
        
        # Validate configuration
        if self.chunk_size <= 0:
            self.logger.warning(f"Invalid chunk_size {self.chunk_size}, using default 1000")
            self.chunk_size = 1000
        
        if self.overlap_size < 0:
            self.logger.warning(f"Invalid overlap_size {self.overlap_size}, using default 200")
            self.overlap_size = 200
        
        self.logger.info(f"TextChunker initialized with chunk_size={self.chunk_size}, overlap_size={self.overlap_size}")
    
    def create_chunks(self, text: str, filename: str, document_id: str, rendered=None) -> List[Dict[str, Any]]:
        """Create optimized text chunks for RAG with semantic coherence and proper page attribution"""
        if not text or not text.strip():
            self.logger.warning(f"Empty or whitespace-only text provided for chunking: {filename}")
            return []
        
        if not document_id or not document_id.strip():
            self.logger.error(f"Invalid document_id provided for chunking: {filename}")
            return []
        
        try:
            # Extract page breaks from Markdown if available
            page_markers = self._extract_page_markers(text)
            
            # Split text into semantic units (sentences/paragraphs)
            semantic_units = self._split_into_semantic_units(text)
            
            if not semantic_units:
                self.logger.warning(f"No semantic units found in text for {filename}")
                return self._create_fallback_chunk(text, document_id, filename)
            
            chunks = []
            current_chunk = ""
            current_chunk_units = []
            current_position = 0
            
            for unit in semantic_units:
                unit_length = len(unit["text"])
                
                # Check if adding this unit would exceed chunk size
                if (len(current_chunk) + unit_length > self.chunk_size and 
                    current_chunk.strip() and 
                    len(current_chunk_units) > 0):
                    
                    # Finalize current chunk
                    chunk_info = self._finalize_chunk(
                        current_chunk_units, current_chunk, document_id, filename, 
                        chunks, page_markers, current_position - len(current_chunk)
                    )
                    chunks.append(chunk_info)
                    
                    # Start new chunk with overlap
                    overlap_units, overlap_text = self._create_overlap(current_chunk_units, self.overlap_size)
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
            
            # Final validation
            if not chunks:
                self.logger.warning(f"No chunks created after post-processing for {filename}, creating fallback")
                chunks = self._create_fallback_chunk(text, document_id, filename)
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error creating text chunks for {filename}: {e}")
            return self._create_fallback_chunk(text, document_id, filename)
    
    def _create_fallback_chunk(self, text: str, document_id: str, filename: str) -> List[Dict[str, Any]]:
        """Create a fallback chunk when normal chunking fails"""
        fallback_chunk = {
            "id": f"{document_id}_chunk_0",
            "text": text.strip()[:self.chunk_size],
            "chunk_index": 0,
            "chunk_type": "text",
            "page_number": 1,
            "block_type": "Text",
            "semantic_units": 1,
            "has_structural_elements": False,
            "source_info": {
                "document_id": document_id,
                "document_name": filename,
                "page_number": 1,
                "block_index": 0,
                "block_type": "text",
                "chunk_type": "text",
                "start_position": 0,
                "end_position": len(text.strip()[:self.chunk_size])
            }
        }
        return [fallback_chunk]
    
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
        # Protected text for common abbreviations
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
        
        # Simple sentence splitting pattern
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
        
        # Generate consistent chunk index
        chunk_index = len(existing_chunks)
        
        # Create chunk info
        chunk = {
            "id": f"{document_id}_chunk_{chunk_index}",
            "text": text.strip(),
            "chunk_index": chunk_index,
            "chunk_type": chunk_type,
            "page_number": page_number,
            "block_type": "Text",
            "semantic_units": len(units),
            "has_structural_elements": has_structural,
            "source_info": {
                "document_id": document_id,
                "document_name": filename,
                "page_number": page_number,
                "block_index": chunk_index,
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
                    if len(prev_chunk["text"]) + len(text) < self.chunk_size:
                        prev_chunk["text"] += "\n\n" + text
                        prev_chunk["semantic_units"] += chunk.get("semantic_units", 1)
                        # Update end position to include merged content
                        prev_chunk["source_info"]["end_position"] = chunk["source_info"]["end_position"]
                        continue
            
            # Clean up text
            chunk["text"] = self._clean_chunk_text(text)
            
            # Update indices after any merging
            final_chunk_index = len(processed_chunks)
            chunk["chunk_index"] = final_chunk_index
            chunk["id"] = f"{chunk['source_info']['document_id']}_chunk_{final_chunk_index}"
            chunk["source_info"]["block_index"] = final_chunk_index
            
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _clean_chunk_text(self, text: str) -> str:
        """Clean up chunk text for better processing"""
        # Remove excessive whitespace
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
        
        for line in lines:
            position += len(line) + 1  # +1 for \n
            
            # Look for Marker's page break patterns: {N}------------------------------------------------
            line_stripped = line.strip()
            page_marker_match = re.match(r'^\{(\d+)\}[-]+$', line_stripped)
            if page_marker_match:
                page_positions.append(position)
                continue
        
        return page_positions
    
    def _get_page_number_for_position(self, position: int, page_markers: List[int]) -> int:
        """Determine page number based on character position in text"""
        if not page_markers or position < 0:
            return 1
        
        # Default to page 1 for content before any markers
        page = 1
        
        try:
            # Check each marker position (skip index 0 which is just document start)
            for i in range(1, len(page_markers)):
                if position >= page_markers[i]:
                    page = i
                else:
                    break
            
            # Ensure page is at least 1 and reasonable
            page = max(1, min(page, 1000))  # Cap at 1000 pages for safety
            
        except Exception as e:
            self.logger.warning(f"Error calculating page number for position {position}: {e}")
            page = 1
        
        return page