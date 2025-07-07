"""
Source Tracker - Citation Management

This module handles precise source attribution and citation tracking
for the RAG system, ensuring accurate PDF and page number references.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

from ..config import get_logger


@dataclass
class SourceReference:
    """Represents a source reference with precise attribution"""
    document_id: str
    document_name: str
    page_number: int  # 1-indexed for user display
    block_index: int
    block_type: str
    confidence_score: float = 1.0
    text_snippet: str = ""
    bbox: Optional[List[float]] = None


class SourceTracker:
    """
    Manages source attribution and citation tracking for RAG responses
    
    Features:
    - Precise page number tracking
    - Confidence scoring for citations
    - Text snippet extraction for verification
    - Formatted citation generation
    """
    
    def __init__(self):
        """Initialize the source tracker"""
        self.logger = get_logger(__name__)
        self._source_registry = {}  # document_id -> document metadata
        
    def register_document(
        self, 
        document_id: str, 
        document_path: str, 
        metadata: Dict[str, Any]
    ):
        """Register a document in the source registry with validation"""
        
        # Extract page count with fallback
        page_count = 0
        if isinstance(metadata.get("content"), dict):
            page_count = metadata["content"].get("page_count", 0)
        elif "page_count" in metadata:
            page_count = metadata.get("page_count", 0)
        
        # Ensure page count is valid
        if not isinstance(page_count, int) or page_count < 0:
            self.logger.warning(f"Invalid page count {page_count} for document {document_id}, defaulting to 0")
            page_count = 0
        
        self._source_registry[document_id] = {
            "document_id": document_id,
            "path": document_path,
            "filename": Path(document_path).name,
            "metadata": metadata,
            "registered_at": metadata.get("processed_at"),
            "page_count": page_count
        }
        
        self.logger.debug(f"Registered document: {document_id} ({Path(document_path).name}, {page_count} pages)")
    
    def create_source_reference(
        self,
        document_id: str,
        page_number: int,
        block_index: int,
        block_type: str,
        text_snippet: str = "",
        bbox: Optional[List[float]] = None,
        confidence_score: float = 1.0
    ) -> SourceReference:
        """Create a source reference for a specific text chunk"""
        
        if document_id not in self._source_registry:
            self.logger.warning(f"Document {document_id} not found in registry")
            document_name = f"Unknown Document ({document_id})"
        else:
            document_name = self._source_registry[document_id]["filename"]
        
        # Ensure page number is valid (at least 1)
        if page_number < 1:
            self.logger.warning(f"Invalid page number {page_number}, adjusting to 1")
            page_number = 1
        
        # Ensure confidence score is valid
        if not 0.0 <= confidence_score <= 1.0:
            self.logger.warning(f"Invalid confidence score {confidence_score}, adjusting to 1.0")
            confidence_score = 1.0
        
        return SourceReference(
            document_id=document_id,
            document_name=document_name,
            page_number=page_number,
            block_index=block_index,
            block_type=block_type,
            confidence_score=confidence_score,
            text_snippet=text_snippet[:200] + "..." if len(text_snippet) > 200 else text_snippet,
            bbox=bbox
        )
    
    def format_citation(
        self, 
        source_ref: SourceReference, 
        style: str = "simple"
    ) -> str:
        """Format a source reference as a citation"""
        
        if style == "simple":
            return f"{source_ref.document_name}, page {source_ref.page_number}"
        
        elif style == "detailed":
            citation = f"{source_ref.document_name}, page {source_ref.page_number}"
            if source_ref.block_type != "Text":
                citation += f" ({source_ref.block_type})"
            if source_ref.confidence_score < 1.0:
                citation += f" (confidence: {source_ref.confidence_score:.2f})"
            return citation
        
        elif style == "academic":
            # Format as academic citation
            doc_name = source_ref.document_name.replace(".pdf", "")
            return f"({doc_name}, p. {source_ref.page_number})"
        
        elif style == "numbered":
            # For numbered reference style
            return f"[{source_ref.document_name}, p.{source_ref.page_number}]"
        
        else:
            return self.format_citation(source_ref, "simple")
    
    def format_multiple_citations(
        self, 
        source_refs: List[SourceReference],
        style: str = "simple",
        max_citations: int = 5
    ) -> str:
        """Format multiple source references as a citation list"""
        
        if not source_refs:
            return "No sources found"
        
        # Sort by document name and page number
        sorted_refs = sorted(
            source_refs, 
            key=lambda x: (x.document_name, x.page_number)
        )
        
        # Group by document
        doc_groups = {}
        for ref in sorted_refs[:max_citations]:
            doc_name = ref.document_name
            if doc_name not in doc_groups:
                doc_groups[doc_name] = []
            doc_groups[doc_name].append(ref.page_number)
        
        # Format grouped citations
        citations = []
        for doc_name, pages in doc_groups.items():
            # Remove duplicates and sort
            unique_pages = sorted(list(set(pages)))
            
            if len(unique_pages) == 1:
                page_str = f"page {unique_pages[0]}"
            elif len(unique_pages) == 2:
                page_str = f"pages {unique_pages[0]} and {unique_pages[1]}"
            else:
                page_str = f"pages {', '.join(map(str, unique_pages[:-1]))} and {unique_pages[-1]}"
            
            citations.append(f"{doc_name}, {page_str}")
        
        if len(source_refs) > max_citations:
            citations.append(f"and {len(source_refs) - max_citations} more sources")
        
        return "; ".join(citations)
    
    def extract_text_snippet(
        self, 
        text: str, 
        max_length: int = 150,
        context_words: int = 10
    ) -> str:
        """Extract a meaningful text snippet for citation"""
        
        if len(text) <= max_length:
            return text.strip()
        
        # Try to find a good breaking point
        words = text.split()
        
        if len(words) <= context_words * 2:
            return " ".join(words)
        
        # Take first and last context_words
        snippet_words = words[:context_words] + ["..."] + words[-context_words:]
        snippet = " ".join(snippet_words)
        
        if len(snippet) <= max_length:
            return snippet
        
        # Fallback: truncate and add ellipsis
        return text[:max_length-3].strip() + "..."
    
    def get_document_info(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered document"""
        return self._source_registry.get(document_id)
    
    def list_registered_documents(self) -> List[Dict[str, Any]]:
        """List all registered documents"""
        return list(self._source_registry.values())
    
    def validate_source_reference(self, source_ref: SourceReference) -> bool:
        """Validate a source reference with improved error handling"""
        
        # Check if document is registered
        if source_ref.document_id not in self._source_registry:
            self.logger.warning(f"Source reference to unregistered document: {source_ref.document_id}")
            return False
        
        doc_info = self._source_registry[source_ref.document_id]
        
        # Check page number validity (more lenient - allow if page_count is 0 or unknown)
        page_count = doc_info.get("page_count", 0)
        if page_count > 0 and (source_ref.page_number < 1 or source_ref.page_number > page_count):
            self.logger.warning(
                f"Page number {source_ref.page_number} may be out of range for document "
                f"{source_ref.document_name} (estimated pages: 1-{page_count})"
            )
            # Don't fail validation - page count might be an estimate
        
        # Check confidence score
        if not 0.0 <= source_ref.confidence_score <= 1.0:
            self.logger.warning(f"Invalid confidence score: {source_ref.confidence_score}")
            return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get source tracking statistics"""
        
        total_docs = len(self._source_registry)
        total_pages = sum(doc.get("page_count", 0) for doc in self._source_registry.values())
        
        doc_types = {}
        for doc in self._source_registry.values():
            filename = doc["filename"]
            if filename.endswith(".pdf"):
                doc_type = "PDF"
            else:
                doc_type = "Other"
            
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        return {
            "total_documents": total_docs,
            "total_pages": total_pages,
            "document_types": doc_types,
            "average_pages_per_document": total_pages / total_docs if total_docs > 0 else 0,
            "documents_with_known_page_count": len([doc for doc in self._source_registry.values() if doc.get("page_count", 0) > 0])
        } 