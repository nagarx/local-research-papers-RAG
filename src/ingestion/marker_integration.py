"""
Marker Integration Module - CLI-based Implementation

This module handles Marker document processing using the CLI approach
to avoid multiprocessing issues and memory leaks.
"""

import os
import time
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import logging

from ..config import get_config, get_logger


class MarkerProcessor:
    """
    Handles Marker document processing using CLI approach
    
    This implementation uses the marker CLI command directly to avoid
    multiprocessing issues and memory leaks that occur with the Python library.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Marker processor"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Try to get enhanced logger for better tracking
        try:
            from ..utils.enhanced_logging import get_enhanced_logger
            self.enhanced_logger = get_enhanced_logger(__name__)
        except ImportError:
            self.enhanced_logger = None
        
        # Verify marker CLI is available
        self._verify_marker_cli()
        
        if self.enhanced_logger:
            self.enhanced_logger.system_ready("MarkerProcessor", "CLI-based implementation")
        else:
            self.logger.info("MarkerProcessor initialized successfully (CLI-based)")
    
    def _verify_marker_cli(self):
        """Verify that marker CLI is available"""
        try:
            result = subprocess.run(
                ["marker", "--help"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode != 0:
                raise RuntimeError("Marker CLI not found or not working properly")
            
            self.logger.info("✅ Marker CLI verified and available")
            
        except Exception as e:
            self.logger.error(f"❌ Marker CLI not available: {e}")
            raise RuntimeError(f"Marker CLI not available: {e}")
    
    def process_document(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Process document using marker CLI"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            raise ValueError(f"File is empty: {file_path}")
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            if self.enhanced_logger:
                self.enhanced_logger.marker_warning(f"Large file detected ({file_size / 1024 / 1024:.1f}MB)", file_path.name)
            else:
                self.logger.warning(f"Large file detected ({file_size / 1024 / 1024:.1f}MB): {file_path}")
        
        # Start processing with enhanced logging
        start_time = time.time()
        
        if self.enhanced_logger:
            self.enhanced_logger.marker_processing_start(file_path.name)
            self.enhanced_logger.marker_processing_stage("CLI Processing", file_path.name, f"Size: {file_size / 1024 / 1024:.1f}MB")
        else:
            self.logger.debug(f"Starting Marker CLI processing for: {file_path}")
        
        try:
            # Create temporary directory for output
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_output_dir = Path(temp_dir) / "marker_output"
                temp_output_dir.mkdir(exist_ok=True)
                
                # Run marker CLI command
                cmd = [
                    "marker",
                    "--output_format", "markdown",
                    "--disable_image_extraction",  # Disable images to avoid complexity
                    "--workers", "1",
                    "--output_dir", str(temp_output_dir),
                    str(file_path)
                ]
                
                if self.enhanced_logger:
                    self.enhanced_logger.marker_processing_stage("CLI Command", file_path.name, f"Running: {' '.join(cmd)}")
                
                # Execute marker CLI
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                    cwd=file_path.parent
                )
                
                if result.returncode != 0:
                    error_msg = f"Marker CLI failed with return code {result.returncode}"
                    if result.stderr:
                        error_msg += f"\nStderr: {result.stderr}"
                    if result.stdout:
                        error_msg += f"\nStdout: {result.stdout}"
                    
                    raise RuntimeError(error_msg)
                
                # Find the output file
                output_files = list(temp_output_dir.glob("*.md"))
                if not output_files:
                    raise RuntimeError(f"No output file generated for {file_path.name}")
                
                output_file = output_files[0]
                
                # Read the processed text
                with open(output_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                processing_time = time.time() - start_time
                
                if self.enhanced_logger:
                    self.enhanced_logger.marker_processing_complete(file_path.name, processing_time, pages=0)
                    self.enhanced_logger.marker_performance("CLI Processing", file_path.name, 
                                                           duration=processing_time, 
                                                           file_size_mb=file_size / 1024 / 1024,
                                                           text_length=len(text))
                else:
                    self.logger.info(f"Marker CLI processing completed for: {file_path} in {processing_time:.2f}s")
                
                # Return result in expected format
                return {
                    "text": text,
                    "format": "markdown",
                    "images": {},  # No images since we disabled extraction
                    "processing_time": processing_time,
                    "file_size": file_size,
                    "output_file": str(output_file)
                }
                
        except subprocess.TimeoutExpired:
            error_msg = f"Marker CLI processing timed out for {file_path.name}"
            if self.enhanced_logger:
                self.enhanced_logger.marker_error(error_msg, file_path.name)
            else:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        except Exception as e:
            if self.enhanced_logger:
                self.enhanced_logger.marker_error(f"CLI processing failed", file_path.name, e)
            else:
                self.logger.error(f"Marker CLI processing failed for {file_path}: {type(e).__name__}: {str(e)}")
            raise RuntimeError(f"Document processing failed: {str(e)}") from e
    
    def extract_text_from_rendered(self, rendered: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """Extract text from CLI processing result"""
        try:
            text = rendered.get("text", "")
            format_type = rendered.get("format", "markdown")
            images = rendered.get("images", {})
            
            if self.enhanced_logger:
                self.enhanced_logger.marker_processing_stage("Text Extraction", "CLI result", f"{len(text)} chars")
                self.enhanced_logger.marker_image_extraction("CLI result", len(images))
            else:
                self.logger.info(f"Extracted: {len(text)} chars, {len(images)} images")
            
            return text, format_type, images
            
        except Exception as e:
            if self.enhanced_logger:
                self.enhanced_logger.marker_warning(f"Text extraction failed, using fallback: {e}")
            else:
                self.logger.warning(f"Text extraction failed, using fallback: {e}")
            return self._fallback_extraction(rendered)
    
    def _fallback_extraction(self, rendered: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """Fallback extraction if normal extraction fails"""
        if self.enhanced_logger:
            self.enhanced_logger.marker_processing_stage("Fallback Extraction", "CLI result", "Using fallback method")
        
        # Try to get text from various possible locations
        text = rendered.get("text", "")
        if not text:
            text = str(rendered)
        
        format_type = rendered.get("format", "txt")
        images = rendered.get("images", {})
        
        return text, format_type, images


def get_global_marker_models():
    """Get global Marker models - CLI implementation doesn't need this"""
    # CLI implementation doesn't need model management
    return {}


# Maintain backward compatibility
class MockRenderedObject:
    """Mock object to maintain compatibility with existing code"""
    
    def __init__(self, text: str, format_type: str = "markdown", images: Dict[str, Any] = None):
        self.text = text
        self.format_type = format_type
        self.images = images or {}
        self.markdown = text if format_type == "markdown" else text
        self.html = text if format_type == "html" else text
        self.page_count = 0  # We don't track page count in CLI mode
        self.metadata = {}
    
    def __getattr__(self, name):
        # Return empty values for any other attributes
        if name in ['page_count', 'metadata']:
            return 0 if name == 'page_count' else {}
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def process_document_with_cli(file_path: Union[str, Path]) -> MockRenderedObject:
    """Process document using CLI and return mock rendered object for compatibility"""
    processor = MarkerProcessor()
    result = processor.process_document(file_path)
    
    # Create mock rendered object for compatibility
    return MockRenderedObject(
        text=result["text"],
        format_type=result["format"],
        images=result["images"]
    )