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
    
    def process_document(self, file_path: Union[str, Path]) -> Tuple[str, str, list]:
        """
        Process a document using the Marker CLI with complete isolation
        
        Args:
            file_path: Path to the PDF file to process
            
        Returns:
            Tuple of (text_content, format_type, images)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if self.enhanced_logger:
            self.enhanced_logger.processing_start(f"Processing {file_path.name}")
        
        start_time = time.time()
        
        # Create isolated temporary directory for this processing
        with tempfile.TemporaryDirectory(prefix="marker_cli_") as temp_dir:
            temp_dir = Path(temp_dir)
            output_dir = temp_dir / "output"
            output_dir.mkdir(exist_ok=True)
            
            # Copy input file to temp directory to avoid path issues
            temp_input = temp_dir / file_path.name
            shutil.copy2(file_path, temp_input)
            
            try:
                # Build CLI command with complete isolation
                cmd = [
                    "marker",
                    "--output_format", "markdown",
                    "--disable_image_extraction",
                    "--workers", "1",
                    "--output_dir", str(output_dir),
                    str(temp_input)
                ]
                
                if self.enhanced_logger:
                    self.enhanced_logger.processing_start(f"Running CLI: {' '.join(cmd)}")
                
                # Run in completely isolated environment
                env = os.environ.copy()
                
                # Add environment variables to prevent multiprocessing issues
                env.update({
                    # Disable multiprocessing completely
                    'MARKER_MAX_WORKERS': '1',
                    'MARKER_PARALLEL_FACTOR': '1',
                    'MARKER_DISABLE_MULTIPROCESSING': '1',
                    'MARKER_SINGLE_THREADED': '1',
                    
                    # Disable GPU usage to avoid memory issues
                    'CUDA_VISIBLE_DEVICES': '',
                    'TORCH_DEVICE': 'cpu',
                    
                    # Disable model caching
                    'MARKER_DISABLE_MODEL_CACHE': '1',
                    'TRANSFORMERS_OFFLINE': '1',
                    
                    # Reduce memory usage
                    'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
                    'TORCH_CPP_LOG_LEVEL': 'ERROR',
                    
                    # Disable progress bars and verbose output
                    'MARKER_DISABLE_TQDM': '1',
                    'MARKER_QUIET': '1',
                    
                    # Force cleanup
                    'MARKER_CLEANUP_ON_EXIT': '1'
                })
                
                # Run the command in a separate process with timeout
                result = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10 minute timeout
                    cwd=temp_dir  # Run in temp directory
                )
                
                if result.returncode != 0:
                    error_msg = f"Marker CLI failed with return code {result.returncode}"
                    if result.stderr:
                        error_msg += f"\nSTDERR: {result.stderr}"
                    if result.stdout:
                        error_msg += f"\nSTDOUT: {result.stdout}"
                    raise RuntimeError(error_msg)
                
                # Find the output markdown file
                markdown_files = list(output_dir.glob("*.md"))
                if not markdown_files:
                    raise RuntimeError(f"No markdown output found in {output_dir}")
                
                # Read the output
                output_file = markdown_files[0]
                with open(output_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                processing_time = time.time() - start_time
                
                if self.enhanced_logger:
                    self.enhanced_logger.processing_complete(
                        f"Processing {file_path.name}", 
                        processing_time,
                        f"Generated {len(text)} characters"
                    )
                else:
                    self.logger.info(f"Processed {file_path.name} in {processing_time:.2f}s")
                
                # Return text, format, and empty images list (since we disable image extraction)
                return text, "markdown", []
                
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"Marker CLI timed out after 10 minutes")
            except Exception as e:
                if self.enhanced_logger:
                    self.enhanced_logger.error_clean(f"Failed to process {file_path.name}", e)
                raise RuntimeError(f"Failed to process document: {e}")
        
        # Temporary directory is automatically cleaned up here
    
    def extract_text_from_rendered(self, rendered: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
        """Extract text from CLI processing result - for backward compatibility"""
        # This method is kept for backward compatibility but not used in CLI approach
        if isinstance(rendered, tuple):
            # New format: (text, format_type, images)
            text, format_type, images = rendered
            return text, format_type, {}
        else:
            # Old format: dictionary (shouldn't happen with CLI approach)
            return rendered.get("text", ""), rendered.get("format", "markdown"), {}
    
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
    """Mock object to maintain backward compatibility with existing code"""
    
    def __init__(self, text: str, format_type: str, images: list):
        self.text = text
        self.format_type = format_type
        self.images = images or []
    
    def __str__(self):
        return self.text
    
    def __repr__(self):
        return f"MockRenderedObject(text_length={len(self.text)}, format={self.format_type})"


def process_document_with_marker(file_path: Union[str, Path]) -> 'MockRenderedObject':
    """
    Process a document using the CLI-based Marker processor
    
    This function maintains backward compatibility with the old interface
    while using the new CLI-based implementation internally.
    """
    processor = MarkerProcessor()
    result = processor.process_document(file_path)
    
    # Create mock rendered object for compatibility
    return MockRenderedObject(
        text=result[0],      # text_content
        format_type=result[1], # format_type
        images=result[2]     # images
    )