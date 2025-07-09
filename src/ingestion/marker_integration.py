"""
Marker Integration Module

This module handles Marker-specific functionality including global model management,
converter setup, and document processing with Marker.
"""

import os
import time
import multiprocessing
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

# Configure multiprocessing early to prevent semaphore leaks
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

# COMPREHENSIVE environment variables to force single-threaded operation
_MARKER_ENV_VARS = {
    'MARKER_MAX_WORKERS': '1',
    'MARKER_PARALLEL_FACTOR': '1',
    'MARKER_WORKERS': '1',
    'MARKER_NUM_WORKERS': '1',
    'MARKER_DISABLE_MULTIPROCESSING': '1',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'VECLIB_MAXIMUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'TOKENIZERS_PARALLELISM': 'false',
    'PYTHONHASHSEED': '0',
    'CUDA_VISIBLE_DEVICES': '0'
}

# Set environment variables
for key, value in _MARKER_ENV_VARS.items():
    os.environ[key] = value

# Configure for headless operation early
try:
    from ..utils.torch_utils import configure_for_headless_operation
    configure_for_headless_operation()
except ImportError:
    pass

# Marker imports
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.config.parser import ConfigParser
from marker.output import text_from_rendered

from ..config import get_config, get_logger

# GLOBAL MODEL CACHE - Initialize once, reuse everywhere
_GLOBAL_MARKER_MODELS = None
_GLOBAL_MODEL_LOAD_TIME = None


def get_global_marker_models():
    """Get global Marker models - initialize once, reuse everywhere"""
    global _GLOBAL_MARKER_MODELS, _GLOBAL_MODEL_LOAD_TIME
    
    if _GLOBAL_MARKER_MODELS is None:
        logger = get_logger(__name__)
        
        # Try to get enhanced logger for better tracking
        try:
            from ..utils.enhanced_logging import get_enhanced_logger
            enhanced_logger = get_enhanced_logger(__name__)
            enhanced_logger.marker_model_loading("Global Models", "One-time setup")
        except ImportError:
            enhanced_logger = None
            logger.info("Loading Marker models globally (one-time setup)...")
        
        # Clean up any existing resources before loading models
        try:
            from ..utils.resource_cleanup import cleanup_multiprocessing_resources, cleanup_semaphore_leaks
            cleanup_multiprocessing_resources()
            cleanup_semaphore_leaks()
        except ImportError:
            pass
        
        start_time = time.time()
        
        try:
            # Load models with single-threaded operation
            _GLOBAL_MARKER_MODELS = create_model_dict()
            _GLOBAL_MODEL_LOAD_TIME = time.time() - start_time
            
            # Clean up after model loading
            try:
                from ..utils.resource_cleanup import cleanup_multiprocessing_resources, cleanup_semaphore_leaks
                cleanup_multiprocessing_resources()
                cleanup_semaphore_leaks()
            except ImportError:
                pass
            
            if enhanced_logger:
                enhanced_logger.marker_model_ready("Global Models", _GLOBAL_MODEL_LOAD_TIME, "All models loaded")
                enhanced_logger.marker_performance("Model Loading", memory_mb=0, load_time=_GLOBAL_MODEL_LOAD_TIME)
            else:
                logger.info(f"Global Marker models loaded in {_GLOBAL_MODEL_LOAD_TIME:.2f}s")
            
        except Exception as e:
            # Clean up on error
            try:
                from ..utils.resource_cleanup import cleanup_multiprocessing_resources, cleanup_semaphore_leaks
                cleanup_multiprocessing_resources()
                cleanup_semaphore_leaks()
            except ImportError:
                pass
            
            if enhanced_logger:
                enhanced_logger.marker_error(f"Failed to load global models", exception=e)
            else:
                logger.error(f"Failed to load global Marker models: {e}")
            raise
    
    return _GLOBAL_MARKER_MODELS


class MarkerProcessor:
    """
    Handles Marker-specific document processing operations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the Marker processor"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Try to get enhanced logger for better tracking
        try:
            from ..utils.enhanced_logging import get_enhanced_logger, configure_marker_logging
            self.enhanced_logger = get_enhanced_logger(__name__)
            
            # Configure logging only once using a class variable to prevent repeated calls
            if not hasattr(MarkerProcessor, '_logging_configured'):
                configure_marker_logging(enable_debug=True, log_level="DEBUG")
                MarkerProcessor._logging_configured = True
        except ImportError:
            self.enhanced_logger = None
        
        # Use global models
        self.marker_models = get_global_marker_models()
        
        # Setup converter
        self._setup_converter()
        
        if self.enhanced_logger:
            self.enhanced_logger.system_ready("MarkerProcessor", "Converter initialized")
        else:
            self.logger.info("MarkerProcessor initialized successfully")
    
    def _setup_converter(self):
        """Setup converter following documentation pattern exactly"""
        if self.enhanced_logger:
            self.enhanced_logger.processing_start("Marker Converter Setup")
        else:
            self.logger.info("Setting up Marker converter...")
        
        # Configuration following documentation best practices
        config = {
            "output_format": "markdown",
            "paginate_output": True,
            "format_lines": True,
            "extract_images": True,
            "use_llm": False,
            "force_ocr": False,
            "disable_tqdm": True,
            
            # CRITICAL: ABSOLUTELY FORCE single-worker operation
            "workers": 1,
            "max_workers": 1,
            "parallel_factor": 1,
            "batch_size": 1,
            "num_workers": 1,
            "disable_multiprocessing": True,
            "single_threaded": True,
            "sequential": True,
            
            # Additional safety settings
            "max_parallel": 1,
            "cpu_count": 1,
            "pool_size": 1,
        }
        
        # Only enable LLM if configured and API key available
        if self.config.marker.use_llm and hasattr(self.config.marker, 'gemini_api_key') and self.config.marker.gemini_api_key:
            config.update({
                "use_llm": True,
                "redo_inline_math": True,
                "llm_service": "marker.services.gemini.GoogleGeminiService",
            })
            if self.enhanced_logger:
                self.enhanced_logger.system_ready("Marker LLM", "Gemini service enabled")
            else:
                self.logger.info("LLM features enabled for Marker")
        else:
            if self.enhanced_logger:
                self.enhanced_logger.system_ready("Marker LLM", "Disabled")
            else:
                self.logger.info("LLM features disabled for Marker")
        
        # Create converter
        config_parser = ConfigParser(config)
        self.converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=self.marker_models,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service()
        )
        
        if self.enhanced_logger:
            self.enhanced_logger.processing_complete("Marker Converter Setup", 0, "Single-worker configuration")
        else:
            self.logger.info("Marker converter initialized with strict single-worker configuration")
    
    def process_document(self, file_path: Union[str, Path]) -> Any:
        """Process document with Marker"""
        try:
            # Validate file
            file_path_obj = Path(file_path)
            if not file_path_obj.exists():
                raise FileNotFoundError(f"File does not exist: {file_path}")
            
            if not file_path_obj.is_file():
                raise ValueError(f"Path is not a file: {file_path}")
            
            # Check file size
            file_size = file_path_obj.stat().st_size
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
                self.enhanced_logger.marker_processing_stage("Validation", file_path.name, f"Size: {file_size / 1024 / 1024:.1f}MB")
            else:
                self.logger.debug(f"Starting Marker processing for: {file_path}")
            
            # Process with Marker
            result = self.converter(str(file_path))
            
            processing_time = time.time() - start_time
            
            if self.enhanced_logger:
                # Try to get page count from result
                page_count = getattr(result, 'page_count', 0) if hasattr(result, 'page_count') else 0
                self.enhanced_logger.marker_processing_complete(file_path.name, processing_time, pages=page_count)
                self.enhanced_logger.marker_performance("Document Processing", file_path.name, 
                                                       duration=processing_time, 
                                                       file_size_mb=file_size / 1024 / 1024,
                                                       pages=page_count)
            else:
                self.logger.debug(f"Marker processing completed for: {file_path}")
            
            return result
            
        except Exception as e:
            if self.enhanced_logger:
                self.enhanced_logger.marker_error(f"Processing failed", file_path.name, e)
            else:
                self.logger.error(f"Marker processing failed for {file_path}: {type(e).__name__}: {str(e)}")
            raise RuntimeError(f"Document processing failed: {str(e)}") from e
    
    def extract_text_from_rendered(self, rendered) -> tuple:
        """Extract text from rendered document"""
        try:
            text, ext, images = text_from_rendered(rendered)
            
            if self.enhanced_logger:
                self.enhanced_logger.marker_processing_stage("Text Extraction", "rendered", f"{len(text)} chars")
                self.enhanced_logger.marker_image_extraction("rendered", len(images))
            else:
                self.logger.info(f"Extracted: {len(text)} chars, {len(images)} images")
            
            return text, ext, images
            
        except Exception as e:
            if self.enhanced_logger:
                self.enhanced_logger.marker_warning(f"text_from_rendered failed, using fallback: {e}")
            else:
                self.logger.warning(f"text_from_rendered failed, using fallback: {e}")
            return self._fallback_extraction(rendered)
    
    def _fallback_extraction(self, rendered) -> tuple:
        """Fallback extraction if text_from_rendered fails"""
        if self.enhanced_logger:
            self.enhanced_logger.marker_processing_stage("Fallback Extraction", "rendered", "Using fallback method")
        
        if hasattr(rendered, 'markdown'):
            return rendered.markdown, "md", getattr(rendered, 'images', {})
        elif hasattr(rendered, 'html'):
            return rendered.html, "html", getattr(rendered, 'images', {})
        else:
            return str(rendered), "txt", {}