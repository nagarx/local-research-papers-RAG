"""
Configuration Management for ArXiv Paper RAG Assistant

This module handles all configuration settings using environment variables
and provides a centralized configuration system.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()


class OllamaConfig(BaseSettings):
    """Ollama LLM configuration"""
    
    base_url: str = Field(
        default="http://localhost:11434",
        env="OLLAMA_BASE_URL",
        description="Ollama server URL"
    )
    model: str = Field(
        default="llama3.2:latest",
        env="OLLAMA_MODEL", 
        description="Ollama model name"
    )
    timeout: int = Field(
        default=300,
        env="OLLAMA_TIMEOUT",
        description="Request timeout in seconds"
    )
    temperature: float = Field(
        default=0.1,
        env="OLLAMA_TEMPERATURE",
        description="Model temperature for responses"
    )
    max_tokens: int = Field(
        default=2048,
        env="OLLAMA_MAX_TOKENS",
        description="Maximum tokens in response"
    )
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration"""
    
    model: str = Field(
        default="all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL",
        description="SentenceTransformers model name"
    )
    batch_size: int = Field(
        default=32,
        env="EMBEDDING_BATCH_SIZE",
        description="Batch size for embedding generation"
    )
    device: str = Field(
        default="auto",
        env="EMBEDDING_DEVICE",
        description="Device for embedding model (auto, cpu, cuda)"
    )
    dimension: int = Field(
        default=384,
        env="VECTOR_DIMENSION",
        description="Embedding vector dimension"
    )
    
    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError('Batch size must be positive')
        return v


class VectorStorageConfig(BaseSettings):
    """Vector storage configuration"""
    
    storage_type: str = Field(
        default="chromadb",
        env="VECTOR_STORAGE_TYPE",
        description="Vector storage backend type"
    )
    collection_name: str = Field(
        default="arxiv_papers",
        env="CHROMA_COLLECTION_NAME",
        description="ChromaDB collection name"
    )
    persist_directory: str = Field(
        default="./data/chroma",
        env="CHROMA_PERSIST_DIR",
        description="ChromaDB persistence directory"
    )
    similarity_threshold: float = Field(
        default=0.25,
        env="SIMILARITY_THRESHOLD",
        description="Minimum similarity score for retrieval"
    )
    distance_function: str = Field(
        default="cosine",
        env="CHROMA_DISTANCE_FUNCTION",
        description="Distance function for similarity (cosine, l2, ip)"
    )
    
    @field_validator('similarity_threshold')
    @classmethod
    def validate_similarity_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Similarity threshold must be between 0.0 and 1.0')
        return v
    
    @field_validator('distance_function')
    @classmethod
    def validate_distance_function(cls, v):
        if v not in ['cosine', 'l2', 'ip']:
            raise ValueError('Distance function must be one of: cosine, l2, ip')
        return v


class DocumentProcessingConfig(BaseSettings):
    """Document processing configuration"""
    
    max_chunk_size: int = Field(
        default=1000,
        env="MAX_CHUNK_SIZE",
        description="Maximum chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=200,
        env="CHUNK_OVERLAP",
        description="Overlap between chunks in tokens"
    )
    min_chunk_size: int = Field(
        default=100,
        env="MIN_CHUNK_SIZE",
        description="Minimum chunk size in tokens"
    )
    use_semantic_chunking: bool = Field(
        default=True,
        env="USE_SEMANTIC_CHUNKING",
        description="Enable semantic boundary detection for chunking"
    )
    
    @field_validator('max_chunk_size')
    @classmethod
    def validate_max_chunk_size(cls, v):
        if v <= 0:
            raise ValueError('Max chunk size must be positive')
        return v
    
    @field_validator('chunk_overlap')
    @classmethod
    def validate_chunk_overlap(cls, v, info):
        if v < 0:
            raise ValueError('Chunk overlap must be non-negative')
        # Note: inter-field validation is more complex in Pydantic V2
        # For now, just validate the field independently
        return v


class MarkerConfig(BaseSettings):
    """Marker document processor configuration"""
    
    use_llm: bool = Field(
        default=True,
        env="MARKER_USE_LLM",
        description="Enable LLM enhancement in Marker"
    )
    extract_images: bool = Field(
        default=True,
        env="MARKER_EXTRACT_IMAGES",
        description="Extract images from documents"
    )
    format_lines: bool = Field(
        default=True,
        env="MARKER_FORMAT_LINES",
        description="Format lines for better text quality"
    )
    force_ocr: bool = Field(
        default=False,
        env="MARKER_FORCE_OCR",
        description="Force OCR on all pages"
    )
    batch_size: int = Field(
        default=4,
        env="MARKER_BATCH_SIZE",
        description="Batch size for Marker processing"
    )


class StoragePathsConfig(BaseSettings):
    """Storage paths configuration"""
    
    data_dir: Path = Field(
        default=Path("./data"),
        env="DATA_DIR",
        description="Base data directory"
    )
    documents_dir: Path = Field(
        default=Path("./data/documents"),
        env="DOCUMENTS_DIR",
        description="Uploaded documents directory"
    )
    processed_dir: Path = Field(
        default=Path("./data/processed"),
        env="PROCESSED_DIR",
        description="Processed documents cache directory"
    )
    embeddings_dir: Path = Field(
        default=Path("./data/embeddings"),
        env="EMBEDDINGS_DIR",
        description="Embeddings storage directory"
    )
    chroma_dir: Path = Field(
        default=Path("./data/chroma"),
        env="CHROMA_DIR",
        description="ChromaDB vector storage directory"
    )
    cache_dir: Path = Field(
        default=Path("./data/cache"),
        env="CACHE_DIR",
        description="General cache directory"
    )
    logs_dir: Path = Field(
        default=Path("./data/logs"),
        env="LOGS_DIR",
        description="Application logs directory"
    )
    
    def ensure_directories(self):
        """Create all necessary directories"""
        for field_name, field in self.model_fields.items():
            directory = getattr(self, field_name)
            if isinstance(directory, Path):
                directory.mkdir(parents=True, exist_ok=True)


class StreamlitConfig(BaseSettings):
    """Streamlit UI configuration"""
    
    port: int = Field(
        default=8501,
        env="STREAMLIT_PORT",
        description="Streamlit server port"
    )
    host: str = Field(
        default="localhost",
        env="STREAMLIT_HOST",
        description="Streamlit server host"
    )
    theme: str = Field(
        default="light",
        env="STREAMLIT_THEME",
        description="UI theme (light/dark)"
    )
    max_upload_size: int = Field(
        default=200,
        env="MAX_UPLOAD_SIZE",
        description="Maximum upload size in MB"
    )


class PerformanceConfig(BaseSettings):
    """Performance and resource configuration"""
    
    max_concurrent_uploads: int = Field(
        default=5,
        env="MAX_CONCURRENT_UPLOADS",
        description="Maximum concurrent file uploads"
    )
    processing_timeout: int = Field(
        default=600,
        env="PROCESSING_TIMEOUT",
        description="Processing timeout in seconds"
    )
    enable_caching: bool = Field(
        default=True,
        env="ENABLE_CACHING",
        description="Enable result caching"
    )
    cache_ttl: int = Field(
        default=3600,
        env="CACHE_TTL",
        description="Cache time-to-live in seconds"
    )


class UIConfig(BaseSettings):
    """User interface configuration"""
    
    show_processing_details: bool = Field(
        default=True,
        env="SHOW_PROCESSING_DETAILS",
        description="Show detailed processing information"
    )
    enable_download_results: bool = Field(
        default=True,
        env="ENABLE_DOWNLOAD_RESULTS",
        description="Allow downloading results"
    )
    default_query_limit: int = Field(
        default=5,
        env="DEFAULT_QUERY_LIMIT",
        description="Default number of results to return"
    )
    max_query_limit: int = Field(
        default=20,
        env="MAX_QUERY_LIMIT",
        description="Maximum number of results to return"
    )


class Config(BaseSettings):
    """Main application configuration"""
    
    # Application metadata
    app_name: str = Field(
        default="ArXiv Paper RAG Assistant",
        env="APP_NAME",
        description="Application name"
    )
    app_version: str = Field(
        default="1.0.0",
        env="APP_VERSION",
        description="Application version"
    )
    environment: str = Field(
        default="development",
        env="ENVIRONMENT",
        description="Environment (development/production)"
    )
    debug: bool = Field(
        default=False,
        env="DEBUG",
        description="Enable debug mode"
    )
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level"
    )
    
    # Component configurations
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_storage: VectorStorageConfig = Field(default_factory=VectorStorageConfig)
    document_processing: DocumentProcessingConfig = Field(default_factory=DocumentProcessingConfig)
    marker: MarkerConfig = Field(default_factory=MarkerConfig)
    storage_paths: StoragePathsConfig = Field(default_factory=StoragePathsConfig)
    streamlit: StreamlitConfig = Field(default_factory=StreamlitConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Allow extra fields and ignore them instead of forbidding
    )
        
    def setup_logging(self):
        """Setup application logging"""
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        
        # Ensure logs directory exists
        self.storage_paths.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    self.storage_paths.logs_dir / 'arxiv_rag.log'
                ),
                logging.StreamHandler()
            ]
        )
        
        # Set specific loggers
        if not self.debug:
            # Suppress verbose logs in production
            logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
            logging.getLogger('transformers').setLevel(logging.WARNING)
            logging.getLogger('torch').setLevel(logging.WARNING)
    
    def ensure_setup(self):
        """Ensure all necessary setup is complete"""
        # Create directories
        self.storage_paths.ensure_directories()
        
        # Setup logging
        self.setup_logging()
        
        # Log configuration
        logger = logging.getLogger(__name__)
        logger.info(f"Starting {self.app_name} v{self.app_version}")
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Debug mode: {self.debug}")
    
    def get_marker_config(self) -> Dict[str, Any]:
        """Get Marker-specific configuration dictionary"""
        return {
            "output_format": "chunks",
            "paginate_output": True,
            "use_llm": self.marker.use_llm,
            "extract_images": self.marker.extract_images,
            "format_lines": self.marker.format_lines,
            "force_ocr": self.marker.force_ocr,
            
            # Ollama integration
            "llm_service": "marker.services.ollama.OllamaService",
            "ollama_base_url": self.ollama.base_url,
            "ollama_model": self.ollama.model,
            
            # Performance settings
            "pdftext_workers": min(4, self.marker.batch_size),
            "detection_batch_size": 10,
            "recognition_batch_size": 64,
            "layout_batch_size": 12,
            
            # Quality settings
            "strip_existing_ocr": True,
            "disable_tqdm": not self.debug,
            "keep_chars": False,  # Not needed for RAG
        }
    
    def get_streamlit_config(self) -> Dict[str, Any]:
        """Get Streamlit-specific configuration"""
        return {
            "server.port": self.streamlit.port,
            "server.address": self.streamlit.host,
            "server.headless": True,
            "browser.gatherUsageStats": False,
            "server.maxUploadSize": self.streamlit.max_upload_size,
            "theme.base": self.streamlit.theme,
        }


# Global configuration instance
_config: Optional[Config] = None

def get_config() -> Config:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = Config()
        _config.ensure_setup()
    return _config

def reload_config() -> Config:
    """Reload configuration from environment"""
    global _config
    _config = Config()
    _config.ensure_setup()
    return _config

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name"""
    return logging.getLogger(name) 