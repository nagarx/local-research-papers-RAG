#!/usr/bin/env python3
"""
Import and Basic Functionality Tests

This module tests all core imports and basic functionality of the RAG system
to ensure the architecture is working correctly.
"""

import pytest
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Test suite imports
from conftest import TestResultTracker


class TestImports:
    """Test all core imports and basic initialization"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.tracker = TestResultTracker()
    
    def test_core_module_imports(self):
        """Test core module imports"""
        print("\n🧪 Testing Core Module Imports")
        print("=" * 50)
        
        # Core modules to test
        modules_to_test = [
            ("src.config", ["get_config", "get_logger", "Config"]),
            ("src.llm", ["OllamaClient"]),
            ("src.storage", ["ChromaVectorStore"]), 
            ("src.chat", ["ChatEngine"]),
            ("src.ingestion", ["DocumentProcessor", "get_global_marker_models"]),
            ("src.embeddings", ["EmbeddingManager"]),
            ("src.tracking", ["SourceTracker", "SourceReference"]),
            ("src.utils.utils", ["FileUtils"]),
            ("src.utils.torch_utils", ["configure_torch_for_production"])
        ]
        
        failed_imports = []
        
        for module_name, classes in modules_to_test:
            start_time = time.time()
            
            try:
                module = __import__(module_name, fromlist=classes)
                
                # Test each class/function
                for class_name in classes:
                    getattr(module, class_name)
                
                duration = time.time() - start_time
                self.tracker.log_result(
                    f"Import {module_name}",
                    True,
                    duration,
                    f"Classes: {', '.join(classes)}"
                )
                
            except ImportError as e:
                duration = time.time() - start_time
                error_msg = f"Import Error: {e}"
                self.tracker.log_result(f"Import {module_name}", False, duration, error_msg)
                failed_imports.append(f"{module_name}: {error_msg}")
                
            except AttributeError as e:
                duration = time.time() - start_time
                error_msg = f"Attribute Error: {e}"
                self.tracker.log_result(f"Import {module_name}", False, duration, error_msg)
                failed_imports.append(f"{module_name}: {error_msg}")
        
        # Assert no failed imports
        assert len(failed_imports) == 0, f"Failed imports: {failed_imports}"
    
    def test_config_initialization(self):
        """Test configuration initialization"""
        print("\n⚙️  Testing Configuration Initialization")
        
        start_time = time.time()
        
        try:
            from src.config import get_config, get_logger
            
            # Test config loading
            config = get_config()
            assert config is not None, "Config should not be None"
            
            # Test logger creation
            logger = get_logger(__name__)
            assert logger is not None, "Logger should not be None"
            
            duration = time.time() - start_time
            self.tracker.log_result(
                "Configuration Initialization",
                True,
                duration,
                f"Config loaded, logger created"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.tracker.log_result(
                "Configuration Initialization",
                False,
                duration,
                str(e)
            )
            raise
    
    def test_ollama_client_creation(self):
        """Test OllamaClient creation and basic functionality"""
        print("\n🤖 Testing Ollama Client Creation")
        
        start_time = time.time()
        
        try:
            from src.llm import OllamaClient
            
            # Create client
            client = OllamaClient()
            assert client is not None, "OllamaClient should be created"
            
            # Test basic attributes
            assert hasattr(client, 'base_url'), "Client should have base_url"
            assert hasattr(client, 'model'), "Client should have model"
            assert hasattr(client, 'test_connection'), "Client should have test_connection method"
            
            duration = time.time() - start_time
            self.tracker.log_result(
                "OllamaClient Creation",
                True,
                duration,
                f"Model: {client.model}, URL: {client.base_url}"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.tracker.log_result(
                "OllamaClient Creation",
                False,
                duration,
                str(e)
            )
            raise
    
    def test_vector_store_creation(self):
        """Test ChromaVectorStore creation and basic functionality"""
        print("\n🔍 Testing Vector Store Creation")
        
        start_time = time.time()
        
        try:
            from src.storage import ChromaVectorStore
            
            # Create store
            store = ChromaVectorStore()
            assert store is not None, "ChromaVectorStore should be created"
            
            # Test basic methods
            assert hasattr(store, 'add_document'), "Store should have add_document method"
            assert hasattr(store, 'search'), "Store should have search method"
            assert hasattr(store, 'get_stats'), "Store should have get_stats method"
            
            # Test stats
            stats = store.get_stats()
            assert isinstance(stats, dict), "Stats should be a dictionary"
            
            duration = time.time() - start_time
            self.tracker.log_result(
                "ChromaVectorStore Creation",
                True,
                duration,
                f"Stats: {stats}"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.tracker.log_result(
                "ChromaVectorStore Creation",
                False,
                duration,
                str(e)
            )
            raise
    
    def test_chat_engine_creation(self):
        """Test ChatEngine creation and initialization"""
        print("\n💬 Testing Chat Engine Creation")
        
        start_time = time.time()
        
        try:
            from src.chat import ChatEngine
            
            # Create engine
            engine = ChatEngine()
            assert engine is not None, "ChatEngine should be created"
            
            # Test basic attributes
            assert hasattr(engine, 'document_processor'), "Engine should have document_processor"
            assert hasattr(engine, 'embedding_manager'), "Engine should have embedding_manager"
            assert hasattr(engine, 'vector_store'), "Engine should have vector_store"
            assert hasattr(engine, 'ollama_client'), "Engine should have ollama_client"
            assert hasattr(engine, 'source_tracker'), "Engine should have source_tracker"
            
            duration = time.time() - start_time
            self.tracker.log_result(
                "ChatEngine Creation",
                True,
                duration,
                "All components initialized"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.tracker.log_result(
                "ChatEngine Creation",
                False,
                duration,
                str(e)
            )
            raise
    
    def test_torch_configuration(self):
        """Test PyTorch configuration and utilities"""
        print("\n🔥 Testing PyTorch Configuration")
        
        start_time = time.time()
        
        try:
            from src.utils.torch_utils import configure_torch_for_production
            
            # Test configuration (should be safe to call multiple times)
            configure_torch_for_production()
            
            duration = time.time() - start_time
            self.tracker.log_result(
                "PyTorch Configuration",
                True,
                duration,
                "Configuration applied"
            )
            
        except Exception as e:
            duration = time.time() - start_time
            self.tracker.log_result(
                "PyTorch Configuration",
                False,
                duration,
                str(e)
            )
            # Don't raise - PyTorch might not be available
            print(f"   Warning: PyTorch configuration failed: {e}")
    
    def teardown_method(self):
        """Cleanup after each test method"""
        self.tracker.print_summary()


def test_directory_structure():
    """Test that the expected directory structure exists"""
    print("\n📁 Testing Directory Structure")
    
    # Expected directories (relative to project root)
    project_root = Path(__file__).parent.parent
    expected_dirs = [
        "src",
        "src/config",
        "src/chat", 
        "src/llm",
        "src/storage",
        "src/ingestion",
        "src/embeddings",
        "src/tracking",
        "src/utils",
        "src/ui"
    ]
    
    missing_dirs = []
    
    for dir_path in expected_dirs:
        path = project_root / dir_path
        if not path.exists():
            missing_dirs.append(dir_path)
        else:
            print(f"   ✅ {dir_path}")
    
    if missing_dirs:
        print(f"   ❌ Missing directories: {missing_dirs}")
        pytest.fail(f"Missing required directories: {missing_dirs}")
    
    print("   ✅ All required directories exist")


def test_essential_files():
    """Test that essential files exist"""
    print("\n📄 Testing Essential Files")
    
    # Essential files (relative to project root)
    project_root = Path(__file__).parent.parent
    essential_files = [
        "requirements.txt",
        "README.md",
        "app.py",
        "src/__init__.py",
        "src/config/config.py",
        "src/chat/chat_engine.py",
        "src/llm/ollama_client.py"
    ]
    
    missing_files = []
    
    for file_path in essential_files:
        path = project_root / file_path
        if not path.exists():
            missing_files.append(file_path)
        else:
            print(f"   ✅ {file_path}")
    
    if missing_files:
        print(f"   ❌ Missing files: {missing_files}")
        pytest.fail(f"Missing essential files: {missing_files}")
    
    print("   ✅ All essential files exist")


if __name__ == "__main__":
    """Run import tests standalone"""
    print("🚀 Running RAG System Import Tests")
    print("=" * 60)
    
    # Run tests
    import pytest
    pytest.main([__file__, "-v"]) 