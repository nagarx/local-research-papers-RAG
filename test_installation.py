#!/usr/bin/env python3
"""
ArXiv Paper RAG Assistant - Installation Test Script

This script performs an end-to-end test of the installation to ensure
everything works correctly for the job interview demonstration.
"""

import sys
import os
import subprocess
import tempfile
import time
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_critical_imports():
    """Test that all critical modules can be imported"""
    print("🧪 Testing critical imports...")
    
    critical_modules = [
        ("streamlit", "streamlit"),
        ("marker", "marker-pdf"),
        ("ollama", "ollama"),
        ("sentence_transformers", "sentence_transformers"),
        ("chromadb", "chromadb"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
    ]
    
    for import_name, display_name in critical_modules:
        try:
            __import__(import_name)
            print(f"   ✅ {display_name}")
        except ImportError as e:
            print(f"   ❌ {display_name}: {e}")
            return False
        except Exception as e:
            print(f"   ⚠️  {display_name}: {e}")
    
    return True

def test_ollama_connection():
    """Test Ollama connection and model availability"""
    print("\n🤖 Testing Ollama connection...")
    
    try:
        # Check if Ollama is running
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            print("   ❌ Ollama service not running")
            return False
        
        print("   ✅ Ollama service is running")
        
        # Check for required model
        if "llama3.2" in result.stdout:
            print("   ✅ llama3.2 model available")
            return True
        else:
            print("   ⚠️  llama3.2 model not found - downloading...")
            
            # Try to download model
            download_result = subprocess.run(
                ["ollama", "pull", "llama3.2:latest"],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes
            )
            
            if download_result.returncode == 0:
                print("   ✅ llama3.2 model downloaded successfully")
                return True
            else:
                print("   ❌ Failed to download llama3.2 model")
                return False
                
    except subprocess.TimeoutExpired:
        print("   ❌ Ollama connection timeout")
        return False
    except FileNotFoundError:
        print("   ❌ Ollama not found")
        return False
    except Exception as e:
        print(f"   ❌ Error testing Ollama: {e}")
        return False

def test_marker_functionality():
    """Test marker CLI functionality"""
    print("\n📄 Testing marker CLI functionality...")
    
    try:
        # Test marker CLI availability
        result = subprocess.run(
            ["marker", "--help"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode == 0:
            print("   ✅ Marker CLI is available")
            return True
        else:
            print("   ❌ Marker CLI not working")
            return False
        
    except FileNotFoundError:
        print("   ❌ Marker CLI not found")
        return False
    except Exception as e:
        print(f"   ❌ Marker CLI test failed: {e}")
        return False

def test_vector_database():
    """Test ChromaDB vector database"""
    print("\n🗄️  Testing vector database...")
    
    try:
        from src.storage.chroma_vector_store import ChromaVectorStore
        
        # Create test vector store (uses default configuration)
        vector_store = ChromaVectorStore()
        
        print("   ✅ Vector store created successfully")
        
        # Just test that we can get stats (basic functionality test)
        stats = vector_store.get_stats()
        if stats:
            print("   ✅ Vector store basic operations working")
            return True
        else:
            print("   ❌ Vector store stats failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Vector database test failed: {e}")
        return False

def test_streamlit_availability():
    """Test that Streamlit can be launched"""
    print("\n🌐 Testing Streamlit availability...")
    
    try:
        # Check if streamlit command works
        result = subprocess.run(
            [sys.executable, "-m", "streamlit", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("   ✅ Streamlit is available")
            return True
        else:
            print("   ❌ Streamlit not available")
            return False
            
    except Exception as e:
        print(f"   ❌ Streamlit test failed: {e}")
        return False

def test_file_permissions():
    """Test file permissions in key directories"""
    print("\n📁 Testing file permissions...")
    
    test_dirs = ["data", "temp_uploads", "data/logs"]
    
    for directory in test_dirs:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            test_file = Path(directory) / ".permission_test"
            test_file.write_text("test")
            test_file.unlink()
            print(f"   ✅ {directory} is writable")
        except Exception as e:
            print(f"   ❌ {directory} permission error: {e}")
            return False
    
    return True

def test_configuration():
    """Test configuration loading"""
    print("\n⚙️  Testing configuration...")
    
    try:
        from src.config.config import get_config
        
        config = get_config()
        print("   ✅ Configuration loaded successfully")
        
        # Test key configuration values
        if hasattr(config, 'ollama') and hasattr(config, 'streamlit'):
            print("   ✅ Required configuration sections present")
            return True
        else:
            print("   ❌ Missing required configuration sections")
            return False
            
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def run_integration_test():
    """Run a complete integration test"""
    print("\n🔗 Running integration test...")
    
    try:
        # Test the main components work together
        from src.storage.chroma_vector_store import ChromaVectorStore
        from src.llm.ollama_client import OllamaClient
        from src.config.config import get_config
        
        config = get_config()
        
        # Test LLM client
        llm_client = OllamaClient()
        print("   ✅ LLM client created")
        
        # Test vector store with default configuration
        vector_store = ChromaVectorStore()
        print("   ✅ Vector store created")
        
        print("   ✅ Integration test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False

def main():
    """Run all installation tests"""
    print("=" * 60)
    print("🧪 ArXiv Paper RAG Assistant - Installation Test")
    print("=" * 60)
    
    tests = [
        ("Critical Imports", test_critical_imports),
        ("Ollama Connection", test_ollama_connection),
        ("Marker Functionality", test_marker_functionality),
        ("Vector Database", test_vector_database),
        ("Streamlit Availability", test_streamlit_availability),
        ("File Permissions", test_file_permissions),
        ("Configuration", test_configuration),
        ("Integration Test", run_integration_test),
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("🚀 Your ArXiv Paper RAG Assistant is ready for the job interview!")
        print("\nTo start the application:")
        print("  python run.py")
        print("  Then open: http://localhost:8501")
        return True
    else:
        print(f"\n❌ {total_tests - passed_tests} test(s) failed")
        print("🔧 Please fix the issues before proceeding")
        print("\nFor help, run:")
        print("  python verify_installation.py")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 