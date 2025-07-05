#!/usr/bin/env python3
"""
Import Test Script

This script tests all imports to ensure the reorganized RAG system is working correctly.
Run this from the project root directory.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test all main imports"""
    
    print("🧪 RAG System Import Test")
    print("=" * 50)
    
    # Show current directory
    current_dir = Path.cwd()
    print(f"📍 Current directory: {current_dir}")
    
    # Verify we're in the right place
    src_dir = current_dir / "src"
    if not src_dir.exists():
        print(f"❌ ERROR: src/ directory not found!")
        print(f"   Make sure you're in the arxiv-paper-rag project root")
        print(f"   Expected: .../arxiv-paper-rag/")
        print(f"   Current:  {current_dir}")
        return False
    
    print(f"✅ Found src/ directory: {src_dir}")
    
    # Add src to Python path
    sys.path.insert(0, str(src_dir))
    print("✅ Added src/ to Python path")
    
    # Test individual module imports
    modules_to_test = [
        ("config", "get_config"),
        ("llm", "OllamaClient"),
        ("storage", "VectorStore"), 
        ("chat", "ChatEngine"),
        ("ingestion", "DocumentProcessor"),
        ("embeddings", "EmbeddingManager"),
        ("tracking", "SourceTracker"),
        ("utils", "FileUtils")
    ]
    
    print(f"\n📦 Testing module imports:")
    failed_imports = []
    
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"   ✅ {module_name}.{class_name}")
        except ImportError as e:
            print(f"   ❌ {module_name}.{class_name} - Import Error: {e}")
            failed_imports.append(f"{module_name}.{class_name}")
        except AttributeError as e:
            print(f"   ❌ {module_name}.{class_name} - Attribute Error: {e}")
            failed_imports.append(f"{module_name}.{class_name}")
        except Exception as e:
            print(f"   ❌ {module_name}.{class_name} - Unexpected Error: {e}")
            failed_imports.append(f"{module_name}.{class_name}")
    
    # Test main package import
    print(f"\n📦 Testing main package import:")
    try:
        # Test importing from main __init__.py
        import config, llm, storage, chat
        print("   ✅ Direct module imports successful")
        
        from llm import OllamaClient
        from storage import VectorStore
        from chat import ChatEngine
        print("   ✅ From imports successful")
        
    except Exception as e:
        print(f"   ❌ Main package import failed: {e}")
        failed_imports.append("main_package")
    
    # Summary
    print(f"\n📊 Test Results:")
    if failed_imports:
        print(f"   ❌ Failed imports: {len(failed_imports)}")
        for failed in failed_imports:
            print(f"      - {failed}")
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Make sure you're in the arxiv-paper-rag project root")
        print(f"   2. Check that src/ directory exists with all modules")
        print(f"   3. Ensure all dependencies are installed: pip install -r requirements.txt")
        return False
    else:
        print(f"   ✅ All imports successful!")
        print(f"   🎉 RAG system is ready to use!")
        return True

def show_directory_structure():
    """Show the expected directory structure"""
    print(f"\n📁 Expected Directory Structure:")
    expected_structure = [
        "src/",
        "├── config/",
        "├── ingestion/", 
        "├── embeddings/",
        "├── storage/",
        "├── llm/",
        "├── chat/",
        "├── tracking/",
        "├── utils/",
        "├── ui/",
        "└── core/"
    ]
    
    for item in expected_structure:
        print(f"   {item}")

if __name__ == "__main__":
    print("🚀 Starting RAG System Import Test...\n")
    
    success = test_imports()
    
    if not success:
        show_directory_structure()
        print(f"\n❌ Import test failed!")
        sys.exit(1)
    else:
        print(f"\n✅ All tests passed! System is ready!") 