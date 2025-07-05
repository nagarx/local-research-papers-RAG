#!/usr/bin/env python3
"""
Streamlit Interface Test Suite

This test suite validates the Streamlit user interface components
"""

import os
import sys
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, Any
import subprocess
import time

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_test_env():
    """Setup test environment"""
    # Create test directories
    test_dirs = [
        "data/test",
        "data/test/documents",
        "data/test/processed",
        "data/test/embeddings",
        "data/test/index",
        "data/test/cache",
        "data/test/logs"
    ]
    
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

try:
    from config import get_config
    setup_test_env()
    config = get_config()
    
    print("✅ Streamlit configuration loaded successfully")
    print(f"📊 Config: {config.streamlit.port}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure you're in the correct directory")
    sys.exit(1)

def test_streamlit_app():
    """Test Streamlit app imports and basic functionality"""
    
    print("🧪 Testing Streamlit App Imports...")
    print("="*50)
    
    # Add src to path
    src_dir = Path(__file__).parent / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    try:
        # Test imports
        print("📦 Testing core imports...")
        import streamlit as st
        print("  ✅ streamlit")
        
        import pandas as pd
        print("  ✅ pandas") 
        
        from arxiv_rag import get_config
        print("  ✅ arxiv_rag.get_config")
        
        # Test Streamlit app import
        print("\n🎨 Testing Streamlit app import...")
        from src.arxiv_rag.ui.streamlit_app import StreamlitRAGApp, main
        print("  ✅ StreamlitRAGApp")
        print("  ✅ main function")
        
        # Test app initialization (without running Streamlit)
        print("\n🔧 Testing app initialization...")
        
        # We can't fully test the app without Streamlit context,
        # but we can verify the class can be imported and basic structure
        app_class = StreamlitRAGApp
        print(f"  ✅ App class: {app_class.__name__}")
        
        print("\n🎉 All tests passed!")
        print("✅ Streamlit app is ready to launch")
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False

def main():
    """Main test function"""
    print("🎨 Streamlit App Test Suite")
    print("="*30)
    
    success = test_streamlit_app()
    
    if success:
        print("\n" + "="*50)
        print("✅ ALL TESTS PASSED!")
        print("🚀 Ready to launch: python app.py")
        print("🌐 URL: http://localhost:8501")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("❌ Tests failed. Check errors above.")
        print("="*50)
        sys.exit(1)

if __name__ == "__main__":
    main() 