#!/usr/bin/env python3
"""
Deployment Issues Fix Script

This script fixes common deployment issues:
1. Missing plotly dependency
2. Import structure problems
3. Verifies the installation

Run this script on any machine where the ArXiv RAG Assistant has issues.
"""

import subprocess
import sys
from pathlib import Path
import os

def print_header():
    """Print script header"""
    print("🔧 ArXiv RAG Assistant - Deployment Fix")
    print("=" * 60)
    print("This script will fix common deployment issues:")
    print("• Install missing dependencies (plotly)")
    print("• Verify import structure")
    print("• Test system components")
    print("=" * 60)

def detect_environment():
    """Detect if we're in a virtual environment"""
    venv_path = Path("venv")
    is_venv = venv_path.exists() and (venv_path / "bin" / "python").exists()
    
    if is_venv:
        python_cmd = str(venv_path / "bin" / "python")
        pip_cmd = [python_cmd, "-m", "pip"]
        print(f"✅ Detected virtual environment: {venv_path}")
    else:
        python_cmd = sys.executable
        pip_cmd = [sys.executable, "-m", "pip"]
        print(f"ℹ️  Using system Python: {python_cmd}")
    
    return python_cmd, pip_cmd

def install_missing_dependencies(pip_cmd):
    """Install missing dependencies"""
    print("\n📦 Installing missing dependencies...")
    
    dependencies = [
        "plotly>=5.15.0",
        "aiohttp>=3.8.0",
        "torch>=2.0.0"
    ]
    
    for dep in dependencies:
        try:
            print(f"   Installing {dep}...", end=" ")
            subprocess.check_call(
                pip_cmd + ["install", dep],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("✅")
        except subprocess.CalledProcessError:
            print("❌")
            print(f"   Failed to install {dep}")
            return False
    
    print("✅ All dependencies installed successfully!")
    return True

def test_imports():
    """Test that all imports work correctly"""
    print("\n🧪 Testing imports...")
    
    # Add project root to Python path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    test_cases = [
        ("src.config", "get_config"),
        ("src.embeddings", "EmbeddingManager"),
        ("src.storage", "ChromaVectorStore"),
        ("src.llm", "OllamaClient"),
        ("src.tracking", "SourceTracker"),
        ("src.ingestion", "DocumentProcessor"),
        ("src.chat", "ChatEngine"),
        ("src.utils", "DocumentStatusChecker"),
    ]
    
    failed_imports = []
    
    for module_name, class_name in test_cases:
        try:
            print(f"   Testing {module_name}.{class_name}...", end=" ")
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print("✅")
        except Exception as e:
            print(f"❌ - {e}")
            failed_imports.append(f"{module_name}.{class_name}: {e}")
    
    if failed_imports:
        print(f"\n❌ Failed imports:")
        for failure in failed_imports:
            print(f"   • {failure}")
        return False
    else:
        print("✅ All imports successful!")
        return True

def test_streamlit_specific():
    """Test Streamlit-specific imports"""
    print("\n🌐 Testing Streamlit app compatibility...")
    
    try:
        # Test the exact imports used by the Streamlit app
        from src.chat import ChatEngine
        from src.config import get_config
        from src.ingestion import get_global_marker_models
        from src.utils import DocumentStatusChecker
        
        # Test plotly import (the one that was causing issues)
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            print("   ✅ Plotly imports successful!")
        except ImportError:
            print("   ⚠️  Plotly import failed - analytics charts may not work")
        
        print("   ✅ Core Streamlit imports successful!")
        return True
        
    except Exception as e:
        print(f"   ❌ Streamlit imports failed: {e}")
        return False

def fix_missing_init_files():
    """Fix missing __init__.py files"""
    print("\n🔧 Fixing missing __init__.py files...")
    
    init_files = {
        "src/embeddings/__init__.py": '''"""
Embeddings Module

This module handles embedding generation and management using SentenceTransformers.
"""

from .embedding_manager import EmbeddingManager

__all__ = ['EmbeddingManager']
''',
        "src/config/__init__.py": '''"""
Configuration Module

This module handles configuration management for the ArXiv RAG system.
"""

from .config import get_config, Config

__all__ = ['get_config', 'Config']
''',
        "src/storage/__init__.py": '''"""
Storage Module

This module handles vector storage and session management.
"""

from .chroma_vector_store import ChromaVectorStore
from .session_manager import SessionManager

__all__ = ['ChromaVectorStore', 'SessionManager']
''',
        "src/llm/__init__.py": '''"""
LLM Module

This module handles local LLM integration using Ollama.
"""

from .ollama_client import OllamaClient

__all__ = ['OllamaClient']
''',
        "src/tracking/__init__.py": '''"""
Tracking Module

This module handles source tracking and citation management.
"""

from .source_tracker import SourceTracker, SourceReference

__all__ = ['SourceTracker', 'SourceReference']
''',
        "src/ingestion/__init__.py": '''"""
Ingestion Module

This module handles document processing and ingestion using Marker.
"""

from .document_processor import DocumentProcessor, get_global_marker_models

__all__ = ['DocumentProcessor', 'get_global_marker_models']
''',
        "src/chat/__init__.py": '''"""
Chat Module

This module handles chat orchestration and query processing for the RAG system.
"""

from .chat_engine import ChatEngine

__all__ = ['ChatEngine']
''',
        "src/utils/__init__.py": '''"""
Utils Module

This module contains utility functions and classes.
"""

from .document_status import DocumentStatusChecker
from .check_documents import main as check_documents_main

try:
    from .enhanced_logging import (
        get_enhanced_logger, startup_banner, startup_complete, suppress_noisy_loggers
    )
except ImportError:
    # Fallback if enhanced logging is not available
    def get_enhanced_logger(name):
        import logging
        return logging.getLogger(name)
    
    def startup_banner(*args, **kwargs):
        pass
    
    def startup_complete(*args, **kwargs):
        pass
    
    def suppress_noisy_loggers():
        pass

__all__ = [
    'DocumentStatusChecker', 
    'check_documents_main',
    'get_enhanced_logger', 
    'startup_banner', 
    'startup_complete', 
    'suppress_noisy_loggers'
]
''',
        "src/ui/__init__.py": '''"""
UI Module

This module contains the Streamlit web interface.
"""

# UI module - no exports needed
__all__ = []
'''
    }
    
    created_files = 0
    
    for file_path, content in init_files.items():
        path = Path(file_path)
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if not path.exists():
            try:
                with open(path, 'w') as f:
                    f.write(content)
                print(f"   ✅ Created {path}")
                created_files += 1
            except Exception as e:
                print(f"   ❌ Failed to create {path}: {e}")
                return False
        else:
            print(f"   ✓ {path} (exists)")
    
    if created_files > 0:
        print(f"✅ Created {created_files} missing __init__.py files!")
    else:
        print("✅ All __init__.py files already present!")
    
    return True

def verify_file_structure():
    """Verify that all necessary files exist"""
    print("\n📁 Verifying file structure...")
    
    critical_files = [
        "src/__init__.py",
        "src/config/__init__.py",
        "src/embeddings/__init__.py",
        "src/storage/__init__.py",
        "src/llm/__init__.py",
        "src/tracking/__init__.py",
        "src/ingestion/__init__.py",
        "src/chat/__init__.py",
        "src/utils/__init__.py",
        "src/ui/streamlit_app.py",
        "requirements.txt",
        "run.py"
    ]
    
    missing_files = []
    
    for file_path in critical_files:
        if Path(file_path).exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing files: {len(missing_files)}")
        return False
    else:
        print("✅ All critical files present!")
        return True

def create_launcher_script():
    """Create or update the launcher script"""
    print("\n🚀 Creating launcher script...")
    
    launcher_content = '''#!/bin/bash
# ArXiv RAG Assistant Launcher
echo "🚀 Starting ArXiv RAG Assistant..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Start the application
python run.py
'''
    
    try:
        with open("start_rag.sh", "w") as f:
            f.write(launcher_content)
        
        # Make it executable
        os.chmod("start_rag.sh", 0o755)
        print("✅ Launcher script created: start_rag.sh")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create launcher script: {e}")
        return False

def main():
    """Main fix function"""
    print_header()
    
    # Step 1: Detect environment
    python_cmd, pip_cmd = detect_environment()
    
    # Step 2: Fix missing __init__.py files
    if not fix_missing_init_files():
        print("\n❌ Failed to create missing __init__.py files.")
        return 1
    
    # Step 3: Verify file structure
    if not verify_file_structure():
        print("\n❌ Critical files are still missing. Please ensure you have the complete repository.")
        return 1
    
    # Step 4: Install dependencies
    if not install_missing_dependencies(pip_cmd):
        print("\n❌ Failed to install dependencies. Try running manually:")
        print("   pip install plotly>=5.15.0 aiohttp>=3.8.0 torch>=2.0.0")
        return 1
    
    # Step 5: Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Check the error messages above.")
        return 1
    
    # Step 6: Test Streamlit compatibility
    if not test_streamlit_specific():
        print("\n❌ Streamlit compatibility test failed.")
        return 1
    
    # Step 7: Create launcher
    if not create_launcher_script():
        print("\n⚠️  Launcher script creation failed, but core issues are fixed.")
    
    # Success message
    print("\n" + "=" * 60)
    print("🎉 All fixes applied successfully!")
    print("=" * 60)
    print("✅ Dependencies installed")
    print("✅ Imports verified")
    print("✅ Streamlit compatibility confirmed")
    print("✅ File structure validated")
    
    print("\n🚀 You can now start the application:")
    print("   ./start_rag.sh")
    print("   or")
    print("   python run.py")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    if exit_code != 0:
        print(f"\n❌ Fix script completed with errors (exit code: {exit_code})")
        print("   Please check the error messages above and try manual fixes.")
    sys.exit(exit_code) 