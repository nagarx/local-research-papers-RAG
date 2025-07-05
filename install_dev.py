#!/usr/bin/env python3
"""
Development Installation Script

Install the ArXiv RAG package in development mode for reliable imports.
This is the most robust way to ensure imports work correctly.
"""

import subprocess
import sys
from pathlib import Path

def install_dev_mode():
    """Install the package in development mode"""
    
    project_root = Path(__file__).parent.absolute()
    setup_py = project_root / "setup.py"
    
    print("🔧 Installing ArXiv RAG package in development mode...")
    print("="*60)
    print(f"📂 Project root: {project_root}")
    print(f"📄 Setup file: {setup_py}")
    
    if not setup_py.exists():
        print(f"❌ setup.py not found at: {setup_py}")
        return False
    
    try:
        # Install in development mode
        print("\n📦 Running: pip install -e .")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], cwd=str(project_root), check=True, capture_output=True, text=True)
        
        print("✅ Package installed successfully!")
        print("\n📋 Installation output:")
        print(result.stdout)
        
        # Test import
        print("\n🧪 Testing import...")
        test_result = subprocess.run([
            sys.executable, "-c", "from arxiv_rag import get_config; print('✅ Import successful!')"
        ], capture_output=True, text=True)
        
        if test_result.returncode == 0:
            print("✅ Import test passed!")
            print("\n🚀 You can now run: python app.py")
            return True
        else:
            print(f"❌ Import test failed: {test_result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Main function"""
    print("🛠️  ArXiv RAG Development Setup")
    print("="*40)
    
    success = install_dev_mode()
    
    if success:
        print("\n" + "="*60)
        print("✅ SETUP COMPLETE!")
        print("🚀 Run the app: python app.py")
        print("🌐 Open browser: http://localhost:8501")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Setup failed. Please check the errors above.")
        print("💡 You may need to install dependencies first:")
        print("   pip install -r requirements.txt")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    main() 