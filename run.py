#!/usr/bin/env python3
"""
ArXiv Paper RAG Assistant - Main Application Launcher

This script launches the Streamlit application and handles initial setup.
"""

import sys
import os
import subprocess
import platform
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def check_python_version():
    """Check if Python version meets requirements"""
    if sys.version_info < (3, 10):
        print("❌ Error: Python 3.10 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✅ Python version: {sys.version.split()[0]}")

def check_ollama():
    """Check if Ollama is installed and running"""
    try:
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode == 0:
            print("✅ Ollama is installed and running")
            return True
        else:
            print("⚠️  Ollama is installed but may not be running")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Ollama not found. Please install Ollama first:")
        print("   Visit: https://ollama.ai")
        return False

def setup_environment():
    """Setup necessary directories and configuration"""
    # Create data directories
    data_dirs = [
        "data",
        "data/documents", 
        "data/processed",
        "data/embeddings",
        "data/index",
        "data/cache",
        "data/logs"
    ]
    
    for dir_path in data_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Create .env from template if it doesn't exist
    env_file = Path(".env")
    env_template = Path("env_example.txt")
    
    if not env_file.exists() and env_template.exists():
        import shutil
        shutil.copy(env_template, env_file)
        print("✅ Created .env configuration file")
    
    print("✅ Environment setup complete")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        "streamlit",
        "marker",
        "ollama", 
        "sentence_transformers",
        "faiss"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
            ])
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            print("Please run: pip install -r requirements.txt")
            sys.exit(1)
    else:
        print("✅ All dependencies are installed")

def download_ollama_model():
    """Download required Ollama model if not present"""
    try:
        # Check if model exists
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if "llama3.2:latest" not in result.stdout:
            print("📥 Downloading Ollama model (this may take a while)...")
            subprocess.run(
                ["ollama", "pull", "llama3.2:latest"],
                check=True,
                timeout=1800  # 30 minutes timeout
            )
            print("✅ Ollama model downloaded successfully")
        else:
            print("✅ Ollama model already available")
            
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Could not download Ollama model automatically")
        print("Please run manually: ollama pull llama3.2:latest")

def launch_streamlit():
    """Launch the Streamlit application"""
    try:
        print("🚀 Launching ArXiv Paper RAG Assistant...")
        print("📖 Open your browser to: http://localhost:8501")
        print("🔄 Starting Streamlit server...")
        
        # Launch Streamlit
        os.environ["PYTHONPATH"] = str(Path(__file__).parent / "src")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "src/ui/main.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down ArXiv Paper RAG Assistant")
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        sys.exit(1)

def main():
    """Main application entry point"""
    print("=" * 60)
    print("🤖 ArXiv Paper RAG Assistant v1.0.0")
    print("=" * 60)
    
    # System checks
    check_python_version()
    check_dependencies()
    
    # Setup environment
    setup_environment()
    
    # Ollama checks
    if check_ollama():
        download_ollama_model()
    else:
        print("⚠️  Continuing without Ollama (limited functionality)")
    
    # Launch application
    print("\n" + "=" * 60)
    launch_streamlit()

if __name__ == "__main__":
    main() 