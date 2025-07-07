#!/usr/bin/env python3
"""
ArXiv Paper RAG Assistant - Main Application Launcher

This script launches the Streamlit application and handles initial setup.
"""

import sys
import os
import subprocess
import platform
import time
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

def check_system_requirements():
    """Check system requirements"""
    print("🔍 Checking system requirements...")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        print(f"   Memory: {total_gb:.1f}GB total, {available_gb:.1f}GB available")
        
        if total_gb < 4:
            print("⚠️  Warning: Less than 4GB RAM detected")
            print("   The system may run slowly with large documents")
        elif total_gb < 8:
            print("⚠️  Warning: Less than 8GB RAM detected")
            print("   Consider processing documents in smaller batches")
        else:
            print("✅ Memory requirements met")
            
    except ImportError:
        print("   Could not check memory (psutil not available)")
    except Exception as e:
        print(f"   Could not check memory: {e}")
    
    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        
        print(f"   Disk space: {free_gb:.1f}GB free")
        
        if free_gb < 2:
            print("⚠️  Warning: Less than 2GB disk space available")
            print("   You may run out of space when processing documents")
        else:
            print("✅ Disk space requirements met")
            
    except Exception as e:
        print(f"   Could not check disk space: {e}")
    
    # Check Python path and executable
    print(f"   Python executable: {sys.executable}")
    print(f"   Python path: {sys.path[0]}")
    
    print("✅ System requirements check complete")

def check_ollama():
    """Check if Ollama is installed and running"""
    try:
        # First check if ollama command exists
        result = subprocess.run(
            ["ollama", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        
        if result.returncode != 0:
            print("❌ Ollama not found. Please install Ollama first:")
            print("   Visit: https://ollama.ai")
            return False
        
        # Check if ollama service is running
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
            print("⚠️  Ollama is installed but service may not be running")
            print("   Trying to start Ollama service...")
            
            # Try to start ollama serve in background
            try:
                if platform.system() == "Windows":
                    subprocess.Popen(["ollama", "serve"], shell=True)
                else:
                    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Wait a bit and check again
                time.sleep(5)
                result = subprocess.run(
                    ["ollama", "list"], 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
                
                if result.returncode == 0:
                    print("✅ Ollama service started successfully")
                    return True
                else:
                    print("⚠️  Could not start Ollama service automatically")
                    return False
                    
            except Exception as e:
                print(f"⚠️  Could not start Ollama service: {e}")
                return False
                
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Ollama not found. Please install Ollama first:")
        print("   Visit: https://ollama.ai")
        return False

def setup_environment():
    """Setup necessary directories and configuration"""
    try:
        # Create data directories
        data_dirs = [
            "data",
            "data/documents", 
            "data/processed",
            "data/embeddings",
            "data/index",
            "data/cache",
            "data/logs",
            "temp_uploads"
        ]
        
        for dir_path in data_dirs:
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            except PermissionError:
                print(f"⚠️  Permission denied creating directory: {dir_path}")
                print("   Please check file permissions or run as administrator")
                continue
            except Exception as e:
                print(f"⚠️  Error creating directory {dir_path}: {e}")
                continue
        
        # Create .env from template if it doesn't exist
        env_file = Path(".env")
        env_template = Path("env_example.txt")
        
        if not env_file.exists() and env_template.exists():
            try:
                import shutil
                shutil.copy(env_template, env_file)
                print("✅ Created .env configuration file")
            except Exception as e:
                print(f"⚠️  Could not create .env file: {e}")
                print("   You may need to manually copy env_example.txt to .env")
        
        # Validate write permissions in key directories
        test_dirs = ["data", "temp_uploads"]
        for test_dir in test_dirs:
            try:
                test_file = Path(test_dir) / ".write_test"
                test_file.write_text("test")
                test_file.unlink()
            except Exception as e:
                print(f"⚠️  Write permission issue in {test_dir}: {e}")
                print("   Please check directory permissions")
        
        print("✅ Environment setup complete")
        
    except Exception as e:
        print(f"❌ Error during environment setup: {e}")
        print("   Some features may not work properly")
        print("   Please check file permissions and disk space")

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        ("streamlit", "streamlit"),
        ("marker-pdf", "marker"),
        ("ollama", "ollama"),
        ("sentence-transformers", "sentence_transformers"),
        ("chromadb", "chromadb"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("pydantic", "pydantic"),
        ("python-dotenv", "dotenv"),
        ("aiofiles", "aiofiles"),
        ("beautifulsoup4", "bs4"),
        ("requests", "requests"),
        ("tqdm", "tqdm"),
        ("loguru", "loguru"),
        ("psutil", "psutil")
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        # Try pip install with retry logic
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Upgrade pip first
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--upgrade", "pip"
                ])
                
                # Install requirements
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
                ])
                print("✅ Dependencies installed successfully")
                return
                
            except subprocess.CalledProcessError as e:
                if attempt == max_attempts - 1:
                    print(f"❌ Failed to install dependencies after {max_attempts} attempts")
                    print(f"Error: {e}")
                    print("Please try manually:")
                    print("  1. python -m pip install --upgrade pip")
                    print("  2. pip install -r requirements.txt")
                    sys.exit(1)
                else:
                    print(f"⚠️  Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2)
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
            "src/ui/streamlit_app.py",
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
    check_system_requirements()
    check_dependencies()
    
    # Setup environment
    setup_environment()
    
    # Ollama checks
    if check_ollama():
        download_ollama_model()
    else:
        print("⚠️  Continuing without Ollama (limited functionality)")
        print("   Install Ollama later from: https://ollama.ai")
    
    # Launch application
    print("\n" + "=" * 60)
    launch_streamlit()

if __name__ == "__main__":
    main() 