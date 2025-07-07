#!/usr/bin/env python3
"""
ArXiv Paper RAG Assistant - Installation Verification Script

This script verifies that all components are properly installed and configured.
Run this after installation to ensure everything is working correctly.
"""

import sys
import os
import subprocess
import importlib
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class InstallationVerifier:
    """Comprehensive installation verification"""
    
    def __init__(self):
        self.results = []
        self.warnings = []
        self.errors = []
        
    def log_result(self, test_name: str, passed: bool, message: str = "", warning: bool = False):
        """Log a test result"""
        status = "✅" if passed else "❌"
        if warning:
            status = "⚠️ "
        
        result = {
            "test": test_name,
            "passed": passed,
            "message": message,
            "warning": warning
        }
        
        self.results.append(result)
        print(f"{status} {test_name}: {message}")
        
        if warning:
            self.warnings.append(result)
        elif not passed:
            self.errors.append(result)
    
    def check_python_version(self) -> bool:
        """Check Python version"""
        try:
            version = sys.version_info
            if version >= (3, 10):
                self.log_result("Python Version", True, f"Python {version.major}.{version.minor}.{version.micro}")
                return True
            else:
                self.log_result("Python Version", False, f"Python {version.major}.{version.minor}.{version.micro} (3.10+ required)")
                return False
        except Exception as e:
            self.log_result("Python Version", False, f"Error checking version: {e}")
            return False
    
    def check_required_packages(self) -> bool:
        """Check if required packages are installed"""
        required_packages = [
            ("streamlit", "streamlit"),
            ("marker", "marker"),
            ("ollama", "ollama"),
            ("sentence_transformers", "sentence_transformers"),
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
            ("psutil", "psutil"),
        ]
        
        all_passed = True
        
        for package_name, import_name in required_packages:
            try:
                importlib.import_module(import_name)
                self.log_result(f"Package: {package_name}", True, "Installed")
            except ImportError:
                self.log_result(f"Package: {package_name}", False, "Not installed")
                all_passed = False
            except Exception as e:
                self.log_result(f"Package: {package_name}", False, f"Error: {e}")
                all_passed = False
        
        return all_passed
    
    def check_ollama_installation(self) -> bool:
        """Check Ollama installation and service"""
        try:
            # Check if ollama command exists
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                self.log_result("Ollama Installation", False, "Ollama not found")
                return False
            
            # Check if service is running
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                self.log_result("Ollama Service", True, "Running")
                
                # Check if required model is available
                if "llama3.2" in result.stdout:
                    self.log_result("Ollama Model", True, "llama3.2 model available")
                else:
                    self.log_result("Ollama Model", False, "llama3.2 model not found", warning=True)
                
                return True
            else:
                self.log_result("Ollama Service", False, "Not running")
                return False
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.log_result("Ollama Installation", False, "Ollama not found")
            return False
        except Exception as e:
            self.log_result("Ollama Installation", False, f"Error: {e}")
            return False
    
    def check_directory_structure(self) -> bool:
        """Check if required directories exist"""
        required_dirs = [
            "src",
            "src/ui",
            "src/chat",
            "src/llm",
            "src/storage",
            "src/ingestion",
            "src/config",
            "src/utils",
            "data",
            "data/documents",
            "data/processed",
            "data/embeddings",
            "data/cache",
            "data/logs",
            "temp_uploads",
            "tests"
        ]
        
        all_passed = True
        
        for directory in required_dirs:
            if Path(directory).exists():
                self.log_result(f"Directory: {directory}", True, "Exists")
            else:
                self.log_result(f"Directory: {directory}", False, "Missing")
                all_passed = False
        
        return all_passed
    
    def check_configuration_files(self) -> bool:
        """Check if configuration files exist"""
        required_files = [
            "requirements.txt",
            "env_example.txt",
            "run.py",
            "app.py",
            "setup.py"
        ]
        
        all_passed = True
        
        for file_path in required_files:
            if Path(file_path).exists():
                self.log_result(f"Config File: {file_path}", True, "Exists")
            else:
                self.log_result(f"Config File: {file_path}", False, "Missing")
                all_passed = False
        
        # Check if .env exists (warning if not)
        if Path(".env").exists():
            self.log_result("Environment File", True, ".env exists")
        else:
            self.log_result("Environment File", False, ".env missing", warning=True)
        
        return all_passed
    
    def check_file_permissions(self) -> bool:
        """Check file permissions for key directories"""
        test_dirs = ["data", "temp_uploads", "data/logs"]
        all_passed = True
        
        for directory in test_dirs:
            try:
                test_file = Path(directory) / ".permission_test"
                test_file.write_text("test")
                test_file.unlink()
                self.log_result(f"Permissions: {directory}", True, "Writable")
            except Exception as e:
                self.log_result(f"Permissions: {directory}", False, f"Not writable: {e}")
                all_passed = False
        
        return all_passed
    
    def check_system_resources(self) -> bool:
        """Check system resources"""
        all_passed = True
        
        try:
            import psutil
            
            # Check memory
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)
            
            if total_gb >= 8:
                self.log_result("Memory", True, f"{total_gb:.1f}GB total, {available_gb:.1f}GB available")
            elif total_gb >= 4:
                self.log_result("Memory", True, f"{total_gb:.1f}GB total (minimum met)", warning=True)
            else:
                self.log_result("Memory", False, f"{total_gb:.1f}GB total (insufficient)")
                all_passed = False
            
            # Check disk space
            disk = psutil.disk_usage('.')
            free_gb = disk.free / (1024**3)
            
            if free_gb >= 5:
                self.log_result("Disk Space", True, f"{free_gb:.1f}GB free")
            elif free_gb >= 2:
                self.log_result("Disk Space", True, f"{free_gb:.1f}GB free (minimum met)", warning=True)
            else:
                self.log_result("Disk Space", False, f"{free_gb:.1f}GB free (insufficient)")
                all_passed = False
            
        except ImportError:
            self.log_result("System Resources", False, "psutil not available", warning=True)
        except Exception as e:
            self.log_result("System Resources", False, f"Error checking resources: {e}")
            all_passed = False
        
        return all_passed
    
    def test_imports(self) -> bool:
        """Test importing key modules"""
        test_modules = [
            ("src.ui.streamlit_app", "Streamlit App"),
            ("src.chat.chat_engine", "Chat Engine"),
            ("src.llm.ollama_client", "Ollama Client"),
            ("src.storage.chroma_vector_store", "Vector Store"),
            ("src.ingestion.document_processor", "Document Processor"),
            ("src.config.config", "Configuration"),
        ]
        
        all_passed = True
        
        for module_name, display_name in test_modules:
            try:
                importlib.import_module(module_name)
                self.log_result(f"Import: {display_name}", True, "Success")
            except ImportError as e:
                self.log_result(f"Import: {display_name}", False, f"Failed: {e}")
                all_passed = False
            except Exception as e:
                self.log_result(f"Import: {display_name}", False, f"Error: {e}")
                all_passed = False
        
        return all_passed
    
    def test_streamlit_launch(self) -> bool:
        """Test if Streamlit can be launched"""
        try:
            # Try to get streamlit version
            result = subprocess.run(
                [sys.executable, "-m", "streamlit", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                self.log_result("Streamlit Launch", True, "Streamlit can be launched")
                return True
            else:
                self.log_result("Streamlit Launch", False, "Streamlit launch failed")
                return False
                
        except Exception as e:
            self.log_result("Streamlit Launch", False, f"Error: {e}")
            return False
    
    def test_basic_functionality(self) -> bool:
        """Test basic functionality"""
        try:
            # Test vector store creation
            from src.storage.chroma_vector_store import ChromaVectorStore
            vector_store = ChromaVectorStore(
                persist_directory="./data/test_chroma",
                collection_name="test_collection"
            )
            self.log_result("Vector Store", True, "Can create vector store")
            
            # Clean up test collection
            try:
                import shutil
                shutil.rmtree("./data/test_chroma", ignore_errors=True)
            except:
                pass
            
            return True
            
        except Exception as e:
            self.log_result("Vector Store", False, f"Error: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all verification tests"""
        print("=" * 60)
        print("🔍 ArXiv Paper RAG Assistant - Installation Verification")
        print("=" * 60)
        
        tests = [
            ("Python Version", self.check_python_version),
            ("Required Packages", self.check_required_packages),
            ("Ollama Installation", self.check_ollama_installation),
            ("Directory Structure", self.check_directory_structure),
            ("Configuration Files", self.check_configuration_files),
            ("File Permissions", self.check_file_permissions),
            ("System Resources", self.check_system_resources),
            ("Module Imports", self.test_imports),
            ("Streamlit Launch", self.test_streamlit_launch),
            ("Basic Functionality", self.test_basic_functionality),
        ]
        
        for test_name, test_func in tests:
            print(f"\n📋 Running {test_name} tests...")
            test_func()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 VERIFICATION SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["passed"])
        warning_tests = len(self.warnings)
        error_tests = len(self.errors)
        
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Warnings: {warning_tests}")
        print(f"Errors: {error_tests}")
        
        if error_tests == 0:
            print("\n✅ Installation verification PASSED!")
            print("🚀 Your ArXiv Paper RAG Assistant is ready to use!")
            if warning_tests > 0:
                print(f"⚠️  {warning_tests} warning(s) noted - system will work but may have reduced performance")
        else:
            print(f"\n❌ Installation verification FAILED with {error_tests} error(s)")
            print("Please fix the errors above before using the system")
        
        if warning_tests > 0:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"   - {warning['test']}: {warning['message']}")
        
        if error_tests > 0:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"   - {error['test']}: {error['message']}")
        
        return {
            "total": total_tests,
            "passed": passed_tests,
            "warnings": warning_tests,
            "errors": error_tests,
            "success": error_tests == 0
        }

def main():
    """Main verification function"""
    verifier = InstallationVerifier()
    results = verifier.run_all_tests()
    
    # Exit with appropriate code
    if results["success"]:
        print("\n🎉 Ready to launch! Run: python run.py")
        sys.exit(0)
    else:
        print("\n🔧 Please fix the errors and run this script again")
        sys.exit(1)

if __name__ == "__main__":
    main() 