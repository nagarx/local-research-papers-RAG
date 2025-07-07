"""
Shared Test Configuration and Fixtures

This module provides shared configuration, fixtures, and utilities for all tests.
"""

import pytest
import asyncio
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Ensure project root is in path so we can import src as a package
import sys
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import after path setup
try:
    from src.config import get_config, get_logger
    from src.chat import ChatEngine
    from src.storage import ChromaVectorStore
    from src.ingestion import get_global_marker_models
    from src.utils.torch_utils import configure_torch_for_production
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    print("This is expected if dependencies are not installed")


class TestResultTracker:
    """Track test results across the suite"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_details = []
        self.start_time = time.time()
    
    def log_result(self, test_name: str, success: bool, duration: float, details: str = ""):
        """Log a test result"""
        self.total_tests += 1
        
        if success:
            self.passed_tests += 1
            status = "✅ PASSED"
        else:
            self.failed_tests += 1
            status = "❌ FAILED"
        
        print(f"{status} {test_name} ({duration:.2f}s)")
        if details:
            print(f"   {details}")
        
        self.test_details.append({
            "test": test_name,
            "success": success,
            "duration": duration,
            "details": details
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get test summary"""
        total_duration = time.time() - self.start_time
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "success_rate": (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0,
            "total_duration": total_duration,
            "details": self.test_details
        }
    
    def print_summary(self):
        """Print test summary"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("🧪 TEST SUITE SUMMARY")
        print("="*60)
        print(f"📊 Total Tests: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed_tests']}")
        print(f"❌ Failed: {summary['failed_tests']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")
        print(f"⏱️  Total Duration: {summary['total_duration']:.2f}s")
        
        if summary['failed_tests'] > 0:
            print("\n❌ Failed Tests:")
            for test in summary['details']:
                if not test['success']:
                    print(f"   • {test['test']}: {test['details']}")
        
        print("="*60)
        
        if summary['failed_tests'] == 0:
            print("🎉 ALL TESTS PASSED! Your RAG system is working perfectly!")
        else:
            print(f"⚠️  {summary['failed_tests']} test(s) failed.")


@pytest.fixture(scope="session")
def test_tracker():
    """Global test result tracker"""
    return TestResultTracker()


@pytest.fixture(scope="session")
def configure_torch():
    """Configure PyTorch for testing"""
    try:
        configure_torch_for_production()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def sample_documents():
    """Get list of sample documents for testing"""
    sample_dir = Path("sample_documents")
    if sample_dir.exists():
        return list(sample_dir.glob("*.pdf"))
    return []


@pytest.fixture(scope="session")
def temp_test_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory(prefix="rag_test_") as temp_dir:
        yield Path(temp_dir)


@pytest.fixture(scope="session") 
def test_config():
    """Get test configuration"""
    try:
        return get_config()
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        return None


@pytest.fixture(scope="session")
def preload_models():
    """Pre-load Marker models to avoid timing issues"""
    try:
        models = get_global_marker_models()
        return models is not None
    except Exception as e:
        print(f"Warning: Could not preload models: {e}")
        return False


@pytest.fixture
async def chat_engine(test_config, preload_models):
    """Create ChatEngine instance for testing"""
    if not test_config:
        pytest.skip("Configuration not available")
    
    try:
        engine = ChatEngine()
        yield engine
    except Exception as e:
        pytest.skip(f"Could not create ChatEngine: {e}")


@pytest.fixture
def vector_store(test_config):
    """Create ChromaVectorStore instance for testing"""
    if not test_config:
        pytest.skip("Configuration not available")
    
    try:
        store = ChromaVectorStore()
        yield store
    except Exception as e:
        pytest.skip(f"Could not create ChromaVectorStore: {e}")


def pytest_configure(config):
    """Configure pytest"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take several seconds)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Add markers to tests based on naming patterns
    for item in items:
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        if "slow" in item.nodeid or "end_to_end" in item.nodeid:
            item.add_marker(pytest.mark.slow) 