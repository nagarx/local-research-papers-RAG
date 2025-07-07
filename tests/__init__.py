"""
ArXiv RAG System - Test Suite

This package contains comprehensive tests for the ArXiv RAG system.

Test Organization:
- test_imports.py: Import and basic functionality tests
- test_rag_system.py: Core RAG functionality tests
- test_streamlit.py: Streamlit UI tests
- conftest.py: Shared test configuration and fixtures
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

__version__ = "1.0.0" 