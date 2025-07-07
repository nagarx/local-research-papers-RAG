#!/usr/bin/env python3
"""
Root Level Test Runner

Simple wrapper to run tests from the project root directory.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Run tests using the test runner in the tests directory"""
    
    # Path to the actual test runner
    test_runner = Path(__file__).parent / "tests" / "run_tests.py"
    
    if not test_runner.exists():
        print("❌ Error: Test runner not found at tests/run_tests.py")
        sys.exit(1)
    
    # Forward all arguments to the test runner
    cmd = [sys.executable, str(test_runner)] + sys.argv[1:]
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n🛑 Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("🧪 ArXiv RAG System - Test Runner")
        print("=" * 50)
        print("Usage: python run_tests.py <test_type>")
        print()
        print("Test types:")
        print("  all         - Run all tests")
        print("  fast        - Run only fast tests")
        print("  imports     - Run import tests")
        print("  rag         - Run RAG system tests")
        print("  ui          - Run UI tests")
        print("  unit        - Run unit tests")
        print("  integration - Run integration tests")
        print("  slow        - Run slow tests")
        print()
        print("Examples:")
        print("  python run_tests.py fast")
        print("  python run_tests.py imports")
        print("  python run_tests.py rag --verbose")
        print()
        print("For more options, see tests/README.md")
        sys.exit(0)
    
    main() 