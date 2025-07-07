#!/usr/bin/env python3
"""
Test Runner for ArXiv RAG System

This script provides an easy way to run different categories of tests.
"""

import sys
import argparse
import subprocess
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and return success status"""
    print(f"\n🚀 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent, capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run ArXiv RAG System Tests")
    parser.add_argument(
        "test_type",
        choices=["all", "unit", "integration", "imports", "rag", "ui", "fast", "slow"],
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-warnings",
        action="store_true",
        help="Disable warnings"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    if args.verbose:
        cmd.append("-v")
    
    if args.no_warnings:
        cmd.append("--disable-warnings")
    
    # Add specific test selection
    if args.test_type == "all":
        cmd.extend(["."])
        description = "Running all tests"
    elif args.test_type == "unit":
        cmd.extend(["-m", "unit"])
        description = "Running unit tests"
    elif args.test_type == "integration":
        cmd.extend(["-m", "integration"])
        description = "Running integration tests"
    elif args.test_type == "imports":
        cmd.extend(["test_imports.py"])
        description = "Running import tests"
    elif args.test_type == "rag":
        cmd.extend(["test_rag_system.py"])
        description = "Running RAG system tests"
    elif args.test_type == "ui":
        cmd.extend(["test_streamlit.py"])
        description = "Running UI tests"
    elif args.test_type == "fast":
        cmd.extend(["-m", "not slow"])
        description = "Running fast tests only"
    elif args.test_type == "slow":
        cmd.extend(["-m", "slow"])
        description = "Running slow tests only"
    
    # Run the tests
    success = run_command(cmd, description)
    
    if success:
        print("\n✅ Tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    print("🧪 ArXiv RAG System Test Runner")
    print("=" * 50)
    
    main() 