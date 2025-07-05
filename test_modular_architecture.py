#!/usr/bin/env python3
"""
Modular Architecture Test

Test script to validate the new modular architecture implementation.
"""

import asyncio
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_basic_imports():
    """Test basic imports"""
    
    print("🔧 Testing Basic Imports and Interfaces")
    print("=" * 50)
    
    # Test 1: Import core interfaces
    print("1. Testing Interface Imports...")
    try:
        from arxiv_rag.core.interfaces import DocumentProcessorProtocol
        print("   ✅ DocumentProcessorProtocol imported successfully")
        
        from arxiv_rag.core.registry import get_registry
        print("   ✅ ComponentRegistry imported successfully")
        
    except Exception as e:
        print(f"   ❌ Interface import failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Test registry
    print("\n2. Testing Component Registry...")
    try:
        registry = get_registry()
        print(f"   ✅ Registry created: {type(registry).__name__}")
        
        stats = registry.get_stats()
        print(f"   ✅ Registry stats: {stats}")
        
    except Exception as e:
        print(f"   ❌ Registry test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Basic Test Completed!")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(test_basic_imports())
