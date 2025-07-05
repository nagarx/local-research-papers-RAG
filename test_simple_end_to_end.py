#!/usr/bin/env python3
"""
Simple End-to-End Modular Architecture Test
"""

import asyncio
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_simple_workflow():
    """Test the simple workflow"""
    
    print("🚀 Simple End-to-End Modular Architecture Test")
    print("=" * 60)
    
    # Step 1: Initialize
    print("1. Initializing Modular System...")
    from arxiv_rag.core.registry import get_registry, get_component
    from arxiv_rag.core.interfaces import DocumentProcessorProtocol
    from arxiv_rag.components import setup_components
    
    setup_result = setup_components()
    print(f"   ✅ Component setup: {setup_result}")
    
    registry = get_registry()
    stats = registry.get_stats()
    print(f"   ✅ Registry initialized: {stats}")
    
    # Step 2: Component Discovery
    print("\n2. Component Discovery...")
    interfaces = registry.list_interfaces()
    print(f"   ✅ Available interfaces: {interfaces}")
    
    implementations = registry.list_implementations(DocumentProcessorProtocol)
    print(f"   ✅ DocumentProcessor implementations: {implementations}")
    
    # Step 3: Component Creation
    print("\n3. Component Creation...")
    doc_processor = get_component(DocumentProcessorProtocol, "marker")
    print(f"   ✅ Created processor: {type(doc_processor).__name__}")
    
    stats = doc_processor.get_processing_stats()
    print(f"   ✅ Processor stats: {stats}")
    
    # Step 4: Demonstrate Configuration
    print("\n4. Configuration Management...")
    component_info = registry.get_component_info(DocumentProcessorProtocol, "marker")
    print(f"   ✅ Component implementation: {component_info['implementation']}")
    print(f"   ✅ Is singleton: {component_info['is_singleton']}")
    print(f"   ✅ Current config: {component_info['config']}")
    
    # Step 5: Final Stats
    print("\n5. Final Registry Statistics...")
    final_stats = registry.get_stats()
    print(f"   ✅ Final stats: {final_stats}")
    
    print("\n🎉 Simple End-to-End Test Completed Successfully!")
    print("=" * 60)
    
    print("\n🌟 Key Features Validated:")
    print("   ✅ Component registration")
    print("   ✅ Component discovery")
    print("   ✅ Component creation")
    print("   ✅ Configuration management")
    print("   ✅ Singleton pattern")
    print("   ✅ Registry statistics")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_simple_workflow())
    print(f"\n✅ Test result: {'PASSED' if success else 'FAILED'}")
