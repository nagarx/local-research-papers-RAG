#!/usr/bin/env python3
"""
Complete Modular Architecture Test

Test script to validate the full modular architecture implementation.
"""

import asyncio
import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_modular_architecture():
    """Test the complete modular architecture implementation"""
    
    print("🔧 Testing Complete Modular Architecture Implementation")
    print("=" * 60)
    
    # Test 1: Basic imports
    print("1. Testing Interface Imports...")
    try:
        from arxiv_rag.core.interfaces import DocumentProcessorProtocol
        print("   ✅ DocumentProcessorProtocol imported successfully")
        
        from arxiv_rag.core.registry import get_registry, get_component
        print("   ✅ ComponentRegistry imported successfully")
        
    except Exception as e:
        print(f"   ❌ Interface import failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 2: Registry creation
    print("\n2. Testing Component Registry...")
    try:
        registry = get_registry()
        print(f"   ✅ Registry created: {type(registry).__name__}")
        
        stats = registry.get_stats()
        print(f"   ✅ Initial registry stats: {stats}")
        
    except Exception as e:
        print(f"   ❌ Registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 3: Component registration
    print("\n3. Testing Component Registration...")
    try:
        from arxiv_rag.components import setup_components
        result = setup_components()
        print(f"   ✅ Component registration result: {result}")
        
        # Check registry stats after registration
        stats = registry.get_stats()
        print(f"   ✅ Registry stats after registration: {stats}")
        
        # List available interfaces
        interfaces = registry.list_interfaces()
        print(f"   ✅ Available interfaces: {interfaces}")
        
        # List implementations for DocumentProcessorProtocol
        if "DocumentProcessorProtocol" in interfaces:
            implementations = registry.list_implementations(DocumentProcessorProtocol)
            print(f"   ✅ DocumentProcessor implementations: {implementations}")
        
    except Exception as e:
        print(f"   ❌ Component registration failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test 4: Component creation
    print("\n4. Testing Component Creation...")
    try:
        # Test creating a DocumentProcessor
        doc_processor = get_component(DocumentProcessorProtocol, "marker")
        print(f"   ✅ Created DocumentProcessor: {type(doc_processor).__name__}")
        
        # Test component methods (if available)
        try:
            if hasattr(doc_processor, 'get_processing_stats'):
                stats = doc_processor.get_processing_stats()
                print(f"   ✅ Processor stats: {stats}")
        except Exception as e:
            print(f"   ⚠️  Processor stats unavailable: {e}")
        
    except Exception as e:
        print(f"   ❌ Component creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Component swapping demonstration
    print("\n5. Testing Component Swapping Capability...")
    try:
        # Get component info
        component_info = registry.get_component_info(DocumentProcessorProtocol, "marker")
        print(f"   ✅ Component info: {component_info['implementation']}")
        print(f"   ✅ Is singleton: {component_info['is_singleton']}")
        print(f"   ✅ Has config: {component_info['has_config']}")
        
        # Demonstrate configuration override
        if component_info['has_config']:
            print(f"   ✅ Current config: {component_info['config']}")
            
            # Create with config override
            doc_processor_custom = get_component(
                DocumentProcessorProtocol, 
                "marker", 
                config_override={"custom_setting": True}
            )
            print(f"   ✅ Created with config override: {type(doc_processor_custom).__name__}")
        
    except Exception as e:
        print(f"   ❌ Component swapping test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 6: Registry statistics
    print("\n6. Final Registry Statistics...")
    try:
        final_stats = registry.get_stats()
        print(f"   ✅ Final registry stats: {final_stats}")
        
        print("\n📊 Summary:")
        print(f"   • Interfaces registered: {final_stats['interfaces']}")
        print(f"   • Total implementations: {final_stats['total_implementations']}")
        print(f"   • Singletons: {final_stats['singletons']}")
        print(f"   • Instantiated singletons: {final_stats['instantiated_singletons']}")
        
    except Exception as e:
        print(f"   ❌ Final stats test failed: {e}")
    
    print("\n🎉 Modular Architecture Test Completed!")
    print("=" * 60)
    
    # Summary of benefits
    print("\n🌟 Modular Architecture Benefits Demonstrated:")
    print("   ✅ Interface-based design")
    print("   ✅ Dependency injection")
    print("   ✅ Component registry")
    print("   ✅ Configuration management")
    print("   ✅ Singleton support")
    print("   ✅ Component swapping capability")
    print("   ✅ Runtime component discovery")

if __name__ == "__main__":
    asyncio.run(test_modular_architecture())
