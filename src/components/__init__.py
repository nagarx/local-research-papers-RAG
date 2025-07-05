"""
Component Registration

This module registers all available component implementations with the registry.
"""

def register_all_components():
    """Register all available component implementations"""
    
    # Import the registry
    from ..core.registry import register_component
    from ..core.interfaces import DocumentProcessorProtocol
    
    # Register DocumentProcessor implementations
    from ..document_processor import DocumentProcessor
    register_component(
        DocumentProcessorProtocol,
        DocumentProcessor,
        "marker",
        config={"use_llm": False, "extract_images": True},
        singleton=True
    )
    
    print("✅ Registered DocumentProcessor as 'marker'")
    
    # For now, only register DocumentProcessor as other components need interface updates
    # TODO: Register other components after updating them to implement interfaces
    
    return True

# Auto-register components when module is imported
def setup_components():
    """Setup and register all components"""
    try:
        register_all_components()
        return True
    except Exception as e:
        print(f"❌ Component registration failed: {e}")
        return False
