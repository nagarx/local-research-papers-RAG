#!/usr/bin/env python3
"""
Test Marker Logging - See ALL marker internal logs

This script demonstrates the full marker logging configuration.
Run this to see all marker internal processing logs.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("🔍 ENABLING ALL MARKER LOGGING - You will see EVERYTHING!")
    print("=" * 60)
    
    # Import and configure logging
    from src.utils.enhanced_logging import configure_marker_logging, suppress_noisy_loggers
    
    # Enable ALL marker logging
    configure_marker_logging(enable_debug=True, log_level="DEBUG")
    
    print("=" * 60)
    print("✅ All marker loggers are now enabled at DEBUG level")
    print("🚀 Now start your application to see full marker logs:")
    print("   python app.py")
    print("")
    print("📋 You will see logs from these marker components:")
    print("   🤖 marker.models - Model loading and management")
    print("   📄 marker.converters - PDF conversion process")
    print("   🏗️  marker.builders - Document structure building")
    print("   ⚙️  marker.processors - Text processing stages")
    print("   🎨 marker.renderers - Output rendering")
    print("   👁️  surya.* - OCR processing (layout, text detection)")
    print("   📐 texify.* - Equation processing")
    print("   📑 pdfium/pypdfium2 - PDF parsing")
    print("   🔧 pdf_postprocessor - Post-processing")
    print("")
    print("💡 TIP: The logs will be VERY verbose - perfect for debugging!")
    print("=" * 60)

if __name__ == "__main__":
    main() 