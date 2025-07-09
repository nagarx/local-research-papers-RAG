#!/usr/bin/env python3
"""
Test ALL Logging - See EVERYTHING from the entire pipeline

This script demonstrates that ALL logging is enabled with NO suppression.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("🚀 ENABLING ALL LOGGING - NO SUPPRESSION WHATSOEVER!")
    print("=" * 70)
    
    # Import and configure logging
    from src.utils.enhanced_logging import enable_all_logging
    
    # Enable ALL logging from ALL libraries
    enable_all_logging()
    
    print("=" * 70)
    print("✅ ALL LOGGERS ENABLED - No suppression applied")
    print("🚀 Now start your application to see COMPLETE pipeline logs:")
    print("   python app.py")
    print("")
    print("📋 You will see EVERYTHING from ALL libraries:")
    print("   🤖 marker.* - All marker internal processing")
    print("   👁️  surya.* - All OCR processing details")
    print("   📐 texify.* - All equation processing")
    print("   🔥 torch.* - All PyTorch operations")
    print("   🤗 transformers.* - All model operations")
    print("   🖼️  PIL.* - All image processing")
    print("   🌐 urllib3.*, requests.* - All network operations")
    print("   🗄️  chromadb.* - All database operations")
    print("   📊 streamlit.* - All UI operations")
    print("   ⚙️  multiprocessing.*, threading.* - All system operations")
    print("   📈 tqdm.* - All progress bars")
    print("   🔧 And EVERYTHING else - complete transparency!")
    print("")
    print("⚠️  WARNING: This will generate MASSIVE amounts of logs!")
    print("💡 Perfect for deep debugging and understanding every detail!")
    print("=" * 70)

if __name__ == "__main__":
    main() 