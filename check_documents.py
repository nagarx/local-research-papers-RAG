#!/usr/bin/env python3
"""
Quick script to check the status of processed and indexed documents
"""

import subprocess
import sys

if __name__ == "__main__":
    print("🔍 Checking document status...")
    print()
    
    try:
        # Run the document status module
        result = subprocess.run([
            sys.executable, "-m", "src.utils.document_status"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            # Filter out warnings and only show the important output
            lines = result.stdout.split('\n')
            filtered_lines = []
            
            for line in lines:
                # Skip PyTorch warnings and other noise
                if any(skip in line for skip in [
                    "PyTorch warnings suppressed",
                    "PyTorch configured",
                    "RuntimeWarning",
                    "DeprecationWarning",
                    "INFO - Starting ArXiv",
                    "INFO - Environment",
                    "INFO - Debug mode"
                ]):
                    continue
                filtered_lines.append(line)
            
            print('\n'.join(filtered_lines))
        else:
            print(f"❌ Error running document status check:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you're running this from the project root directory.") 