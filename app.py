#!/usr/bin/env python3
"""
ArXiv RAG System - Streamlit App Launcher

Simple launcher script for the Streamlit web interface.
Run this script to start the web application.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit application"""
    
    # Get the path to the Streamlit app
    app_path = Path(__file__).parent / "src" / "ui" / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"❌ Streamlit app not found at: {app_path}")
        sys.exit(1)
    
    print("🚀 Starting ArXiv RAG Assistant...")
    print("📚 Web interface will open in your browser")
    print("🔗 URL: http://localhost:8501")
    print("\n" + "="*50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.address", "localhost",
            "--server.port", "8501",
            "--theme.base", "light",
            "--theme.primaryColor", "#1e3c72"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 