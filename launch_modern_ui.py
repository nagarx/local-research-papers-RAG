#!/usr/bin/env python3
"""
Launch script for the modern ArXiv RAG UI
Starts both the FastAPI backend and Next.js frontend
"""

import subprocess
import sys
import os
import time
import signal
import webbrowser
from pathlib import Path

class UILauncher:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        
    def start_backend(self):
        """Start the FastAPI backend server"""
        print("🚀 Starting FastAPI backend...")
        
        # Check if uvicorn is installed
        try:
            import uvicorn
        except ImportError:
            print("❌ uvicorn not installed. Installing dependencies...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        # Start the backend
        self.backend_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn",
            "src.api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
        
        # Wait for backend to start
        print("⏳ Waiting for backend to start...")
        time.sleep(3)
        
        # Check if backend is running
        import requests
        max_retries = 10
        for i in range(max_retries):
            try:
                response = requests.get("http://localhost:8000/api/health")
                if response.status_code == 200:
                    print("✅ Backend is running!")
                    break
            except:
                time.sleep(1)
                if i == max_retries - 1:
                    print("❌ Backend failed to start")
                    return False
        
        return True
    
    def start_frontend(self):
        """Start the Next.js frontend"""
        print("\n🎨 Starting Next.js frontend...")
        
        frontend_dir = Path("frontend")
        if not frontend_dir.exists():
            print("❌ Frontend directory not found")
            return False
        
        # Check if node_modules exists
        if not (frontend_dir / "node_modules").exists():
            print("📦 Installing frontend dependencies...")
            subprocess.run(["npm", "install"], cwd=frontend_dir)
        
        # Start the frontend
        self.frontend_process = subprocess.Popen([
            "npm", "run", "dev"
        ], cwd=frontend_dir)
        
        print("✅ Frontend is starting...")
        print("\n" + "="*50)
        print("🌟 Modern RAG UI is running!")
        print("="*50)
        print("\n📍 Access the application at: http://localhost:3000")
        print("📍 API documentation at: http://localhost:8000/docs")
        print("\n⚠️  Press Ctrl+C to stop both servers\n")
        
        # Open browser after a delay
        time.sleep(3)
        webbrowser.open("http://localhost:3000")
        
        return True
    
    def stop(self):
        """Stop both servers"""
        print("\n🛑 Stopping servers...")
        
        if self.backend_process:
            self.backend_process.terminate()
            print("✅ Backend stopped")
            
        if self.frontend_process:
            self.frontend_process.terminate()
            print("✅ Frontend stopped")
    
    def run(self):
        """Run both servers"""
        try:
            # Start backend
            if not self.start_backend():
                print("❌ Failed to start backend")
                return
            
            # Start frontend
            if not self.start_frontend():
                print("❌ Failed to start frontend")
                self.stop()
                return
            
            # Wait for processes
            if self.frontend_process:
                self.frontend_process.wait()
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Received interrupt signal")
        finally:
            self.stop()
            print("\n👋 Goodbye!")

def main():
    """Main entry point"""
    print("="*50)
    print("🚀 ArXiv RAG Modern UI Launcher")
    print("="*50)
    
    launcher = UILauncher()
    launcher.run()

if __name__ == "__main__":
    main()