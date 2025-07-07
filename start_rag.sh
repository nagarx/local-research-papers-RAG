#!/bin/bash
# ArXiv RAG Assistant Launcher
echo "🚀 Starting ArXiv RAG Assistant..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Start the application
python run.py
