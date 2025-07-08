#!/bin/bash
# ArXiv Paper RAG Assistant Launcher

cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Check if Ollama is running
if ! pgrep -f "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 5
fi

# Launch the application
python run.py
