# ArXiv Paper RAG Assistant

## Technical Overview

**RAG Pipeline Architecture**: Multi-stage document processing system implementing structure-aware PDF extraction → semantic chunking → dense retrieval → contextualized generation for academic corpora.

**Document Processing**: Marker-based extraction preserving LaTeX notation, citations, and hierarchical structure. Implements sliding window chunking with configurable overlap to maintain semantic coherence across section boundaries.

**Embedding Framework**: BAAI/bge-large-en-v1.5 model generates dense vector representations (1024d) optimized for academic text similarity with 512-token context windows. ChromaDB vector store implements HNSW indexing with cosine similarity for sub-linear retrieval complexity.

**Generation**: Local LLaMA 3.2 (7B parameters) via Ollama for privacy-preserving inference. Retrieval-augmented prompting with top-k similar chunks and structured citation metadata for source attribution.

## Quick Start

### Prerequisites

- **Python**: 3.10 or higher with pip
- **Memory**: 8GB+ RAM (4GB minimum, recommended for large document collections)
- **Storage**: 5GB+ free space (for models and processed documents)
- **Ollama**: Local LLM service (installed automatically by installer)
- **System Dependencies**: Build tools and libraries (installed automatically)

### Installation

1. **Extract the package**:
   ```bash
   unzip arxiv-paper-rag-v1.0.zip
   cd arxiv-paper-rag
   ```

2. **Run the automated installer**:
   ```bash
   # Windows
   install.bat
   
   # macOS/Linux
   chmod +x install.sh && ./install.sh
   ```
   
   The installer will:
   - Check and install Python 3.10+ if needed
   - Install Ollama automatically
   - Create a virtual environment
   - Install all Python dependencies
   - Download the required AI model
   - Set up directories and configuration
   - Run verification tests

3. **Verify installation** (recommended):
   ```bash
   # Quick verification
   python verify_installation.py
   
   # Comprehensive test (recommended for job interview)
   python test_installation.py
   ```

4. **Launch the application**:
   ```bash
   # Windows - Use the launcher
   start_rag.bat
   
   # macOS/Linux - Use the launcher
   ./start_rag.sh
   
   # Or manual launch with system checks
   python run.py
   ```

5. **Access the interface**:
   Open your browser to `http://localhost:8501`

## Manual Installation

If the automated installer doesn't work:

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate     # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and setup Ollama**:
   ```bash
   # Install Ollama (see https://ollama.ai)
   ollama pull llama3.2:latest
   ```

4. **Configure environment**:
   ```bash
   cp env_example.txt .env
   # Edit .env with your preferred settings
   ```

## Usage

1. **Upload Documents**: Use the "Upload PDFs" page to add your research papers
2. **Wait for Processing**: The system will process and index your documents
3. **Start Chatting**: Use the "Chat" page to ask questions about your documents
4. **View Sources**: Each answer includes precise citations with PDF names and page numbers

## Vector Storage

The system uses **ChromaDB** for efficient vector storage and retrieval:
- **Persistent storage**: Documents are stored locally in ChromaDB collections
- **Similarity search**: Cosine similarity for accurate document retrieval
- **Metadata filtering**: Filter searches by document, page number, or other criteria
- **Scalable**: Handles thousands of documents efficiently

## Troubleshooting

### Quick Fixes

**Installation Issues:**
- Run `python verify_installation.py` to diagnose problems
- Ensure Python 3.10+ is installed and in PATH
- Make sure Ollama is installed from https://ollama.ai

**Common Problems:**
- **Port 8501 in use**: Use `python run.py --port 8502` or kill existing process
- **Memory errors**: Reduce batch size in configuration or close other applications
- **Permission errors**: Run installer as administrator (Windows) or fix file permissions
- **Slow processing**: Ensure you have sufficient RAM and use SSD if possible
- **Python not found**: Install Python 3.10+ from python.org and ensure it's in PATH
- **Ollama issues**: Install from https://ollama.ai and run `ollama serve` to start service
- **Package install fails**: Upgrade pip with `python -m pip install --upgrade pip`

**Getting Help:**
- Run `python verify_installation.py` for comprehensive diagnostics
- Check logs in `data/logs/` directory
- Review the troubleshooting section above for common solutions

## License

MIT License - see LICENSE file for details.

## Contributing

This is a research project. For issues or suggestions, please refer to the documentation. 