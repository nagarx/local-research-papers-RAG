# ArXiv Paper RAG Assistant

An intelligent RAG (Retrieval-Augmented Generation) system designed specifically for academic research papers. Upload multiple PDFs, process them with state-of-the-art document understanding, and chat with your documents using local LLMs.

## Features

- 🔄 **Batch PDF Processing**: Upload and process hundreds of PDFs simultaneously
- 🧠 **Local LLM Integration**: Uses Ollama for private, offline AI processing
- 📄 **Advanced Document Understanding**: Powered by Marker for superior text extraction
- 🎯 **Precise Source Attribution**: Exact PDF and page number citations
- 💬 **Interactive Chat Interface**: Streamlit-based user-friendly interface
- 📚 **Persistent Storage**: ChromaDB-based vector storage for efficient retrieval
- 🔍 **Semantic Search**: Intelligent context retrieval for accurate answers

## Quick Start

### Prerequisites

- Python 3.10 or higher
- 8GB+ RAM (recommended for large document collections)
- Ollama installed and running

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

3. **Launch the application**:
   ```bash
   # Full launcher with system checks and setup
   python run.py
   
   # Simple launcher (if dependencies already installed)
   python app.py
   ```

4. **Access the interface**:
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
   ollama pull llama3.1:8b
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

## Sample Papers Included

The package includes 10+ carefully selected ArXiv research papers covering:
- Neural Networks and Deep Learning
- Transformer Architectures
- Computer Vision Advances
- Natural Language Processing
- Machine Learning Theory
- AI Ethics and Explainability

## System Requirements

- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 2GB free space for models and processing
- **Network**: Required for initial Ollama model download
- **OS**: Windows 10+, macOS 10.15+, or Linux

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md) for common issues and solutions.

## License

MIT License - see LICENSE file for details.

## Contributing

This is a research project. For issues or suggestions, please refer to the documentation. 