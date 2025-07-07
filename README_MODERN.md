# ArXiv Paper RAG Assistant - Modern UI Edition

A state-of-the-art RAG (Retrieval-Augmented Generation) system for academic research papers with a modern, Notion-like user interface. Upload PDFs, process them with advanced document understanding, and chat with your documents using local LLMs.

## 🎯 Features

- 🎨 **Modern Notion-like UI**: Clean, minimal interface with resizable panels
- 🔄 **Batch PDF Processing**: Upload and process multiple PDFs simultaneously
- 🧠 **Local LLM Integration**: Uses Ollama for private, offline AI processing
- 📄 **Advanced Document Understanding**: Powered by Marker for superior text extraction
- 🎯 **Precise Source Attribution**: Exact PDF and page number citations
- 💬 **Interactive Chat Interface**: Real-time responses with source tracking
- 📚 **Persistent Storage**: ChromaDB-based vector storage for efficient retrieval
- 🔍 **Semantic Search**: Intelligent context retrieval for accurate answers
- 📊 **System Analytics**: Monitor documents, chunks, and system performance

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Node.js 18 or higher
- 8GB+ RAM (16GB recommended for large document collections)
- Ollama installed and running

### Installation

1. **Clone or extract the repository**:
   ```bash
   git clone <repository-url>
   cd arxiv-paper-rag
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install and setup Ollama**:
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull llama3.2:latest
   ```

4. **Setup the frontend** (optional - launcher will do this):
   ```bash
   cd frontend
   npm install
   cd ..
   ```

5. **Launch the application**:
   ```bash
   python launch_modern_ui.py
   ```

6. **Access the interface**:
   - Frontend UI: `http://localhost:3000`
   - API Documentation: `http://localhost:8000/docs`

## 🏗️ Architecture

### Backend (FastAPI)
- RESTful API for all RAG operations
- Async document processing
- Real-time health monitoring
- Session management for temporary/permanent documents

### Frontend (Next.js)
- Modern React-based UI with TypeScript
- Tailwind CSS for sleek styling
- Resizable panels for flexible workspace
- Real-time updates with React Query

### Core Components
- **Document Processor**: Marker-based PDF extraction
- **Embedding Manager**: Sentence transformers for semantic search
- **Vector Store**: ChromaDB for efficient retrieval
- **LLM Client**: Ollama integration for local inference
- **Session Manager**: Handle temporary and permanent documents

## 📱 User Interface

### Three-Panel Layout

1. **Left Panel - Settings**
   - System initialization
   - LLM model selection
   - Chunking parameters
   - Retrieval settings

2. **Center Panel - Main Workspace**
   - **Upload Tab**: Drag-and-drop PDF upload
   - **Chat Tab**: Interactive Q&A with documents
   - **Analytics Tab**: System statistics and monitoring

3. **Right Panel - Document Management**
   - List of indexed documents
   - Search and filter
   - Storage type indicators
   - Quick actions

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the frontend directory:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### System Configuration
The system uses a comprehensive configuration system. Key settings:

- **LLM Model**: Default is `llama3.2:latest`
- **Chunk Size**: 1000 tokens (adjustable)
- **Retrieval Top-K**: 5 results (adjustable)
- **Storage Path**: `./data/` (configurable)

## 📚 API Documentation

### Key Endpoints

- `POST /api/sessions/start` - Start a new document session
- `POST /api/upload` - Upload and process PDFs
- `POST /api/chat` - Send chat queries
- `GET /api/documents` - List all documents
- `GET /api/analytics/overview` - Get system statistics
- `GET /api/health` - System health check

Full API documentation available at `http://localhost:8000/docs` when the backend is running.

## 🧪 Development

### Running in Development Mode

1. **Backend**:
   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```

2. **Frontend**:
   ```bash
   cd frontend
   npm run dev
   ```

### Adding New Features

1. Backend: Add endpoints in `src/api/main.py`
2. Frontend: Create components in `frontend/src/components/`
3. Update API types in `frontend/src/lib/api.ts`

## 🐛 Troubleshooting

### Common Issues

1. **"System not initialized"**
   - Click "Initialize System" in the Settings panel
   - Ensure all services are running

2. **"Ollama not connected"**
   - Verify Ollama is running: `ollama list`
   - Check the Ollama URL in settings

3. **Upload failures**
   - Ensure PDFs are not corrupted
   - Check file size limits
   - Verify storage permissions

### Debug Mode

Set environment variable for detailed logging:
```bash
export DEBUG=true
python launch_modern_ui.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Marker](https://github.com/VikParuchuri/marker) for PDF processing
- [Ollama](https://ollama.ai) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [Sentence Transformers](https://www.sbert.net/) for embeddings