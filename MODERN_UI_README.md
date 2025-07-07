# Modern UI for ArXiv RAG System

A sleek, Notion-like user interface for the ArXiv RAG system built with Next.js and FastAPI.

## Features

### 🎯 Key Features
- **Three-panel layout** with resizable panels
- **Left Panel**: System settings and configuration
- **Center Panel**: Document upload, chat interface, and analytics
- **Right Panel**: Document management system
- **Modern, minimal design** inspired by Notion
- **Real-time updates** and responsive interactions

### 🔧 Technical Stack
- **Frontend**: Next.js 14, React, TypeScript, Tailwind CSS
- **Backend**: FastAPI, Python
- **UI Components**: react-resizable-panels, lucide-react, recharts
- **State Management**: React Query

## Installation

### Prerequisites
- Python 3.8+
- Node.js 18+
- npm or yarn

### Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
```

## Running the Application

### Quick Start
Use the launcher script to start both backend and frontend:

```bash
python launch_modern_ui.py
```

This will:
1. Start the FastAPI backend on http://localhost:8000
2. Start the Next.js frontend on http://localhost:3000
3. Open your browser automatically

### Manual Start

#### Backend
```bash
uvicorn src.api.main:app --reload --port 8000
```

#### Frontend
```bash
cd frontend
npm run dev
```

## UI Components

### Settings Panel (Left)
- System initialization controls
- LLM model selection and configuration
- Chunking parameters
- Retrieval settings

### Main Panel (Center)
- **Upload Tab**: Drag-and-drop PDF upload with storage type selection
- **Chat Tab**: Interactive chat interface with source citations
- **Analytics Tab**: System statistics and visualizations

### Document Panel (Right)
- List of all indexed documents
- Search and filter functionality
- Document metadata display
- Quick delete actions

## API Endpoints

The FastAPI backend provides the following endpoints:

- `GET /api/health` - System health check
- `POST /api/sessions/start` - Start a new session
- `GET /api/documents` - List all documents
- `POST /api/upload` - Upload PDF files
- `POST /api/chat` - Send chat messages
- `GET /api/analytics/overview` - Get system analytics
- `GET /api/config` - Get system configuration

Full API documentation available at http://localhost:8000/docs

## Configuration

### Environment Variables
Create a `.env` file in the frontend directory:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Customization
- Tailwind configuration: `frontend/tailwind.config.ts`
- Global styles: `frontend/src/app/globals.css`
- API client: `frontend/src/lib/api.ts`

## Development

### Adding New Features
1. Backend: Add endpoints in `src/api/main.py`
2. Frontend: Create components in `frontend/src/components/`
3. Update API types in `frontend/src/lib/api.ts`

### Styling
The UI uses Tailwind CSS with custom design tokens for a consistent look:
- Primary colors defined in CSS variables
- Notion-like spacing and typography
- Smooth transitions and animations

## Troubleshooting

### Backend Issues
- Ensure Python dependencies are installed
- Check if port 8000 is available
- Verify RAG system components are properly initialized

### Frontend Issues
- Clear Next.js cache: `rm -rf frontend/.next`
- Reinstall dependencies: `cd frontend && rm -rf node_modules && npm install`
- Check if port 3000 is available

### Connection Issues
- Verify CORS settings in FastAPI
- Check API URL in frontend configuration
- Ensure both servers are running

## Future Enhancements
- Dark mode support
- WebSocket for real-time updates
- Advanced document preview
- Collaborative features
- Export functionality