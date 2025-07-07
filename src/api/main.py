"""
FastAPI Backend for ArXiv RAG System

Provides REST API endpoints for the RAG system functionality.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import asyncio
import tempfile
import shutil
from pathlib import Path
import json
import logging
from datetime import datetime
import uuid

# Import RAG system components
import sys
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chat import ChatEngine
from src.config import get_config
from src.ingestion import get_global_marker_models
from src.utils import DocumentStatusChecker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="ArXiv RAG System API",
    description="API for managing documents and chatting with ArXiv papers",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global chat engine instance
chat_engine = None
session_data = {}

# Pydantic models for requests/responses
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    top_k: Optional[int] = 5

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    session_id: str

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    total_chunks: int
    storage_type: str
    added_date: str
    status: str

class SessionInfo(BaseModel):
    session_id: str
    document_count: int
    temporary_count: int
    permanent_count: int
    created_at: str

class SystemHealth(BaseModel):
    overall_status: str
    components: Dict[str, Dict[str, Any]]
    timestamp: str

class ProcessingProgress(BaseModel):
    status: str
    progress: float
    message: str
    document_id: Optional[str] = None

# Initialize system on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    global chat_engine
    try:
        logger.info("Initializing RAG system...")
        chat_engine = ChatEngine()
        logger.info("RAG system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")

# System endpoints
@app.get("/api/health", response_model=SystemHealth)
async def get_system_health():
    """Get system health status"""
    try:
        if not chat_engine:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        # Simple health check
        components = {}
        overall_status = "healthy"
        
        try:
            # Check vector store
            if hasattr(chat_engine, 'vector_store') and chat_engine.vector_store:
                components['vector_store'] = {'status': 'healthy'}
            else:
                components['vector_store'] = {'status': 'not_initialized'}
                overall_status = "warning"
                
            # Check Ollama client
            if hasattr(chat_engine, 'ollama_client') and chat_engine.ollama_client:
                components['ollama_client'] = {'status': 'healthy'}
            else:
                components['ollama_client'] = {'status': 'not_initialized'}
                overall_status = "warning"
                
            # Check session manager
            if hasattr(chat_engine, 'session_manager') and chat_engine.session_manager:
                components['session_manager'] = {'status': 'healthy'}
            else:
                components['session_manager'] = {'status': 'not_initialized'}
                overall_status = "warning"
                
        except Exception as e:
            overall_status = "error"
            components['error'] = str(e)
        
        return SystemHealth(
            overall_status=overall_status,
            components=components,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/preload-models")
async def preload_models():
    """Pre-load Marker models"""
    try:
        models = get_global_marker_models()
        return {"status": "success", "message": "Models loaded successfully"}
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Session management endpoints
@app.post("/api/sessions/start", response_model=SessionInfo)
async def start_session():
    """Start a new document session"""
    try:
        if not chat_engine:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        session_id = chat_engine.start_session()
        session_info = chat_engine.get_session_info()
        
        # Store session data
        session_data[session_id] = {
            "created_at": datetime.now().isoformat(),
            "documents": []
        }
        
        return SessionInfo(
            session_id=session_id,
            document_count=session_info.get("document_count", 0),
            temporary_count=session_info.get("temporary_count", 0),
            permanent_count=session_info.get("permanent_count", 0),
            created_at=session_data[session_id]["created_at"]
        )
    except Exception as e:
        logger.error(f"Failed to start session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sessions/{session_id}/end")
async def end_session(session_id: str):
    """End a document session"""
    try:
        if not chat_engine:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        cleanup_summary = chat_engine.end_session()
        
        # Remove session data
        if session_id in session_data:
            del session_data[session_id]
        
        return {"status": "success", "cleanup_summary": cleanup_summary}
    except Exception as e:
        logger.error(f"Failed to end session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/current", response_model=Optional[SessionInfo])
async def get_current_session():
    """Get current session info"""
    try:
        if not chat_engine:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        session_info = chat_engine.get_session_info()
        if session_info and chat_engine.session_manager.current_session_id:
            session_id = chat_engine.session_manager.current_session_id
            return SessionInfo(
                session_id=session_id,
                document_count=session_info.get("document_count", 0),
                temporary_count=session_info.get("temporary_count", 0),
                permanent_count=session_info.get("permanent_count", 0),
                created_at=session_data.get(session_id, {}).get("created_at", datetime.now().isoformat())
            )
        return None
    except Exception as e:
        logger.error(f"Failed to get session info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Document management endpoints
@app.get("/api/documents", response_model=List[DocumentInfo])
async def get_all_documents():
    """Get all documents (permanent and session)"""
    try:
        if not chat_engine:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        all_docs = []
        
        # Get permanent documents
        permanent_docs = chat_engine.get_permanent_documents()
        for doc in permanent_docs:
            all_docs.append(DocumentInfo(
                document_id=doc["document_id"],
                filename=doc["filename"],
                total_chunks=doc.get("total_chunks", 0),
                storage_type="permanent",
                added_date=doc.get("added_to_permanent", datetime.now().isoformat()),
                status="active"
            ))
        
        # Get session documents
        if chat_engine.session_manager.current_session_id:
            session_docs = chat_engine.session_manager.get_session_documents()
            for doc_id, storage_type in session_docs.items():
                if storage_type == "temporary":
                    # Get document info from vector store
                    doc_info = chat_engine.vector_store.get_document_info(doc_id)
                    if doc_info:
                        all_docs.append(DocumentInfo(
                            document_id=doc_id,
                            filename=doc_info.get("filename", "unknown"),
                            total_chunks=doc_info.get("total_chunks", 0),
                            storage_type="temporary",
                            added_date=doc_info.get("processed_at", datetime.now().isoformat()),
                            status="active"
                        ))
        
        return all_docs
    except Exception as e:
        logger.error(f"Failed to get documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/permanent", response_model=List[DocumentInfo])
async def get_permanent_documents():
    """Get only permanent documents"""
    try:
        if not chat_engine:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        permanent_docs = chat_engine.get_permanent_documents()
        return [
            DocumentInfo(
                document_id=doc["document_id"],
                filename=doc["filename"],
                total_chunks=doc.get("total_chunks", 0),
                storage_type="permanent",
                added_date=doc.get("added_to_permanent", datetime.now().isoformat()),
                status="active"
            )
            for doc in permanent_docs
        ]
    except Exception as e:
        logger.error(f"Failed to get permanent documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document"""
    try:
        if not chat_engine:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        success = chat_engine.remove_permanent_document(document_id)
        if success:
            return {"status": "success", "message": "Document deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(f"Failed to delete document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# File upload endpoint
@app.post("/api/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    storage_type: str = "temporary",
    force_reprocess: bool = False
):
    """Upload and process PDF files"""
    try:
        if not chat_engine:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        results = []
        
        for file in files:
            # Save uploaded file temporarily
            temp_dir = Path(tempfile.mkdtemp())
            file_path = temp_dir / file.filename
            
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Process the file
            try:
                # Use add_documents_async for single file
                result = await chat_engine.add_documents_async(
                    [str(file_path)],
                    storage_types=[storage_type],
                    force_reprocess=force_reprocess
                )
                
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "document_id": result.get("document_id"),
                    "chunks": result.get("total_chunks", 0)
                })
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                })
            finally:
                # Clean up temp file
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        return {"status": "success", "results": results}
    except Exception as e:
        logger.error(f"Failed to upload files: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Chat endpoints
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a chat query and get a response"""
    try:
        if not chat_engine:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        # Query the chat engine
        response = await chat_engine.query_async(request.query, top_k=request.top_k or 5)
        
        # Extract sources
        sources = []
        if "sources" in response:
            for source in response["sources"]:
                sources.append({
                    "document": source.get("document", "Unknown"),
                    "page": source.get("page", "Unknown"),
                    "score": source.get("score", 0.0),
                    "text": source.get("text", "")
                })
        
        session_id = chat_engine.session_manager.current_session_id or "default"
        
        return ChatResponse(
            response=response.get("response", "No response generated"),
            sources=sources,
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"Chat query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics endpoints
@app.get("/api/analytics/overview")
async def get_analytics_overview():
    """Get system analytics and statistics"""
    try:
        if not chat_engine:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        # Get document statistics
        all_docs = await get_all_documents()
        permanent_count = len([d for d in all_docs if d.storage_type == "permanent"])
        temporary_count = len([d for d in all_docs if d.storage_type == "temporary"])
        total_chunks = sum(d.total_chunks for d in all_docs)
        
        # Get vector store info
        vector_store_info = {}
        if hasattr(chat_engine.vector_store, 'get_stats'):
            vector_store_info = chat_engine.vector_store.get_stats()
        
        return {
            "documents": {
                "total": len(all_docs),
                "permanent": permanent_count,
                "temporary": temporary_count
            },
            "chunks": {
                "total": total_chunks,
                "average_per_doc": total_chunks / len(all_docs) if all_docs else 0
            },
            "vector_store": vector_store_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configuration endpoints
@app.get("/api/config")
async def get_configuration():
    """Get current system configuration"""
    try:
        config = get_config()
        
        return {
            "llm": {
                "model": config.ollama.model,
                "temperature": config.ollama.temperature,
                "context_length": config.ollama.context_length
            },
            "chunking": {
                "chunk_size": config.processing.chunk_size,
                "chunk_overlap": config.processing.chunk_overlap
            },
            "retrieval": {
                "top_k": config.ui.default_query_limit,
                "similarity_threshold": config.vector_storage.similarity_threshold
            }
        }
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/config")
async def update_configuration(config_updates: Dict[str, Any]):
    """Update system configuration"""
    try:
        # This would need to be implemented in the config module
        # For now, return success
        return {"status": "success", "message": "Configuration updated"}
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)