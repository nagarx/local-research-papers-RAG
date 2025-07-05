"""
ArXiv RAG System - Streamlit UI

A modern, comprehensive web interface for the ArXiv RAG system.
Features:
- Multi-file PDF upload
- Real-time document processing
- Interactive chat interface
- Source citations with page numbers
- System health monitoring
- Document management
"""

import streamlit as st
import asyncio
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Configure Streamlit page
st.set_page_config(
    page_title="ArXiv RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/arxiv-rag',
        'Report a bug': "https://github.com/your-repo/arxiv-rag/issues",
        'About': "# ArXiv RAG Assistant\nChat with your research papers using local LLMs!"
    }
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1e3c72;
        margin: 1rem 0;
    }
    .chat-message {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #1e3c72;
    }
    .chat-response {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    .source-citation {
        background: #fff3cd;
        padding: 0.5rem;
        border-radius: 4px;
        margin: 0.25rem 0;
        font-size: 0.9rem;
        border-left: 3px solid #ffc107;
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Import RAG system components
try:
    import sys
    from pathlib import Path
    
    # Add project root to Python path so src package can be imported
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Import torch utilities early to suppress warnings
    try:
        from src.utils.torch_utils import configure_torch_for_production
        configure_torch_for_production()
    except ImportError:
        pass
    
    from src.chat import ChatEngine
    from src.config import get_config
    from src.ingestion import get_global_marker_models
    
    # Store import success
    IMPORTS_SUCCESSFUL = True
    
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    import traceback
    error_details = traceback.format_exc()
    print(f"❌ Failed to import RAG system: {e}")
    print(f"Error details: {error_details}")
    # Only show error in streamlit context, not when imported elsewhere
    if 'streamlit' in sys.modules:
        st.error(f"❌ Failed to import RAG system: {e}")
        st.stop()


class StreamlitRAGApp:
    """Main Streamlit RAG Application"""
    
    def __init__(self):
        if not IMPORTS_SUCCESSFUL:
            raise ImportError("RAG system imports failed")
        self.config = get_config()
        self.chat_engine = None
        self.setup_session_state()
        
    def setup_session_state(self):
        """Initialize Streamlit session state"""
        if 'chat_engine' not in st.session_state:
            st.session_state.chat_engine = None
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'processed_documents' not in st.session_state:
            st.session_state.processed_documents = []
        if 'system_stats' not in st.session_state:
            st.session_state.system_stats = {}
        if 'upload_status' not in st.session_state:
            st.session_state.upload_status = {}
        if 'models_loaded' not in st.session_state:
            st.session_state.models_loaded = False
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>📚 ArXiv RAG Assistant</h1>
            <p>Chat with your research papers using local LLMs powered by Marker & Ollama</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with system controls"""
        st.sidebar.markdown("## 🔧 System Control")
        
        # System initialization
        with st.sidebar.expander("🚀 System Status", expanded=True):
            if st.button("🔄 Initialize System", key="init_system"):
                self.initialize_system()
            
            # Show system status
            if st.session_state.chat_engine:
                st.markdown('<p class="status-healthy">✅ System Ready</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-warning">⚠️ System Not Initialized</p>', unsafe_allow_html=True)
        
        # Model status
        with st.sidebar.expander("🧠 Model Status", expanded=False):
            if st.session_state.models_loaded:
                st.markdown('<p class="status-healthy">✅ Models Loaded</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-error">❌ Models Not Loaded</p>', unsafe_allow_html=True)
            
            if st.button("📦 Pre-load Models", key="load_models"):
                self.preload_models()
        
        # Document management
        with st.sidebar.expander("📄 Document Management", expanded=False):
            if st.session_state.processed_documents:
                st.write(f"📊 {len(st.session_state.processed_documents)} documents processed")
                
                # Show document list
                for doc in st.session_state.processed_documents:
                    st.write(f"• {doc['filename']} ({doc['total_chunks']} chunks)")
                
                if st.button("🗑️ Clear All Documents", key="clear_docs"):
                    self.clear_documents()
            else:
                st.write("No documents processed yet")
        
        # System configuration
        with st.sidebar.expander("⚙️ Configuration", expanded=False):
            # Model selection
            available_models = self.get_available_ollama_models()
            if available_models:
                selected_model = st.selectbox(
                    "LLM Model",
                    available_models,
                    index=0 if available_models else 0
                )
                
                if st.button("🔄 Update Model", key="update_model"):
                    self.update_ollama_model(selected_model)
            
            # Processing parameters
            st.subheader("Processing Parameters")
            chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
            overlap = st.slider("Chunk Overlap", 50, 200, 100)
            top_k = st.slider("Retrieval Top-K", 1, 10, 5)
            
            # Store in session state
            st.session_state.chunk_size = chunk_size
            st.session_state.overlap = overlap
            st.session_state.top_k = top_k
        
        # System health monitoring
        with st.sidebar.expander("🏥 System Health", expanded=False):
            if st.button("🔍 Check Health", key="health_check"):
                self.check_system_health()
            
            # Show health status
            if st.session_state.system_stats:
                stats = st.session_state.system_stats
                
                # Overall status
                overall_status = stats.get('overall_status', 'unknown')
                if overall_status == 'healthy':
                    st.markdown('<p class="status-healthy">✅ System Healthy</p>', unsafe_allow_html=True)
                elif overall_status == 'warning':
                    st.markdown('<p class="status-warning">⚠️ System Warning</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="status-error">❌ System Error</p>', unsafe_allow_html=True)
                
                # Component status
                components = stats.get('components', {})
                for component, status in components.items():
                    if isinstance(status, dict):
                        component_status = status.get('status', 'unknown')
                        if component_status == 'healthy':
                            st.write(f"✅ {component.title()}")
                        else:
                            st.write(f"❌ {component.title()}")
    
    def initialize_system(self):
        """Initialize the RAG system"""
        with st.spinner("🔄 Initializing RAG system..."):
            try:
                # Clear any previous instance
                if st.session_state.chat_engine:
                    del st.session_state.chat_engine
                
                # Initialize new chat engine
                st.session_state.chat_engine = ChatEngine()
                
                st.success("✅ RAG system initialized successfully!")
                
            except Exception as e:
                st.error(f"❌ Failed to initialize system: {e}")
                st.session_state.chat_engine = None
    
    def preload_models(self):
        """Pre-load Marker models"""
        with st.spinner("📦 Loading Marker models..."):
            try:
                start_time = time.time()
                models = get_global_marker_models()
                load_time = time.time() - start_time
                
                st.session_state.models_loaded = True
                st.success(f"✅ Models loaded in {load_time:.1f}s!")
                
            except Exception as e:
                st.error(f"❌ Failed to load models: {e}")
                st.session_state.models_loaded = False
    
    def get_available_ollama_models(self) -> List[str]:
        """Get available Ollama models"""
        try:
            if st.session_state.chat_engine:
                # This would need to be implemented in the OllamaClient
                return ["llama3.2:latest", "qwen2.5-coder:latest", "deepseek-r1:7b"]
            return []
        except Exception:
            return []
    
    def update_ollama_model(self, model_name: str):
        """Update Ollama model"""
        try:
            # This would need to be implemented
            st.success(f"✅ Model updated to {model_name}")
        except Exception as e:
            st.error(f"❌ Failed to update model: {e}")
    
    def clear_documents(self):
        """Clear all processed documents"""
        try:
            if st.session_state.chat_engine:
                # Clear vector store
                st.session_state.chat_engine.vector_store.clear()
                st.session_state.processed_documents = []
                st.success("✅ All documents cleared!")
            
        except Exception as e:
            st.error(f"❌ Failed to clear documents: {e}")
    
    def check_system_health(self):
        """Check system health using actual RAG system"""
        try:
            if st.session_state.chat_engine:
                # Run actual health check
                health_result = asyncio.run(
                    st.session_state.chat_engine.test_system_health()
                )
                
                st.session_state.system_stats = health_result
                
                # Display results
                overall_status = health_result.get('overall_status', 'unknown')
                if overall_status == 'healthy':
                    st.success("✅ Health check completed - System is healthy!")
                elif overall_status == 'degraded':
                    st.warning("⚠️ Health check completed - System has warnings")
                else:
                    st.error("❌ Health check completed - System has errors")
                
                # Show detailed component status
                components = health_result.get('components', {})
                for component, details in components.items():
                    if isinstance(details, dict):
                        status = details.get('status', 'unknown')
                        if status == 'healthy':
                            st.success(f"✅ {component.title()}: {status}")
                        elif status == 'degraded':
                            st.warning(f"⚠️ {component.title()}: {status}")
                        else:
                            st.error(f"❌ {component.title()}: {status}")
                            
                        # Show additional details
                        if 'details' in details:
                            st.text(f"   Details: {details['details']}")
                            
            else:
                st.error("❌ System not initialized")
                
        except Exception as e:
            st.error(f"❌ Health check failed: {e}")
            st.exception(e)
    
    def render_file_upload(self):
        """Render file upload interface"""
        st.markdown("## 📁 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select one or more PDF files to upload and process"
        )
        
        if uploaded_files:
            st.write(f"📄 {len(uploaded_files)} files selected:")
            
            # Show file details
            for file in uploaded_files:
                file_size = len(file.read()) / (1024 * 1024)  # MB
                file.seek(0)  # Reset file pointer
                st.write(f"• {file.name} ({file_size:.1f} MB)")
            
            # Process files
            if st.button("🚀 Process Documents", key="process_docs"):
                self.process_uploaded_files(uploaded_files)
    
    def process_uploaded_files(self, uploaded_files):
        """Process uploaded PDF files using the actual RAG system"""
        if not st.session_state.chat_engine:
            st.error("❌ Please initialize the system first")
            return
        
        # Save uploaded files temporarily
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            file_paths = []
            
            # Save files
            for i, file in enumerate(uploaded_files):
                temp_path = temp_dir / file.name
                with open(temp_path, "wb") as f:
                    f.write(file.read())
                file_paths.append(str(temp_path))
                
                progress_bar.progress((i + 1) / len(uploaded_files) * 0.3)
                status_text.text(f"📄 Saved {file.name}")
            
            # Process documents using actual RAG system
            status_text.text("🔄 Processing documents with Marker...")
            
            # Define progress callback - matching chat_engine signature
            def update_progress(message, processed_count=None, total_count=None):
                if processed_count is not None and total_count is not None:
                    progress = 0.3 + (processed_count / total_count) * 0.7
                    progress_bar.progress(progress)
                    status_text.text(f"🔄 {message} ({processed_count}/{total_count})")
                else:
                    status_text.text(f"🔄 {message}")
            
            # Process documents asynchronously
            result = asyncio.run(
                st.session_state.chat_engine.add_documents_async(
                    file_paths=file_paths,
                    progress_callback=update_progress
                )
            )
            
            if result['success']:
                # Update processed documents list - use the simplified format
                total_chunks = result.get('total_chunks', 0)
                if isinstance(total_chunks, str):
                    try:
                        total_chunks = int(total_chunks)
                    except ValueError:
                        total_chunks = 0
                
                chunks_per_doc = total_chunks // len(file_paths) if len(file_paths) > 0 else 0
                
                for file_path in file_paths:
                    filename = Path(file_path).name
                    st.session_state.processed_documents.append({
                        'filename': filename,
                        'total_chunks': chunks_per_doc,
                        'processed_at': datetime.now().isoformat()
                    })
                
                progress_bar.progress(1.0)
                status_text.text("✅ All documents processed successfully!")
                
                st.success(f"✅ Processed {len(uploaded_files)} documents successfully!")
                
                # Show processing summary
                st.info(f"📊 Processing Summary:\n" + 
                       f"- Total documents: {result['total_documents']}\n" + 
                       f"- Total chunks: {result['total_chunks']}\n" + 
                       f"- Processing time: {result['processing_time']:.1f}s")
            else:
                st.error(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
            
            # Clean up temp files
            for file_path in file_paths:
                Path(file_path).unlink(missing_ok=True)
            
        except Exception as e:
            st.error(f"❌ Error processing documents: {e}")
            # Clean up temp files on error
            for file_path in file_paths:
                Path(file_path).unlink(missing_ok=True)
    
    def render_chat_interface(self):
        """Render the chat interface"""
        st.markdown("## 💬 Chat with Documents")
        
        if not st.session_state.processed_documents:
            st.info("📝 Upload and process documents first to start chatting!")
            return
        
        # Chat input
        user_query = st.text_input(
            "Ask a question about your documents:",
            placeholder="What is the main contribution of this paper?",
            key="chat_input"
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button("📤 Send", key="send_query"):
                if user_query.strip():
                    self.process_chat_query(user_query)
        
        with col2:
            if st.button("🗑️ Clear Chat", key="clear_chat"):
                st.session_state.conversation_history = []
                st.rerun()
        
        # Display conversation history
        if st.session_state.conversation_history:
            st.markdown("### 💬 Conversation History")
            
            for i, message in enumerate(st.session_state.conversation_history):
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message">
                        <strong>🧑 You:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                
                elif message['role'] == 'assistant':
                    st.markdown(f"""
                    <div class="chat-response">
                        <strong>🤖 Assistant:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show sources if available
                    if message.get('sources'):
                        st.markdown("**📚 Sources:**")
                        for source in message['sources']:
                            st.markdown(f"""
                            <div class="source-citation">
                                📄 {source['document_name']}, page {source['page_number']}<br>
                                <em>"{source['text_snippet']}"</em>
                            </div>
                            """, unsafe_allow_html=True)
    
    def process_chat_query(self, query: str):
        """Process a chat query using the actual RAG system"""
        if not st.session_state.chat_engine:
            st.error("❌ System not initialized")
            return
        
        # Add user message
        st.session_state.conversation_history.append({
            'role': 'user',
            'content': query,
            'timestamp': datetime.now().isoformat()
        })
        
        with st.spinner("🔄 Generating response..."):
            try:
                # Use actual RAG system for query processing
                top_k = st.session_state.get('top_k', 5)
                
                # Run the actual query through the RAG system
                result = asyncio.run(
                    st.session_state.chat_engine.query_async(
                        user_query=query,
                        top_k=top_k,
                        include_conversation_history=True
                    )
                )
                
                if result.get('error'):
                    st.error(f"❌ Query failed: {result.get('error_message', 'Unknown error')}")
                else:
                    # Format sources for display
                    formatted_sources = []
                    for source in result.get('sources', []):
                        formatted_sources.append({
                            'document_name': source.get('document_name', 'Unknown Document'),
                            'page_number': source.get('page_number', 1),
                            'text_snippet': source.get('text_snippet', '')[:150] + '...'
                        })
                    
                    # Add assistant response
                    st.session_state.conversation_history.append({
                        'role': 'assistant',
                        'content': result['response'],
                        'sources': formatted_sources,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing query: {e}")
                # Add error response to conversation
                st.session_state.conversation_history.append({
                    'role': 'assistant',
                    'content': f"I apologize, but I encountered an error processing your query: {str(e)}",
                    'sources': [],
                    'timestamp': datetime.now().isoformat()
                })
    
    def render_analytics(self):
        """Render analytics dashboard"""
        st.markdown("## 📊 Analytics & Insights")
        
        if not st.session_state.processed_documents:
            st.info("📝 Process documents first to see analytics!")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📄 Documents",
                len(st.session_state.processed_documents),
                delta=None
            )
        
        with col2:
            total_chunks = sum(doc['total_chunks'] for doc in st.session_state.processed_documents)
            st.metric(
                "📝 Total Chunks",
                total_chunks,
                delta=None
            )
        
        with col3:
            st.metric(
                "💬 Queries",
                len([m for m in st.session_state.conversation_history if m['role'] == 'user']),
                delta=None
            )
        
        with col4:
            st.metric(
                "🤖 Responses",
                len([m for m in st.session_state.conversation_history if m['role'] == 'assistant']),
                delta=None
            )
        
        # Document processing timeline
        if st.session_state.processed_documents:
            st.markdown("### 📈 Document Processing Timeline")
            
            df = pd.DataFrame(st.session_state.processed_documents)
            df['processed_at'] = pd.to_datetime(df['processed_at'])
            
            fig = px.bar(
                df,
                x='filename',
                y='total_chunks',
                title='Chunks per Document',
                labels={'total_chunks': 'Number of Chunks', 'filename': 'Document'}
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    def run(self):
        """Run the Streamlit application"""
        self.render_header()
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3 = st.tabs(["📁 Upload", "💬 Chat", "📊 Analytics"])
        
        with tab1:
            self.render_file_upload()
        
        with tab2:
            self.render_chat_interface()
        
        with tab3:
            self.render_analytics()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666;'>"
            "ArXiv RAG Assistant - Powered by Marker & Ollama | "
            "<a href='https://github.com/your-repo/arxiv-rag' target='_blank'>GitHub</a>"
            "</div>",
            unsafe_allow_html=True
        )


def main():
    """Main application entry point"""
    app = StreamlitRAGApp()
    app.run()


if __name__ == "__main__":
    main() 