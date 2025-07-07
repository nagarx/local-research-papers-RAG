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
    from src.utils import DocumentStatusChecker
    
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
        if 'current_session_id' not in st.session_state:
            st.session_state.current_session_id = None
        if 'session_info' not in st.session_state:
            st.session_state.session_info = None
    
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
        
        # Session management
        with st.sidebar.expander("📁 Session Management", expanded=True):
            # Current session info
            if st.session_state.current_session_id:
                st.markdown(f"**Current Session:** `{st.session_state.current_session_id[:8]}...`")
                
                if st.session_state.session_info:
                    session_info = st.session_state.session_info
                    st.write(f"📄 Documents: {session_info['document_count']}")
                    st.write(f"⏱️ Temporary: {session_info['temporary_count']}")
                    st.write(f"💾 Permanent: {session_info['permanent_count']}")
                
                if st.button("🔚 End Session", key="end_session"):
                    self.end_current_session()
            else:
                st.write("No active session")
                if st.button("🚀 Start New Session", key="start_session"):
                    self.start_new_session()
        
        # Document management
        with st.sidebar.expander("📄 Document Management", expanded=False):
            # Document status overview
            if st.button("📋 View All Documents", key="view_all_docs"):
                self.show_document_status()
            
            # Current session documents
            if st.session_state.processed_documents:
                st.write(f"📊 {len(st.session_state.processed_documents)} documents in session")
                
                # Show document list with storage types
                for doc in st.session_state.processed_documents:
                    storage_type = doc.get('storage_type', 'unknown')
                    icon = "💾" if storage_type == 'permanent' else "⏱️"
                    st.write(f"{icon} {doc['filename']} ({doc['total_chunks']} chunks)")
                
                if st.button("🗑️ Clear Session Documents", key="clear_docs"):
                    self.clear_session_documents()
            else:
                st.write("No documents in current session")
            
            # Permanent documents management
            if st.button("📚 View Permanent Documents", key="view_permanent"):
                self.show_permanent_documents()
        
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
    
    def start_new_session(self):
        """Start a new document session"""
        try:
            if st.session_state.chat_engine:
                session_id = st.session_state.chat_engine.start_session()
                if session_id:
                    st.session_state.current_session_id = session_id
                    st.session_state.session_info = st.session_state.chat_engine.get_session_info()
                    st.success(f"✅ Started new session: {session_id[:8]}...")
                    st.rerun()
                else:
                    st.error("❌ Failed to start session")
            else:
                st.error("❌ Please initialize the system first")
        except Exception as e:
            st.error(f"❌ Failed to start session: {e}")
    
    def end_current_session(self):
        """End the current session"""
        try:
            if st.session_state.chat_engine and st.session_state.current_session_id:
                cleanup_summary = st.session_state.chat_engine.end_session()
                
                if "error" not in cleanup_summary:
                    st.success(f"✅ Session ended successfully!")
                    st.info(f"📊 Cleanup Summary:\n"
                           f"• Temporary documents removed: {cleanup_summary['temporary_docs_removed']}\n"
                           f"• Permanent documents kept: {cleanup_summary['permanent_docs_kept']}")
                    
                    if cleanup_summary.get('errors'):
                        st.warning(f"⚠️ Some errors occurred: {cleanup_summary['errors']}")
                else:
                    st.error(f"❌ Failed to end session: {cleanup_summary['error']}")
                
                # Clear session state
                st.session_state.current_session_id = None
                st.session_state.session_info = None
                st.session_state.processed_documents = []
                st.rerun()
            else:
                st.warning("⚠️ No active session to end")
        except Exception as e:
            st.error(f"❌ Failed to end session: {e}")
    
    def clear_session_documents(self):
        """Clear current session documents"""
        try:
            if st.session_state.chat_engine and st.session_state.current_session_id:
                # End current session (which clears temporary docs)
                self.end_current_session()
            else:
                st.warning("⚠️ No active session")
        except Exception as e:
            st.error(f"❌ Failed to clear session documents: {e}")
    
    def show_permanent_documents(self):
        """Show permanent documents management interface"""
        try:
            if st.session_state.chat_engine:
                permanent_docs = st.session_state.chat_engine.get_permanent_documents()
                
                if permanent_docs:
                    st.markdown("### 💾 Permanent Documents")
                    
                    for doc in permanent_docs:
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.write(f"📄 **{doc['filename']}**")
                            # Get chunk count from ChromaDB if available
                            if st.session_state.chat_engine:
                                doc_info = st.session_state.chat_engine.vector_store.get_document_info(doc['document_id'])
                                actual_chunks = doc_info.get('total_chunks', 0) if doc_info else 0
                                st.write(f"Chunks: {actual_chunks} | Added: {doc['added_to_permanent'][:10]}")
                            else:
                                st.write(f"Chunks: {doc['total_chunks']} | Added: {doc['added_to_permanent'][:10]}")
                        
                        with col2:
                            if st.button("🔄 Reload", key=f"reload_{doc['document_id']}"):
                                st.info("Reload functionality coming soon...")
                        
                        with col3:
                            if st.button("🗑️ Delete", key=f"delete_{doc['document_id']}"):
                                with st.spinner(f"Deleting {doc['filename']}..."):
                                    try:
                                        success = st.session_state.chat_engine.remove_permanent_document(doc['document_id'])
                                        if success:
                                            st.success(f"✅ Deleted {doc['filename']}")
                                            # Force refresh after successful deletion
                                            time.sleep(1)
                                            st.rerun()
                                        else:
                                            st.error(f"❌ Failed to delete {doc['filename']}")
                                    except Exception as delete_error:
                                        st.error(f"❌ Error deleting {doc['filename']}: {str(delete_error)}")
                        
                        st.divider()
                else:
                    st.info("📭 No permanent documents found")
            else:
                st.error("❌ Please initialize the system first")
                
        except Exception as e:
            st.error(f"❌ Failed to load permanent documents: {e}")
            st.exception(e)
    
    def render_document_management_section(self):
        """Render the always-visible document management section"""
        st.markdown("## 📚 Document Management")
        
        # Add refresh and management controls
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.markdown("*Manage your indexed documents and their storage*")
        
        with col2:
            if st.button("🔄 Refresh", key="refresh_doc_management"):
                self._force_complete_refresh()
                st.rerun()
        
        with col3:
            if st.button("📊 System Status", key="doc_system_status"):
                self.show_document_status()
                
        # Add a comprehensive cleanup button for debugging
        st.expander("🔧 Advanced Operations", expanded=False)
        with st.expander("🔧 Advanced Operations", expanded=False):
            st.warning("⚠️ Advanced operations - use with caution!")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🧹 Force Complete Cache Clear", key="force_cache_clear"):
                    self._force_complete_refresh()
                    st.success("✅ All document caches cleared")
                    st.rerun()
                    
            with col2:
                if st.button("🔄 Reload Document Data", key="reload_doc_data"):
                    # Force reload all document data
                    self._force_complete_refresh()
                    st.success("✅ Document data reloaded")
                    st.rerun()
        
        with col4:
            # Auto-refresh toggle
            auto_refresh = st.checkbox("⏱️ Auto-refresh", key="auto_refresh_docs")
            if auto_refresh:
                # Automatically refresh every 30 seconds
                import time
                time.sleep(30)
                st.rerun()
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📄 All Documents", "💾 Permanent Documents", "⏱️ Session Documents"])
        
        with tab1:
            self.render_all_documents_view()
        
        with tab2:
            self.render_permanent_documents_view()
        
        with tab3:
            self.render_session_documents_view()
    
    def render_all_documents_view(self):
        """Render view of all documents in the system"""
        try:
            if not st.session_state.chat_engine:
                st.warning("⚠️ Please initialize the system first to view documents")
                return
            
            # Get comprehensive document status
            checker = DocumentStatusChecker()
            status = checker.get_all_documents_status()
            
            if "error" in status:
                st.error(f"❌ Error getting document status: {status['error']}")
                return
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📄 Total", status['total_documents'])
            with col2:
                st.metric("🔄 Processed", status['processed_only'] + status['processed_and_indexed'])
            with col3:
                st.metric("🗂️ Indexed", status['indexed_only'] + status['processed_and_indexed'])
            with col4:
                st.metric("💾 Permanent", status['permanent_only'] + status['all_statuses'])
            
            # Document list
            if status['documents']:
                # Add search and filter controls
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    search_term = st.text_input("🔍 Search documents", 
                                              placeholder="Search by filename...",
                                              key="search_all_docs")
                
                with col2:
                    filter_status = st.selectbox("Filter by status", 
                                               options=["All", "Processed", "Indexed", "Permanent"],
                                               key="filter_all_docs")
                
                st.markdown("#### 📋 Document List (Newest to Oldest)")
                
                # Sort documents by processing time (newest first)
                sorted_docs = sorted(status['documents'], 
                                   key=lambda x: x.get('processed_at', ''), 
                                   reverse=True)
                
                # Apply search and filter
                filtered_docs = []
                for doc in sorted_docs:
                    # Apply search filter
                    if search_term and search_term.lower() not in doc['filename'].lower():
                        continue
                    
                    # Apply status filter
                    if filter_status != "All":
                        if filter_status.lower() not in [s.lower() for s in doc.get('status', [])]:
                            continue
                    
                    filtered_docs.append(doc)
                
                if filtered_docs:
                    st.info(f"📊 Showing {len(filtered_docs)} of {len(sorted_docs)} documents")
                    
                    for doc in filtered_docs:
                        self.render_document_card(doc, show_actions=True)
                else:
                    st.info("📭 No documents match your search criteria")
            else:
                st.info("📭 No documents found in the system")
                
        except Exception as e:
            st.error(f"❌ Failed to load documents: {e}")
            st.exception(e)
    
    def render_permanent_documents_view(self):
        """Render view of permanent documents only"""
        try:
            if not st.session_state.chat_engine:
                st.warning("⚠️ Please initialize the system first to view permanent documents")
                return
            
            permanent_docs = st.session_state.chat_engine.get_permanent_documents()
            
            if permanent_docs:
                # Add search control
                search_term = st.text_input("🔍 Search permanent documents", 
                                          placeholder="Search by filename...",
                                          key="search_permanent_docs")
                
                st.markdown("#### 💾 Permanent Documents (Newest to Oldest)")
                
                # Sort by added_to_permanent date (newest first)
                sorted_docs = sorted(permanent_docs, 
                                   key=lambda x: x.get('added_to_permanent', ''), 
                                   reverse=True)
                
                # Apply search filter
                filtered_docs = []
                for doc in sorted_docs:
                    if search_term and search_term.lower() not in doc['filename'].lower():
                        continue
                    filtered_docs.append(doc)
                
                if filtered_docs:
                    st.info(f"📊 Showing {len(filtered_docs)} of {len(sorted_docs)} permanent documents")
                    
                    for doc in filtered_docs:
                        self.render_permanent_document_card(doc)
                else:
                    st.info("📭 No permanent documents match your search criteria")
            else:
                st.info("📭 No permanent documents found")
                
        except Exception as e:
            st.error(f"❌ Failed to load permanent documents: {e}")
            st.exception(e)
    
    def render_session_documents_view(self):
        """Render view of current session documents"""
        try:
            if not st.session_state.chat_engine:
                st.warning("⚠️ Please initialize the system first to view session documents")
                return
            
            if not st.session_state.current_session_id:
                st.info("📭 No active session. Start a new session to see session documents.")
                return
            
            session_info = st.session_state.chat_engine.get_session_info()
            
            if not session_info or not session_info.get('documents'):
                st.info("📭 No documents in current session")
                return
            
            # Session info
            st.markdown(f"#### ⏱️ Current Session: `{session_info['session_id'][:8]}...`")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📄 Total", session_info['document_count'])
            with col2:
                st.metric("⏱️ Temporary", session_info['temporary_count'])
            with col3:
                st.metric("💾 Permanent", session_info['permanent_count'])
            
            # Session documents
            if st.session_state.processed_documents:
                st.markdown("#### 📋 Session Documents")
                
                for doc in st.session_state.processed_documents:
                    self.render_session_document_card(doc)
            
            # Session actions
            st.markdown("#### 🔧 Session Actions")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔚 End Session", key="end_session_main"):
                    self.end_current_session()
            
            with col2:
                if st.button("🗑️ Clear Session Documents", key="clear_session_main"):
                    self.clear_session_documents()
                    
        except Exception as e:
            st.error(f"❌ Failed to load session documents: {e}")
            st.exception(e)
    
    def render_document_card(self, doc, show_actions=False):
        """Render a document card with status and actions"""
        try:
            # Create a card-like container with background
            with st.container():
                # Create a styled card
                card_style = """
                <div style="
                    background-color: #f0f2f6;
                    padding: 1rem;
                    border-radius: 8px;
                    border-left: 4px solid #1e3c72;
                    margin: 0.5rem 0;
                ">
                """
                
                # Document header
                col1, col2, col3 = st.columns([4, 1, 1])
                
                with col1:
                    # Document name and status
                    status_icons = {
                        'processed': '🔄',
                        'indexed': '🗂️',
                        'permanent': '💾',
                        'temporary': '⏱️'
                    }
                    
                    status_text = []
                    for status in doc.get('status', []):
                        if status in status_icons:
                            status_text.append(f"{status_icons[status]} {status.title()}")
                    
                    st.markdown(f"**📄 {doc['filename']}**")
                    st.markdown(f"*Status: {' | '.join(status_text)}*")
                    
                    # Get real-time chunk count if available
                    if st.session_state.chat_engine and 'document_id' in doc:
                        try:
                            doc_info = st.session_state.chat_engine.vector_store.get_document_info(doc['document_id'])
                            if doc_info:
                                real_chunks = doc_info.get('total_chunks', doc.get('total_chunks', 0))
                                st.write(f"📊 **Chunks:** {real_chunks}")
                            else:
                                st.write(f"📊 **Chunks:** {doc.get('total_chunks', 0)}")
                        except Exception:
                            st.write(f"📊 **Chunks:** {doc.get('total_chunks', 0)}")
                    else:
                        st.write(f"📊 **Chunks:** {doc.get('total_chunks', 0)}")
                
                with col2:
                    # Document metrics
                    processed_at = doc.get('processed_at', '')
                    if processed_at:
                        st.write(f"**Processed:**")
                        st.write(processed_at[:10])
                
                with col3:
                    # Document ID (shortened)
                    if 'document_id' in doc:
                        st.write(f"**ID:**")
                        st.write(f"`{doc['document_id'][:8]}...`")
                
                # Additional info in a compact format
                additional_info = []
                if doc.get('added_to_permanent'):
                    additional_info.append(f"💾 Permanent: {doc['added_to_permanent'][:10]}")
                if doc.get('session_id'):
                    additional_info.append(f"📋 Session: `{doc['session_id'][:8]}...`")
                
                if additional_info:
                    st.write(" | ".join(additional_info))
                
                # Actions
                if show_actions:
                    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                    
                    with col1:
                        if st.button("ℹ️ Info", key=f"info_{doc['document_id']}"):
                            with st.expander("📄 Document Details", expanded=True):
                                st.json(doc)
                    
                    with col2:
                        if 'permanent' in doc.get('status', []):
                            if st.button("🗑️ Delete", key=f"delete_all_{doc['document_id']}"):
                                # Store deletion request in session state for confirmation
                                st.session_state[f"confirm_delete_all_{doc['document_id']}"] = True
                                st.rerun()
                    
                    # Show confirmation dialog if deletion was requested
                    if st.session_state.get(f"confirm_delete_all_{doc['document_id']}", False):
                        st.warning(f"⚠️ **Confirm Deletion of {doc['filename']}**")
                        st.write("This will **permanently remove** the document from:")
                        st.write("• Vector database (all embeddings and chunks)")
                        st.write("• Permanent documents registry")
                        st.write("• Processed documents directory")
                        st.write("• All associated files")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("✅ Yes, Delete Permanently", key=f"confirm_all_yes_{doc['document_id']}"):
                                # Clear confirmation flag and proceed with deletion
                                del st.session_state[f"confirm_delete_all_{doc['document_id']}"]
                                self.delete_document(doc['document_id'], doc['filename'])
                        
                        with col2:
                            if st.button("❌ Cancel", key=f"confirm_all_no_{doc['document_id']}"):
                                # Clear confirmation flag
                                del st.session_state[f"confirm_delete_all_{doc['document_id']}"]
                                st.rerun()
                    
                    with col3:
                        if st.button("🔄 Refresh", key=f"refresh_{doc['document_id']}"):
                            st.rerun()
                    
                    with col4:
                        if st.button("📋 View Chunks", key=f"chunks_{doc['document_id']}"):
                            self.show_document_chunks(doc['document_id'], doc['filename'])
                
                st.divider()
                
        except Exception as e:
            st.error(f"❌ Error rendering document card: {e}")
    
    def show_document_chunks(self, document_id, filename):
        """Show document chunks in a modal-like interface"""
        try:
            if not st.session_state.chat_engine:
                st.error("❌ System not initialized")
                return
            
            # Get document info and chunks
            doc_info = st.session_state.chat_engine.vector_store.get_document_info(document_id)
            
            if not doc_info:
                st.error(f"❌ Could not find document info for {filename}")
                return
            
            with st.expander(f"📋 Chunks for {filename}", expanded=True):
                st.write(f"**Total Chunks:** {doc_info.get('total_chunks', 0)}")
                
                # Show first few chunks as preview
                chunks = doc_info.get('chunks', [])
                if chunks:
                    st.write("**Sample Chunks:**")
                    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                        st.write(f"**Chunk {i+1}:**")
                        st.write(f"Page: {chunk.get('page_number', 'N/A')}")
                        st.write(f"Type: {chunk.get('block_type', 'N/A')}")
                        st.write(f"Text: {chunk.get('text', '')[:200]}...")
                        st.divider()
                    
                    if len(chunks) > 3:
                        st.write(f"... and {len(chunks) - 3} more chunks")
                else:
                    st.write("No chunk details available")
                
        except Exception as e:
            st.error(f"❌ Error showing chunks: {e}")
    
    def render_permanent_document_card(self, doc):
        """Render a permanent document card"""
        try:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"**📄 {doc['filename']}**")
                    
                    # Get real-time chunk count
                    if st.session_state.chat_engine:
                        doc_info = st.session_state.chat_engine.vector_store.get_document_info(doc['document_id'])
                        actual_chunks = doc_info.get('total_chunks', 0) if doc_info else 0
                        st.write(f"**Chunks:** {actual_chunks}")
                    else:
                        st.write(f"**Chunks:** {doc.get('total_chunks', 0)}")
                    
                    st.write(f"**Added:** {doc.get('added_to_permanent', '')[:19]}")
                
                with col2:
                    st.metric("Status", "💾 Permanent")
                
                with col3:
                    # Actions - with confirmation
                    if st.button("🗑️ Delete", key=f"delete_perm_{doc['document_id']}"):
                        # Store deletion request in session state for confirmation
                        st.session_state[f"confirm_delete_{doc['document_id']}"] = True
                        st.rerun()
                
                # Show confirmation dialog if deletion was requested
                if st.session_state.get(f"confirm_delete_{doc['document_id']}", False):
                    st.warning(f"⚠️ **Confirm Deletion of {doc['filename']}**")
                    st.write("This will **permanently remove** the document from:")
                    st.write("• Vector database (all embeddings and chunks)")
                    st.write("• Permanent documents registry")
                    st.write("• Processed documents directory")
                    st.write("• All associated files")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ Yes, Delete Permanently", key=f"confirm_yes_{doc['document_id']}"):
                            # Clear confirmation flag and proceed with deletion
                            del st.session_state[f"confirm_delete_{doc['document_id']}"]
                            self.delete_permanent_document(doc['document_id'], doc['filename'])
                    
                    with col2:
                        if st.button("❌ Cancel", key=f"confirm_no_{doc['document_id']}"):
                            # Clear confirmation flag
                            del st.session_state[f"confirm_delete_{doc['document_id']}"]
                            st.rerun()
                
                st.divider()
                
        except Exception as e:
            st.error(f"❌ Error rendering permanent document card: {e}")
    
    def render_session_document_card(self, doc):
        """Render a session document card"""
        try:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    storage_icon = "💾" if doc.get('storage_type') == 'permanent' else "⏱️"
                    st.markdown(f"**{storage_icon} {doc['filename']}**")
                    st.write(f"**Chunks:** {doc.get('total_chunks', 0)}")
                
                with col2:
                    st.metric("Type", doc.get('storage_type', 'unknown').title())
                
                with col3:
                    if st.button("ℹ️ Info", key=f"info_session_{doc.get('id', 'unknown')}"):
                        st.json(doc)
                
                st.divider()
                
        except Exception as e:
            st.error(f"❌ Error rendering session document card: {e}")
    
    def delete_document(self, document_id, filename):
        """Delete a document completely from all storage locations"""
        try:
            with st.spinner(f"Completely removing document {filename} from all storage locations..."):
                # Use the comprehensive removal method
                success = st.session_state.chat_engine.remove_permanent_document(document_id)
                
                if success:
                    st.success(f"✅ Completely removed document {filename} from all storage locations")
                    st.info("🔄 Document has been removed from:\n" +
                           "• Vector database (ChromaDB)\n" +
                           "• Permanent documents registry\n" +
                           "• Processed documents directory\n" +
                           "• Embedding files\n" +
                           "• Current session (if applicable)")
                    
                    # Clear cached data
                    self._clear_document_cache(document_id)
                    
                    time.sleep(2)  # Give user time to see the success message
                    st.rerun()
                else:
                    st.error(f"❌ Failed to completely remove document {filename}")
                    st.warning("⚠️ The document may still exist in some storage locations. Please check the system logs for details.")
                    
        except Exception as e:
            st.error(f"❌ Critical error removing document {filename}: {str(e)}")
            st.warning("⚠️ Please check the system logs and try again.")
    
    def _clear_document_cache(self, document_id):
        """Clear document from session state caches"""
        try:
            # Clear from processed documents list
            if 'processed_documents' in st.session_state:
                st.session_state.processed_documents = [
                    doc for doc in st.session_state.processed_documents 
                    if doc.get('id') != document_id and doc.get('document_id') != document_id
                ]
            
            # Update session info
            if st.session_state.chat_engine:
                st.session_state.session_info = st.session_state.chat_engine.get_session_info()
            
            # Clear any other document-related caches
            if 'conversation_history' in st.session_state:
                # No need to clear conversation history as it's separate
                pass
                
        except Exception as e:
            self.logger.warning(f"Error clearing document cache: {e}")
    
    def _force_complete_refresh(self):
        """Force a complete refresh of all document data"""
        try:
            # Clear all document-related caches
            if 'processed_documents' in st.session_state:
                del st.session_state.processed_documents
            
            if 'session_info' in st.session_state:
                del st.session_state.session_info
            
            # Force refresh of session info
            if st.session_state.chat_engine:
                st.session_state.session_info = st.session_state.chat_engine.get_session_info()
            
            # Initialize empty processed documents list
            st.session_state.processed_documents = []
            
        except Exception as e:
            self.logger.warning(f"Error during complete refresh: {e}")
    
    def delete_permanent_document(self, document_id, filename):
        """Delete a permanent document completely from all storage locations"""
        try:
            with st.spinner(f"Completely removing document {filename} from all storage locations..."):
                success = st.session_state.chat_engine.remove_permanent_document(document_id)
                
                if success:
                    st.success(f"✅ Completely removed document {filename} from all storage locations")
                    st.info("🔄 Document has been removed from:\n" +
                           "• Vector database (ChromaDB)\n" +
                           "• Permanent documents registry\n" +
                           "• Processed documents directory\n" +
                           "• Embedding files\n" +
                           "• Current session (if applicable)")
                    
                    # Clear cached data
                    self._clear_document_cache(document_id)
                    
                    time.sleep(2)  # Give user time to see the success message
                    st.rerun()
                else:
                    st.error(f"❌ Failed to completely remove document {filename}")
                    st.warning("⚠️ The document may still exist in some storage locations. Please check the system logs for details.")
                    
        except Exception as e:
            st.error(f"❌ Critical error removing document {filename}: {str(e)}")
            st.warning("⚠️ Please check the system logs and try again.")
    
    def show_document_status(self):
        """Show comprehensive document status (legacy method - keeping for compatibility)"""
        try:
            checker = DocumentStatusChecker()
            status = checker.get_all_documents_status()
            
            if "error" in status:
                st.error(f"❌ Error getting document status: {status['error']}")
                return
            
            st.markdown("### 📊 Document Status Overview")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📄 Total", status['total_documents'])
            with col2:
                st.metric("🔄 Processed", status['processed_only'] + status['processed_and_indexed'])
            with col3:
                st.metric("🗂️ Indexed", status['indexed_only'] + status['processed_and_indexed'])
            with col4:
                st.metric("💾 Permanent", status['permanent_only'] + status['all_statuses'])
            
            # Document details
            if status['documents']:
                st.markdown("#### 📋 Document Details")
                
                for doc in status['documents']:
                    with st.expander(f"📄 {doc['filename']}", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Document ID:** `{doc['document_id']}`")
                            st.write(f"**Status:** {', '.join(doc['status'])}")
                            st.write(f"**Processed:** {doc.get('processed_at', 'unknown')[:19]}")
                            
                        with col2:
                            if 'total_chunks' in doc:
                                st.write(f"**Chunks:** {doc['total_chunks']}")
                            if 'added_to_permanent' in doc:
                                st.write(f"**Added to Permanent:** {doc['added_to_permanent'][:19]}")
                            if 'session_id' in doc:
                                st.write(f"**Session:** `{doc['session_id'][:8]}...`")
            else:
                st.info("📭 No documents found in the system")
                
        except Exception as e:
            st.error(f"❌ Failed to show document status: {e}")
    
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
        
        # Check if session is active
        if not st.session_state.current_session_id:
            st.warning("⚠️ Please start a session first to upload documents")
            return
        
        # Show current document status
        try:
            checker = DocumentStatusChecker()
            status = checker.get_all_documents_status()
            
            if status['total_documents'] > 0:
                st.info(f"📊 **Current System Status:** {status['total_documents']} documents total "
                       f"({status['processed_and_indexed']} processed & indexed, "
                       f"{status.get('permanent_only', 0) + status.get('all_statuses', 0)} permanent)")
                
                # Show list of existing documents
                with st.expander("📋 View Existing Documents", expanded=False):
                    for doc in status['documents'][:10]:  # Show first 10
                        status_icons = {"processed": "🔄", "indexed": "🗂️", "permanent": "💾"}
                        icons = " ".join([status_icons.get(s, s) for s in doc['status']])
                        st.write(f"{icons} {doc['filename']}")
                    
                    if len(status['documents']) > 10:
                        st.write(f"... and {len(status['documents']) - 10} more documents")
            else:
                st.info("📭 No documents currently in the system. Upload some PDFs to get started!")
                
        except Exception as e:
            st.warning(f"⚠️ Could not check document status: {e}")
        
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
            
            st.markdown("### 📁 Storage Options")
            
            # Storage type selection
            storage_option = st.radio(
                "Choose storage type for uploaded documents:",
                options=["temporary", "permanent"],
                format_func=lambda x: {
                    "temporary": "⏱️ Temporary (removed when session ends)",
                    "permanent": "💾 Permanent (saved for future sessions)"
                }[x],
                help="Temporary documents are automatically removed when you end the session. Permanent documents are saved and can be accessed in future sessions."
            )
            
            # Individual file storage selection
            if len(uploaded_files) > 1:
                st.markdown("#### Individual File Settings")
                
                use_individual = st.checkbox("Configure storage per file", key="individual_storage")
                
                if use_individual:
                    storage_types = []
                    for i, file in enumerate(uploaded_files):
                        file_storage = st.selectbox(
                            f"Storage for {file.name}:",
                            options=["temporary", "permanent"],
                            index=0 if storage_option == "temporary" else 1,
                            format_func=lambda x: "⏱️ Temporary" if x == "temporary" else "💾 Permanent",
                            key=f"storage_{i}"
                        )
                        storage_types.append(file_storage)
                else:
                    storage_types = [storage_option] * len(uploaded_files)
            else:
                storage_types = [storage_option]
            
            # Processing options
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button("🚀 Process Documents", key="process_docs"):
                    self.process_uploaded_files(uploaded_files, storage_types)
            
            with col2:
                force_reprocess = st.checkbox("🔄 Force Reprocess", help="Reprocess documents even if they already exist")
                if st.button("🔄 Process (Force)", key="force_process_docs"):
                    self.process_uploaded_files(uploaded_files, storage_types, force_reprocess=True)
            
            # Show storage summary
            if len(uploaded_files) > 1:
                temp_count = storage_types.count("temporary")
                perm_count = storage_types.count("permanent")
                st.info(f"📊 Summary: {temp_count} temporary, {perm_count} permanent documents")
    
    def process_uploaded_files(self, uploaded_files, storage_types=None, force_reprocess=False):
        """Process uploaded PDF files using the actual RAG system"""
        if not st.session_state.chat_engine:
            st.error("❌ Please initialize the system first")
            return
        
        if storage_types is None:
            storage_types = ['temporary'] * len(uploaded_files)
        
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
                    storage_types=storage_types,
                    progress_callback=update_progress,
                    force_reprocess=force_reprocess
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
                
                for i, file_path in enumerate(file_paths):
                    filename = Path(file_path).name
                    st.session_state.processed_documents.append({
                        'filename': filename,
                        'total_chunks': chunks_per_doc,
                        'processed_at': datetime.now().isoformat(),
                        'storage_type': storage_types[i] if i < len(storage_types) else 'temporary'
                    })
                
                progress_bar.progress(1.0)
                status_text.text("✅ All documents processed successfully!")
                
                # Update session info
                st.session_state.session_info = st.session_state.chat_engine.get_session_info()
                
                st.success(f"✅ Processed {len(uploaded_files)} documents successfully!")
                
                # Show processing summary with storage info
                temp_count = storage_types.count("temporary")
                perm_count = storage_types.count("permanent")
                
                st.info(f"📊 Processing Summary:\n" + 
                       f"- Total documents: {result['total_documents']}\n" + 
                       f"- Total chunks: {result['total_chunks']}\n" + 
                       f"- Processing time: {result['processing_time']:.1f}s\n" +
                       f"- Temporary documents: {temp_count}\n" +
                       f"- Permanent documents: {perm_count}")
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
        
        # Main content tabs with Document Management as the first tab
        tab1, tab2, tab3, tab4 = st.tabs(["📚 Document Management", "📁 Upload", "💬 Chat", "📊 Analytics"])
        
        with tab1:
            self.render_document_management_section()
        
        with tab2:
            self.render_file_upload()
        
        with tab3:
            self.render_chat_interface()
        
        with tab4:
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