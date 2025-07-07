# Session Management System Guide

## 🎯 **Overview**

The ArXiv RAG Assistant now includes a comprehensive session management system that allows users to choose between **temporary** and **permanent** storage for their processed documents.

## 🔧 **Key Features**

### 1. **Session Management**
- **Start New Session**: Creates a unique session ID for document management
- **Session Info**: Shows current session details (document count, storage types)
- **End Session**: Automatically cleans up temporary documents while preserving permanent ones

### 2. **Storage Options**
- **⏱️ Temporary Documents**: Automatically removed when session ends
- **💾 Permanent Documents**: Saved permanently and accessible across sessions

### 3. **Document Management**
- **Individual Storage Selection**: Choose storage type per document
- **Batch Storage Selection**: Apply same storage type to all documents
- **Permanent Documents Library**: View and manage all permanent documents

## 🚀 **How to Use**

### **Step 1: Start a Session**
1. Initialize the RAG system if not already done
2. Click "🚀 Start New Session" in the sidebar
3. A unique session ID will be created and displayed

### **Step 2: Upload Documents**
1. Navigate to the "📁 Upload" tab
2. Select PDF files to upload
3. Choose storage type:
   - **Temporary**: Documents will be removed when session ends
   - **Permanent**: Documents will be saved for future use
4. For multiple files, optionally configure storage per file
5. Click "🚀 Process Documents"

### **Step 3: Chat with Documents**
1. Navigate to the "💬 Chat" tab
2. Ask questions about your documents
3. Both temporary and permanent documents are available for queries

### **Step 4: Session Management**
- **View Session Info**: See document counts and storage types in sidebar
- **End Session**: Removes temporary documents, keeps permanent ones
- **Manage Permanent Documents**: View, reload, or delete permanent documents

## 📊 **Session Information**

The sidebar shows:
- **Current Session ID**: First 8 characters of the session UUID
- **Document Count**: Total documents in current session
- **⏱️ Temporary**: Count of temporary documents
- **💾 Permanent**: Count of permanent documents

## 🗂️ **Storage Architecture**

### **Session Data Structure**
```
data/
├── sessions/
│   └── {session-id}/
│       └── session.json          # Session metadata
├── permanent_documents.json      # Permanent documents registry
└── chroma/                      # Vector database (all documents)
```

### **Document Lifecycle**
1. **Upload**: Document processed with Marker
2. **Storage Assignment**: Tagged as temporary or permanent
3. **Session Tracking**: Added to current session
4. **Vector Storage**: Embeddings stored in ChromaDB
5. **Session End**: Temporary documents removed, permanent kept

## 🔄 **Session Cleanup Process**

When ending a session:
1. **Temporary Documents**: Removed from vector store and metadata
2. **Permanent Documents**: Kept in vector store and added to permanent registry
3. **Session File**: Marked as ended with cleanup summary
4. **UI State**: Session state cleared

## 📚 **Permanent Documents Management**

### **View Permanent Documents**
- Click "📚 View Permanent Documents" in sidebar
- Shows all permanent documents with metadata
- Displays filename, chunks, and date added

### **Delete Permanent Documents**
- Click "🗑️ Delete" next to any permanent document
- Removes from vector store and permanent registry
- Cannot be undone

### **Reload Permanent Documents**
- "🔄 Reload" functionality (coming soon)
- Will allow re-indexing permanent documents

## ⚠️ **Important Notes**

### **Session Requirements**
- Must start a session before uploading documents
- Only one active session at a time
- Session ID is unique and cannot be reused

### **Storage Considerations**
- **Temporary documents** are completely removed when session ends
- **Permanent documents** persist across all sessions
- **Vector embeddings** are stored in ChromaDB for both types
- **Raw text files** are saved for both types during processing

### **Performance Impact**
- **Temporary documents**: Lower storage usage, automatic cleanup
- **Permanent documents**: Higher storage usage, manual cleanup required
- **Session overhead**: Minimal, only metadata tracking

## 🛠️ **Technical Implementation**

### **Key Components**
1. **SessionManager**: Handles session lifecycle and document tracking
2. **ChatEngine**: Integrates session management with document processing
3. **ChromaVectorStore**: Stores embeddings for both temporary and permanent documents
4. **StreamlitUI**: Provides user interface for session management

### **Storage Types Integration**
- Documents are tagged with storage type during processing
- Session manager tracks document-to-storage-type mapping
- Cleanup process uses storage type to determine removal

### **Error Handling**
- Session start/end failures are gracefully handled
- Document processing errors don't affect session state
- Storage type validation prevents invalid configurations

## 🔮 **Future Enhancements**

### **Planned Features**
- **Session History**: View past sessions and their documents
- **Document Sharing**: Share permanent documents between users
- **Automatic Cleanup**: Configurable cleanup of old sessions
- **Storage Quotas**: Limits on permanent document storage
- **Document Versioning**: Track document updates and changes

### **Advanced Options**
- **Batch Operations**: Bulk operations on permanent documents
- **Document Categories**: Organize permanent documents by topic
- **Access Control**: User-specific permanent document libraries
- **Backup/Restore**: Export/import permanent document collections

## 📋 **Troubleshooting**

### **Common Issues**
1. **"No active session"**: Start a new session before uploading
2. **Storage type errors**: Ensure valid storage type selection
3. **Session cleanup failures**: Check vector store connectivity
4. **Permanent document access**: Verify permanent documents registry

### **Recovery Actions**
- **Restart session**: End current session and start new one
- **Manual cleanup**: Remove orphaned files from data directories
- **Reset permanent registry**: Clear permanent_documents.json if corrupted
- **Vector store reset**: Clear ChromaDB if embeddings are corrupted

This session management system provides flexible document storage options while maintaining the performance and reliability of the RAG system. 