"""
Document Status Utility

This module provides utilities to check the status of processed and indexed documents
before processing new ones.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..config import get_config, get_logger


class DocumentStatusChecker:
    """Utility to check the status of processed and indexed documents"""
    
    def __init__(self):
        """Initialize the document status checker"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # Data paths
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.chroma_dir = self.data_dir / "chroma"
        self.permanent_docs_file = self.data_dir / "permanent_documents.json"
        
        # ChromaDB metadata file
        self.chroma_metadata_file = self.chroma_dir / "document_metadata.json"
    
    def get_processed_documents(self) -> List[Dict[str, Any]]:
        """Get list of processed documents (raw text extracted)"""
        processed_docs = []
        
        try:
            if not self.processed_dir.exists():
                return processed_docs
            
            # Find all metadata files
            metadata_files = list(self.processed_dir.glob("*_metadata.json"))
            
            for metadata_file in metadata_files:
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    processed_docs.append({
                        "document_id": metadata.get("document_id", "unknown"),
                        "filename": metadata.get("original_filename", "unknown"),
                        "processed_at": metadata.get("extracted_at", "unknown"),
                        "text_length": metadata.get("text_length", 0),
                        "chunks_count": metadata.get("extraction_stats", {}).get("paragraphs", 0),
                        "source": "processed"
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Failed to read metadata file {metadata_file}: {e}")
                    continue
            
            # Sort by processed date (most recent first)
            processed_docs.sort(key=lambda x: x["processed_at"], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error getting processed documents: {e}")
        
        return processed_docs
    
    def get_indexed_documents(self) -> List[Dict[str, Any]]:
        """Get list of indexed documents (in vector database)"""
        indexed_docs = []
        
        try:
            if not self.chroma_metadata_file.exists():
                return indexed_docs
            
            with open(self.chroma_metadata_file, 'r', encoding='utf-8') as f:
                chroma_metadata = json.load(f)
            
            for doc_id, doc_info in chroma_metadata.items():
                indexed_docs.append({
                    "document_id": doc_id,
                    "filename": doc_info.get("filename", "unknown"),
                    "processed_at": doc_info.get("processed_at", "unknown"),
                    "total_chunks": doc_info.get("total_chunks", 0),
                    "chunk_count": len(doc_info.get("chunk_ids", [])),
                    "source": "indexed"
                })
            
            # Sort by processed date (most recent first)
            indexed_docs.sort(key=lambda x: x["processed_at"], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Error getting indexed documents: {e}")
        
        return indexed_docs
    
    def get_permanent_documents(self) -> List[Dict[str, Any]]:
        """Get list of permanent documents"""
        permanent_docs = []
        
        try:
            if not self.permanent_docs_file.exists():
                return permanent_docs
            
            with open(self.permanent_docs_file, 'r', encoding='utf-8') as f:
                perm_data = json.load(f)
            
            for doc_id, doc_info in perm_data.items():
                # Handle both old and new metadata formats
                total_chunks = doc_info.get("total_chunks", 0)
                if total_chunks == 0:
                    # Try to get from content.blocks if available
                    content = doc_info.get("content", {})
                    if isinstance(content, dict) and "blocks" in content:
                        total_chunks = len(content["blocks"])
                
                permanent_docs.append({
                    "document_id": doc_id,
                    "filename": doc_info.get("filename", "unknown"),
                    "processed_at": doc_info.get("processed_at", "unknown"),
                    "added_to_permanent": doc_info.get("added_to_permanent", "unknown"),
                    "total_chunks": total_chunks,
                    "session_id": doc_info.get("session_id", "unknown"),
                    "source": "permanent"
                })
            
            # Sort by added to permanent date (most recent first)
            permanent_docs.sort(key=lambda x: x["added_to_permanent"], reverse=True)
            
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self.logger.error(f"Error parsing permanent documents JSON: {e}")
            # If JSON is corrupted, try to recover by creating a backup and starting fresh
            try:
                import shutil
                backup_file = self.permanent_docs_file.with_suffix('.json.backup')
                shutil.copy2(self.permanent_docs_file, backup_file)
                self.logger.info(f"Created backup of corrupted file: {backup_file}")
                
                # Create empty permanent documents file
                with open(self.permanent_docs_file, 'w', encoding='utf-8') as f:
                    json.dump({}, f)
                self.logger.info("Reset permanent documents file due to corruption")
            except Exception as backup_error:
                self.logger.error(f"Failed to create backup and reset: {backup_error}")
                
        except Exception as e:
            self.logger.error(f"Error getting permanent documents: {e}")
        
        return permanent_docs
    
    def get_all_documents_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all documents"""
        try:
            processed_docs = self.get_processed_documents()
            indexed_docs = self.get_indexed_documents()
            permanent_docs = self.get_permanent_documents()
            
            # Create a comprehensive view
            all_documents = {}
            
            # Add processed documents
            for doc in processed_docs:
                doc_id = doc["document_id"]
                all_documents[doc_id] = {
                    **doc,
                    "status": ["processed"]
                }
            
            # Add indexed status
            for doc in indexed_docs:
                doc_id = doc["document_id"]
                if doc_id in all_documents:
                    all_documents[doc_id]["status"].append("indexed")
                    all_documents[doc_id]["total_chunks"] = doc["total_chunks"]
                    all_documents[doc_id]["chunk_count"] = doc["chunk_count"]
                else:
                    all_documents[doc_id] = {
                        **doc,
                        "status": ["indexed"]
                    }
            
            # Add permanent status
            for doc in permanent_docs:
                doc_id = doc["document_id"]
                if doc_id in all_documents:
                    all_documents[doc_id]["status"].append("permanent")
                    all_documents[doc_id]["added_to_permanent"] = doc["added_to_permanent"]
                    all_documents[doc_id]["session_id"] = doc["session_id"]
                else:
                    all_documents[doc_id] = {
                        **doc,
                        "status": ["permanent"]
                    }
            
            # Convert to list and sort
            documents_list = list(all_documents.values())
            documents_list.sort(key=lambda x: x.get("processed_at", ""), reverse=True)
            
            return {
                "total_documents": len(documents_list),
                "processed_only": len([d for d in documents_list if d["status"] == ["processed"]]),
                "indexed_only": len([d for d in documents_list if d["status"] == ["indexed"]]),
                "permanent_only": len([d for d in documents_list if d["status"] == ["permanent"]]),
                "processed_and_indexed": len([d for d in documents_list if set(d["status"]) == {"processed", "indexed"}]),
                "all_statuses": len([d for d in documents_list if len(d["status"]) >= 3]),
                "documents": documents_list,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting all documents status: {e}")
            return {
                "total_documents": 0,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def print_documents_status(self, show_details: bool = True):
        """Print a formatted status report of all documents"""
        try:
            status = self.get_all_documents_status()
            
            if "error" in status:
                print(f"❌ Error getting document status: {status['error']}")
                return
            
            print("📊 Document Status Report")
            print("=" * 50)
            print(f"📄 Total Documents: {status['total_documents']}")
            print(f"🔄 Processed Only: {status['processed_only']}")
            print(f"🗂️ Indexed Only: {status['indexed_only']}")
            print(f"💾 Permanent Only: {status['permanent_only']}")
            print(f"✅ Processed & Indexed: {status['processed_and_indexed']}")
            print(f"🌟 All Statuses: {status['all_statuses']}")
            print(f"⏰ Generated: {status['timestamp']}")
            print()
            
            if show_details and status['documents']:
                print("📋 Document Details:")
                print("-" * 50)
                
                for i, doc in enumerate(status['documents'], 1):
                    status_icons = {
                        "processed": "🔄",
                        "indexed": "🗂️",
                        "permanent": "💾"
                    }
                    
                    status_str = " ".join([status_icons.get(s, s) for s in doc['status']])
                    
                    print(f"{i:2d}. {status_str} {doc['filename']}")
                    print(f"     ID: {doc['document_id']}")
                    print(f"     Processed: {doc.get('processed_at', 'unknown')[:19]}")
                    
                    if 'total_chunks' in doc:
                        print(f"     Chunks: {doc['total_chunks']}")
                    
                    if 'added_to_permanent' in doc:
                        print(f"     Added to Permanent: {doc['added_to_permanent'][:19]}")
                    
                    print()
            
            elif not status['documents']:
                print("📭 No documents found in the system.")
            
        except Exception as e:
            print(f"❌ Error printing document status: {e}")
    
    def check_document_exists(self, filename: str) -> Optional[Dict[str, Any]]:
        """Check if a document with the given filename already exists"""
        try:
            status = self.get_all_documents_status()
            
            for doc in status.get('documents', []):
                if doc['filename'] == filename:
                    return doc
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error checking document existence: {e}")
            return None
    
    def get_storage_usage(self) -> Dict[str, Any]:
        """Get storage usage statistics"""
        try:
            usage = {
                "processed_dir": 0,
                "chroma_dir": 0,
                "embeddings_dir": 0,
                "total_size": 0,
                "file_counts": {
                    "processed": 0,
                    "metadata": 0,
                    "embeddings": 0,
                    "chroma": 0
                }
            }
            
            # Calculate processed directory size
            if self.processed_dir.exists():
                for file_path in self.processed_dir.rglob("*"):
                    if file_path.is_file():
                        size = file_path.stat().st_size
                        usage["processed_dir"] += size
                        
                        if file_path.name.endswith("_metadata.json"):
                            usage["file_counts"]["metadata"] += 1
                        elif file_path.name.endswith("_raw.md"):
                            usage["file_counts"]["processed"] += 1
            
            # Calculate chroma directory size
            if self.chroma_dir.exists():
                for file_path in self.chroma_dir.rglob("*"):
                    if file_path.is_file():
                        size = file_path.stat().st_size
                        usage["chroma_dir"] += size
                        usage["file_counts"]["chroma"] += 1
            
            # Calculate embeddings directory size
            embeddings_dir = self.data_dir / "embeddings"
            if embeddings_dir.exists():
                for file_path in embeddings_dir.rglob("*"):
                    if file_path.is_file():
                        size = file_path.stat().st_size
                        usage["embeddings_dir"] += size
                        usage["file_counts"]["embeddings"] += 1
            
            usage["total_size"] = usage["processed_dir"] + usage["chroma_dir"] + usage["embeddings_dir"]
            
            return usage
            
        except Exception as e:
            self.logger.error(f"Error calculating storage usage: {e}")
            return {"error": str(e)}


def main():
    """Main function for command-line usage"""
    checker = DocumentStatusChecker()
    checker.print_documents_status(show_details=True)
    
    # Also show storage usage
    usage = checker.get_storage_usage()
    if "error" not in usage:
        print("\n💾 Storage Usage:")
        print("-" * 30)
        print(f"Processed Files: {usage['processed_dir'] / 1024 / 1024:.1f} MB")
        print(f"ChromaDB: {usage['chroma_dir'] / 1024 / 1024:.1f} MB")
        print(f"Embeddings: {usage['embeddings_dir'] / 1024 / 1024:.1f} MB")
        print(f"Total: {usage['total_size'] / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main() 