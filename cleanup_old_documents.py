#!/usr/bin/env python3
"""
Cleanup Old Documents - RAG System Maintenance

This script cleans up old documents from the RAG system and fixes any registry issues.
It can be run periodically to maintain system performance and consistency.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.storage import VectorStore
from src.tracking import SourceTracker
from src.config import get_config, get_logger


class DocumentCleanup:
    """Handles cleanup and maintenance of RAG system documents"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.vector_store = VectorStore()
        self.source_tracker = SourceTracker()
        
    def cleanup_old_documents(self, days_old: int = 7) -> dict:
        """Remove documents older than specified days"""
        
        self.logger.info(f"Starting cleanup of documents older than {days_old} days...")
        
        # Use the vector store's cleanup method
        result = self.vector_store.cleanup_old_documents(days_old)
        
        # Log results
        if result.get("removed_documents", 0) > 0:
            self.logger.info(f"Cleanup completed: removed {result['removed_documents']} documents")
        else:
            self.logger.info("No old documents found to remove")
            
        return result
    
    def fix_registry_issues(self) -> dict:
        """Fix document registry mismatches"""
        
        self.logger.info("Fixing document registry issues...")
        
        # Re-register all existing documents
        registered_count = self.vector_store.re_register_existing_documents(self.source_tracker)
        
        result = {
            "re_registered_documents": registered_count,
            "status": "success" if registered_count >= 0 else "error"
        }
        
        if registered_count > 0:
            self.logger.info(f"Fixed registry for {registered_count} documents")
        else:
            self.logger.info("No registry issues found")
            
        return result
    
    def get_system_status(self) -> dict:
        """Get comprehensive system status"""
        
        # Vector store stats
        vector_stats = self.vector_store.get_enhanced_stats()
        
        # Source tracker stats  
        source_stats = self.source_tracker.get_statistics()
        
        # Calculate potential issues
        total_vector_docs = vector_stats.get("total_documents", 0)
        total_source_docs = source_stats.get("total_documents", 0)
        
        registry_mismatch = total_vector_docs != total_source_docs
        
        status = {
            "vector_store": vector_stats,
            "source_tracker": source_stats,
            "potential_issues": {
                "registry_mismatch": registry_mismatch,
                "mismatch_count": abs(total_vector_docs - total_source_docs) if registry_mismatch else 0
            },
            "recommendations": []
        }
        
        # Add recommendations
        if registry_mismatch:
            status["recommendations"].append("Run --fix-registry to resolve document registry mismatches")
        
        if vector_stats.get("storage_size_mb", 0) > 1000:  # > 1GB
            status["recommendations"].append("Consider running cleanup to reduce storage size")
            
        return status
    
    def print_status_report(self, status: dict):
        """Print a formatted status report"""
        
        print("\n" + "="*60)
        print("📊 RAG SYSTEM STATUS REPORT")
        print("="*60)
        
        # Vector Store Info
        vs = status["vector_store"]
        print(f"\n📁 Vector Store:")
        print(f"   Documents: {vs.get('total_documents', 0)}")
        print(f"   Chunks: {vs.get('total_chunks', 0)}")
        print(f"   Vectors: {vs.get('total_vectors', 0)}")
        print(f"   Storage: {vs.get('storage_size_mb', 0):.1f} MB")
        
        # Source Tracker Info
        st = status["source_tracker"]
        print(f"\n📚 Source Tracker:")
        print(f"   Registered Documents: {st.get('total_documents', 0)}")
        print(f"   Total Pages: {st.get('total_pages', 0)}")
        print(f"   Avg Pages/Doc: {st.get('average_pages_per_document', 0):.1f}")
        
        # Issues
        issues = status["potential_issues"]
        if issues["registry_mismatch"]:
            print(f"\n⚠️  Issues Detected:")
            print(f"   Registry Mismatch: {issues['mismatch_count']} documents")
        else:
            print(f"\n✅ No Issues Detected")
        
        # Recommendations
        recommendations = status["recommendations"]
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("\n" + "="*60)


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description="RAG System Document Cleanup and Maintenance Tool"
    )
    
    parser.add_argument(
        "--cleanup",
        type=int,
        metavar="DAYS",
        help="Remove documents older than DAYS (default: 7)"
    )
    
    parser.add_argument(
        "--fix-registry",
        action="store_true",
        help="Fix document registry mismatches"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show system status report"
    )
    
    parser.add_argument(
        "--full-maintenance",
        action="store_true", 
        help="Run full maintenance (cleanup + fix registry + status)"
    )
    
    args = parser.parse_args()
    
    # Default to status if no arguments
    if not any([args.cleanup, args.fix_registry, args.status, args.full_maintenance]):
        args.status = True
    
    try:
        cleanup = DocumentCleanup()
        
        # Full maintenance mode
        if args.full_maintenance:
            print("🔧 Running full system maintenance...")
            
            # Fix registry issues first
            registry_result = cleanup.fix_registry_issues()
            print(f"✅ Registry fix: {registry_result['re_registered_documents']} documents")
            
            # Clean up old documents (default 7 days)
            cleanup_result = cleanup.cleanup_old_documents(7)
            print(f"✅ Cleanup: {cleanup_result['removed_documents']} documents removed")
            
            # Show final status
            status = cleanup.get_system_status()
            cleanup.print_status_report(status)
            
            return
        
        # Individual operations
        if args.fix_registry:
            result = cleanup.fix_registry_issues()
            print(f"✅ Fixed registry for {result['re_registered_documents']} documents")
        
        if args.cleanup is not None:
            days = args.cleanup if args.cleanup > 0 else 7
            result = cleanup.cleanup_old_documents(days)
            print(f"✅ Removed {result['removed_documents']} documents older than {days} days")
        
        if args.status:
            status = cleanup.get_system_status()
            cleanup.print_status_report(status)
    
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 