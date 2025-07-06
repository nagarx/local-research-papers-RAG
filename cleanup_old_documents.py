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
import asyncio
from typing import Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.storage import ChromaVectorStore
from src.tracking import SourceTracker
from src.config import get_config, get_logger


class DocumentCleanupManager:
    """Manages cleanup of old documents"""
    
    def __init__(self):
        """Initialize the cleanup manager"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.vector_store = ChromaVectorStore()
        self.source_tracker = SourceTracker()
        
    async def cleanup_old_documents(self, days_old: int = 7) -> Dict[str, Any]:
        """Clean up documents older than specified days"""
        try:
            self.logger.info(f"Starting cleanup of documents older than {days_old} days")
            
            # Initialize vector store
            await self.vector_store.initialize()
            
            # Get initial stats
            initial_stats = self.vector_store.get_stats()
            self.logger.info(f"Initial stats: {initial_stats}")
            
            # Perform cleanup
            cleanup_result = self.vector_store.cleanup_old_documents(days_old)
            
            # Get final stats
            final_stats = self.vector_store.get_stats()
            self.logger.info(f"Final stats: {final_stats}")
            
            # Return comprehensive result
            return {
                "cleanup_result": cleanup_result,
                "initial_stats": initial_stats,
                "final_stats": final_stats,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
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


async def main():
    """Main cleanup function"""
    parser = argparse.ArgumentParser(description="Clean up old documents from vector store")
    parser.add_argument(
        "--days", 
        type=int, 
        default=7,
        help="Remove documents older than this many days (default: 7)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually removing it"
    )
    
    args = parser.parse_args()
    
    cleanup_manager = DocumentCleanupManager()
    
    if args.dry_run:
        print(f"DRY RUN: Would remove documents older than {args.days} days")
        # TODO: Implement dry run functionality
        return
    
    print(f"Cleaning up documents older than {args.days} days...")
    result = await cleanup_manager.cleanup_old_documents(args.days)
    
    if "error" in result:
        print(f"Error during cleanup: {result['error']}")
        return
    
    cleanup_result = result["cleanup_result"]
    print(f"Cleanup completed:")
    print(f"  - Documents removed: {cleanup_result['removed_documents']}")
    print(f"  - Documents remaining: {cleanup_result['remaining_documents']}")
    print(f"  - Cutoff date: {cleanup_result['cutoff_date']}")


if __name__ == "__main__":
    asyncio.run(main()) 