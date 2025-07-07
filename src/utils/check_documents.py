#!/usr/bin/env python3
"""
Document Status Checker Utility

This module provides a convenient interface to check the status of processed 
and indexed documents in the RAG pipeline.
"""

import sys
from pathlib import Path

# Add the project root to the Python path if needed
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from .document_status import DocumentStatusChecker


def main():
    """Main function to check document status with comprehensive reporting"""
    print("🔍 Checking document status...")
    print()
    
    try:
        # Create status checker and print report
        checker = DocumentStatusChecker()
        checker.print_documents_status(show_details=True)
        
        # Storage usage report
        usage = checker.get_storage_usage()
        if "error" not in usage:
            print("\n💾 Storage Usage:")
            print("-" * 30)
            print(f"Processed Files: {usage['processed_dir'] / 1024 / 1024:.1f} MB ({usage['file_counts']['processed']} files)")
            print(f"ChromaDB: {usage['chroma_dir'] / 1024 / 1024:.1f} MB ({usage['file_counts']['chroma']} files)")
            print(f"Embeddings: {usage['embeddings_dir'] / 1024 / 1024:.1f} MB ({usage['file_counts']['embeddings']} files)")
            print(f"Metadata: {usage['file_counts']['metadata']} files")
            print(f"Total: {usage['total_size'] / 1024 / 1024:.1f} MB")
        
        # Duplicate detection report
        duplicate_report = checker.get_duplicate_detection_report()
        if "error" not in duplicate_report and duplicate_report['total_documents'] > 0:
            print("\n🔍 Duplicate Detection Report:")
            print("-" * 30)
            print(f"Documents with Hash: {duplicate_report['documents_with_hash']}")
            print(f"Unique Hashes: {duplicate_report['unique_hashes']}")
            
            if duplicate_report['duplicate_groups'] > 0:
                print(f"⚠️  Duplicate Groups: {duplicate_report['duplicate_groups']}")
                print(f"⚠️  Duplicate Documents: {duplicate_report['duplicate_documents']}")
                
                # Show some duplicate examples
                print("\n🔗 Duplicate Examples:")
                for i, (hash_val, docs) in enumerate(duplicate_report['duplicates'].items()):
                    if i >= 3:  # Limit to first 3 examples
                        break
                    print(f"   Hash: {hash_val[:12]}... ({len(docs)} copies)")
                    for doc in docs:
                        print(f"     - {doc['filename']}")
            else:
                print("✅ No duplicates detected")
        
        # Pipeline health check
        status = checker.get_all_documents_status()
        if "error" not in status and status['total_documents'] > 0:
            print("\n🏥 Pipeline Health Check:")
            print("-" * 30)
            
            # Check for documents that might be stuck in processing
            processed_only = status['processed_only']
            indexed_only = status['indexed_only']
            
            if processed_only > 0:
                print(f"⚠️  {processed_only} documents processed but not indexed")
            
            if indexed_only > 0:
                print(f"⚠️  {indexed_only} documents indexed but not in processed cache")
            
            complete_pipeline = status['processed_and_indexed'] + status['all_statuses']
            total = status['total_documents']
            
            if complete_pipeline == total:
                print("✅ All documents are properly processed and indexed")
            else:
                incomplete = total - complete_pipeline
                print(f"⚠️  {incomplete} documents have incomplete processing")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you're running this from the project root directory.")


def check_specific_document(filename: str = None, content_hash: str = None):
    """Check if a specific document exists by filename or content hash"""
    try:
        checker = DocumentStatusChecker()
        
        if filename:
            result = checker.check_document_exists(filename)
            if result:
                print(f"✅ Document '{filename}' found:")
                print(f"   ID: {result['document_id']}")
                print(f"   Status: {' + '.join(result['status'])}")
                print(f"   Processed: {result.get('processed_at', 'unknown')}")
                return result
            else:
                print(f"❌ Document '{filename}' not found")
                return None
        
        elif content_hash:
            result = checker.check_document_exists_by_hash(content_hash)
            if result:
                print(f"✅ Document with hash '{content_hash[:12]}...' found:")
                print(f"   Filename: {result['filename']}")
                print(f"   ID: {result['document_id']}")
                print(f"   Status: {' + '.join(result['status'])}")
                return result
            else:
                print(f"❌ Document with hash '{content_hash[:12]}...' not found")
                return None
        
        else:
            print("❌ Please provide either filename or content_hash")
            return None
            
    except Exception as e:
        print(f"❌ Error checking document: {e}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check document status in RAG pipeline")
    parser.add_argument("--filename", help="Check specific document by filename")
    parser.add_argument("--hash", help="Check specific document by content hash")
    parser.add_argument("--full-report", action="store_true", help="Show full comprehensive report")
    
    args = parser.parse_args()
    
    if args.filename or args.hash:
        check_specific_document(filename=args.filename, content_hash=args.hash)
    else:
        main()
        
        if args.full_report:
            # Additional detailed reporting
            try:
                checker = DocumentStatusChecker()
                status = checker.get_all_documents_status()
                
                if "error" not in status:
                    print("\n📈 Detailed Statistics:")
                    print("-" * 30)
                    
                    total_chunks = sum(doc.get('total_chunks', 0) for doc in status['documents'])
                    total_text = sum(doc.get('text_length', 0) for doc in status['documents'])
                    total_images = sum(doc.get('image_count', 0) for doc in status['documents'])
                    
                    print(f"Total Chunks: {total_chunks:,}")
                    print(f"Total Text: {total_text:,} characters")
                    print(f"Total Images: {total_images}")
                    
                    if status['documents']:
                        avg_chunks = total_chunks / len(status['documents'])
                        avg_text = total_text / len(status['documents'])
                        print(f"Avg Chunks/Doc: {avg_chunks:.1f}")
                        print(f"Avg Text/Doc: {avg_text:,.0f} chars")
                        
                        # Show format breakdown
                        formats = {}
                        for doc in status['documents']:
                            fmt = doc.get('extraction_format', 'unknown')
                            formats[fmt] = formats.get(fmt, 0) + 1
                        
                        if formats:
                            print(f"\nFormat Breakdown:")
                            for fmt, count in formats.items():
                                print(f"   {fmt}: {count} documents")
                    
            except Exception as e:
                print(f"❌ Error generating detailed statistics: {e}") 