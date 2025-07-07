#!/usr/bin/env python3
"""
Migration Script: FAISS to ChromaDB

This script migrates data from the old FAISS-based vector store to the new ChromaDB implementation.
"""

import asyncio
import json
import argparse
from pathlib import Path
from datetime import datetime
import shutil
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import get_config, get_logger
from src.storage import ChromaVectorStore


class FaissToChromaDBMigrator:
    """Handles migration from FAISS to ChromaDB"""
    
    def __init__(self):
        """Initialize the migrator"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        self.chroma_store = ChromaVectorStore()
        
    async def migrate_from_faiss(self, faiss_data_dir: Path, backup_dir: Path = None) -> dict:
        """Migrate data from FAISS to ChromaDB"""
        try:
            self.logger.info("Starting migration from FAISS to ChromaDB")
            
            # Create backup if requested
            if backup_dir:
                self.logger.info(f"Creating backup at {backup_dir}")
                self._create_backup(faiss_data_dir, backup_dir)
            
            # Initialize ChromaDB
            await self.chroma_store.initialize()
            
            # Perform migration
            migration_result = await self.chroma_store.migrate_from_faiss_data(faiss_data_dir)
            
            # Get final stats
            final_stats = self.chroma_store.get_stats()
            
            # Return comprehensive result
            return {
                "migration_result": migration_result,
                "final_stats": final_stats,
                "timestamp": datetime.utcnow().isoformat(),
                "success": "error" not in migration_result
            }
            
        except Exception as e:
            self.logger.error(f"Error during migration: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "success": False
            }
    
    def _create_backup(self, source_dir: Path, backup_dir: Path):
        """Create a backup of the FAISS data"""
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy FAISS data
            if source_dir.exists():
                backup_faiss_dir = backup_dir / "faiss_data"
                shutil.copytree(source_dir, backup_faiss_dir, dirs_exist_ok=True)
                
                # Create backup manifest
                manifest = {
                    "backup_date": datetime.utcnow().isoformat(),
                    "source_directory": str(source_dir),
                    "backup_directory": str(backup_faiss_dir),
                    "migration_type": "faiss_to_chromadb"
                }
                
                manifest_file = backup_dir / "backup_manifest.json"
                with open(manifest_file, 'w') as f:
                    json.dump(manifest, f, indent=2)
                
                self.logger.info(f"Backup created successfully at {backup_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            raise
    
    def check_faiss_data_exists(self, faiss_data_dir: Path) -> bool:
        """Check if FAISS data exists"""
        metadata_file = faiss_data_dir / "metadata.json"
        index_file = faiss_data_dir / "vector_index.faiss"
        
        return metadata_file.exists() and index_file.exists()
    
    def get_faiss_stats(self, faiss_data_dir: Path) -> dict:
        """Get statistics about FAISS data"""
        try:
            metadata_file = faiss_data_dir / "metadata.json"
            
            if not metadata_file.exists():
                return {"error": "No FAISS metadata file found"}
            
            with open(metadata_file, 'r') as f:
                data = json.load(f)
            
            metadata = data.get("metadata", {})
            
            total_documents = len(metadata)
            total_chunks = sum(
                len(doc_meta.get("chunks", {}))
                for doc_meta in metadata.values()
            )
            
            return {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "last_saved": data.get("last_saved", "unknown"),
                "id_to_faiss_index_count": len(data.get("id_to_faiss_index", {}))
            }
            
        except Exception as e:
            return {"error": str(e)}


async def main():
    """Main migration function"""
    parser = argparse.ArgumentParser(
        description="Migrate from FAISS to ChromaDB vector storage"
    )
    parser.add_argument(
        "--faiss-data-dir",
        type=Path,
        default=Path("./data/index"),
        help="Path to FAISS data directory (default: ./data/index)"
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        help="Directory to create backup (optional)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check FAISS data without migrating"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force migration even if ChromaDB already has data"
    )
    
    args = parser.parse_args()
    
    migrator = FaissToChromaDBMigrator()
    
    # Check if FAISS data exists
    if not migrator.check_faiss_data_exists(args.faiss_data_dir):
        print(f"❌ No FAISS data found at {args.faiss_data_dir}")
        print("   Make sure the path is correct and contains metadata.json and vector_index.faiss")
        return
    
    # Get FAISS stats
    faiss_stats = migrator.get_faiss_stats(args.faiss_data_dir)
    print(f"📊 FAISS Data Summary:")
    print(f"   - Documents: {faiss_stats.get('total_documents', 0)}")
    print(f"   - Chunks: {faiss_stats.get('total_chunks', 0)}")
    print(f"   - Last saved: {faiss_stats.get('last_saved', 'unknown')}")
    
    if args.check_only:
        return
    
    # Check if ChromaDB already has data
    try:
        await migrator.chroma_store.initialize()
        chroma_stats = migrator.chroma_store.get_stats()
        
        if chroma_stats.get("total_documents", 0) > 0 and not args.force:
            print(f"⚠️  ChromaDB already contains {chroma_stats['total_documents']} documents")
            print("   Use --force to proceed anyway (this will add to existing data)")
            response = input("Continue? (y/N): ").strip().lower()
            if response != 'y':
                print("Migration cancelled")
                return
    except Exception as e:
        print(f"⚠️  Could not check ChromaDB status: {e}")
        print("Proceeding with migration...")
    
    # Set up backup directory
    backup_dir = None
    if args.backup_dir:
        backup_dir = args.backup_dir
    else:
        # Create default backup directory
        backup_dir = Path("./backups") / f"faiss_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"🔄 Starting migration...")
    print(f"   - Source: {args.faiss_data_dir}")
    print(f"   - Backup: {backup_dir}")
    print(f"   - Target: ChromaDB")
    
    # Perform migration
    result = await migrator.migrate_from_faiss(args.faiss_data_dir, backup_dir)
    
    if result["success"]:
        migration_result = result["migration_result"]
        final_stats = result["final_stats"]
        
        print(f"✅ Migration completed successfully!")
        print(f"   - Documents migrated: {migration_result['documents_migrated']}")
        print(f"   - Chunks migrated: {migration_result['chunks_migrated']}")
        
        if migration_result.get("errors"):
            print(f"   - Errors: {len(migration_result['errors'])}")
            for error in migration_result["errors"][:5]:  # Show first 5 errors
                print(f"     • {error}")
        
        print(f"📊 Final ChromaDB Stats:")
        print(f"   - Total documents: {final_stats.get('total_documents', 0)}")
        print(f"   - Total vectors: {final_stats.get('total_vectors', 0)}")
        print(f"   - Collection: {final_stats.get('collection_name', 'unknown')}")
        
        print(f"📁 Backup created at: {backup_dir}")
        
    else:
        print(f"❌ Migration failed: {result.get('error', 'Unknown error')}")
        return
    
    # Clean up old FAISS files (optional)
    response = input("\n🗑️  Remove old FAISS files? (y/N): ").strip().lower()
    if response == 'y':
        try:
            # Move instead of delete for safety
            old_faiss_dir = Path("./data/old_faiss_data")
            if old_faiss_dir.exists():
                shutil.rmtree(old_faiss_dir)
            
            shutil.move(str(args.faiss_data_dir), str(old_faiss_dir))
            print(f"✅ Old FAISS files moved to {old_faiss_dir}")
            print("   (You can safely delete this directory after verifying the migration)")
            
        except Exception as e:
            print(f"❌ Error moving old files: {e}")


if __name__ == "__main__":
    asyncio.run(main())