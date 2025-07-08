#!/usr/bin/env python3
"""
Reset ChromaDB Database

This utility script resets the ChromaDB database to fix schema compatibility issues.
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Optional

from ..config import get_config, get_logger


def reset_chroma_database(force: bool = False) -> bool:
    """
    Reset the ChromaDB database by removing all existing data.
    
    Args:
        force: If True, skip confirmation prompts
        
    Returns:
        True if reset was successful, False otherwise
    """
    try:
        config = get_config()
        logger = get_logger(__name__)
        
        # Get ChromaDB persist directory
        chroma_dir = Path(config.vector_storage.persist_directory)
        
        if not chroma_dir.exists():
            logger.info(f"ChromaDB directory {chroma_dir} does not exist - nothing to reset")
            return True
        
        # Check if user wants to proceed
        if not force:
            print(f"⚠️  WARNING: This will delete all vector data in {chroma_dir}")
            print("All uploaded documents will need to be re-processed.")
            response = input("Are you sure you want to continue? (y/N): ")
            if response.lower() != 'y':
                print("Reset cancelled.")
                return False
        
        # Reset the database
        logger.info(f"Resetting ChromaDB database at {chroma_dir}")
        
        # Remove the entire ChromaDB directory
        shutil.rmtree(chroma_dir)
        
        # Recreate the directory
        chroma_dir.mkdir(parents=True, exist_ok=True)
        
        # Also reset document metadata if it exists
        metadata_file = chroma_dir / "document_metadata.json"
        if metadata_file.exists():
            metadata_file.unlink()
        
        logger.info("ChromaDB database reset successfully")
        print("✅ ChromaDB database reset successfully")
        print("📝 Note: All documents will need to be re-uploaded and processed")
        
        return True
        
    except Exception as e:
        logger.error(f"Error resetting ChromaDB database: {e}")
        print(f"❌ Error resetting database: {e}")
        return False


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reset ChromaDB database")
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Skip confirmation prompts"
    )
    
    args = parser.parse_args()
    
    success = reset_chroma_database(force=args.force)
    exit(0 if success else 1)


if __name__ == "__main__":
    main() 