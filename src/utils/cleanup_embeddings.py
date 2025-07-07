#!/usr/bin/env python3
"""
Cleanup temporary embedding files that failed to be renamed
"""

import os
import glob
from pathlib import Path
from typing import List


def cleanup_temp_embedding_files(embeddings_dir: str = "data/embeddings") -> dict:
    """
    Clean up temporary embedding files that failed to be renamed
    
    Args:
        embeddings_dir: Directory containing embedding files
        
    Returns:
        Dictionary with cleanup results
    """
    results = {
        'cleaned': 0,
        'errors': [],
        'kept': 0
    }
    
    embeddings_path = Path(embeddings_dir)
    
    if not embeddings_path.exists():
        results['errors'].append(f"Embeddings directory {embeddings_dir} does not exist")
        return results
    
    # Find all temporary files
    temp_files = list(embeddings_path.glob("*.tmp.npy"))
    
    print(f"Found {len(temp_files)} temporary embedding files to clean up")
    
    for temp_file in temp_files:
        try:
            # Extract the hash from the filename
            hash_part = temp_file.stem.replace('.tmp', '')
            
            # Check if final file already exists
            final_file = embeddings_path / f"{hash_part}.npy"
            
            if final_file.exists():
                # Final file exists, safe to remove temp file
                temp_file.unlink()
                results['cleaned'] += 1
                print(f"Removed temporary file: {temp_file.name}")
            else:
                # Try to rename temp file to final file
                try:
                    temp_file.rename(final_file)
                    results['kept'] += 1
                    print(f"Renamed temporary file: {temp_file.name} -> {final_file.name}")
                except Exception as e:
                    results['errors'].append(f"Failed to rename {temp_file.name}: {e}")
                    print(f"Failed to rename {temp_file.name}: {e}")
                    
        except Exception as e:
            results['errors'].append(f"Error processing {temp_file.name}: {e}")
            print(f"Error processing {temp_file.name}: {e}")
    
    return results


def main():
    """Main cleanup function"""
    print("🧹 Cleaning up temporary embedding files...")
    
    results = cleanup_temp_embedding_files()
    
    print(f"\n✅ Cleanup complete!")
    print(f"📊 Results:")
    print(f"  - Cleaned up: {results['cleaned']} files")
    print(f"  - Kept/Renamed: {results['kept']} files")
    print(f"  - Errors: {len(results['errors'])} errors")
    
    if results['errors']:
        print(f"\n❌ Errors:")
        for error in results['errors']:
            print(f"  - {error}")


if __name__ == "__main__":
    main() 