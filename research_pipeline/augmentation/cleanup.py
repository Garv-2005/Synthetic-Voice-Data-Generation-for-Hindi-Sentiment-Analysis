#!/usr/bin/env python3
"""
Clean up augmentation folder: remove debug/test files and cache
"""

import os
from pathlib import Path
import shutil

aug_dir = Path(__file__).parent

def cleanup():
    """Remove unnecessary files from augmentation folders"""
    
    files_to_remove = [
        # Classical debug files
        "classical/pipeline_output.txt",
        "classical/training_output.txt", 
        "classical/train_direct.py",
        "classical/__pycache__",
        
        # Transformer debug files
        "transformer/debug_gen.py",
        "transformer/test_mae.py",
        "transformer/__pycache__",
        
        # VAE
        "vae_gan/__pycache__",
        
        # Root
        "__pycache__",
    ]
    
    for file_path in files_to_remove:
        full_path = aug_dir / file_path
        if full_path.exists():
            if full_path.is_dir():
                print(f"Removing directory: {file_path}")
                shutil.rmtree(full_path)
            else:
                print(f"Removing file: {file_path}")
                full_path.unlink()
        else:
            print(f"Not found (skip): {file_path}")
    
    print("\nCleanup complete!")
    print("\nRemaining structure:")
    print("augmentation/")
    for approach in ["classical", "transformer", "vae_gan"]:
        path = aug_dir / approach
        if path.exists():
            print(f"  {approach}/")
            for file in sorted(path.glob("*.py")):
                print(f"    - {file.name}")
            for file in sorted(path.glob("*.md")):
                print(f"    - {file.name}")

if __name__ == "__main__":
    cleanup()
