#!/usr/bin/env python3
"""
Script to copy candidate_XX.wav files from training directories to a unified training dataset.
Renames files to include their source path information for uniqueness.
"""

import os
import shutil
import re
from pathlib import Path
from typing import List, Tuple


def find_candidate_files(source_dir: str) -> List[Path]:
    """
    Find all candidate_XX.wav files in the source directory and its subdirectories.
    
    Args:
        source_dir: Root directory to search in
        
    Returns:
        List of Path objects for found candidate files
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Source directory {source_dir} does not exist!")
        return []
    
    # Find all candidate_XX.wav files recursively
    pattern = "candidate_*.wav"
    candidate_files = list(source_path.rglob(pattern))
    
    print(f"Found {len(candidate_files)} candidate files")
    return candidate_files


def generate_unique_filename(source_file: Path, source_root: Path) -> str:
    """
    Generate a unique filename based on the source file path.
    
    Args:
        source_file: Path to the source file
        source_root: Root directory of the source
        
    Returns:
        Unique filename string
    """
    # Get relative path from source root
    relative_path = source_file.relative_to(source_root)
    
    # Convert path to filename components
    path_parts = relative_path.parts
    
    # Extract the training session name (first directory after source_root)
    if len(path_parts) >= 1:
        training_session = path_parts[0]
    else:
        training_session = "unknown"
    
    # Extract chunk and candidate info
    chunk_info = ""
    candidate_info = ""
    
    for part in path_parts:
        if part.startswith("chunk_"):
            chunk_info = part.replace("_", "-")  # chunk_001 -> chunk-001
        elif part.startswith("candidate_"):
            # Remove .wav extension from candidate name to avoid double extension
            candidate_info = part.replace("_", "-").replace(".wav", "")  # candidate_01.wav -> candidate-01
    
    # Build the new filename
    new_filename = f"{training_session}_{chunk_info}_{candidate_info}.wav"
    
    return new_filename


def copy_candidate_files(source_dir: str, noisy_dir: str, clean_dir: str) -> None:
    """
    Copy all candidate files from source to both noisy and clean directories with unique names.
    
    Args:
        source_dir: Source directory containing training data
        noisy_dir: Target directory for noisy files
        clean_dir: Target directory for clean files
    """
    source_path = Path(source_dir)
    noisy_path = Path(noisy_dir)
    clean_path = Path(clean_dir)
    
    # Create target directories if they don't exist
    noisy_path.mkdir(parents=True, exist_ok=True)
    clean_path.mkdir(parents=True, exist_ok=True)
    
    # Find all candidate files
    candidate_files = find_candidate_files(source_dir)
    
    if not candidate_files:
        print("No candidate files found!")
        return
    
    # Track copied files for reporting
    copied_count = 0
    skipped_count = 0
    
    for source_file in candidate_files:
        # Generate unique filename
        new_filename = generate_unique_filename(source_file, source_path)
        noisy_file = noisy_path / new_filename
        clean_file = clean_path / new_filename
        
        # Check if target files already exist
        if noisy_file.exists() and clean_file.exists():
            print(f"Skipping {source_file.name} -> {new_filename} (already exists in both directories)")
            skipped_count += 1
            continue
        
        try:
            # Copy to noisy directory
            if not noisy_file.exists():
                shutil.copy2(source_file, noisy_file)
                print(f"Copied to noisy: {source_file.name} -> {new_filename}")
            
            # Copy to clean directory
            if not clean_file.exists():
                shutil.copy2(source_file, clean_file)
                print(f"Copied to clean: {source_file.name} -> {new_filename}")
            
            copied_count += 1
            
        except Exception as e:
            print(f"Error copying {source_file}: {e}")
    
    print(f"\nSummary:")
    print(f"  Files copied: {copied_count}")
    print(f"  Files skipped: {skipped_count}")
    print(f"  Total processed: {copied_count + skipped_count}")


def main():
    """Main function to execute the script."""
    # Define source and target directories
    source_directory = "data/output/training"
    noisy_directory = "Chatterbox-CleanUNet/data/raw/noisy"
    clean_directory = "Chatterbox-CleanUNet/data/raw/clean"
    
    print(f"Source directory: {source_directory}")
    print(f"Noisy directory: {noisy_directory}")
    print(f"Clean directory: {clean_directory}")
    print("-" * 50)
    
    # Execute the copy operation
    copy_candidate_files(source_directory, noisy_directory, clean_directory)


if __name__ == "__main__":
    main() 