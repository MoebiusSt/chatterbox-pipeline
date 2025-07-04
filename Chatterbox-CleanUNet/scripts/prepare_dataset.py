#!/usr/bin/env python3
"""
Dataset preparation script for CleanUNet TTS artifact removal
Automatically splits clean/noisy audio pairs into train/validation/test sets
"""
import argparse
import os
import shutil
import random
from pathlib import Path
from typing import List, Tuple
import torchaudio

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Prepare dataset for CleanUNet training')
    
    # Input directories
    parser.add_argument('--clean_dir', required=True, 
                       help='Directory containing clean audio files')
    parser.add_argument('--noisy_dir', required=True, 
                       help='Directory containing noisy audio files')
    
    # Output directory
    parser.add_argument('--output_dir', default='data/processed', 
                       help='Output directory for processed dataset')
    
    # Split ratios
    parser.add_argument('--train_ratio', type=float, default=0.8, 
                       help='Ratio for training set (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1, 
                       help='Ratio for validation set (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1, 
                       help='Ratio for test set (default: 0.1)')
    
    # Audio options
    parser.add_argument('--target_sample_rate', type=int, default=24000, 
                       help='Target sample rate for all audio files')
    parser.add_argument('--file_pattern', default='*.wav', 
                       help='File pattern to match (default: *.wav)')
    
    # Processing options
    parser.add_argument('--copy_files', action='store_true', 
                       help='Copy files instead of creating symlinks')
    parser.add_argument('--validate_pairs', action='store_true', 
                       help='Validate that audio pairs have similar length')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducible splits')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    
    return parser.parse_args()

def find_audio_pairs(clean_dir: Path, noisy_dir: Path, pattern: str = "*.wav") -> List[Tuple[Path, Path]]:
    """Find matching clean/noisy audio file pairs"""
    clean_files = list(clean_dir.glob(pattern))
    pairs = []
    missing_files = []
    
    for clean_file in clean_files:
        noisy_file = noisy_dir / clean_file.name
        if noisy_file.exists():
            pairs.append((clean_file, noisy_file))
        else:
            missing_files.append(clean_file.name)
    
    if missing_files:
        print(f"Warning: {len(missing_files)} clean files have no matching noisy file:")
        for missing in missing_files[:10]:  # Show first 10
            print(f"  - {missing}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    
    return pairs

def validate_audio_pair(clean_path: Path, noisy_path: Path, max_length_diff: float = 0.1) -> bool:
    """Validate that audio pair has compatible properties"""
    try:
        # Load audio metadata
        clean_info = torchaudio.info(clean_path)
        noisy_info = torchaudio.info(noisy_path)
        
        # Check sample rates
        if clean_info.sample_rate != noisy_info.sample_rate:
            print(f"Warning: Sample rate mismatch in {clean_path.name}")
            return True  # Still usable with resampling
        
        # Check length similarity
        clean_duration = clean_info.num_frames / clean_info.sample_rate
        noisy_duration = noisy_info.num_frames / noisy_info.sample_rate
        
        length_diff = abs(clean_duration - noisy_duration) / max(clean_duration, noisy_duration)
        if length_diff > max_length_diff:
            print(f"Warning: Length mismatch in {clean_path.name} ({length_diff:.2%} difference)")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error validating {clean_path.name}: {e}")
        return False

def create_directory_structure(output_dir: Path):
    """Create the dataset directory structure"""
    subdirs = [
        'train/clean', 'train/noisy',
        'validation/clean', 'validation/noisy', 
        'test/clean', 'test/noisy'
    ]
    
    for subdir in subdirs:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

def copy_or_link_file(src: Path, dst: Path, copy_files: bool):
    """Copy or create symlink for file"""
    if copy_files:
        shutil.copy2(src, dst)
    else:
        # Create relative symlink if possible
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            # Fallback to copying if symlinks not supported
            shutil.copy2(src, dst)

def split_dataset(pairs: List[Tuple[Path, Path]], train_ratio: float, 
                 val_ratio: float, test_ratio: float, seed: int) -> Tuple[List, List, List]:
    """Split audio pairs into train/validation/test sets"""
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Shuffle pairs deterministically
    random.seed(seed)
    shuffled_pairs = pairs.copy()
    random.shuffle(shuffled_pairs)
    
    # Calculate split indices
    total = len(shuffled_pairs)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    # Split
    train_pairs = shuffled_pairs[:train_end]
    val_pairs = shuffled_pairs[train_end:val_end]
    test_pairs = shuffled_pairs[val_end:]
    
    return train_pairs, val_pairs, test_pairs

def process_split(pairs: List[Tuple[Path, Path]], output_dir: Path, 
                 split_name: str, copy_files: bool, verbose: bool):
    """Process a single split (train/val/test)"""
    if verbose:
        print(f"Processing {split_name} split ({len(pairs)} pairs)...")
    
    clean_dir = output_dir / split_name / 'clean'
    noisy_dir = output_dir / split_name / 'noisy'
    
    successful = 0
    failed = 0
    
    for clean_path, noisy_path in pairs:
        try:
            clean_dst = clean_dir / clean_path.name
            noisy_dst = noisy_dir / noisy_path.name
            
            copy_or_link_file(clean_path, clean_dst, copy_files)
            copy_or_link_file(noisy_path, noisy_dst, copy_files)
            
            successful += 1
            
        except Exception as e:
            print(f"Error processing {clean_path.name}: {e}")
            failed += 1
    
    if verbose:
        print(f"  Successfully processed: {successful}")
        if failed > 0:
            print(f"  Failed: {failed}")

def create_dataset_info(output_dir: Path, args, total_pairs: int, 
                       train_count: int, val_count: int, test_count: int):
    """Create dataset information file"""
    info = {
        'dataset_info': {
            'total_pairs': total_pairs,
            'train_pairs': train_count,
            'validation_pairs': val_count,
            'test_pairs': test_count,
            'target_sample_rate': args.target_sample_rate,
            'file_pattern': args.file_pattern,
            'random_seed': args.seed,
            'ratios': {
                'train': args.train_ratio,
                'validation': args.val_ratio,
                'test': args.test_ratio
            }
        },
        'source_directories': {
            'clean_dir': str(Path(args.clean_dir).resolve()),
            'noisy_dir': str(Path(args.noisy_dir).resolve())
        }
    }
    
    info_file = output_dir / 'dataset_info.txt'
    with open(info_file, 'w') as f:
        f.write("CleanUNet Dataset Information\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total audio pairs: {total_pairs}\n")
        f.write(f"Training pairs: {train_count} ({train_count/total_pairs:.1%})\n")
        f.write(f"Validation pairs: {val_count} ({val_count/total_pairs:.1%})\n")
        f.write(f"Test pairs: {test_count} ({test_count/total_pairs:.1%})\n\n")
        f.write(f"Target sample rate: {args.target_sample_rate} Hz\n")
        f.write(f"File pattern: {args.file_pattern}\n")
        f.write(f"Random seed: {args.seed}\n\n")
        f.write(f"Source directories:\n")
        f.write(f"  Clean: {args.clean_dir}\n")
        f.write(f"  Noisy: {args.noisy_dir}\n")

def main():
    """Main dataset preparation function"""
    args = parse_args()
    
    # Validate input directories
    clean_dir = Path(args.clean_dir)
    noisy_dir = Path(args.noisy_dir)
    output_dir = Path(args.output_dir)
    
    if not clean_dir.exists():
        raise FileNotFoundError(f"Clean directory does not exist: {clean_dir}")
    if not noisy_dir.exists():
        raise FileNotFoundError(f"Noisy directory does not exist: {noisy_dir}")
    
    # Find audio pairs
    print(f"Searching for audio pairs...")
    print(f"  Clean directory: {clean_dir}")
    print(f"  Noisy directory: {noisy_dir}")
    print(f"  File pattern: {args.file_pattern}")
    
    pairs = find_audio_pairs(clean_dir, noisy_dir, args.file_pattern)
    
    if len(pairs) == 0:
        raise ValueError("No matching audio pairs found!")
    
    print(f"Found {len(pairs)} audio pairs")
    
    # Validate pairs if requested
    if args.validate_pairs:
        print("Validating audio pairs...")
        valid_pairs = []
        for clean_path, noisy_path in pairs:
            if validate_audio_pair(clean_path, noisy_path):
                valid_pairs.append((clean_path, noisy_path))
        
        if len(valid_pairs) < len(pairs):
            print(f"Removed {len(pairs) - len(valid_pairs)} invalid pairs")
            pairs = valid_pairs
    
    # Split dataset
    print(f"\nSplitting dataset...")
    print(f"  Train: {args.train_ratio:.1%}")
    print(f"  Validation: {args.val_ratio:.1%}")
    print(f"  Test: {args.test_ratio:.1%}")
    
    train_pairs, val_pairs, test_pairs = split_dataset(
        pairs, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(train_pairs)} pairs")
    print(f"  Validation: {len(val_pairs)} pairs")
    print(f"  Test: {len(test_pairs)} pairs")
    
    # Create directory structure
    print(f"\nCreating directory structure in: {output_dir}")
    create_directory_structure(output_dir)
    
    # Process splits
    process_split(train_pairs, output_dir, 'train', args.copy_files, args.verbose)
    process_split(val_pairs, output_dir, 'validation', args.copy_files, args.verbose)
    process_split(test_pairs, output_dir, 'test', args.copy_files, args.verbose)
    
    # Create dataset info
    create_dataset_info(output_dir, args, len(pairs), 
                       len(train_pairs), len(val_pairs), len(test_pairs))
    
    print(f"\n✅ Dataset preparation completed successfully!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Dataset info saved to: {output_dir}/dataset_info.txt")
    print(f"\n🚀 Ready for training:")
    print(f"   python scripts/train.py --train_data {output_dir}/train --val_data {output_dir}/validation")

if __name__ == '__main__':
    main() 