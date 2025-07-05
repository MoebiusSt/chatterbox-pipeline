#!/usr/bin/env python3
"""
Filter training data to keep only pairs with significant artifacts.
This script helps improve model training by focusing on clear examples.
"""
import argparse
import os
import shutil
from pathlib import Path
import torch
import torchaudio
import numpy as np
from tqdm import tqdm
from typing import Tuple, List

def compute_spectral_distance(clean_audio: torch.Tensor, noisy_audio: torch.Tensor) -> float:
    """Compute spectral distance between clean and noisy audio"""
    # Ensure same length
    min_length = min(clean_audio.shape[1], noisy_audio.shape[1])
    clean_audio = clean_audio[:, :min_length]
    noisy_audio = noisy_audio[:, :min_length]
    
    # Compute STFT
    clean_stft = torch.stft(clean_audio.squeeze(), n_fft=1024, hop_length=256, return_complex=True)
    noisy_stft = torch.stft(noisy_audio.squeeze(), n_fft=1024, hop_length=256, return_complex=True)
    
    # Compute magnitude spectrograms
    clean_mag = torch.abs(clean_stft)
    noisy_mag = torch.abs(noisy_stft)
    
    # Compute mean squared difference
    spectral_diff = torch.mean((clean_mag - noisy_mag) ** 2)
    return spectral_diff.item()

def compute_rms_difference(clean_audio: torch.Tensor, noisy_audio: torch.Tensor) -> float:
    """Compute RMS difference between clean and noisy audio"""
    # Ensure same length
    min_length = min(clean_audio.shape[1], noisy_audio.shape[1])
    clean_audio = clean_audio[:, :min_length]
    noisy_audio = noisy_audio[:, :min_length]
    
    # Compute RMS
    clean_rms = torch.sqrt(torch.mean(clean_audio ** 2))
    noisy_rms = torch.sqrt(torch.mean(noisy_audio ** 2))
    
    # Compute relative difference
    if clean_rms > 0:
        return abs(clean_rms - noisy_rms) / clean_rms
    return 0.0

def compute_artifact_score(clean_audio: torch.Tensor, noisy_audio: torch.Tensor) -> float:
    """Compute overall artifact score for audio pair"""
    # Ensure same length
    min_length = min(clean_audio.shape[1], noisy_audio.shape[1])
    clean_audio = clean_audio[:, :min_length]
    noisy_audio = noisy_audio[:, :min_length]
    
    # Compute difference signal
    diff_signal = noisy_audio - clean_audio
    
    # Compute energy of difference signal
    diff_energy = torch.mean(diff_signal ** 2)
    
    # Compute energy of clean signal
    clean_energy = torch.mean(clean_audio ** 2)
    
    # Return relative artifact energy
    if clean_energy > 0:
        return (diff_energy / clean_energy).item()
    return 0.0

def analyze_audio_pair(clean_path: Path, noisy_path: Path) -> Tuple[float, float, float]:
    """Analyze an audio pair and return artifact metrics"""
    try:
        # Load audio files
        clean_audio, _ = torchaudio.load(clean_path)
        noisy_audio, _ = torchaudio.load(noisy_path)
        
        # Convert to mono if necessary
        if clean_audio.shape[0] > 1:
            clean_audio = clean_audio.mean(dim=0, keepdim=True)
        if noisy_audio.shape[0] > 1:
            noisy_audio = noisy_audio.mean(dim=0, keepdim=True)
        
        # Compute metrics
        artifact_score = compute_artifact_score(clean_audio, noisy_audio)
        spectral_distance = compute_spectral_distance(clean_audio, noisy_audio)
        rms_difference = compute_rms_difference(clean_audio, noisy_audio)
        
        return artifact_score, spectral_distance, rms_difference
        
    except Exception as e:
        print(f"Error analyzing {clean_path}: {e}")
        return 0.0, 0.0, 0.0

def filter_training_data(clean_dir: Path, noisy_dir: Path, output_dir: Path, 
                        threshold: float = 0.01, min_duration: float = 1.0) -> List[Tuple[Path, Path]]:
    """Filter training data based on artifact score"""
    
    # Find all audio pairs
    clean_files = list(clean_dir.glob("*.wav"))
    pairs = []
    
    for clean_file in clean_files:
        noisy_file = noisy_dir / clean_file.name
        if noisy_file.exists():
            pairs.append((clean_file, noisy_file))
    
    print(f"Found {len(pairs)} audio pairs")
    
    # Analyze all pairs
    filtered_pairs = []
    artifact_scores = []
    
    for clean_path, noisy_path in tqdm(pairs, desc="Analyzing pairs"):
        artifact_score, spectral_distance, rms_difference = analyze_audio_pair(clean_path, noisy_path)
        
        # Check duration
        try:
            info = torchaudio.info(clean_path)
            duration = info.num_frames / info.sample_rate
            
            if duration < min_duration:
                continue
                
        except Exception:
            continue
        
        artifact_scores.append(artifact_score)
        
        # Filter based on threshold
        if artifact_score >= threshold:
            filtered_pairs.append((clean_path, noisy_path))
    
    # Print statistics
    print(f"\nArtifact Score Statistics:")
    print(f"Mean: {np.mean(artifact_scores):.4f}")
    print(f"Median: {np.median(artifact_scores):.4f}")
    print(f"Std: {np.std(artifact_scores):.4f}")
    print(f"Min: {np.min(artifact_scores):.4f}")
    print(f"Max: {np.max(artifact_scores):.4f}")
    
    print(f"\nFiltered {len(filtered_pairs)} pairs out of {len(pairs)} total")
    print(f"Kept {len(filtered_pairs)/len(pairs)*100:.1f}% of original data")
    
    return filtered_pairs

def copy_filtered_data(filtered_pairs: List[Tuple[Path, Path]], output_dir: Path):
    """Copy filtered pairs to output directory"""
    clean_output = output_dir / "clean"
    noisy_output = output_dir / "noisy"
    
    clean_output.mkdir(parents=True, exist_ok=True)
    noisy_output.mkdir(parents=True, exist_ok=True)
    
    for clean_path, noisy_path in tqdm(filtered_pairs, desc="Copying files"):
        # Copy clean file
        clean_dest = clean_output / clean_path.name
        shutil.copy2(clean_path, clean_dest)
        
        # Copy noisy file
        noisy_dest = noisy_output / noisy_path.name
        shutil.copy2(noisy_path, noisy_dest)

def main():
    parser = argparse.ArgumentParser(description='Filter training data for significant artifacts')
    parser.add_argument('--clean_dir', required=True, help='Directory with clean audio files')
    parser.add_argument('--noisy_dir', required=True, help='Directory with noisy audio files')
    parser.add_argument('--output_dir', required=True, help='Output directory for filtered data')
    parser.add_argument('--threshold', type=float, default=0.01, 
                       help='Minimum artifact score threshold (default: 0.01)')
    parser.add_argument('--min_duration', type=float, default=1.0,
                       help='Minimum audio duration in seconds (default: 1.0)')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze data, do not copy files')
    
    args = parser.parse_args()
    
    clean_dir = Path(args.clean_dir)
    noisy_dir = Path(args.noisy_dir)
    output_dir = Path(args.output_dir)
    
    # Validate input directories
    if not clean_dir.exists():
        raise FileNotFoundError(f"Clean directory not found: {clean_dir}")
    if not noisy_dir.exists():
        raise FileNotFoundError(f"Noisy directory not found: {noisy_dir}")
    
    # Filter data
    filtered_pairs = filter_training_data(
        clean_dir, noisy_dir, output_dir, 
        threshold=args.threshold, min_duration=args.min_duration
    )
    
    # Copy filtered data (unless analyze_only)
    if not args.analyze_only:
        copy_filtered_data(filtered_pairs, output_dir)
        print(f"\nFiltered data saved to: {output_dir}")
    
    print(f"\nRecommendation:")
    print(f"- Use threshold >= {np.percentile([analyze_audio_pair(p[0], p[1])[0] for p in filtered_pairs[:50]], 75):.4f} for high-quality artifacts")
    print(f"- Use threshold >= {np.percentile([analyze_audio_pair(p[0], p[1])[0] for p in filtered_pairs[:50]], 50):.4f} for medium-quality artifacts")

if __name__ == "__main__":
    main() 