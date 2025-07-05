#!/usr/bin/env python3
"""
Script to enhance weak artifacts in training data to match real-world artifact strength
"""
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.audio_utils import load_audio, save_audio
from pathlib import Path
import random

def enhance_training_artifacts(noisy_dir, clean_dir, output_dir, amplification_factor=50):
    """
    Enhance weak artifacts in training data by amplifying the differences
    
    Args:
        noisy_dir: Directory with noisy training files
        clean_dir: Directory with clean training files  
        output_dir: Output directory for enhanced training pairs
        amplification_factor: How much to amplify the artifacts
    """
    
    noisy_path = Path(noisy_dir)
    clean_path = Path(clean_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    (output_path / 'noisy').mkdir(parents=True, exist_ok=True)
    (output_path / 'clean').mkdir(parents=True, exist_ok=True)
    
    noisy_files = list(noisy_path.glob('*.wav'))
    print(f"Found {len(noisy_files)} noisy files")
    
    enhanced_count = 0
    
    for noisy_file in noisy_files:
        clean_file = clean_path / noisy_file.name
        
        if not clean_file.exists():
            print(f"Warning: No matching clean file for {noisy_file.name}")
            continue
            
        try:
            # Load audio pairs
            noisy_audio, sr = load_audio(str(noisy_file), 24000)
            clean_audio, _ = load_audio(str(clean_file), 24000)
            
            # Calculate the difference (artifacts)
            artifact = noisy_audio - clean_audio
            
            # Check if there are any artifacts
            artifact_rms = torch.sqrt(torch.mean(artifact ** 2))
            if artifact_rms < 1e-6:
                print(f"Skipping {noisy_file.name} - no artifacts detected")
                continue
            
            # Amplify the artifacts
            enhanced_artifact = artifact * amplification_factor
            
            # Create new noisy audio with enhanced artifacts
            enhanced_noisy = clean_audio + enhanced_artifact
            
            # Clip to prevent distortion
            enhanced_noisy = torch.clamp(enhanced_noisy, -0.95, 0.95)
            
            # Save enhanced pair
            save_audio(enhanced_noisy, str(output_path / 'noisy' / noisy_file.name), sr)
            save_audio(clean_audio, str(output_path / 'clean' / noisy_file.name), sr)
            
            enhanced_count += 1
            
            if enhanced_count <= 3:  # Show details for first 3 files
                new_artifact_rms = torch.sqrt(torch.mean((enhanced_noisy - clean_audio) ** 2))
                print(f"Enhanced {noisy_file.name}:")
                print(f"  Original artifact RMS: {artifact_rms:.6f}")
                print(f"  Enhanced artifact RMS: {new_artifact_rms:.6f}")
                print(f"  Amplification: {new_artifact_rms/artifact_rms:.1f}x")
                
        except Exception as e:
            print(f"Error processing {noisy_file.name}: {e}")
            continue
    
    print(f"\n✅ Enhanced {enhanced_count} training pairs")
    print(f"📁 Output: {output_path}")
    return enhanced_count

def create_mixed_training_data():
    """Create training data with both weak and strong artifacts"""
    
    print("🔧 Creating enhanced training data with stronger artifacts")
    print("=" * 60)
    
    # Process test data (smaller set for quick testing)
    enhanced_count = enhance_training_artifacts(
        'data/processed/test/noisy',
        'data/processed/test/clean', 
        'data/enhanced_training_test',
        amplification_factor=50  # Make artifacts 50x stronger
    )
    
    if enhanced_count > 0:
        print(f"\n🧪 Test with enhanced training data:")
        print(f"   Original training artifacts: ~0.0004 RMS")
        print(f"   Enhanced training artifacts: ~0.02 RMS (50x stronger)")
        print(f"   Candidate_02 artifacts:      0.061 RMS")
        print(f"   Gap reduced from 151x to 3x!")
        
        # Create a sample pair for manual inspection
        test_files = list(Path('data/enhanced_training_test/noisy').glob('*.wav'))
        if test_files:
            test_file = test_files[0]
            print(f"\n🔍 Inspect sample: {test_file.name}")
            
            # Load and analyze
            enhanced_noisy, _ = load_audio(str(test_file), 24000)
            clean_file = Path('data/enhanced_training_test/clean') / test_file.name
            clean_audio, _ = load_audio(str(clean_file), 24000)
            
            difference = enhanced_noisy - clean_audio
            diff_rms = torch.sqrt(torch.mean(difference ** 2))
            
            print(f"   Enhanced artifact strength: {diff_rms:.6f} RMS")
            print(f"   Target (candidate_02): 0.061204 RMS")
            print(f"   Ratio: {0.061204/diff_rms:.1f}x")
    
    return enhanced_count > 0

if __name__ == '__main__':
    success = create_mixed_training_data()
    
    if success:
        print(f"\n🚀 NEXT STEPS:")
        print(f"1. Listen to enhanced training samples")
        print(f"2. If good: Process full training set") 
        print(f"3. Retrain model with enhanced data")
        print(f"4. Test on candidate_02.wav")
    else:
        print(f"\n❌ No training data found or enhanced") 