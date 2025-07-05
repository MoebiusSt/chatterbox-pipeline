#!/usr/bin/env python3
"""
Enhanced audio using SimpleUNet model
"""
import argparse
import torch
import os
import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.simple_unet import SimpleUNet
from utils.audio_utils import load_audio, save_audio, normalize_audio

def load_model(model_path, device):
    """Load trained SimpleUNet model"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        # Default config
        model_config = {
            'sample_rate': 24000,
            'n_fft': 1024,
            'hop_length': 256,
            'win_length': 1024,
            'crop_length': 48000,
            'l1_weight': 0.8,
            'stft_weight': 0.2
        }
    
    # Create model
    model = SimpleUNet(model_config)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model, model_config

def enhance_audio(model, audio, chunk_size=48000, overlap=0.25, device='cuda'):
    """Enhance audio using the model"""
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    original_length = audio.shape[1]
    
    # Process in chunks if audio is too long
    if original_length <= chunk_size:
        # Process entire audio
        audio = audio.to(device)
        
        # Pad if necessary
        if audio.shape[1] < chunk_size:
            pad_length = chunk_size - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad_length))
        
        with torch.no_grad():
            enhanced = model(audio.unsqueeze(0))  # Add batch dimension
            enhanced = enhanced.squeeze(0)  # Remove batch dimension
        
        # Trim back to original length
        enhanced = enhanced[:, :original_length]
        
        return enhanced.cpu()
    
    else:
        # Process in overlapping chunks
        hop_size = int(chunk_size * (1 - overlap))
        enhanced_chunks = []
        
        for start in range(0, original_length - chunk_size + 1, hop_size):
            end = start + chunk_size
            chunk = audio[:, start:end].to(device)
            
            with torch.no_grad():
                enhanced_chunk = model(chunk.unsqueeze(0))
                enhanced_chunk = enhanced_chunk.squeeze(0)
            
            enhanced_chunks.append(enhanced_chunk.cpu())
        
        # Handle last chunk if needed
        if original_length % hop_size != 0:
            start = original_length - chunk_size
            if start >= 0:
                chunk = audio[:, start:].to(device)
                # Pad if necessary
                if chunk.shape[1] < chunk_size:
                    pad_length = chunk_size - chunk.shape[1]
                    chunk = torch.nn.functional.pad(chunk, (0, pad_length))
                
                with torch.no_grad():
                    enhanced_chunk = model(chunk.unsqueeze(0))
                    enhanced_chunk = enhanced_chunk.squeeze(0)
                
                # Trim to actual length
                actual_length = original_length - start
                enhanced_chunk = enhanced_chunk[:, :actual_length]
                enhanced_chunks.append(enhanced_chunk.cpu())
        
        # Reconstruct audio with simple overlap-add
        reconstructed = torch.zeros(1, original_length)
        window_sum = torch.zeros(1, original_length)
        
        # Create simple window
        window = torch.ones(chunk_size)
        
        for i, chunk in enumerate(enhanced_chunks):
            start = i * hop_size
            end = min(start + chunk.shape[1], original_length)
            
            if start < original_length:
                chunk_length = end - start
                reconstructed[:, start:end] += chunk[:, :chunk_length]
                window_sum[:, start:end] += window[:chunk_length]
        
        # Normalize by window sum
        window_sum = torch.clamp(window_sum, min=1e-8)
        reconstructed = reconstructed / window_sum
        
        return reconstructed

def main():
    parser = argparse.ArgumentParser(description='Enhance audio using SimpleUNet')
    parser.add_argument('input', help='Input audio file')
    parser.add_argument('output', help='Output audio file')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--chunk_size', type=int, default=48000, help='Chunk size for processing')
    parser.add_argument('--overlap', type=float, default=0.25, help='Overlap ratio for chunks')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model}")
    model, model_config = load_model(args.model, device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Load audio
    print(f"Loading audio: {args.input}")
    audio, sr = load_audio(args.input, model_config['sample_rate'])
    
    print(f"Audio length: {audio.shape[1] / sr:.2f} seconds")
    
    # Enhance audio
    print("Enhancing audio...")
    start_time = time.time()
    
    enhanced_audio = enhance_audio(
        model, audio, 
        chunk_size=args.chunk_size, 
        overlap=args.overlap, 
        device=device
    )
    
    processing_time = time.time() - start_time
    
    # Normalize output using peak normalization (consistent with training)
    enhanced_audio = normalize_audio(enhanced_audio, method='peak')
    
    # Save enhanced audio
    print(f"Saving enhanced audio: {args.output}")
    save_audio(enhanced_audio, args.output, model_config['sample_rate'])
    
    # Print statistics
    audio_length = audio.shape[1] / sr
    real_time_factor = processing_time / audio_length if audio_length > 0 else 0
    
    print(f"\nProcessing completed!")
    print(f"Processing time: {processing_time:.2f}s")
    print(f"Real-time factor: {real_time_factor:.2f}x")

if __name__ == "__main__":
    main() 