import torch
import torch.nn as nn
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import time

from ..models.cleanunet import CleanUNet
from ..utils.audio_utils import (
    load_audio, save_audio, normalize_audio, 
    chunk_audio, reconstruct_audio
)

class AudioEnhancer:
    """Audio enhancement using trained CleanUNet model"""
    
    def __init__(self, inference_config: Dict, model_config: Dict):
        """
        Initialize audio enhancer
        
        Args:
            inference_config: Inference configuration
            model_config: Model configuration
        """
        self.inference_config = inference_config['inference']
        self.model_config = model_config['model']
        self.device = torch.device(self.inference_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Audio parameters
        self.sample_rate = self.inference_config.get('sample_rate', 16000)
        self.chunk_size = self.inference_config.get('chunk_size', 32768)
        self.overlap = self.inference_config.get('overlap', 0.25)
        self.normalize_output = self.inference_config.get('normalize_output', True)
        
        # Load model
        self.model = self._load_model()
        
        print(f"AudioEnhancer initialized on device: {self.device}")
        print(f"Model loaded from: {self.inference_config['model_path']}")
    
    def _load_model(self) -> nn.Module:
        """Load trained CleanUNet model"""
        model_path = self.inference_config['model_path']
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create model
        model = CleanUNet(self.model_config)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Move to device and set to eval mode
        model.to(self.device)
        model.eval()
        
        return model
    
    def enhance_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Enhance audio using the trained model
        
        Args:
            audio: Input audio tensor (1, length)
            
        Returns:
            Enhanced audio tensor (1, length)
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        original_length = audio.shape[1]
        
        # Process audio in chunks if it's too long
        if original_length > self.chunk_size:
            return self._enhance_chunks(audio)
        else:
            return self._enhance_single(audio)
    
    def _enhance_single(self, audio: torch.Tensor) -> torch.Tensor:
        """Enhance single audio chunk"""
        # Move to device
        audio = audio.to(self.device)
        
        # Pad if necessary
        if audio.shape[1] < self.chunk_size:
            pad_length = self.chunk_size - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad_length))
            original_length = audio.shape[1] - pad_length
        else:
            original_length = audio.shape[1]
        
        # Enhance
        with torch.no_grad():
            enhanced = self.model(audio)
        
        # Trim back to original length
        if original_length < enhanced.shape[1]:
            enhanced = enhanced[:, :original_length]
        
        return enhanced.cpu()
    
    def _enhance_chunks(self, audio: torch.Tensor) -> torch.Tensor:
        """Enhance audio by processing in overlapping chunks"""
        # Split into chunks
        chunks = chunk_audio(audio, self.chunk_size, self.overlap)
        enhanced_chunks = []
        
        print(f"Processing {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            # Move to device
            chunk = chunk.to(self.device)
            
            # Enhance chunk
            with torch.no_grad():
                enhanced_chunk = self.model(chunk)
            
            enhanced_chunks.append(enhanced_chunk.cpu())
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(chunks)} chunks")
        
        # Reconstruct audio
        hop_size = int(self.chunk_size * (1 - self.overlap))
        reconstructed = reconstruct_audio(enhanced_chunks, hop_size, audio.shape[1])
        
        return reconstructed
    
    def enhance_file(self, input_path: str, output_path: str) -> Dict[str, float]:
        """
        Enhance audio file
        
        Args:
            input_path: Path to input audio file
            output_path: Path to output enhanced audio file
            
        Returns:
            Dictionary with processing statistics
        """
        print(f"Enhancing: {input_path} -> {output_path}")
        
        start_time = time.time()
        
        # Load audio
        audio, sr = load_audio(input_path, self.sample_rate)
        
        # Enhance audio
        enhanced_audio = self.enhance_audio(audio)
        
        # Normalize if requested
        if self.normalize_output:
            enhanced_audio = normalize_audio(enhanced_audio)
        
        # Save enhanced audio
        save_audio(enhanced_audio, output_path, self.sample_rate)
        
        processing_time = time.time() - start_time
        audio_length = audio.shape[1] / self.sample_rate
        
        stats = {
            'processing_time': processing_time,
            'audio_length': audio_length,
            'real_time_factor': processing_time / audio_length if audio_length > 0 else 0
        }
        
        print(f"Enhanced audio saved to: {output_path}")
        print(f"Processing time: {processing_time:.2f}s")
        print(f"Audio length: {audio_length:.2f}s")
        print(f"Real-time factor: {stats['real_time_factor']:.2f}x")
        
        return stats
    
    def enhance_directory(self, input_dir: str, output_dir: str, 
                         file_pattern: str = "*.wav") -> Dict[str, float]:
        """
        Enhance all audio files in a directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            file_pattern: File pattern to match (default: "*.wav")
            
        Returns:
            Dictionary with overall processing statistics
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find audio files
        audio_files = list(input_dir.glob(file_pattern))
        
        if not audio_files:
            print(f"No audio files found in {input_dir} matching pattern {file_pattern}")
            return {}
        
        print(f"Found {len(audio_files)} audio files to enhance")
        
        total_processing_time = 0
        total_audio_length = 0
        processed_files = 0
        failed_files = []
        
        for audio_file in audio_files:
            try:
                output_file = output_dir / f"{audio_file.stem}_enhanced{audio_file.suffix}"
                
                stats = self.enhance_file(str(audio_file), str(output_file))
                
                total_processing_time += stats['processing_time']
                total_audio_length += stats['audio_length']
                processed_files += 1
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                failed_files.append(str(audio_file))
        
        overall_stats = {
            'total_files': len(audio_files),
            'processed_files': processed_files,
            'failed_files': len(failed_files),
            'total_processing_time': total_processing_time,
            'total_audio_length': total_audio_length,
            'average_real_time_factor': total_processing_time / total_audio_length if total_audio_length > 0 else 0
        }
        
        print(f"\nBatch processing completed:")
        print(f"Processed: {processed_files}/{len(audio_files)} files")
        print(f"Failed: {len(failed_files)} files")
        print(f"Total processing time: {total_processing_time:.2f}s")
        print(f"Total audio length: {total_audio_length:.2f}s")
        print(f"Average real-time factor: {overall_stats['average_real_time_factor']:.2f}x")
        
        if failed_files:
            print(f"Failed files: {failed_files}")
        
        return overall_stats
    
    def enhance_with_comparison(self, input_path: str, output_dir: str, 
                               reference_path: str = None) -> Dict[str, float]:
        """
        Enhance audio and optionally compare with reference
        
        Args:
            input_path: Path to input audio file
            output_dir: Output directory path
            reference_path: Optional path to reference clean audio
            
        Returns:
            Dictionary with processing statistics and metrics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define output paths
        input_filename = Path(input_path).stem
        enhanced_path = output_dir / f"{input_filename}_enhanced.wav"
        
        # Enhance audio
        stats = self.enhance_file(input_path, str(enhanced_path))
        
        # If reference provided, compute metrics
        if reference_path:
            from ..utils.metrics import compute_all_metrics
            
            # Load reference and enhanced audio
            reference_audio, _ = load_audio(reference_path, self.sample_rate)
            enhanced_audio, _ = load_audio(str(enhanced_path), self.sample_rate)
            
            # Compute metrics
            metrics = compute_all_metrics(reference_audio, enhanced_audio, self.sample_rate)
            
            # Add metrics to stats
            stats.update(metrics)
            
            print(f"\nQuality metrics:")
            print(f"PESQ: {metrics['pesq']:.3f}")
            print(f"STOI: {metrics['stoi']:.3f}")
            print(f"SI-SNR: {metrics['si_snr']:.3f} dB")
        
        return stats
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_path': self.inference_config['model_path'],
            'device': str(self.device),
            'sample_rate': self.sample_rate,
            'chunk_size': self.chunk_size,
            'overlap': self.overlap,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        }

def create_enhancer(inference_config_path: str, model_config_path: str) -> AudioEnhancer:
    """
    Factory function to create AudioEnhancer
    
    Args:
        inference_config_path: Path to inference configuration
        model_config_path: Path to model configuration
        
    Returns:
        AudioEnhancer instance
    """
    with open(inference_config_path, 'r') as f:
        inference_config = yaml.safe_load(f)
    
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    
    return AudioEnhancer(inference_config, model_config) 