import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, List, Optional
from pathlib import Path

def load_audio(file_path: str, sample_rate: int = 16000) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and resample if necessary
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        
    Returns:
        Tuple of (audio_tensor, sample_rate)
    """
    try:
        audio, sr = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            audio = resampler(audio)
        
        return audio, sample_rate
        
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        raise

def save_audio(audio: torch.Tensor, file_path: str, sample_rate: int = 16000):
    """
    Save audio tensor to file
    
    Args:
        audio: Audio tensor (1, length) or (length,)
        file_path: Output file path
        sample_rate: Sample rate
    """
    try:
        # Ensure audio is 2D
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Normalize audio to prevent clipping
        audio = normalize_audio(audio)
        
        # Save audio
        torchaudio.save(file_path, audio, sample_rate)
        
    except Exception as e:
        print(f"Error saving audio file {file_path}: {e}")
        raise

def normalize_audio(audio: torch.Tensor, method: str = 'peak') -> torch.Tensor:
    """
    Normalize audio tensor
    
    Args:
        audio: Input audio tensor
        method: Normalization method ('peak', 'rms', 'lufs')
        
    Returns:
        Normalized audio tensor
    """
    if method == 'peak':
        # Peak normalization
        max_val = torch.max(torch.abs(audio))
        if max_val > 0:
            audio = audio / max_val
            
    elif method == 'rms':
        # RMS normalization
        rms = torch.sqrt(torch.mean(audio ** 2))
        if rms > 0:
            audio = audio / rms
            
    elif method == 'lufs':
        # LUFS normalization (simplified)
        # This is a basic implementation; for production use a proper LUFS meter
        mean_square = torch.mean(audio ** 2)
        if mean_square > 0:
            audio = audio / torch.sqrt(mean_square)
    
    # Clip to [-1, 1] range
    audio = torch.clamp(audio, -1.0, 1.0)
    
    return audio

def chunk_audio(audio: torch.Tensor, chunk_size: int, overlap: float = 0.0) -> List[torch.Tensor]:
    """
    Split audio into overlapping chunks
    
    Args:
        audio: Input audio tensor (1, length)
        chunk_size: Size of each chunk
        overlap: Overlap ratio (0.0 to 1.0)
        
    Returns:
        List of audio chunks
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    audio_length = audio.shape[1]
    hop_size = int(chunk_size * (1 - overlap))
    chunks = []
    
    for start in range(0, audio_length - chunk_size + 1, hop_size):
        end = start + chunk_size
        chunks.append(audio[:, start:end])
    
    # Handle remaining audio
    if audio_length % hop_size != 0:
        start = audio_length - chunk_size
        if start >= 0:
            chunks.append(audio[:, start:])
    
    return chunks

def reconstruct_audio(chunks: List[torch.Tensor], hop_size: int, total_length: int) -> torch.Tensor:
    """
    Reconstruct audio from overlapping chunks using overlap-add
    
    Args:
        chunks: List of audio chunks
        hop_size: Hop size between chunks
        total_length: Total length of reconstructed audio
        
    Returns:
        Reconstructed audio tensor
    """
    if not chunks:
        return torch.zeros(1, total_length)
    
    reconstructed = torch.zeros(1, total_length)
    window_sum = torch.zeros(1, total_length)
    
    chunk_size = chunks[0].shape[1]
    
    # Create window for overlap-add
    window = torch.hann_window(chunk_size).unsqueeze(0)
    
    for i, chunk in enumerate(chunks):
        start = i * hop_size
        end = min(start + chunk_size, total_length)
        
        if start < total_length:
            # Apply window and add to reconstruction
            windowed_chunk = chunk * window
            reconstructed[:, start:end] += windowed_chunk[:, :end-start]
            window_sum[:, start:end] += window[:, :end-start]
    
    # Normalize by window sum to avoid amplitude changes
    window_sum = torch.clamp(window_sum, min=1e-8)
    reconstructed = reconstructed / window_sum
    
    return reconstructed

def compute_snr(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    """
    Compute Signal-to-Noise Ratio
    
    Args:
        clean: Clean audio signal
        noisy: Noisy audio signal
        
    Returns:
        SNR in dB
    """
    signal_power = torch.mean(clean ** 2)
    noise_power = torch.mean((noisy - clean) ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()

def add_noise(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """
    Add noise to clean audio at specified SNR
    
    Args:
        clean: Clean audio signal
        noise: Noise signal
        snr_db: Target SNR in dB
        
    Returns:
        Noisy audio signal
    """
    # Calculate current power
    signal_power = torch.mean(clean ** 2)
    noise_power = torch.mean(noise ** 2)
    
    # Calculate scaling factor for desired SNR
    snr_linear = 10 ** (snr_db / 10)
    noise_scaling = torch.sqrt(signal_power / (noise_power * snr_linear))
    
    # Scale noise and add to signal
    scaled_noise = noise * noise_scaling
    noisy = clean + scaled_noise
    
    return noisy

def apply_gain(audio: torch.Tensor, gain_db: float) -> torch.Tensor:
    """
    Apply gain to audio signal
    
    Args:
        audio: Input audio tensor
        gain_db: Gain in dB
        
    Returns:
        Audio with applied gain
    """
    gain_linear = 10 ** (gain_db / 20)
    return audio * gain_linear

def high_pass_filter(audio: torch.Tensor, cutoff_freq: float, sample_rate: int = 16000) -> torch.Tensor:
    """
    Apply high-pass filter to audio
    
    Args:
        audio: Input audio tensor
        cutoff_freq: Cutoff frequency in Hz
        sample_rate: Sample rate
        
    Returns:
        Filtered audio
    """
    # Convert to numpy for librosa processing
    audio_np = audio.squeeze().numpy()
    
    # Apply high-pass filter
    filtered = librosa.effects.preemphasis(audio_np, coef=0.97)
    
    # Convert back to tensor
    return torch.tensor(filtered).unsqueeze(0)

def compute_spectral_centroid(audio: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
    """
    Compute spectral centroid of audio
    
    Args:
        audio: Input audio tensor
        sample_rate: Sample rate
        
    Returns:
        Spectral centroid values
    """
    # Convert to numpy for librosa processing
    audio_np = audio.squeeze().numpy()
    
    # Compute spectral centroid
    centroid = librosa.feature.spectral_centroid(y=audio_np, sr=sample_rate)
    
    # Convert back to tensor
    return torch.tensor(centroid)

def detect_voice_activity(audio: torch.Tensor, frame_length: int = 1024, hop_length: int = 512) -> torch.Tensor:
    """
    Simple voice activity detection based on energy
    
    Args:
        audio: Input audio tensor
        frame_length: Frame length for analysis
        hop_length: Hop length between frames
        
    Returns:
        Binary mask indicating voice activity
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # Compute frame-wise energy
    audio_padded = torch.nn.functional.pad(audio, (frame_length//2, frame_length//2))
    frames = audio_padded.unfold(1, frame_length, hop_length)
    energy = torch.mean(frames ** 2, dim=2)
    
    # Threshold-based VAD
    threshold = torch.mean(energy) * 0.1  # Adjust threshold as needed
    vad_mask = energy > threshold
    
    return vad_mask

def trim_silence(audio: torch.Tensor, threshold: float = 0.01) -> torch.Tensor:
    """
    Trim silence from beginning and end of audio
    
    Args:
        audio: Input audio tensor
        threshold: Amplitude threshold for silence detection
        
    Returns:
        Trimmed audio
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # Find non-silent regions
    non_silent = torch.abs(audio) > threshold
    non_silent_indices = torch.where(non_silent[0])[0]
    
    if len(non_silent_indices) == 0:
        return audio
    
    # Trim to non-silent region
    start_idx = non_silent_indices[0].item()
    end_idx = non_silent_indices[-1].item() + 1
    
    return audio[:, start_idx:end_idx] 