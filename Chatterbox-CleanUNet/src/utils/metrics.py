import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import librosa
from pesq import pesq
from pystoi import stoi
import scipy.signal

def compute_pesq(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int = 24000) -> float:
    """
    Compute PESQ score between clean and enhanced audio
    
    Args:
        clean: Clean reference audio
        enhanced: Enhanced audio
        sample_rate: Sample rate (8000 or 16000)
        
    Returns:
        PESQ score
    """
    try:
        # Convert to numpy and ensure 1D
        clean_np = clean.squeeze().cpu().numpy()
        enhanced_np = enhanced.squeeze().cpu().numpy()
        
        # Ensure same length
        min_length = min(len(clean_np), len(enhanced_np))
        clean_np = clean_np[:min_length]
        enhanced_np = enhanced_np[:min_length]
        
        # PESQ expects specific sample rates (8000 or 16000)
        if sample_rate not in [8000, 16000]:
            # Resample to 16000 for PESQ (closest to 24000)
            clean_np = librosa.resample(clean_np, orig_sr=sample_rate, target_sr=16000)
            enhanced_np = librosa.resample(enhanced_np, orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000
        
        # Compute PESQ
        pesq_score = pesq(sample_rate, clean_np, enhanced_np, 'wb')
        return pesq_score
        
    except Exception as e:
        print(f"Error computing PESQ: {e}")
        return 0.0

def compute_stoi(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int = 24000) -> float:
    """
    Compute STOI score between clean and enhanced audio
    
    Args:
        clean: Clean reference audio
        enhanced: Enhanced audio
        sample_rate: Sample rate
        
    Returns:
        STOI score
    """
    try:
        # Convert to numpy and ensure 1D
        clean_np = clean.squeeze().cpu().numpy()
        enhanced_np = enhanced.squeeze().cpu().numpy()
        
        # Ensure same length
        min_length = min(len(clean_np), len(enhanced_np))
        clean_np = clean_np[:min_length]
        enhanced_np = enhanced_np[:min_length]
        
        # Compute STOI
        stoi_score = stoi(clean_np, enhanced_np, sample_rate, extended=False)
        return stoi_score
        
    except Exception as e:
        print(f"Error computing STOI: {e}")
        return 0.0

def compute_estoi(clean: torch.Tensor, enhanced: torch.Tensor, sample_rate: int = 24000) -> float:
    """
    Compute Extended STOI score between clean and enhanced audio
    
    Args:
        clean: Clean reference audio
        enhanced: Enhanced audio
        sample_rate: Sample rate
        
    Returns:
        Extended STOI score
    """
    try:
        # Convert to numpy and ensure 1D
        clean_np = clean.squeeze().cpu().numpy()
        enhanced_np = enhanced.squeeze().cpu().numpy()
        
        # Ensure same length
        min_length = min(len(clean_np), len(enhanced_np))
        clean_np = clean_np[:min_length]
        enhanced_np = enhanced_np[:min_length]
        
        # Compute Extended STOI
        estoi_score = stoi(clean_np, enhanced_np, sample_rate, extended=True)
        return estoi_score
        
    except Exception as e:
        print(f"Error computing Extended STOI: {e}")
        return 0.0

def compute_snr(clean: torch.Tensor, noisy: torch.Tensor) -> float:
    """
    Compute Signal-to-Noise Ratio in dB
    
    Args:
        clean: Clean signal
        noisy: Noisy signal
        
    Returns:
        SNR in dB
    """
    signal_power = torch.mean(clean ** 2)
    noise_power = torch.mean((noisy - clean) ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()

def compute_si_snr(clean: torch.Tensor, enhanced: torch.Tensor) -> float:
    """
    Compute Scale-Invariant Signal-to-Noise Ratio
    
    Args:
        clean: Clean reference signal
        enhanced: Enhanced signal
        
    Returns:
        SI-SNR in dB
    """
    # Zero-mean signals
    clean = clean - torch.mean(clean)
    enhanced = enhanced - torch.mean(enhanced)
    
    # Compute target scaling factor
    alpha = torch.sum(enhanced * clean) / torch.sum(clean ** 2)
    
    # Compute SI-SNR
    target = alpha * clean
    noise = enhanced - target
    
    signal_power = torch.sum(target ** 2)
    noise_power = torch.sum(noise ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    si_snr = 10 * torch.log10(signal_power / noise_power)
    return si_snr.item()

def compute_lsd(clean: torch.Tensor, enhanced: torch.Tensor, n_fft: int = 512) -> float:
    """
    Compute Log-Spectral Distance
    
    Args:
        clean: Clean reference signal
        enhanced: Enhanced signal
        n_fft: FFT size
        
    Returns:
        LSD value
    """
    # Compute spectrograms
    clean_spec = torch.stft(clean.squeeze(), n_fft=n_fft, return_complex=True)
    enhanced_spec = torch.stft(enhanced.squeeze(), n_fft=n_fft, return_complex=True)
    
    # Compute magnitude spectrograms
    clean_mag = torch.abs(clean_spec)
    enhanced_mag = torch.abs(enhanced_spec)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    clean_mag = torch.clamp(clean_mag, min=epsilon)
    enhanced_mag = torch.clamp(enhanced_mag, min=epsilon)
    
    # Compute LSD
    lsd = torch.mean((torch.log(clean_mag) - torch.log(enhanced_mag)) ** 2)
    return lsd.item()

def compute_spectral_convergence(clean: torch.Tensor, enhanced: torch.Tensor, n_fft: int = 512) -> float:
    """
    Compute Spectral Convergence
    
    Args:
        clean: Clean reference signal
        enhanced: Enhanced signal
        n_fft: FFT size
        
    Returns:
        Spectral convergence value
    """
    # Compute spectrograms
    clean_spec = torch.stft(clean.squeeze(), n_fft=n_fft, return_complex=True)
    enhanced_spec = torch.stft(enhanced.squeeze(), n_fft=n_fft, return_complex=True)
    
    # Compute magnitude spectrograms
    clean_mag = torch.abs(clean_spec)
    enhanced_mag = torch.abs(enhanced_spec)
    
    # Compute spectral convergence
    numerator = torch.sum((clean_mag - enhanced_mag) ** 2)
    denominator = torch.sum(clean_mag ** 2)
    
    if denominator == 0:
        return float('inf')
    
    sc = numerator / denominator
    return sc.item()

def compute_mcd(clean: torch.Tensor, enhanced: torch.Tensor, n_fft: int = 512) -> float:
    """
    Compute Mel-Cepstral Distortion
    
    Args:
        clean: Clean reference signal
        enhanced: Enhanced signal
        n_fft: FFT size
        
    Returns:
        MCD value
    """
    # Convert to numpy for librosa processing
    clean_np = clean.squeeze().cpu().numpy()
    enhanced_np = enhanced.squeeze().cpu().numpy()
    
    # Compute mel-frequency cepstral coefficients
    clean_mfcc = librosa.feature.mfcc(y=clean_np, n_fft=n_fft, n_mfcc=13)
    enhanced_mfcc = librosa.feature.mfcc(y=enhanced_np, n_fft=n_fft, n_mfcc=13)
    
    # Ensure same dimensions
    min_frames = min(clean_mfcc.shape[1], enhanced_mfcc.shape[1])
    clean_mfcc = clean_mfcc[:, :min_frames]
    enhanced_mfcc = enhanced_mfcc[:, :min_frames]
    
    # Compute MCD
    mcd = np.mean(np.sqrt(np.sum((clean_mfcc - enhanced_mfcc) ** 2, axis=0)))
    return mcd

def compute_all_metrics(clean: torch.Tensor, enhanced: torch.Tensor, 
                       sample_rate: int = 24000) -> Dict[str, float]:
    """
    Compute all audio quality metrics
    
    Args:
        clean: Clean reference audio
        enhanced: Enhanced audio
        sample_rate: Sample rate
        
    Returns:
        Dictionary of metric values
    """
    metrics = {}
    
    try:
        metrics['pesq'] = compute_pesq(clean, enhanced, sample_rate)
    except Exception as e:
        print(f"PESQ computation failed: {e}")
        metrics['pesq'] = 0.0
    
    try:
        metrics['stoi'] = compute_stoi(clean, enhanced, sample_rate)
    except Exception as e:
        print(f"STOI computation failed: {e}")
        metrics['stoi'] = 0.0
    
    try:
        metrics['estoi'] = compute_estoi(clean, enhanced, sample_rate)
    except Exception as e:
        print(f"ESTOI computation failed: {e}")
        metrics['estoi'] = 0.0
    
    try:
        metrics['snr'] = compute_snr(clean, enhanced)
    except Exception as e:
        print(f"SNR computation failed: {e}")
        metrics['snr'] = 0.0
    
    try:
        metrics['si_snr'] = compute_si_snr(clean, enhanced)
    except Exception as e:
        print(f"SI-SNR computation failed: {e}")
        metrics['si_snr'] = 0.0
    
    try:
        metrics['lsd'] = compute_lsd(clean, enhanced)
    except Exception as e:
        print(f"LSD computation failed: {e}")
        metrics['lsd'] = float('inf')
    
    try:
        metrics['spectral_convergence'] = compute_spectral_convergence(clean, enhanced)
    except Exception as e:
        print(f"Spectral convergence computation failed: {e}")
        metrics['spectral_convergence'] = float('inf')
    
    try:
        metrics['mcd'] = compute_mcd(clean, enhanced)
    except Exception as e:
        print(f"MCD computation failed: {e}")
        metrics['mcd'] = float('inf')
    
    return metrics

class MetricsCalculator:
    """Class for batch computation of audio quality metrics"""
    
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.reset()
    
    def reset(self):
        """Reset accumulated metrics"""
        self.metrics = {
            'pesq': [],
            'stoi': [],
            'estoi': [],
            'snr': [],
            'si_snr': [],
            'lsd': [],
            'spectral_convergence': [],
            'mcd': []
        }
    
    def update(self, clean: torch.Tensor, enhanced: torch.Tensor):
        """Update metrics with new batch"""
        batch_metrics = compute_all_metrics(clean, enhanced, self.sample_rate)
        
        for key, value in batch_metrics.items():
            if not (np.isnan(value) or np.isinf(value)):
                self.metrics[key].append(value)
    
    def compute_average(self) -> Dict[str, float]:
        """Compute average metrics"""
        avg_metrics = {}
        
        for key, values in self.metrics.items():
            if values:
                avg_metrics[key] = np.mean(values)
                avg_metrics[f'{key}_std'] = np.std(values)
            else:
                avg_metrics[key] = 0.0
                avg_metrics[f'{key}_std'] = 0.0
        
        return avg_metrics
    
    def get_summary(self) -> str:
        """Get formatted summary of metrics"""
        avg_metrics = self.compute_average()
        
        summary = "Audio Quality Metrics Summary:\n"
        summary += "=" * 40 + "\n"
        summary += f"PESQ:     {avg_metrics['pesq']:.3f} ± {avg_metrics['pesq_std']:.3f}\n"
        summary += f"STOI:     {avg_metrics['stoi']:.3f} ± {avg_metrics['stoi_std']:.3f}\n"
        summary += f"ESTOI:    {avg_metrics['estoi']:.3f} ± {avg_metrics['estoi_std']:.3f}\n"
        summary += f"SNR:      {avg_metrics['snr']:.3f} ± {avg_metrics['snr_std']:.3f} dB\n"
        summary += f"SI-SNR:   {avg_metrics['si_snr']:.3f} ± {avg_metrics['si_snr_std']:.3f} dB\n"
        summary += f"LSD:      {avg_metrics['lsd']:.3f} ± {avg_metrics['lsd_std']:.3f}\n"
        summary += f"Spec Conv: {avg_metrics['spectral_convergence']:.3f} ± {avg_metrics['spectral_convergence_std']:.3f}\n"
        summary += f"MCD:      {avg_metrics['mcd']:.3f} ± {avg_metrics['mcd_std']:.3f}\n"
        
        return summary 