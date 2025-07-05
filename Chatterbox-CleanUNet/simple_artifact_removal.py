#!/usr/bin/env python3
"""
Simple, working artifact removal for TTS hallucinations
Alternatives to the broken CleanUNet model
"""
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.audio_utils import load_audio, save_audio
import numpy as np
import scipy.signal
from scipy.signal import butter, filtfilt

def spectral_gate_artifact_removal(audio, sr=24000, threshold_db=-40, 
                                  gate_freq_range=(50, 8000)):
    """
    Simple spectral gating - remove frequencies where artifacts dominate
    """
    # Convert to numpy
    if isinstance(audio, torch.Tensor):
        audio_np = audio.squeeze().numpy()
    else:
        audio_np = audio
    
    # Compute spectrogram
    f, t, Sxx = scipy.signal.spectrogram(audio_np, sr, nperseg=1024, noverlap=512)
    
    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # Create frequency mask for artifact range
    freq_mask = (f >= gate_freq_range[0]) & (f <= gate_freq_range[1])
    
    # Find time regions with strong artifacts
    artifact_power = np.mean(Sxx_db[freq_mask, :], axis=0)
    artifact_regions = artifact_power > threshold_db
    
    # Create time-frequency gate
    gate = np.ones_like(Sxx_db)
    gate[freq_mask[:, None] & artifact_regions[None, :]] *= 0.1  # Reduce by 90%
    
    # Apply gate
    Sxx_gated = Sxx * (10 ** (gate / 10))
    
    # Reconstruct audio
    _, audio_clean = scipy.signal.istft(Sxx_gated, sr, nperseg=1024, noverlap=512)
    
    return torch.tensor(audio_clean).unsqueeze(0).float()

def silence_detection_removal(audio, sr=24000, silence_threshold=0.01, 
                             min_speech_length=0.5, artifact_start_time=20.0):
    """
    Remove artifacts by detecting the end of speech and silencing everything after
    """
    if isinstance(audio, torch.Tensor):
        audio_np = audio.squeeze().numpy()
    else:
        audio_np = audio
    
    # Calculate RMS in small windows
    window_size = int(0.1 * sr)  # 100ms windows
    hop_size = int(0.05 * sr)    # 50ms hop
    
    rms_values = []
    for i in range(0, len(audio_np) - window_size, hop_size):
        window = audio_np[i:i + window_size]
        rms = np.sqrt(np.mean(window ** 2))
        rms_values.append(rms)
    
    rms_values = np.array(rms_values)
    
    # Find speech regions
    speech_regions = rms_values > silence_threshold
    
    # Find the last significant speech
    last_speech_idx = 0
    min_speech_samples = int(min_speech_length / 0.05)  # Convert to window units
    
    for i in range(len(speech_regions) - min_speech_samples, -1, -1):
        if np.all(speech_regions[i:i + min_speech_samples]):
            last_speech_idx = i
            break
    
    # Convert back to audio samples
    last_speech_sample = last_speech_idx * hop_size + window_size
    
    # If we found speech ending before the expected artifact time, use that
    artifact_start_sample = int(artifact_start_time * sr)
    cutoff_sample = min(last_speech_sample, artifact_start_sample)
    
    # Create cleaned audio
    audio_clean = audio_np.copy()
    audio_clean[cutoff_sample:] = 0  # Silence everything after speech ends
    
    return torch.tensor(audio_clean).unsqueeze(0).float()

def bandpass_filter_removal(audio, sr=24000, 
                           speech_range=(80, 7000), 
                           artifact_range=(7000, 12000)):
    """
    Use frequency filtering to reduce artifacts
    """
    if isinstance(audio, torch.Tensor):
        audio_np = audio.squeeze().numpy()
    else:
        audio_np = audio
    
    # Design bandpass filter for speech
    nyquist = sr / 2
    low = speech_range[0] / nyquist
    high = speech_range[1] / nyquist
    
    # Butterworth bandpass filter
    b, a = butter(4, [low, high], btype='band')
    audio_filtered = filtfilt(b, a, audio_np)
    
    return torch.tensor(audio_filtered.copy()).unsqueeze(0).float()

def test_simple_methods():
    """Test simple artifact removal methods on candidate_02.wav"""
    
    print("🔧 Testing SIMPLE artifact removal methods")
    print("=" * 60)
    
    # Load candidate_02.wav
    audio, sr = load_audio('candidate_02.wav', 24000)
    print(f"📂 Loaded: candidate_02.wav ({audio.shape[1]/sr:.1f}s)")
    
    # Method 1: Spectral Gating
    print("\n🎛️  Method 1: Spectral Gating")
    audio_spectral = spectral_gate_artifact_removal(audio, sr)
    save_audio(audio_spectral, 'candidate_02_spectral_gate.wav', sr)
    
    # Analyze results
    artifact_start = int(20 * sr)
    original_artifact = audio[:, artifact_start:]
    cleaned_artifact = audio_spectral[:, artifact_start:]
    
    orig_rms = torch.sqrt(torch.mean(original_artifact ** 2))
    clean_rms = torch.sqrt(torch.mean(cleaned_artifact ** 2))
    reduction = (orig_rms - clean_rms) / orig_rms * 100
    
    print(f"   Artifact RMS reduction: {reduction:.1f}%")
    
    # Method 2: Silence Detection
    print("\n🔇 Method 2: Silence Detection")
    audio_silence = silence_detection_removal(audio, sr)
    save_audio(audio_silence, 'candidate_02_silence_detection.wav', sr)
    
    cleaned_artifact_2 = audio_silence[:, artifact_start:]
    clean_rms_2 = torch.sqrt(torch.mean(cleaned_artifact_2 ** 2))
    reduction_2 = (orig_rms - clean_rms_2) / orig_rms * 100
    
    print(f"   Artifact RMS reduction: {reduction_2:.1f}%")
    
    # Method 3: Bandpass Filter
    print("\n🎚️  Method 3: Bandpass Filter")
    audio_bandpass = bandpass_filter_removal(audio, sr)
    save_audio(audio_bandpass, 'candidate_02_bandpass.wav', sr)
    
    cleaned_artifact_3 = audio_bandpass[:, artifact_start:]
    clean_rms_3 = torch.sqrt(torch.mean(cleaned_artifact_3 ** 2))
    reduction_3 = (orig_rms - clean_rms_3) / orig_rms * 100
    
    print(f"   Artifact RMS reduction: {reduction_3:.1f}%")
    
    # Summary
    print("\n📊 SUMMARY:")
    print("=" * 30)
    print(f"Original artifact RMS:     {orig_rms:.6f}")
    print(f"Spectral gate result:      {clean_rms:.6f} ({reduction:.1f}% reduction)")
    print(f"Silence detection result:  {clean_rms_2:.6f} ({reduction_2:.1f}% reduction)")
    print(f"Bandpass filter result:    {clean_rms_3:.6f} ({reduction_3:.1f}% reduction)")
    
    # Find best method
    best_method = ""
    best_reduction = 0
    if reduction > best_reduction:
        best_reduction = reduction
        best_method = "Spectral Gating"
    if reduction_2 > best_reduction:
        best_reduction = reduction_2
        best_method = "Silence Detection"
    if reduction_3 > best_reduction:
        best_reduction = reduction_3
        best_method = "Bandpass Filter"
    
    print(f"\n🏆 BEST METHOD: {best_method} ({best_reduction:.1f}% reduction)")
    
    print(f"\n💾 SAVED FILES:")
    print(f"   candidate_02_spectral_gate.wav")
    print(f"   candidate_02_silence_detection.wav") 
    print(f"   candidate_02_bandpass.wav")
    
    print(f"\n🎯 RECOMMENDATION:")
    if best_reduction > 80:
        print("✅ Simple methods work well! Use best method.")
    elif best_reduction > 50:
        print("⚠️  Moderate success. Consider combining methods.")
    else:
        print("❌ Simple methods insufficient. Need advanced ML approach.")
        print("💡 Consider: Audacity noise reduction, professional tools")

if __name__ == '__main__':
    test_simple_methods() 