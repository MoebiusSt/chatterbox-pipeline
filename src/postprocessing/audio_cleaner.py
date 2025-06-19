"""
Advanced audio cleaning module with multiple artifact removal techniques.
Provides spectral gating, normalization, and advanced filtering.
"""

import logging
import math

# Use absolute imports
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio

sys.path.append(str(Path(__file__).resolve().parents[1]))
from postprocessing.noise_analyzer import NoiseAnalyzer, NoiseProfile


@dataclass
class CleaningSettings:
    """Settings for audio cleaning operations."""

    spectral_gating: bool = True
    spectral_gate_threshold: float = 0.02
    normalize_audio: bool = True
    target_rms: float = 0.2
    highpass_freq: Optional[float] = 80.0
    lowpass_freq: Optional[float] = 8000.0
    remove_dc_offset: bool = True
    apply_gentle_compression: bool = True
    compression_ratio: float = 2.0


class AudioCleaner:
    """
    Advanced audio cleaning with multiple artifact removal techniques.
    Combines spectral gating, filtering, and normalization.
    """

    def __init__(
        self, sample_rate: int = 24000, settings: Optional[CleaningSettings] = None
    ):
        """
        Initialize AudioCleaner.

        Args:
            sample_rate: Audio sample rate
            settings: Cleaning settings (uses defaults if None)
        """
        self.sample_rate = sample_rate
        self.settings = settings or CleaningSettings()
        self.noise_analyzer = NoiseAnalyzer(sample_rate=sample_rate)
        self.logger = logging.getLogger(__name__)

    def clean_audio(
        self, audio: torch.Tensor, noise_profile: Optional[NoiseProfile] = None
    ) -> torch.Tensor:
        """
        Apply comprehensive audio cleaning.

        Args:
            audio: Input audio tensor (1, num_samples) or (num_samples,)
            noise_profile: Optional noise profile for adaptive processing

        Returns:
            Cleaned audio tensor
        """
        try:
            # Ensure audio is in correct format
            if audio.dim() == 2:
                audio = audio.squeeze(0)

            original_duration = len(audio) / self.sample_rate
            self.logger.info(
                f"Cleaning audio: {original_duration:.2f}s, {len(audio)} samples"
            )

            # Analyze noise if profile not provided
            if noise_profile is None:
                noise_profile = self.noise_analyzer.analyze_noise_floor(
                    audio.unsqueeze(0)
                )

            cleaned_audio = audio.clone()

            # Step 1: Remove DC offset
            if self.settings.remove_dc_offset:
                cleaned_audio = self._remove_dc_offset(cleaned_audio)
                self.logger.debug("Applied DC offset removal")

            # Step 2: Apply high-pass filter
            if self.settings.highpass_freq is not None:
                cleaned_audio = self._apply_highpass_filter(
                    cleaned_audio, self.settings.highpass_freq
                )
                self.logger.debug(
                    f"Applied highpass filter: {self.settings.highpass_freq}Hz"
                )

            # Step 3: Apply low-pass filter
            if self.settings.lowpass_freq is not None:
                cleaned_audio = self._apply_lowpass_filter(
                    cleaned_audio, self.settings.lowpass_freq
                )
                self.logger.debug(
                    f"Applied lowpass filter: {self.settings.lowpass_freq}Hz"
                )

            # Step 4: Spectral gating for noise reduction
            if self.settings.spectral_gating:
                cleaned_audio = self._apply_spectral_gating(
                    cleaned_audio, noise_profile
                )
                self.logger.debug("Applied spectral gating")

            # Step 5: Gentle compression
            if self.settings.apply_gentle_compression:
                cleaned_audio = self._apply_gentle_compression(
                    cleaned_audio, self.settings.compression_ratio
                )
                self.logger.debug(
                    f"Applied compression: {self.settings.compression_ratio}:1"
                )

            # Step 6: Normalize audio
            if self.settings.normalize_audio:
                cleaned_audio = self._normalize_audio(
                    cleaned_audio, self.settings.target_rms
                )
                self.logger.debug(f"Normalized to RMS: {self.settings.target_rms}")

            self.logger.info("Audio cleaning completed successfully")
            return cleaned_audio

        except Exception as e:
            self.logger.error(f"Audio cleaning failed: {e}")
            return audio

    def remove_artifacts(
        self,
        audio: torch.Tensor,
        threshold: float,
        preserve_margins: Tuple[float, float] = (0.1, 0.1),
    ) -> torch.Tensor:
        """
        Remove low-volume artifacts while preserving natural sounds.

        Args:
            audio: Input audio tensor
            threshold: Amplitude threshold for artifact removal
            preserve_margins: (before, after) margins in seconds

        Returns:
            Audio with artifacts removed
        """
        try:
            # Frame-based analysis
            frame_size = 1024
            hop_size = 512

            # Create frames
            audio_padded = torch.nn.functional.pad(
                audio, (frame_size // 2, frame_size // 2), mode="constant"
            )
            frames = audio_padded.unfold(0, frame_size, hop_size)

            # Calculate frame RMS
            frame_rms = torch.sqrt(torch.mean(frames**2, dim=1))

            # Identify frames to keep
            keep_frames = frame_rms > threshold

            # Apply margins around speech segments
            margin_frames_before = int(
                preserve_margins[0] * self.sample_rate / hop_size
            )
            margin_frames_after = int(preserve_margins[1] * self.sample_rate / hop_size)

            # Extend keep regions with margins
            extended_keep = keep_frames.clone()
            for i in range(len(keep_frames)):
                if keep_frames[i]:
                    start = max(0, i - margin_frames_before)
                    end = min(len(keep_frames), i + margin_frames_after + 1)
                    extended_keep[start:end] = True

            # Reconstruct audio
            output_audio = torch.zeros_like(audio)
            for i, keep in enumerate(extended_keep):
                if keep:
                    start_sample = i * hop_size
                    end_sample = min(start_sample + hop_size, len(audio))
                    if start_sample < len(audio):
                        output_audio[start_sample:end_sample] = audio[
                            start_sample:end_sample
                        ]

            return output_audio

        except Exception as e:
            self.logger.error(f"Artifact removal failed: {e}")
            return audio

    def _remove_dc_offset(self, audio: torch.Tensor) -> torch.Tensor:
        """Remove DC offset from audio."""
        dc_offset = torch.mean(audio)
        return audio - dc_offset

    def _apply_highpass_filter(
        self, audio: torch.Tensor, cutoff_freq: float
    ) -> torch.Tensor:
        """Apply high-pass filter."""
        # Simple high-pass filter using torchaudio
        nyquist = self.sample_rate / 2
        normalized_cutoff = cutoff_freq / nyquist

        # Create a simple high-pass filter
        # For simplicity, using a basic implementation
        b = torch.tensor([1.0, -1.0])
        a = torch.tensor([1.0, -0.95])  # Pole at 0.95

        # Apply filter (simplified - in practice would use proper filter design)
        # For now, just apply a basic high-pass effect
        filtered_audio = audio.clone()
        alpha = math.exp(-2 * math.pi * cutoff_freq / self.sample_rate)

        for i in range(1, len(filtered_audio)):
            filtered_audio[i] = alpha * filtered_audio[i - 1] + alpha * (
                audio[i] - audio[i - 1]
            )

        return filtered_audio

    def _apply_lowpass_filter(
        self, audio: torch.Tensor, cutoff_freq: float
    ) -> torch.Tensor:
        """Apply low-pass filter."""
        # Simple low-pass filter
        alpha = math.exp(-2 * math.pi * cutoff_freq / self.sample_rate)
        filtered_audio = audio.clone()

        for i in range(1, len(filtered_audio)):
            filtered_audio[i] = alpha * filtered_audio[i - 1] + (1 - alpha) * audio[i]

        return filtered_audio

    def _apply_spectral_gating(
        self, audio: torch.Tensor, noise_profile: NoiseProfile
    ) -> torch.Tensor:
        """Apply spectral gating for noise reduction."""
        try:
            # STFT for frequency domain processing
            n_fft = 1024
            hop_length = 256

            # Convert to complex spectrogram
            stft = torch.stft(
                audio,
                n_fft=n_fft,
                hop_length=hop_length,
                window=torch.hann_window(n_fft),
                return_complex=True,
            )

            # Calculate magnitude and phase
            magnitude = torch.abs(stft)
            phase = torch.angle(stft)

            # Estimate noise floor in frequency domain
            noise_floor = torch.quantile(magnitude, 0.1, dim=1, keepdim=True)

            # Create gating mask
            gate_threshold = noise_floor * (1.0 + self.settings.spectral_gate_threshold)
            gating_mask = magnitude > gate_threshold

            # Apply soft gating (gradual reduction rather than hard cut)
            gating_factor = torch.where(
                gating_mask,
                torch.ones_like(magnitude),
                magnitude / gate_threshold * 0.1,  # Reduce by 90% below threshold
            )

            # Apply gating to magnitude
            gated_magnitude = magnitude * gating_factor

            # Reconstruct complex spectrogram
            gated_stft = gated_magnitude * torch.exp(1j * phase)

            # ISTFT back to time domain
            cleaned_audio = torch.istft(
                gated_stft,
                n_fft=n_fft,
                hop_length=hop_length,
                window=torch.hann_window(n_fft),
                length=len(audio),
            )

            return cleaned_audio

        except Exception as e:
            self.logger.warning(f"Spectral gating failed: {e}, skipping")
            return audio

    def _apply_gentle_compression(
        self, audio: torch.Tensor, ratio: float
    ) -> torch.Tensor:
        """Apply gentle dynamic range compression."""
        try:
            # Calculate RMS in overlapping windows
            window_size = int(0.01 * self.sample_rate)  # 10ms windows
            hop_size = window_size // 2

            # Pad audio
            padded_audio = torch.nn.functional.pad(
                audio, (window_size // 2, window_size // 2), mode="reflect"
            )

            # Calculate windowed RMS
            windows = padded_audio.unfold(0, window_size, hop_size)
            rms = torch.sqrt(torch.mean(windows**2, dim=1))

            # Smooth RMS
            alpha = 0.9
            smoothed_rms = torch.zeros_like(rms)
            smoothed_rms[0] = rms[0]
            for i in range(1, len(rms)):
                smoothed_rms[i] = alpha * smoothed_rms[i - 1] + (1 - alpha) * rms[i]

            # Calculate compression gain
            threshold = 0.3  # Compression threshold
            compressed_rms = torch.where(
                smoothed_rms > threshold,
                threshold + (smoothed_rms - threshold) / ratio,
                smoothed_rms,
            )

            # Calculate gain adjustment
            gain = torch.where(
                smoothed_rms > 1e-8,
                compressed_rms / smoothed_rms,
                torch.ones_like(smoothed_rms),
            )

            # Interpolate gain to original audio length
            gain_interp = torch.nn.functional.interpolate(
                gain.unsqueeze(0).unsqueeze(0),
                size=len(audio),
                mode="linear",
                align_corners=False,
            ).squeeze()

            # Apply compression
            compressed_audio = audio * gain_interp

            return compressed_audio

        except Exception as e:
            self.logger.warning(f"Compression failed: {e}, skipping")
            return audio

    def _normalize_audio(self, audio: torch.Tensor, target_rms: float) -> torch.Tensor:
        """Normalize audio to target RMS level."""
        current_rms = torch.sqrt(torch.mean(audio**2))

        if current_rms > 1e-8:  # Avoid division by zero
            gain = target_rms / current_rms
            # Limit gain to prevent clipping
            peak_after_gain = torch.max(torch.abs(audio)) * gain
            if peak_after_gain > 0.95:
                gain = 0.95 / torch.max(torch.abs(audio))

            normalized_audio = audio * gain
        else:
            normalized_audio = audio

        return normalized_audio

    def preserve_natural_sounds(
        self, audio: torch.Tensor, margins: Tuple[float, float] = (0.1, 0.1)
    ) -> torch.Tensor:
        """
        Preserve natural breathing and lip sounds with careful margin handling.

        Args:
            audio: Input audio tensor
            margins: (before, after) margins in seconds

        Returns:
            Audio with preserved natural sounds
        """
        try:
            # Detect speech segments
            segments = self.noise_analyzer.detect_speech_segments(audio.unsqueeze(0))

            if not segments:
                return audio

            # Create mask for preserved audio
            preserved_mask = torch.zeros(len(audio), dtype=torch.bool)

            margin_samples_before = int(margins[0] * self.sample_rate)
            margin_samples_after = int(margins[1] * self.sample_rate)

            for segment in segments:
                start = max(0, segment.start_sample - margin_samples_before)
                end = min(len(audio), segment.end_sample + margin_samples_after)
                preserved_mask[start:end] = True

            # Apply mask
            preserved_audio = torch.where(
                preserved_mask, audio, torch.zeros_like(audio)
            )

            return preserved_audio

        except Exception as e:
            self.logger.error(f"Natural sound preservation failed: {e}")
            return audio

    def get_cleaning_stats(
        self, original: torch.Tensor, cleaned: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate statistics about the cleaning process.

        Args:
            original: Original audio
            cleaned: Cleaned audio

        Returns:
            Dictionary of cleaning statistics
        """
        original_rms = torch.sqrt(torch.mean(original**2)).item()
        cleaned_rms = torch.sqrt(torch.mean(cleaned**2)).item()

        original_peak = torch.max(torch.abs(original)).item()
        cleaned_peak = torch.max(torch.abs(cleaned)).item()

        # Signal-to-noise ratio improvement (rough estimate)
        noise_reduction = original_rms / cleaned_rms if cleaned_rms > 0 else 1.0

        stats = {
            "original_rms": original_rms,
            "cleaned_rms": cleaned_rms,
            "original_peak": original_peak,
            "cleaned_peak": cleaned_peak,
            "rms_ratio": cleaned_rms / original_rms if original_rms > 0 else 1.0,
            "peak_ratio": cleaned_peak / original_peak if original_peak > 0 else 1.0,
            "estimated_noise_reduction_db": (
                20 * math.log10(noise_reduction) if noise_reduction > 0 else 0.0
            ),
        }

        return stats
