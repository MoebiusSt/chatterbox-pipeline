"""
Noise analysis module for determining optimal thresholds and speech segments.
Analyzes audio characteristics to guide post-processing decisions.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio


@dataclass
class NoiseProfile:
    """Profile of noise characteristics in audio."""

    noise_floor: float
    speech_threshold: float
    silence_threshold: float
    peak_amplitude: float
    rms_level: float
    dynamic_range: float
    recommended_threshold: float
    confidence: float


@dataclass
class SpeechSegment:
    """Detected speech segment with timestamps."""

    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    peak_amplitude: float
    rms_level: float
    confidence: float


class NoiseAnalyzer:
    """
    Analyzes audio for noise characteristics and speech detection.
    Provides thresholds and parameters for Auto-Editor processing.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        frame_size: int = 2048,
        hop_size: int = 512,
        min_speech_duration: float = 0.1,
        min_silence_duration: float = 0.05,
    ):
        """
        Initialize NoiseAnalyzer.

        Args:
            sample_rate: Audio sample rate
            frame_size: Size of analysis frames
            hop_size: Hop size between frames
            min_speech_duration: Minimum duration for speech segments (seconds)
            min_silence_duration: Minimum duration for silence segments (seconds)
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.logger = logging.getLogger(__name__)

    def analyze_noise_floor(self, audio: torch.Tensor) -> NoiseProfile:
        """
        Analyze audio to determine noise floor and optimal thresholds.

        Args:
            audio: Audio tensor (1, num_samples) or (num_samples,)

        Returns:
            NoiseProfile with analysis results
        """
        try:
            # Ensure audio is in correct format
            if audio.dim() == 2:
                audio = audio.squeeze(0)

            audio_np = audio.detach().cpu().numpy()

            # Calculate basic statistics
            peak_amplitude = float(torch.max(torch.abs(audio)).item())
            rms_level = float(torch.sqrt(torch.mean(audio**2)).item())

            # Frame-based analysis
            frames = self._create_frames(audio_np)
            frame_energies = self._calculate_frame_energies(frames)
            frame_rms = np.sqrt(frame_energies)

            # Noise floor estimation (using percentile of quietest frames)
            noise_floor = float(np.percentile(frame_rms, 10))  # 10th percentile

            # Speech threshold (adaptive based on content)
            speech_threshold = self._calculate_speech_threshold(frame_rms, noise_floor)

            # Silence threshold (between noise floor and speech)
            silence_threshold = noise_floor + (speech_threshold - noise_floor) * 0.3

            # Dynamic range
            dynamic_range = peak_amplitude / max(noise_floor, 1e-8)
            dynamic_range_db = (
                20 * math.log10(dynamic_range) if dynamic_range > 0 else 0
            )

            # Recommended threshold for Auto-Editor
            recommended_threshold = self._calculate_recommended_threshold(
                noise_floor, speech_threshold, dynamic_range_db
            )

            # Confidence based on dynamic range and consistency
            confidence = self._calculate_confidence(frame_rms, dynamic_range_db)

            profile = NoiseProfile(
                noise_floor=noise_floor,
                speech_threshold=speech_threshold,
                silence_threshold=silence_threshold,
                peak_amplitude=peak_amplitude,
                rms_level=rms_level,
                dynamic_range=dynamic_range_db,
                recommended_threshold=recommended_threshold,
                confidence=confidence,
            )

            self.logger.info(
                f"Noise analysis: floor={noise_floor:.6f}, "
                f"speech_thresh={speech_threshold:.6f}, "
                f"recommended={recommended_threshold:.6f}, "
                f"confidence={confidence:.3f}"
            )

            return profile

        except Exception as e:
            self.logger.error(f"Noise analysis failed: {e}")
            # Return safe defaults
            return NoiseProfile(
                noise_floor=0.001,
                speech_threshold=0.01,
                silence_threshold=0.005,
                peak_amplitude=1.0,
                rms_level=0.1,
                dynamic_range=40.0,
                recommended_threshold=0.01,
                confidence=0.0,
            )

    def detect_speech_segments(
        self, audio: torch.Tensor, threshold: Optional[float] = None
    ) -> List[SpeechSegment]:
        """
        Detect speech segments in audio based on energy analysis.

        Args:
            audio: Audio tensor
            threshold: Optional custom threshold (uses auto-detected if None)

        Returns:
            List of detected speech segments
        """
        try:
            # Ensure audio is in correct format
            if audio.dim() == 2:
                audio = audio.squeeze(0)

            audio_np = audio.detach().cpu().numpy()

            # Get threshold
            if threshold is None:
                profile = self.analyze_noise_floor(audio)
                threshold = profile.speech_threshold

            # Frame-based analysis
            frames = self._create_frames(audio_np)
            frame_energies = self._calculate_frame_energies(frames)
            frame_rms = np.sqrt(frame_energies)

            # Detect speech frames
            speech_frames = frame_rms > threshold

            # Convert frame indices to sample indices
            frame_times = (
                np.arange(len(speech_frames)) * self.hop_size / self.sample_rate
            )

            # Find continuous speech segments
            segments = self._find_continuous_segments(
                speech_frames, frame_times, frame_rms
            )

            # Filter by minimum duration
            min_samples = int(self.min_speech_duration * self.sample_rate)
            filtered_segments = []

            for segment in segments:
                duration_samples = segment.end_sample - segment.start_sample
                if duration_samples >= min_samples:
                    filtered_segments.append(segment)

            self.logger.info(f"Detected {len(filtered_segments)} speech segments")
            return filtered_segments

        except Exception as e:
            self.logger.error(f"Speech segment detection failed: {e}")
            return []

    def analyze_reference_audio(self, reference_path: str) -> NoiseProfile:
        """
        Analyze reference audio file to establish baseline characteristics.

        Args:
            reference_path: Path to reference audio file

        Returns:
            NoiseProfile for reference audio
        """
        try:
            if not Path(reference_path).exists():
                raise FileNotFoundError(f"Reference audio not found: {reference_path}")

            # Load reference audio
            audio, sr = torchaudio.load(reference_path)

            # Resample if needed
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio = resampler(audio)

            # Analyze
            profile = self.analyze_noise_floor(audio)

            self.logger.info(f"Reference audio analyzed: {reference_path}")
            return profile

        except Exception as e:
            self.logger.error(f"Reference audio analysis failed: {e}")
            raise

    def _create_frames(self, audio_np: np.ndarray) -> np.ndarray:
        """Create overlapping frames from audio."""
        num_frames = (len(audio_np) - self.frame_size) // self.hop_size + 1
        frames = np.zeros((num_frames, self.frame_size))

        for i in range(num_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            if end <= len(audio_np):
                frames[i] = audio_np[start:end]

        return frames

    def _calculate_frame_energies(self, frames: np.ndarray) -> np.ndarray:
        """Calculate energy for each frame."""
        return np.sum(frames**2, axis=1) / frames.shape[1]

    def _calculate_speech_threshold(
        self, frame_rms: np.ndarray, noise_floor: float
    ) -> float:
        """Calculate adaptive speech threshold."""
        # Use median of upper half as speech level estimate
        sorted_rms = np.sort(frame_rms)
        upper_half = sorted_rms[len(sorted_rms) // 2 :]
        speech_level = np.median(upper_half)

        # Threshold between noise floor and speech level
        threshold = noise_floor + (speech_level - noise_floor) * 0.4

        # Ensure minimum threshold above noise floor
        min_threshold = noise_floor * 2.0
        threshold = max(threshold, min_threshold)

        return float(threshold)

    def _calculate_recommended_threshold(
        self, noise_floor: float, speech_threshold: float, dynamic_range_db: float
    ) -> float:
        """Calculate recommended threshold for Auto-Editor."""
        # Base on speech threshold but adjust for dynamic range
        base_threshold = speech_threshold

        # Adjust based on dynamic range
        if dynamic_range_db > 50:  # High dynamic range
            multiplier = 0.8  # Can be more aggressive
        elif dynamic_range_db > 30:  # Medium dynamic range
            multiplier = 1.0  # Use as-is
        else:  # Low dynamic range
            multiplier = 1.3  # Be more conservative

        recommended = base_threshold * multiplier

        # Ensure it's above noise floor
        recommended = max(recommended, noise_floor * 1.5)

        return float(recommended)

    def _calculate_confidence(
        self, frame_rms: np.ndarray, dynamic_range_db: float
    ) -> float:
        """Calculate confidence in the analysis."""
        # Base confidence on dynamic range
        if dynamic_range_db > 40:
            range_confidence = 1.0
        elif dynamic_range_db > 20:
            range_confidence = 0.8
        else:
            range_confidence = 0.5

        # Consistency confidence (based on variance)
        rms_variance = np.var(frame_rms)
        rms_mean = np.mean(frame_rms)
        cv = rms_variance / (rms_mean**2) if rms_mean > 0 else 1.0

        if cv < 0.5:
            consistency_confidence = 1.0
        elif cv < 1.0:
            consistency_confidence = 0.8
        else:
            consistency_confidence = 0.6

        # Combined confidence
        overall_confidence = (range_confidence + consistency_confidence) / 2.0
        return min(1.0, max(0.0, overall_confidence))

    def _find_continuous_segments(
        self, speech_frames: np.ndarray, frame_times: np.ndarray, frame_rms: np.ndarray
    ) -> List[SpeechSegment]:
        """Find continuous speech segments from frame analysis."""
        segments = []
        in_speech = False
        start_idx = 0

        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                # Start of speech segment
                start_idx = i
                in_speech = True
            elif not is_speech and in_speech:
                # End of speech segment
                segment = self._create_speech_segment(
                    start_idx, i - 1, frame_times, frame_rms
                )
                segments.append(segment)
                in_speech = False

        # Handle case where speech continues to end
        if in_speech:
            segment = self._create_speech_segment(
                start_idx, len(speech_frames) - 1, frame_times, frame_rms
            )
            segments.append(segment)

        return segments

    def _create_speech_segment(
        self,
        start_frame: int,
        end_frame: int,
        frame_times: np.ndarray,
        frame_rms: np.ndarray,
    ) -> SpeechSegment:
        """Create SpeechSegment from frame indices."""
        start_time = frame_times[start_frame]
        end_time = frame_times[end_frame]

        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)

        # Calculate segment statistics
        segment_rms = frame_rms[start_frame : end_frame + 1]
        peak_amplitude = float(np.max(segment_rms))
        avg_rms = float(np.mean(segment_rms))

        # Simple confidence based on amplitude consistency
        rms_std = np.std(segment_rms)
        confidence = 1.0 / (1.0 + rms_std / avg_rms) if avg_rms > 0 else 0.5

        return SpeechSegment(
            start_sample=start_sample,
            end_sample=end_sample,
            start_time=start_time,
            end_time=end_time,
            peak_amplitude=peak_amplitude,
            rms_level=avg_rms,
            confidence=confidence,
        )

    def get_auto_editor_params(
        self,
        profile: NoiseProfile,
        margin_before: float = 0.1,
        margin_after: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Generate Auto-Editor parameters based on noise analysis.

        Args:
            profile: Noise profile from analysis
            margin_before: Margin before cuts (seconds)
            margin_after: Margin after cuts (seconds)

        Returns:
            Dictionary of Auto-Editor parameters
        """
        # Convert threshold to decibels (Auto-Editor uses dB)
        threshold_db = 20 * math.log10(max(profile.recommended_threshold, 1e-8))

        params = {
            "silent_threshold": f"{threshold_db:.1f}dB",
            "margin_before": f"{margin_before}s",
            "margin_after": f"{margin_after}s",
            "frame_margin": 2,  # Frame margin for precision
            "min_clip_length": f"{self.min_speech_duration}s",
            "min_cut_length": f"{self.min_silence_duration}s",
        }

        self.logger.debug(f"Auto-Editor params: {params}")
        return params
