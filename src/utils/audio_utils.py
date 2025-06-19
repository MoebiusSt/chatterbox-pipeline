"""
Audio utility functions for processing and manipulation.
Provides common audio operations like concatenation and silence insertion.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio


def concatenate_audio_segments(
    segments: List[torch.Tensor], sample_rate: Optional[int] = None
) -> torch.Tensor:
    """
    Concatenate multiple audio segments into a single tensor.

    Args:
        segments: List of audio tensors to concatenate
        sample_rate: Sample rate of audio (for validation)

    Returns:
        Concatenated audio tensor
    """
    if not segments:
        return torch.empty(0)

    # Ensure all segments have same dimensions
    processed_segments = []
    for segment in segments:
        if segment.dim() == 1:
            processed_segments.append(segment.unsqueeze(0))
        else:
            processed_segments.append(segment)

    # Concatenate along time dimension
    return torch.cat(processed_segments, dim=1)


def add_silence_between_segments(
    segments: List[torch.Tensor],
    silence_duration: float,
    sample_rate: Optional[int] = None,
    device: Optional[str] = None,
) -> List[torch.Tensor]:
    """
    Add silence between audio segments.

    Args:
        segments: List of audio segments
        silence_duration: Duration of silence in seconds
        sample_rate: Sample rate
        device: Device for silence tensors

    Returns:
        List with silence inserted between segments
    """
    if not segments:
        return segments

    if device is None:
        device = segments[0].device if len(segments) > 0 else "cpu"

    if sample_rate is None:
        raise ValueError(
            "sample_rate must be provided explicitly. Check pipeline config for 'audio.sample_rate'."
        )

    silence_samples = int(silence_duration * sample_rate)
    silence = torch.zeros(1, silence_samples, device=device)

    result = []
    for i, segment in enumerate(segments):
        result.append(segment)
        if i < len(segments) - 1:  # Don't add silence after last segment
            result.append(silence)

    return result


def normalize_audio(
    audio: torch.Tensor, target_rms: float = 0.2, prevent_clipping: bool = True
) -> torch.Tensor:
    """
    Normalize audio to target RMS level.

    Args:
        audio: Input audio tensor
        target_rms: Target RMS level
        prevent_clipping: Whether to prevent clipping

    Returns:
        Normalized audio
    """
    current_rms = torch.sqrt(torch.mean(audio**2))

    if current_rms < 1e-8:
        return audio

    gain = target_rms / current_rms

    if prevent_clipping:
        max_amplitude = torch.max(torch.abs(audio))
        max_gain = 0.95 / max_amplitude if max_amplitude > 0 else 1.0
        gain = min(gain, max_gain)

    return audio * gain


def apply_fade_in_out(
    audio: torch.Tensor,
    fade_in_duration: float = 0.01,
    fade_out_duration: float = 0.01,
    sample_rate: Optional[int] = None,
) -> torch.Tensor:
    """
    Apply fade in and fade out to audio.

    Args:
        audio: Input audio tensor
        fade_in_duration: Fade in duration in seconds
        fade_out_duration: Fade out duration in seconds
        sample_rate: Sample rate

    Returns:
        Audio with fades applied
    """
    if audio.dim() == 2:
        audio = audio.squeeze(0)

    if sample_rate is None:
        raise ValueError(
            "sample_rate must be provided explicitly. Check pipeline config for 'audio.sample_rate'."
        )

    fade_in_samples = int(fade_in_duration * sample_rate)
    fade_out_samples = int(fade_out_duration * sample_rate)

    audio_length = len(audio)
    faded_audio = audio.clone()

    # Apply fade in
    if fade_in_samples > 0 and fade_in_samples < audio_length:
        fade_in_curve = torch.linspace(0, 1, fade_in_samples)
        faded_audio[:fade_in_samples] *= fade_in_curve

    # Apply fade out
    if fade_out_samples > 0 and fade_out_samples < audio_length:
        fade_out_curve = torch.linspace(1, 0, fade_out_samples)
        faded_audio[-fade_out_samples:] *= fade_out_curve

    return faded_audio


def resample_audio(audio: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    """
    Resample audio to a new sample rate.

    Args:
        audio: Input audio tensor
        orig_freq: Original sample rate
        new_freq: Target sample rate

    Returns:
        Resampled audio
    """
    if orig_freq == new_freq:
        return audio

    resampler = torchaudio.transforms.Resample(orig_freq, new_freq)
    return resampler(audio)


def save_audio_tensor(
    audio: torch.Tensor,
    file_path: str,
    sample_rate: Optional[int] = None,  # Remove hardcoded default
) -> None:
    """
    Save audio tensor to file.

    Args:
        audio: Audio tensor to save
        file_path: Output file path
        sample_rate: Audio sample rate (REQUIRED - must be provided by caller)
    """
    # Require sample_rate to be provided explicitly
    if sample_rate is None:
        raise ValueError(
            "sample_rate must be provided explicitly. Check pipeline config for 'audio.sample_rate'."
        )

    try:
        # Ensure audio is 2D (channels, samples)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # Ensure consistent data type
        audio = audio.float()

        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Save audio
        torchaudio.save(file_path, audio, sample_rate)

    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save audio to {file_path}: {e}")
        raise


def load_audio_tensor(
    file_path: str, target_sample_rate: Optional[int] = None  # Remove hardcoded default
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file as tensor.

    Args:
        file_path: Path to audio file
        target_sample_rate: Target sample rate for resampling (optional)

    Returns:
        Tuple of (audio_tensor, actual_sample_rate)
    """
    try:
        audio, sample_rate = torchaudio.load(file_path)

        # Resample if target rate specified and different
        if target_sample_rate is not None and sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_sample_rate
            )
            audio = resampler(audio)
            sample_rate = target_sample_rate

        return audio, sample_rate

    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load audio from {file_path}: {e}")
        raise


def calculate_duration(
    audio: torch.Tensor, sample_rate: Optional[int] = None  # Remove hardcoded default
) -> float:
    """
    Calculate audio duration in seconds.

    Args:
        audio: Audio tensor
        sample_rate: Audio sample rate (REQUIRED - must be provided by caller)

    Returns:
        Duration in seconds
    """
    # Require sample_rate to be provided explicitly
    if sample_rate is None:
        raise ValueError(
            "sample_rate must be provided explicitly. Check pipeline config for 'audio.sample_rate'."
        )

    if audio.dim() == 2:
        num_samples = audio.shape[1]
    else:
        num_samples = audio.shape[0]

    return num_samples / sample_rate


def create_silence(
    duration: float, sample_rate: Optional[int] = None  # Remove hardcoded default
) -> torch.Tensor:
    """
    Create silence tensor of specified duration.

    Args:
        duration: Duration in seconds
        sample_rate: Audio sample rate (REQUIRED - must be provided by caller)

    Returns:
        Silence tensor
    """
    # Require sample_rate to be provided explicitly
    if sample_rate is None:
        raise ValueError(
            "sample_rate must be provided explicitly. Check pipeline config for 'audio.sample_rate'."
        )

    num_samples = int(duration * sample_rate)
    return torch.zeros(1, num_samples)


def normalize_audio(audio: torch.Tensor, target_peak: float = 0.9) -> torch.Tensor:
    """
    Normalize audio to target peak amplitude.

    Args:
        audio: Audio tensor to normalize
        target_peak: Target peak amplitude (0.0 to 1.0)

    Returns:
        Normalized audio tensor
    """
    # Find current peak
    current_peak = audio.abs().max()

    if current_peak > 0:
        # Calculate normalization factor
        norm_factor = target_peak / current_peak
        audio = audio * norm_factor

    return audio


def detect_silence(
    audio: torch.Tensor,
    threshold: float = 0.01,
    min_duration: float = 0.1,
    sample_rate: Optional[int] = None,  # Remove hardcoded default
) -> List[Tuple[float, float]]:
    """
    Detect silence regions in audio.

    Args:
        audio: Audio tensor
        threshold: Amplitude threshold for silence detection
        min_duration: Minimum duration of silence to detect (seconds)
        sample_rate: Audio sample rate (REQUIRED - must be provided by caller)

    Returns:
        List of (start_time, end_time) tuples for silence regions
    """
    # Require sample_rate to be provided explicitly
    if sample_rate is None:
        raise ValueError(
            "sample_rate must be provided explicitly. Check pipeline config for 'audio.sample_rate'."
        )

    # Convert to mono if stereo
    if audio.dim() == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0)
    elif audio.dim() == 2:
        audio = audio.squeeze(0)

    # Find samples below threshold
    silence_mask = audio.abs() < threshold

    # Find continuous silence regions
    silence_regions = []
    in_silence = False
    silence_start = 0

    min_samples = int(min_duration * sample_rate)

    for i, is_silent in enumerate(silence_mask):
        if is_silent and not in_silence:
            # Start of silence
            silence_start = i
            in_silence = True
        elif not is_silent and in_silence:
            # End of silence
            silence_duration_samples = i - silence_start
            if silence_duration_samples >= min_samples:
                start_time = silence_start / sample_rate
                end_time = i / sample_rate
                silence_regions.append((start_time, end_time))
            in_silence = False

    # Handle silence extending to end of audio
    if in_silence:
        silence_duration_samples = len(silence_mask) - silence_start
        if silence_duration_samples >= min_samples:
            start_time = silence_start / sample_rate
            end_time = len(silence_mask) / sample_rate
            silence_regions.append((start_time, end_time))

    return silence_regions
