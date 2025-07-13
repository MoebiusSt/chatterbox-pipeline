import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchaudio as ta

from utils.file_manager.io_handlers.candidate_io import AudioCandidate

# Use the centralized logging configuration from cbpipe.py
logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Handles audio processing operations including concatenation,
    silence insertion, and file I/O operations.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        normal_silence_duration: float = 0.20,
        paragraph_silence_duration: float = 0.20,
        device: str = "cpu",
    ):
        self.sample_rate = sample_rate
        self.normal_silence_duration = normal_silence_duration
        self.paragraph_silence_duration = paragraph_silence_duration
        self.device = device

        # Pre-calculate silence tensors
        self.normal_silence = self._create_silence_tensor(normal_silence_duration)
        self.paragraph_silence = self._create_silence_tensor(paragraph_silence_duration)

        logger.info(f"AudioProcessor initialized: sr={sample_rate}, device={device}")

    def _create_silence_tensor(self, duration: float) -> torch.Tensor:
        """
        Creates a silence tensor of specified duration in seconds.

        Returns:
            Silence tensor.
        """
        num_samples = int(self.sample_rate * duration)
        silence = torch.zeros((1, num_samples), device=self.device)
        return silence

    def add_silence(
        self, audio: torch.Tensor, duration: float, position: str = "after"
    ) -> torch.Tensor:

        silence = self._create_silence_tensor(duration)

        if position == "before":
            return torch.cat([silence, audio], dim=1)
        elif position == "after":
            return torch.cat([audio, silence], dim=1)
        elif position == "both":
            return torch.cat([silence, audio, silence], dim=1)
        else:
            logger.warning(f"Unknown position '{position}', using 'after'")
            return torch.cat([audio, silence], dim=1)

    def concatenate_segments(
        self,
        audio_segments: List[torch.Tensor],
        has_paragraph_breaks: Optional[List[bool]] = None,
    ) -> torch.Tensor:
        """
        Concatenates audio segments with appropriate silence insertion.

        """
        if not audio_segments:
            logger.warning("No audio segments provided for concatenation")
            return torch.zeros((1, 1000), device=self.device)

        logger.info(f"Concatenating {len(audio_segments)} audio segments")

        # Ensure all segments have consistent dimensions and are on the same device
        processed_segments = []

        for i, segment in enumerate(audio_segments):
            # Ensure tensor is 2D (batch_size, length)
            if segment.ndim == 1:
                segment = segment.unsqueeze(0)

            # Move to correct device
            segment = segment.to(self.device)

            processed_segments.append(segment)

            # Add silence after segment (except for the last one)
            if i < len(audio_segments) - 1:
                if has_paragraph_breaks and i < len(has_paragraph_breaks):
                    if has_paragraph_breaks[i]:
                        processed_segments.append(self.paragraph_silence)
                    else:
                        processed_segments.append(self.normal_silence)
                else:
                    # Default to normal silence
                    processed_segments.append(self.normal_silence)

        # Concatenate all segments
        try:
            final_audio = torch.cat(processed_segments, dim=1)
            logger.info(f"Concatenation successful: final shape {final_audio.shape}")
            return final_audio
        except Exception as e:
            logger.error(f"Error concatenating audio segments: {e}")
            # Return silence as fallback
            return torch.zeros((1, self.sample_rate), device=self.device)

    def concatenate_candidates(
        self,
        candidates: List[AudioCandidate],
        has_paragraph_breaks: Optional[List[bool]] = None,
    ) -> torch.Tensor:
        """
        Concatenates audio from a list of AudioCandidate objects.

        """
        audio_segments = [candidate.audio_tensor for candidate in candidates]
        return self.concatenate_segments(audio_segments, has_paragraph_breaks)

    def save_audio(
        self, audio: torch.Tensor, output_path: str, sample_rate: Optional[int] = None
    ) -> bool:
        """
        Saves audio tensor to file.

        Returns:
            True if successful, False otherwise.
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        try:
            # Ensure audio is on CPU for saving
            audio_cpu = audio.cpu()

            # Ensure parent directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Save audio file
            ta.save(output_path, audio_cpu, sample_rate)
            logger.info(f"Audio saved successfully to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving audio to {output_path}: {e}")
            return False

    def load_audio(
        self, input_path: str
    ) -> Tuple[Optional[torch.Tensor], Optional[int]]:
        """
        Loads audio file and returns tensor and sample rate.

        Returns:
            Tuple of (audio_tensor, sample_rate) or (None, None) if failed.
        """
        try:
            audio, sr = ta.load(input_path)

            # Move to device and ensure correct dimensions
            audio = audio.to(self.device)
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)

            logger.info(
                f"Audio loaded successfully from {input_path}: shape={audio.shape}, sr={sr}"
            )
            return audio, sr

        except Exception as e:
            logger.error(f"Error loading audio from {input_path}: {e}")
            return None, None

    def get_audio_duration(self, audio: torch.Tensor) -> float:
        """
        Calculates the duration of an audio tensor in seconds.

        """
        if audio.ndim == 1:
            length = audio.shape[0]
        else:
            length = audio.shape[1]

        duration = length / self.sample_rate
        return duration

    def trim_silence(
        self, audio: torch.Tensor, threshold: float = 0.01, frame_length: int = 2048
    ) -> torch.Tensor:
        """
        Trims silence from the beginning and end of audio.

        Args:
            audio: Input audio tensor.
            threshold: Amplitude threshold for silence detection.
            frame_length: Frame length for analysis.

        Returns:
            Trimmed audio tensor.
        """
        try:
            # Ensure audio is 1D for trimming
            if audio.ndim > 1:
                audio_1d = audio.squeeze()
            else:
                audio_1d = audio

            # Find non-silent regions
            non_silent = torch.abs(audio_1d) > threshold

            if not non_silent.any():
                logger.warning("No non-silent audio found")
                return audio

            # Find start and end indices
            nonzero_indices = torch.nonzero(non_silent)
            if len(nonzero_indices) == 0:
                logger.warning("No non-silent frames found")
                return audio

            start_idx = int(nonzero_indices[0].item())
            end_idx = int(nonzero_indices[-1].item()) + 1

            # Trim audio
            trimmed = audio_1d[start_idx:end_idx]

            # Restore original dimensions
            if audio.ndim > 1:
                trimmed = trimmed.unsqueeze(0)

            logger.debug(f"Trimmed audio: {audio.shape} -> {trimmed.shape}")
            return trimmed

        except Exception as e:
            logger.error(f"Error trimming silence: {e}")
            return audio
