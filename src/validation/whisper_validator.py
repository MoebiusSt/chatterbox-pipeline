"""
Whisper-based audio validation module for quality assurance.
Transcribes generated audio and compares against original text.
"""

import logging

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import torch
import torchaudio
import whisper

from utils.file_manager.io_handlers.candidate_io import AudioCandidate

from .quality_calculator import QualityCalculator
from .transcription_io import TranscriptionIO

@dataclass
class ValidationResult:
    """Result of audio validation process."""

    is_valid: bool
    transcription: str
    similarity_score: float
    quality_score: float
    validation_time: float
    error_message: Optional[str] = None

class WhisperValidator:
    """
    Validates audio candidates using Whisper speech-to-text.
    Transcribes audio and compares against original text for quality assurance.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        similarity_threshold: float = 0.90,
        min_quality_score: float = 0.6,
    ):
        self.model_size = model_size
        self.similarity_threshold = similarity_threshold
        self.min_quality_score = min_quality_score
        self.logger = logging.getLogger(__name__)

        # Auto-detect device if needed
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model = None
        self.quality_calculator = QualityCalculator()
        self.transcription_io = TranscriptionIO()
        self._load_model()

    def _load_model(self):
        """Load Whisper model."""
        try:
            self.logger.debug(
                f"Loading Whisper model '{self.model_size}' on device '{self.device}'..."
            )
            self.model = whisper.load_model(self.model_size, device=self.device)
            self.logger.debug("Whisper model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe_audio(
        self, audio: torch.Tensor, sample_rate: int = 24000, language: str = "en"
    ) -> str:
        """
        Transcribe audio tensor using Whisper.

        Args:
            audio: Audio tensor (1, num_samples) or (num_samples,)
            sample_rate: Sample rate of audio
            language: Language code for transcription

        Returns:
            Transcribed text
        """
        start_time = datetime.now()

        try:
            # Ensure audio is in correct format
            if audio.dim() == 2:
                audio = audio.squeeze(0)  # Remove channel dimension

            # Convert to numpy and ensure float32
            audio_np = audio.detach().cpu().numpy().astype("float32")

            # Resample to 16kHz if needed (Whisper requirement)
            if sample_rate != 16000:
                audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=16000
                )
                audio_resampled = resampler(audio_tensor).squeeze(0)
                audio_np = audio_resampled.numpy()

            if self.model is None:
                raise RuntimeError(
                    "Whisper model not loaded. Call _load_model() first."
                )

            result = self.model.transcribe(
                audio_np,
                language=language,
                task="transcribe",
                fp16=False,
            )

            transcription = result["text"].strip()

            duration = (datetime.now() - start_time).total_seconds()
            self.logger.debug(
                f"Transcription completed in {duration:.2f}s: '{transcription[:100]}...'"
            )

            return transcription

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise

    def validate_candidate(
        self, candidate: AudioCandidate, original_text: str, sample_rate: int = 24000
    ) -> ValidationResult:
        """
        Validate an audio candidate against original text.

        Args:
            candidate: Audio candidate to validate
            original_text: Original text for comparison
            sample_rate: Sample rate of audio

        Returns:
            ValidationResult with validation outcome
        """
        start_time = datetime.now()

        try:
            transcription = self.transcribe_audio(
                candidate.audio_tensor, sample_rate=sample_rate
            )

            # Use QualityCalculator for scoring
            similarity_score = self.quality_calculator.calculate_similarity(
                original_text, transcription
            )
            quality_score = self.quality_calculator.calculate_quality_score(
                candidate, transcription, similarity_score
            )

            # Validation logic with flexibility
            strict_validation = (
                similarity_score >= self.similarity_threshold
                and quality_score >= self.min_quality_score
            )

            flexible_validation = (
                (
                    quality_score >= self.min_quality_score + 0.02
                    and similarity_score >= self.similarity_threshold - 0.1
                )
                or (
                    similarity_score >= self.similarity_threshold + 0.02
                    and quality_score >= self.min_quality_score - 0.1
                )
                or (
                    similarity_score >= self.similarity_threshold - 0.05
                    and quality_score >= self.min_quality_score - 0.05
                    and (similarity_score + quality_score)
                    >= (self.similarity_threshold + self.min_quality_score) - 0.05
                )
            )

            is_valid = strict_validation or flexible_validation

            validation_time = (datetime.now() - start_time).total_seconds()

            result = ValidationResult(
                is_valid=is_valid,
                transcription=transcription,
                similarity_score=similarity_score,
                quality_score=quality_score,
                validation_time=validation_time,
            )

            return result

        except Exception as e:
            validation_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Validation failed: {e}")

            return ValidationResult(
                is_valid=False,
                transcription="",
                similarity_score=0.0,
                quality_score=0.0,
                validation_time=validation_time,
                error_message=str(e),
            )

    def batch_validate(
        self,
        candidates: list[AudioCandidate],
        original_texts: list[str],
        sample_rate: int = 24000,
    ) -> list[ValidationResult]:
        """
        Validate multiple candidates in batch.

        Args:
            candidates: List of audio candidates
            original_texts: List of original texts (same length as candidates)
            sample_rate: Sample rate of audio

        Returns:
            List of validation results
        """
        if len(candidates) != len(original_texts):
            raise ValueError("Number of candidates must match number of original texts")

        results = []
        for i, (candidate, original_text) in enumerate(zip(candidates, original_texts)):
            self.logger.info(f"Validating candidate {i+1}/{len(candidates)}...")
            result = self.validate_candidate(candidate, original_text, sample_rate)
            results.append(result)

        return results

    def save_transcriptions_to_disk(
        self,
        transcriptions: List[str],
        chunk_index: int,
        candidate_indices: List[int],
        validation_data: Optional[List[dict]] = None,
        output_dir: Optional[str] = None,
    ) -> List[str]:
        """Delegate to TranscriptionIO for saving transcriptions."""
        return self.transcription_io.save_transcriptions_to_disk(
            transcriptions, chunk_index, candidate_indices, validation_data, output_dir
        )
