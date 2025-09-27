"""
Whisper-based audio validation module for quality assurance.
Transcribes generated audio and compares against original text.
"""

import logging

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import torchaudio
import whisper

from utils.file_manager.io_handlers.candidate_io import AudioCandidate

from .quality_calculator import QualityCalculator
from .number_normalization import normalize_text_for_numbers
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
    # Optional normalization metadata
    normalized_transcription: Optional[str] = None
    normalization_language: Optional[str] = None
    numbers_normalization_mode: Optional[str] = None

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
        self.models: Dict[str, Any] = {}  # Cache for language-specific models
        self.quality_calculator = QualityCalculator()
        self.transcription_io = TranscriptionIO()
        
        # Load default model (backwards compatibility)
        self.model = None
        self._load_model()

    def _load_model(self, language: str = "en"):
        """Load default Whisper model for backwards compatibility."""
        try:
            self.logger.debug(
                f"Loading default Whisper model '{self.model_size}' on device '{self.device}'..."
            )
            self.model = whisper.load_model(self.model_size, device=self.device)
            self.models["default"] = self.model
            self.logger.debug("Default Whisper model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load default Whisper model: {e}")
            raise
    
    def _get_model_for_language(self, language: str = "en"):
        """
        Get or load appropriate Whisper model for the specified language.
        
        Args:
            language: Language code (e.g., "en", "de", "fr")
            
        Returns:
            Loaded Whisper model
        """
        try:
            # Check if we already have a cached model for this language
            if language in self.models:
                return self.models[language]
            
            # Determine best model for language
            model_name = self._get_model_name_for_language(language)
            
            self.logger.debug(f"Loading Whisper model '{model_name}' for language '{language}'...")
            model = whisper.load_model(model_name, device=self.device)
            
            # Cache the model
            self.models[language] = model
            self.logger.debug(f"✅ Loaded {model_name} model for language '{language}'")
            
            return model
            
        except Exception as e:
            self.logger.warning(f"Failed to load language-specific model for '{language}': {e}")
            # Fallback to default model
            if "default" in self.models:
                self.logger.warning(f"Using default model as fallback for language '{language}'")
                return self.models["default"]
            else:
                raise
    
    def _get_model_name_for_language(self, language: str) -> str:
        """
        Determine the best Whisper model name for a given language.
        
        Args:
            language: Language code
            
        Returns:
            Model name to use
        """
        # For English, prefer the English-only model variant when available.
        # For all other languages, use the multilingual model.
        lang = (language or "en").lower()
        if lang == "en":
            # Only these sizes have dedicated ".en" variants in Whisper
            sizes_with_en_variant = {"tiny", "base", "small", "medium"}
            if self.model_size in sizes_with_en_variant:
                return f"{self.model_size}.en"
            # e.g., "large" has no ".en" variant
            return self.model_size

        return self.model_size

    def transcribe_audio(
        self,
        audio: torch.Tensor,
        sample_rate: int = 24000,
        language: str = "en",
        initial_prompt: Optional[str] = None,
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

            # Get language-appropriate model
            model = self._get_model_for_language(language)
            
            kwargs = {
                "language": language,
                "task": "transcribe",
                "fp16": False,
            }
            if initial_prompt:
                kwargs["initial_prompt"] = initial_prompt

            result: Dict[str, Any] = model.transcribe(audio_np, **kwargs)

            transcription = str(result.get("text", "")).strip()

            duration = (datetime.now() - start_time).total_seconds()
            self.logger.debug(
                f"Transcription completed in {duration:.2f}s: '{transcription[:100]}...'"
            )

            return transcription

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise

    def validate_candidate(
        self,
        candidate: AudioCandidate,
        original_text: str,
        sample_rate: int = 24000,
        language: str = "en",
    ) -> ValidationResult:
        """
        Validate an audio candidate against original text.

        Args:
            candidate: Audio candidate to validate
            original_text: Original text for comparison
            sample_rate: Sample rate of audio
            language: Language code for transcription

        Returns:
            ValidationResult with validation outcome
        """
        start_time = datetime.now()

        try:
            # Prepare optional initial prompt from config
            validation_cfg = getattr(self, "_config", {}).get("validation", {}) if hasattr(self, "_config") else {}
            prompt_enabled = bool(validation_cfg.get("whisper_initial_prompt_enabled", False))
            prompt_text = str(validation_cfg.get("whisper_initial_prompt_text", "")).strip() if prompt_enabled else None

            # Guard language against None or empty strings
            safe_language = language or "en"

            if candidate.audio_tensor is None:
                raise ValueError("AudioCandidate.audio_tensor is None; cannot transcribe")

            transcription = self.transcribe_audio(
                candidate.audio_tensor,
                sample_rate=sample_rate,
                language=safe_language,
                initial_prompt=prompt_text if prompt_enabled and prompt_text else None,
            )

            # Use QualityCalculator for scoring
            # Apply number normalization for non-English if configured
            numbers_mode = str(validation_cfg.get("numbers_normalization_mode", "placeholder")).lower() if validation_cfg else "placeholder"
            apply_norm = safe_language.lower() != "en" and numbers_mode in {"placeholder", "digits", "words"}

            if apply_norm:
                original_for_sim = normalize_text_for_numbers(original_text, safe_language, numbers_mode)
                transcr_for_sim = normalize_text_for_numbers(transcription, safe_language, numbers_mode)
            else:
                original_for_sim = original_text
                transcr_for_sim = transcription

            similarity_score = self.quality_calculator.calculate_similarity(
                original_for_sim, transcr_for_sim
            )
            # For length comparison, we want to use the same normalization (if applied)
            base_original_for_length = original_for_sim if apply_norm else original_text
            quality_score = self.quality_calculator.calculate_quality_score(
                candidate,
                transcription,
                similarity_score,
                original_text_for_length=base_original_for_length,
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
                normalized_transcription=(transcr_for_sim if apply_norm else None),
                normalization_language=(safe_language if apply_norm else None),
                numbers_normalization_mode=(numbers_mode if apply_norm else None),
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
