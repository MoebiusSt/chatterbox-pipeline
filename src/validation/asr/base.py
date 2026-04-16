"""ASR backend base class and shared data structures."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

from utils.file_manager.io_handlers.candidate_io import AudioCandidate

from ..number_normalization import normalize_text_for_numbers
from ..quality_calculator import QualityCalculator
from ..whisper_validator import ValidationResult


@dataclass
class ASRWord:
    """Single aligned word with timestamps (seconds)."""

    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    probability: Optional[float] = None


@dataclass
class ASRResult:
    """Common result type for all ASR backends.

    words is None when the backend does not provide word-level alignment
    (e.g. OpenAI Whisper). Consumers fall back to their legacy behaviour in
    that case.
    """

    transcription: str
    language: str
    duration_s: float
    backend: str
    words: Optional[List[ASRWord]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class ASRBackend(ABC):
    """Abstract ASR backend used by the validation pipeline.

    Concrete implementations wrap OpenAI Whisper or VibeVoice-ASR. All backends
    share the similarity/length scoring logic implemented here, so switching
    the backend only changes the transcription source (and optionally adds
    word-level alignment), not the validation mathematics.
    """

    backend_name: str = "abstract"
    supports_alignment: bool = False

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        validation_cfg = self.config.get("validation", {}) or {}
        self.similarity_threshold: float = float(validation_cfg.get("similarity_threshold", 0.7))
        self.min_quality_score: float = float(validation_cfg.get("min_quality_score", 0.75))
        self.quality_calculator = QualityCalculator()

    # ------------------------------------------------------------------
    # Abstract primitives
    # ------------------------------------------------------------------

    @abstractmethod
    def transcribe_with_alignment(
        self,
        audio: torch.Tensor,
        language: str = "en",
        sample_rate: int = 24000,
    ) -> ASRResult:
        """Transcribe audio and (if supported) return word-level timestamps."""

    # ------------------------------------------------------------------
    # Default high-level API
    # ------------------------------------------------------------------

    def validate_candidate(
        self,
        candidate: AudioCandidate,
        original_text: str,
        sample_rate: int = 24000,
        language: str = "en",
    ) -> ValidationResult:
        """Transcribe and score the candidate against the original text."""
        start_time = datetime.now()
        try:
            if candidate.audio_tensor is None:
                raise ValueError("AudioCandidate.audio_tensor is None; cannot transcribe")

            asr_result = self.transcribe_with_alignment(
                candidate.audio_tensor,
                language=language or "en",
                sample_rate=sample_rate,
            )
            return self.score_transcription(
                candidate=candidate,
                original_text=original_text,
                transcription=asr_result.transcription,
                language=language or "en",
                start_time=start_time,
            )
        except Exception as e:
            validation_time = (datetime.now() - start_time).total_seconds()
            return ValidationResult(
                is_valid=False,
                transcription="",
                similarity_score=0.0,
                quality_score=0.0,
                validation_time=validation_time,
                error_message=str(e),
                asr_backend=self.backend_name,
            )

    def score_transcription(
        self,
        candidate: AudioCandidate,
        original_text: str,
        transcription: str,
        language: str = "en",
        start_time: Optional[datetime] = None,
    ) -> ValidationResult:
        """Compute ValidationResult from a pre-existing transcription.

        Shared by all backends to keep scoring identical regardless of ASR
        source. Mirrors ``WhisperValidator.validate_candidate`` scoring.
        """
        start_time = start_time or datetime.now()
        safe_language = language or "en"
        validation_cfg: Dict[str, Any] = self.config.get("validation", {}) or {}

        numbers_mode = str(validation_cfg.get("numbers_normalization_mode", "placeholder")).lower()
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
        base_original_for_length = original_for_sim if apply_norm else original_text
        quality_score = self.quality_calculator.calculate_quality_score(
            candidate,
            transcription,
            similarity_score,
            original_text_for_length=base_original_for_length,
        )

        eff_thr = self._compute_effective_similarity_threshold(original_text)
        strict_validation = (
            similarity_score >= eff_thr and quality_score >= self.min_quality_score
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

        return ValidationResult(
            is_valid=is_valid,
            transcription=transcription,
            similarity_score=similarity_score,
            quality_score=quality_score,
            validation_time=validation_time,
            normalized_transcription=(transcr_for_sim if apply_norm else None),
            normalization_language=(safe_language if apply_norm else None),
            numbers_normalization_mode=(numbers_mode if apply_norm else None),
            base_similarity_threshold=self.similarity_threshold,
            effective_similarity_threshold=eff_thr,
            asr_backend=self.backend_name,
        )

    def _compute_effective_similarity_threshold(self, text: str) -> float:
        """Dynamic threshold based on text length and punctuation density.

        Applies a capped downward adjustment so long texts do not get arbitrarily
        lenient. Honours ``validation.similarity_threshold_bonus_max`` (default
        0.08) mirroring :class:`WhisperValidator`.
        """
        try:
            base = float(self.similarity_threshold)
            if not text:
                return max(0.0, min(1.0, base))
            length = len(text)
            import re
            punct_count = len(re.findall(r"[\.,;:!\?\-\(\)\[\]\{\}\"'«»„“”‚‘’]", text))
            punct_density = punct_count / max(1, length)
            val_cfg = self.config.get("validation", {}) if isinstance(self.config, dict) else {}
            try:
                bonus_max = float(val_cfg.get("similarity_threshold_bonus_max", 0.08))
            except Exception:
                bonus_max = 0.08
            bonus_max = max(0.0, min(0.2, bonus_max))
            raw = 0.05 * min(1.0, length / 300.0) + 0.03 * min(1.0, punct_density * 10)
            adj = min(bonus_max, raw)
            thr = max(0.0, base - adj)
            return float(thr)
        except Exception:
            return float(self.similarity_threshold)
