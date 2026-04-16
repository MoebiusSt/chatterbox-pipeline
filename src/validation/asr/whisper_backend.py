"""WhisperBackend: wraps the existing WhisperValidator.

Preserves the exact legacy behaviour for all non-VibeVoice model types. Word
timestamps are not provided by OpenAI Whisper natively, so ``words`` is always
None. Consumers fall back to their WhisperX-based legacy paths in that case.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import torch

from utils.file_manager.io_handlers.candidate_io import AudioCandidate

from ..whisper_validator import ValidationResult, WhisperValidator
from .base import ASRBackend, ASRResult


class WhisperBackend(ASRBackend):
    """Adapter around OpenAI Whisper via the existing ``WhisperValidator``."""

    backend_name = "whisper"
    supports_alignment = False

    def __init__(self, config: Dict[str, Any], validator: Optional[WhisperValidator] = None):
        super().__init__(config)
        validation_cfg = self.config.get("validation", {}) or {}
        if validator is not None:
            self.validator = validator
        else:
            self.validator = WhisperValidator(
                model_size=validation_cfg.get("whisper_model", "base"),
                device="auto",
                similarity_threshold=validation_cfg.get("similarity_threshold", 0.7),
                min_quality_score=validation_cfg.get("min_quality_score", 0.75),
            )
        try:
            setattr(self.validator, "_config", self.config)
        except Exception:
            pass

    def transcribe_with_alignment(
        self,
        audio: torch.Tensor,
        language: str = "en",
        sample_rate: int = 24000,
    ) -> ASRResult:
        validation_cfg = self.config.get("validation", {}) or {}
        prompt_enabled = bool(validation_cfg.get("whisper_initial_prompt_enabled", False))
        prompt_text = str(validation_cfg.get("whisper_initial_prompt_text", "")).strip() if prompt_enabled else None

        transcription = self.validator.transcribe_audio(
            audio,
            sample_rate=sample_rate,
            language=language or "en",
            initial_prompt=prompt_text if prompt_enabled and prompt_text else None,
        )
        duration_s = 0.0
        try:
            if audio is not None and audio.numel() > 0:
                duration_s = float(audio.shape[-1]) / float(sample_rate)
        except Exception:
            duration_s = 0.0
        return ASRResult(
            transcription=transcription,
            language=language or "en",
            duration_s=duration_s,
            backend=self.backend_name,
            words=None,
        )

    def validate_candidate(
        self,
        candidate: AudioCandidate,
        original_text: str,
        sample_rate: int = 24000,
        language: str = "en",
    ) -> ValidationResult:
        # Delegate to the wrapped validator to preserve legacy semantics
        # (including prompt handling, number normalization, thresholds).
        return self.validator.validate_candidate(
            candidate,
            original_text,
            sample_rate=sample_rate,
            language=language or "en",
        )
