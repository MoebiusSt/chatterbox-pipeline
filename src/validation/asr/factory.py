"""ASR backend factory with auto-routing by model type."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..whisper_validator import WhisperValidator
from .base import ASRBackend
from .whisper_backend import WhisperBackend

logger = logging.getLogger(__name__)


def resolve_asr_backend(
    config: Dict[str, Any],
    whisper_validator: Optional[WhisperValidator] = None,
) -> ASRBackend:
    """Pick the ASR backend based on config and TTS model type.

    Precedence:
      1. ``validation.asr_backend`` explicit ``whisper`` / ``vibevoice_asr``
      2. ``validation.asr_backend: auto`` (default) → ``whisper`` for **all**
         model types (including VibeVoice).

    VibeVoice-ASR remains available as opt-in via
    ``validation.asr_backend: vibevoice_asr`` but is no longer auto-selected:
    on consumer GPUs (≤16 GB) its ~16 GB weight footprint forces VRAM
    spillover after the TTS model, and the vendored long-form inference path
    currently hits an off-by-one mask bug on ≥4-min clips. Whisper is faster,
    stable, and delivers adequate alignment for our scorer stack.
    """
    validation_cfg: Dict[str, Any] = config.get("validation", {}) or {}
    sel = str(validation_cfg.get("asr_backend", "auto")).strip().lower()
    model_type = str(config.get("generation", {}).get("model_type", "")).strip().lower()

    if sel == "auto":
        sel = "whisper"

    if sel == "vibevoice_asr":
        try:
            from .vibevoice_asr_backend import VibeVoiceASRBackend
            logger.info("🎙️  ASR backend: vibevoice_asr (model_type=%s)", model_type or "n/a")
            return VibeVoiceASRBackend(config)
        except Exception as e:
            logger.warning(
                "VibeVoice-ASR backend unavailable (%s) - falling back to Whisper", e
            )
            return WhisperBackend(config, validator=whisper_validator)

    logger.info("🎙️  ASR backend: whisper (model_type=%s)", model_type or "n/a")
    return WhisperBackend(config, validator=whisper_validator)
