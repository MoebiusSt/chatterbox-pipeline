"""ASR backend abstraction for validation pipeline.

Provides a pluggable transcription backend used by validation, tail-trim and
prosody scoring. Default backend wraps OpenAI Whisper (existing behaviour).
VibeVoiceASRBackend adds long-form transcription with word-level timestamps
for use with VibeVoice TTS.
"""

from .base import ASRBackend, ASRResult, ASRWord
from .factory import resolve_asr_backend
from .whisper_backend import WhisperBackend

__all__ = [
    "ASRBackend",
    "ASRResult",
    "ASRWord",
    "WhisperBackend",
    "resolve_asr_backend",
]
