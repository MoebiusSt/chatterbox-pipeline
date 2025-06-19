"""
Post-processing module for audio artifact removal and enhancement.
Provides Auto-Editor integration, noise analysis, and advanced cleaning.
"""

from .audio_cleaner import AudioCleaner, CleaningSettings
from .auto_editor_wrapper import AutoEditorWrapper, ProcessingResult
from .noise_analyzer import NoiseAnalyzer, NoiseProfile, SpeechSegment

__all__ = [
    "NoiseAnalyzer",
    "NoiseProfile",
    "SpeechSegment",
    "AutoEditorWrapper",
    "ProcessingResult",
    "AudioCleaner",
    "CleaningSettings",
]
