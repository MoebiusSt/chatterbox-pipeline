"""
Audio generation module for TTS candidate generation and management.
Provides TTS generation, candidate management, and audio processing.
"""

from .audio_processor import AudioProcessor
from .batch_processor import BatchProcessor, GenerationResult
from .candidate_manager import CandidateManager
from .model_cache import ChatterboxModelCache
from .selection_strategies import SelectionStrategies
from .tts_generator import TTSGenerator

__all__ = [
    "TTSGenerator",
    "CandidateManager",
    "AudioProcessor",
    "BatchProcessor",
    "GenerationResult",
    "SelectionStrategies",
    "ChatterboxModelCache",
]
