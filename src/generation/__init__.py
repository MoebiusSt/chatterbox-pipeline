"""
Audio generation module for TTS candidate generation and management.
Provides TTS generation, candidate management, and audio processing.
"""

from .candidate_manager import CandidateManager
from .batch_processor import BatchProcessor, GenerationResult
from .selection_strategies import SelectionStrategies
from .tts_generator import TTSGenerator
from .audio_processor import AudioProcessor
from .model_cache import ChatterboxModelCache

__all__ = [
    "TTSGenerator", 
    "CandidateManager", 
    "AudioProcessor", 
    "BatchProcessor", 
    "GenerationResult", 
    "SelectionStrategies",
    "ChatterboxModelCache"
]
