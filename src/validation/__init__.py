"""
Validation module for the Enhanced TTS Pipeline.
Provides Whisper-based audio validation, fuzzy text matching, and quality scoring.
"""

from .fuzzy_matcher import FuzzyMatcher, MatchResult
from .quality_scorer import QualityScore, QualityScorer, ScoringStrategy
from .whisper_validator import ValidationResult, WhisperValidator

__all__ = [
    "WhisperValidator",
    "ValidationResult",
    "FuzzyMatcher",
    "MatchResult",
    "QualityScorer",
    "QualityScore",
    "ScoringStrategy",
]
