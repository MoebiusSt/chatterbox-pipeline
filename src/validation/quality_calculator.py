"""
Quality calculation utilities for audio validation.
Handles similarity scoring and overall quality assessment.
"""

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Use absolute import pattern like existing modules
sys.path.append(str(Path(__file__).resolve().parents[1]))

if TYPE_CHECKING:
    from utils.file_manager import AudioCandidate


class QualityCalculator:
    """Calculates quality metrics for audio validation."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_similarity(self, original: str, transcription: str) -> float:
        """
        Calculate similarity between original text and transcription.
        This is a simplified implementation - will be enhanced by FuzzyMatcher.

        Args:
            original: Original text
            transcription: Transcribed text

        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            original_tokens = set(original.lower().split())
            transcription_tokens = set(transcription.lower().split())

            if not original_tokens and not transcription_tokens:
                return 1.0
            if not original_tokens or not transcription_tokens:
                return 0.0

            intersection = original_tokens.intersection(transcription_tokens)
            union = original_tokens.union(transcription_tokens)

            similarity = len(intersection) / len(union) if union else 0.0
            return min(1.0, max(0.0, similarity))

        except Exception as e:
            self.logger.warning(f"Similarity calculation failed: {e}")
            return 0.0

    def calculate_quality_score(
        self, candidate: "AudioCandidate", transcription: str, similarity_score: float
    ) -> float:
        """
        Calculate overall quality score for the candidate.

        Args:
            candidate: Audio candidate
            transcription: Transcribed text
            similarity_score: Text similarity score

        Returns:
            Quality score (0.0 to 1.0)
        """
        try:
            if len(candidate.chunk_text) > 0:
                length_score = min(1.0, len(transcription) / len(candidate.chunk_text))
            else:
                length_score = 1.0 if transcription else 0.0

            quality_score = (
                similarity_score * 0.7  # 70% similarity (increased from 60%)
                + length_score * 0.30   # 30% text length comparison
            )

            return min(1.0, max(0.0, quality_score))

        except Exception as e:
            self.logger.warning(f"Quality score calculation failed: {e}")
            return similarity_score 