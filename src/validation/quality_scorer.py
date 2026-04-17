"""
Quality scoring module for evaluating and ranking audio candidates.
Combines multiple metrics to determine the best audio candidate.
"""

import logging

# Use absolute imports to avoid relative import issues
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, cast

from utils.file_manager.io_handlers.candidate_io import AudioCandidate
from utils.language_registry import format_generation_params_summary
from validation.fuzzy_matcher import MatchResult
from validation.whisper_validator import ValidationResult

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
    module="pkg_resources",
)
warnings.filterwarnings(
    "ignore",
    message="`LoRACompatibleLinear` is deprecated",
    category=FutureWarning,
    module="diffusers",
)


class ScoringStrategy(Enum):
    """Scoring strategy for combining scores."""

    WEIGHTED_AVERAGE = "weighted_average"


@dataclass
class QualityScore:
    """Comprehensive quality score for an audio candidate."""

    overall_score: float
    similarity_score: float
    length_score: float
    penalty_score: float
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"QualityScore(overall={self.overall_score:.3f}, "
            f"sim={self.similarity_score:.3f}, "
            f"length={self.length_score:.3f}, "
            f"penalty={self.penalty_score:.3f})"
        )


class QualityScorer:
    """
    Evaluates and scores audio candidates based on multiple quality metrics.
    Used for selecting the best candidate from multiple generations.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        sample_rate: int = 24000,
    ):
        """
        Initialize QualityScorer.

        Args:
            weights: Custom weights for different score components
            sample_rate: Audio sample rate for duration calculations
        """
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)

        # Default weights for weighted average strategy
        self.weights = weights or {
            "similarity": 0.75,  # 75% - text similarity from FuzzyMatcher or WhisperValidator
            "length": 0.25,  # 25% - text length comparison (original vs transcribed text)
        }

        # Normalize weights to ensure they sum to 1.0
        weight_sum = sum(self.weights.values())
        if abs(weight_sum - 1.0) > 1e-6:
            self.logger.info(f"Normalizing weights from {weight_sum:.6f} to 1.0")
            for key in self.weights:
                self.weights[key] = self.weights[key] / weight_sum

    def score_candidate(
        self,
        candidate: AudioCandidate,
        validation_result: ValidationResult,
        match_result: Optional[MatchResult] = None,
        expected_duration: Optional[float] = None,
    ) -> QualityScore:
        """
        Score a single audio candidate.

        Args:
            candidate: Audio candidate to score
            validation_result: Whisper validation result
            match_result: Optional fuzzy matching result
            expected_duration: Expected audio duration in seconds

        Returns:
            QualityScore with detailed scoring breakdown
        """
        try:
            # Calculate individual score components
            similarity_score = self._calculate_similarity_score(
                validation_result, match_result
            )

            length_score = self._calculate_length_score(candidate, validation_result)

            penalty_score = self._calculate_penalty_score(candidate, validation_result)

            # Combine scores according to strategy
            overall_score = self._combine_scores(
                similarity_score,
                length_score,
                penalty_score,
            )

            # Create detailed score object without redundant weights
            quality_score = QualityScore(
                overall_score=overall_score,
                similarity_score=similarity_score,
                length_score=length_score,
                penalty_score=penalty_score,
                details={
                    "candidate_id": f"chunk_{candidate.chunk_idx}_candidate_{candidate.candidate_idx}",
                    "audio_duration": self._get_audio_duration(candidate),
                    "expected_duration": expected_duration,
                    "validation_passed": validation_result.is_valid,
                    # Candidate-specific calculated scores
                    "individual_scores": {
                        "similarity_score": similarity_score,
                        "length_score": length_score,
                        "penalty_score": penalty_score,
                        "overall_score": overall_score,
                    },
                    # Raw validation metrics from the ASR backend (Whisper or
                    # VibeVoice-ASR). The fields remain ``whisper_*`` for
                    # historical compatibility; the actual source is recorded
                    # in ``asr_backend``.
                    "validation_metrics": {
                        "asr_backend": getattr(validation_result, "asr_backend", None),
                        "whisper_similarity": validation_result.similarity_score,
                        "whisper_quality": validation_result.quality_score,
                        "transcription_length": len(validation_result.transcription),
                        "original_text_length": (
                            len(candidate.chunk_text) if candidate.chunk_text else 0
                        ),
                        # diagnostics
                        "normalized_transcription": getattr(validation_result, "normalized_transcription", None),
                        "normalization_language": getattr(validation_result, "normalization_language", None),
                        "numbers_normalization_mode": getattr(validation_result, "numbers_normalization_mode", None),
                        "base_similarity_threshold": getattr(validation_result, "base_similarity_threshold", None),
                        "effective_similarity_threshold": getattr(validation_result, "effective_similarity_threshold", None),
                    },
                },
            )

            self.logger.debug(
                f"Scored candidate chunk_{candidate.chunk_idx}_candidate_{candidate.candidate_idx}: {quality_score}"
            )
            return quality_score

        except Exception as e:
            self.logger.error(f"Failed to score candidate: {e}")
            # Return minimal score on error
            return QualityScore(
                overall_score=0.0,
                similarity_score=0.0,
                length_score=0.0,
                penalty_score=1.0,
                details={},
            )

    def _calculate_similarity_score(
        self,
        validation_result: ValidationResult,
        match_result: Optional[MatchResult] = None,
    ) -> float:
        """Calculate similarity-based score."""
        # Use fuzzy match result if available, otherwise validation similarity
        if match_result is not None:
            return match_result.similarity
        else:
            return validation_result.similarity_score

    def _calculate_length_score(
        self, candidate: AudioCandidate, validation_result: ValidationResult
    ) -> float:
        """Calculate text length comparison score (original text vs transcribed text length)."""
        try:
            transcription = validation_result.transcription
            original_text = candidate.chunk_text

            if not original_text:
                return 1.0 if not transcription else 0.0

            # Completeness: ratio of transcription length to original
            length_ratio = len(transcription) / len(original_text)
            completeness = min(1.0, length_ratio)  # Cap at 1.0

            # Penalize very short transcriptions
            if length_ratio < 0.3:
                completeness *= 0.5

            # Word count similarity
            original_words = len(original_text.split())
            transcribed_words = len(transcription.split())

            if original_words > 0:
                word_ratio = transcribed_words / original_words
                word_score = 1.0 / (1.0 + abs(word_ratio - 1.0))
            else:
                word_score = 1.0 if transcribed_words == 0 else 0.0

            # Combine completeness and word count
            length_score = completeness * 0.7 + word_score * 0.3

            return min(1.0, max(0.0, length_score))

        except Exception as e:
            self.logger.warning(f"Length score calculation failed: {e}")
            return 0.5

    def _calculate_penalty_score(
        self, candidate: AudioCandidate, validation_result: ValidationResult
    ) -> float:
        """Calculate penalty factors that reduce overall score."""
        penalty = 0.0

        try:
            # Penalty for validation errors
            if validation_result.error_message:
                penalty += 0.3

            # Penalty for empty transcription
            if not validation_result.transcription.strip():
                penalty += 0.4

            # Penalty for very low similarity
            if validation_result.similarity_score < 0.3:
                penalty += 0.2

            return min(1.0, penalty)

        except Exception as e:
            self.logger.warning(f"Penalty score calculation failed: {e}")
            return 0.0

    def _combine_scores(
        self,
        similarity_score: float,
        length_score: float,
        penalty_score: float,
    ) -> float:
        """Combine individual scores using weighted average."""
        try:
            weighted_sum = (
                similarity_score * self.weights["similarity"]
                + length_score * self.weights["length"]
            )
            # Apply penalty
            overall_score = weighted_sum * (1.0 - penalty_score)

            return min(1.0, max(0.0, overall_score))

        except Exception as e:
            self.logger.error(f"Score combination failed: {e}")
            return 0.0

    def _get_audio_duration(self, candidate: AudioCandidate) -> float:
        """Get audio duration in seconds."""
        try:
            num_samples = candidate.audio_tensor.shape[-1]
            return num_samples / self.sample_rate
        except Exception:
            return 0.0

    def _create_failed_score(
        self, candidate: AudioCandidate, error: str
    ) -> QualityScore:
        """Create a failed quality score."""
        return QualityScore(
            overall_score=0.0,
            similarity_score=0.0,
            length_score=0.0,
            penalty_score=1.0,
            details={
                "error": error,
                "candidate_id": f"chunk_{candidate.chunk_idx}_candidate_{candidate.candidate_idx}",
                "failed": True,
            },
        )

    def rank_candidates(
        self,
        candidates: List[AudioCandidate],
        validation_results: List[ValidationResult],
        match_results: Optional[List[Optional[MatchResult]]] = None,  # Updated annotation
        expected_durations: Optional[List[Optional[float]]] = None,
    ) -> List[Tuple[AudioCandidate, QualityScore]]:
        """
        Rank multiple candidates by quality score.

        Candidates are sorted by overall_score (descending).
        In case of tied scores, shorter audio duration is preferred as tie-breaker.

        Args:
            candidates: List of audio candidates
            validation_results: List of validation results
            match_results: Optional list of fuzzy match results
            expected_durations: Optional list of expected durations

        Returns:
            List of (candidate, score) tuples, sorted by score (best first)
        """
        if len(candidates) != len(validation_results):
            raise ValueError("Number of candidates must match validation results")

        # Prepare optional parameters
        match_results = match_results or [None] * len(candidates)  # Now compatible with annotation
        expected_durations = expected_durations or [None] * len(candidates)

        # Score all candidates
        scored_candidates = []
        for i, (candidate, validation, match, duration) in enumerate(
            zip(candidates, validation_results, match_results, expected_durations)
        ):
            score = self.score_candidate(candidate, validation, match, duration)
            scored_candidates.append((candidate, score))

        # Sort by overall score (descending), then by audio duration (ascending) as tie-breaker
        scored_candidates.sort(
            key=lambda x: (x[1].overall_score, -self._get_audio_duration(x[0])),
            reverse=True,
        )

        # Get chunk info for better logging
        if candidates:
            chunk_idx = candidates[0].chunk_idx
            chunk_num = chunk_idx + 1
        else:
            chunk_idx = -1
            chunk_num = 'unknown'

        best_score = scored_candidates[0][1].overall_score if scored_candidates else 0.0
        worst_score = scored_candidates[-1][1].overall_score if scored_candidates else 0.0

        # Get TTS parameters from best candidate if available
        if scored_candidates:
            best_candidate = scored_candidates[0][0]
            best_params = best_candidate.generation_params or {}
        else:
            best_candidate = None
            best_params = {}
            best_idx = 0

        best_idx = best_candidate.candidate_idx + 1 if best_candidate else 1  # Display as 1-based

        # Model-aware parameter summary (Chatterbox/Qwen3/VibeVoice each have
        # distinct param surfaces; QualityScorer has no config so we infer
        # from the keys present in ``best_params``).
        params_summary = format_generation_params_summary(best_params)

        chunk_str = f"{chunk_num:02d}" if isinstance(chunk_num, int) else str(chunk_num)
        self.logger.info(
            f"Chunk_{chunk_str} - "
            f"Best candidate: {best_idx} of {len(candidates)} (score: {best_score:.3f} worst: {worst_score:.3f}) "
            f"– {params_summary}"
        )

        return scored_candidates

    def select_best_candidate(
        self,
        candidates: List[AudioCandidate],
        validation_results: List[ValidationResult],
        match_results: Optional[List[MatchResult]] = None,
        expected_durations: Optional[List[float]] = None,
    ) -> Tuple[AudioCandidate, QualityScore]:
        """
        Select the best candidate from a list.

        Args:
            candidates: List of audio candidates
            validation_results: List of validation results
            match_results: Optional list of fuzzy match results
            expected_durations: Optional list of expected durations

        Returns:
            Tuple of (best_candidate, best_score)
        """
        if not candidates:
            raise ValueError("No candidates provided")

        expected_durations_typed = cast(Optional[List[Optional[float]]], expected_durations)
        ranked_candidates = self.rank_candidates(
            candidates,
            validation_results,
            match_results,
            expected_durations_typed,
        )

        best_candidate, best_score = ranked_candidates[0]

        # The detailed logging is now done in rank_candidates method above

        return best_candidate, best_score
