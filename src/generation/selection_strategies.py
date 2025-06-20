import logging
from typing import List, Optional, Tuple

from utils.file_manager import AudioCandidate

logger = logging.getLogger(__name__)


class SelectionStrategies:
    """Implements various candidate selection strategies."""

    @staticmethod
    def select_best_candidate(
        candidates: List[AudioCandidate], selection_strategy: str = "shortest"
    ) -> Optional[AudioCandidate]:
        """Selects the best audio candidate from a list based on a specified strategy."""
        if not candidates:
            logger.warning("No candidates provided for selection")
            return None

        if selection_strategy == "shortest":
            selected = min(
                candidates,
                key=lambda c: (
                    c.audio_tensor.shape[-1]
                    if c.audio_tensor is not None
                    else float("inf")
                ),
            )
            logger.debug(
                f"Selected shortest candidate with length: {selected.audio_tensor.shape[-1] if selected.audio_tensor is not None else 0}"
            )

        elif selection_strategy == "first":
            selected = candidates[0]
            logger.debug(f"Selected first candidate")

        elif selection_strategy == "random":
            import random
            selected = random.choice(candidates)
            logger.debug(f"Selected random candidate")

        else:
            logger.warning(
                f"Unknown selection strategy: {selection_strategy}, using first"
            )
            selected = candidates[0]

        return selected

    @staticmethod
    def select_best_candidate_with_validation(
        candidates_with_validation: List[Tuple],  # [(candidate, validation_result, quality_score), ...]
        prefer_valid: bool = True,
    ) -> Optional[Tuple]:
        """Selects the best candidate based on validation results and quality scores."""
        if not candidates_with_validation:
            logger.warning("No candidates provided for selection")
            return None

        logger.debug(
            f"🔍 Selecting best from {len(candidates_with_validation)} validated candidates..."
        )

        if prefer_valid:
            valid_candidates = [
                (c, v, q) for c, v, q in candidates_with_validation if v.is_valid
            ]
            invalid_candidates = [
                (c, v, q) for c, v, q in candidates_with_validation if not v.is_valid
            ]

            candidates_to_consider = (
                valid_candidates if valid_candidates else invalid_candidates
            )
            selection_pool = "valid" if valid_candidates else "invalid (fallback)"
        else:
            candidates_to_consider = candidates_with_validation
            selection_pool = "all"

        logger.debug(
            f"🎯 Considering {len(candidates_to_consider)} {selection_pool} candidates"
        )

        def sort_key(item):
            candidate, validation_result, quality_score = item
            audio_duration = (
                candidate.audio_tensor.shape[-1]
                if candidate.audio_tensor is not None
                else float("inf")
            )
            return (-quality_score, audio_duration)

        candidates_to_consider.sort(key=sort_key)

        best_candidate_tuple = candidates_to_consider[0]
        candidate, validation_result, quality_score = best_candidate_tuple

        audio_duration = (
            candidate.audio_tensor.shape[-1]
            if candidate.audio_tensor is not None
            else 0
        )

        logger.debug(
            f"✅ Selected candidate with quality={quality_score:.3f}, "
            f"valid={validation_result.is_valid}, audio_duration={audio_duration} samples"
        )

        return best_candidate_tuple 