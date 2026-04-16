"""
Quality calculation utilities for audio validation.
Handles similarity scoring and overall quality assessment.
"""

import logging
import math
import re
from typing import List

try:
    from rapidfuzz import fuzz as _rf_fuzz  # type: ignore
    _RAPIDFUZZ_AVAILABLE = True
except Exception:
    _RAPIDFUZZ_AVAILABLE = False

from typing import TYPE_CHECKING

# Use absolute import pattern like existing modules

if TYPE_CHECKING:
    from utils.file_manager.io_handlers.candidate_io import AudioCandidate

class QualityCalculator:
    """Calculates quality metrics for audio validation."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_similarity(self, original: str, transcription: str) -> float:
        """
        Calculate similarity between original text and transcription.
        Uses robust normalization and optionally RapidFuzz token ratios.

        Args:
            original: Original text
            transcription: Transcribed text

        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            norm_original = self._normalize_text_base(original)
            norm_transcr = self._normalize_text_base(transcription)

            if not norm_original and not norm_transcr:
                return 1.0
            if not norm_original or not norm_transcr:
                return 0.0

            if _RAPIDFUZZ_AVAILABLE:
                try:
                    # For micro-chunks (≤2 tokens), use character-based ratio to avoid 1.0 saturation
                    tokens_o = self._tokenize(norm_original)
                    tokens_t = self._tokenize(norm_transcr)
                    if len(tokens_o) <= 2 or len(tokens_t) <= 2:
                        char_ratio = _rf_fuzz.ratio(norm_original, norm_transcr) / 100.0
                        # Cap at 0.92 to preserve discriminability for micro-chunks
                        similarity = min(0.92, char_ratio)
                    elif len(norm_original) > 800:
                        # Hardened long-text score: combines token-based similarity
                        # with partial_ratio (hallucination-aware) and a
                        # length-robust log-scaled length score.
                        tsr = _rf_fuzz.token_set_ratio(norm_original, norm_transcr) / 100.0
                        tor = _rf_fuzz.token_sort_ratio(norm_original, norm_transcr) / 100.0
                        pr = _rf_fuzz.partial_ratio(norm_original, norm_transcr) / 100.0
                        token_sim = max(tsr, tor)
                        len_o = max(1, len(norm_original))
                        len_t = max(1, len(norm_transcr))
                        length_score = min(
                            math.log(min(len_o, len_t) + 1.0)
                            / math.log(max(len_o, len_t) + 1.0),
                            1.0,
                        )
                        similarity = 0.55 * token_sim + 0.30 * pr + 0.15 * length_score
                    else:
                        tsr = _rf_fuzz.token_set_ratio(norm_original, norm_transcr)
                        tor = _rf_fuzz.token_sort_ratio(norm_original, norm_transcr)
                        similarity = max(tsr, tor) / 100.0
                    return float(max(0.0, min(1.0, similarity)))
                except Exception:
                    # fall back to jaccard below
                    pass

            tokens_o = self._tokenize(norm_original)
            tokens_t = self._tokenize(norm_transcr)
            if not tokens_o and not tokens_t:
                return 1.0
            if not tokens_o or not tokens_t:
                return 0.0
            set_o = set(tokens_o)
            set_t = set(tokens_t)
            union = set_o | set_t
            inter = set_o & set_t
            similarity = (len(inter) / len(union)) if union else 0.0
            return float(max(0.0, min(1.0, similarity)))

        except Exception as e:
            self.logger.warning(f"Similarity calculation failed: {e}")
            return 0.0

    def calculate_quality_score(
        self,
        candidate: "AudioCandidate",
        transcription: str,
        similarity_score: float,
        original_text_for_length: str | None = None,
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
            base_original = (
                original_text_for_length
                if original_text_for_length is not None
                else candidate.chunk_text
            )

            if base_original and len(base_original) > 0:
                # Calculate progressive length score that tolerates fine differences
                # but increasingly penalizes larger deviations in both directions
                raw_ratio = len(transcription) / len(base_original)
                length_deviation = abs(1.0 - raw_ratio)
                
                # Use a progressive penalty: mild for small deviations, harsh for large ones

                length_score = max(0.0, 1.0 - (length_deviation ** 1.4))
            else:
                length_score = 1.0 if transcription else 0.0

            quality_score = (
                similarity_score * 0.7  # 70% similarity (increased from 60%)
                + length_score * 0.30  # 30% text length comparison
            )

            return min(1.0, max(0.0, quality_score))

        except Exception as e:
            self.logger.warning(f"Quality score calculation failed: {e}")
            return similarity_score

    # --- Helpers ---

    def _normalize_text_base(self, text: str) -> str:
        if not text:
            return ""

        t = text.lower()

        # Normalize common unicode quotes/dashes to spaces or ascii
        quotes_map = {
            "\u2018": "'", "\u2019": "'", "\u201a": ",", "\u201b": "'",
            "\u201c": '"', "\u201d": '"', "\u201e": '"',
            "\xab": '"', "\xbb": '"',
        }
        for k, v in quotes_map.items():
            t = t.replace(k, v)

        dashes = ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212", "-"]
        for d in dashes:
            t = t.replace(d, " ")

        # German sharp s tolerance: map ß to ss
        t = t.replace("ß", "ss")

        # Remove remaining punctuation (keep letters, digits, underscore)
        t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)

        # Collapse whitespace
        t = re.sub(r"\s+", " ", t, flags=re.UNICODE).strip()
        return t

    def _tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return [tok for tok in text.split(" ") if tok]
