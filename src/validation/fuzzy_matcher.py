"""
Fuzzy text matching module for comparing transcriptions with original text.
Implements multiple similarity algorithms for robust text comparison.
"""

import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

try:
    from fuzzywuzzy import fuzz

    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False
    logging.warning("fuzzywuzzy not available - using fallback similarity methods")

try:
    import Levenshtein

    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False
    logging.warning(
        "python-Levenshtein not available - using fallback similarity methods"
    )


@dataclass
class MatchResult:
    """Result of text matching operation."""

    similarity: float
    is_match: bool
    original_text: str
    compared_text: str
    method: str
    details: dict


class FuzzyMatcher:
    """
    Advanced text similarity matcher with multiple algorithms.
    Compares transcribed text against original text for validation.
    """

    def __init__(
        self,
        threshold: float = 0.95,
        case_sensitive: bool = False,
        normalize_whitespace: bool = True,
        remove_punctuation: bool = False,
    ):
        """
        Initialize FuzzyMatcher.

        Args:
            threshold: Minimum similarity score for match (0.0 to 1.0)
            case_sensitive: Whether to consider case in comparisons
            normalize_whitespace: Whether to normalize whitespace
            remove_punctuation: Whether to remove punctuation for comparison
        """
        self.threshold = threshold
        self.case_sensitive = case_sensitive
        self.normalize_whitespace = normalize_whitespace
        self.remove_punctuation = remove_punctuation
        import logging
        self.logger = logging.getLogger(__name__)

        # Pre-compile regex for punctuation removal
        if self.remove_punctuation:
            self.punctuation_regex = re.compile(r"[^\w\s]")

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for comparison.

        Args:
            text: Input text

        Returns:
            Preprocessed text
        """
        if not text:
            return ""

        # Case normalization
        if not self.case_sensitive:
            text = text.lower()

        # Whitespace normalization
        if self.normalize_whitespace:
            text = " ".join(text.split())

        # Punctuation removal
        if self.remove_punctuation:
            text = self.punctuation_regex.sub(" ", text)
            text = " ".join(text.split())  # Remove extra spaces

        return text.strip()

    def match_texts(self, text1: str, text2: str, method: str = "auto") -> MatchResult:
        """
        Compare two texts and return similarity score.

        Args:
            text1: First text (typically original)
            text2: Second text (typically transcription)
            method: Similarity method ("auto", "ratio", "partial", "token", "levenshtein", "sequence")

        Returns:
            MatchResult with similarity score and match status
        """
        # Preprocess texts
        proc_text1 = self._preprocess_text(text1)
        proc_text2 = self._preprocess_text(text2)

        # Choose method
        if method == "auto":
            method = self._choose_best_method(proc_text1, proc_text2)

        # Calculate similarity
        try:
            similarity, details = self._calculate_similarity(
                proc_text1, proc_text2, method
            )
            is_match = similarity >= self.threshold

            result = MatchResult(
                similarity=similarity,
                is_match=is_match,
                original_text=text1,
                compared_text=text2,
                method=method,
                details=details,
            )

            self.logger.debug(
                f"Text match ({method}): {similarity:.3f} "
                f"({'PASS' if is_match else 'FAIL'}) - '{text1[:50]}...' vs '{text2[:50]}...'"
            )

            return result

        except Exception as e:
            self.logger.error(f"Text matching failed: {e}")
            return MatchResult(
                similarity=0.0,
                is_match=False,
                original_text=text1,
                compared_text=text2,
                method=method,
                details={"error": str(e)},
            )

    def _choose_best_method(self, text1: str, text2: str) -> str:
        """
        Choose the best similarity method based on text characteristics.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Best method name
        """
        len1, len2 = len(text1), len(text2)

        # For very short texts, use exact ratio
        if max(len1, len2) < 50:
            return "ratio"

        # For significantly different lengths, use partial matching
        if abs(len1 - len2) / max(len1, len2, 1) > 0.3:
            return "partial"

        # For similar length texts, use token-based matching
        return "token"

    def _calculate_similarity(
        self, text1: str, text2: str, method: str
    ) -> Tuple[float, dict]:
        """
        Calculate similarity using specified method.

        Args:
            text1: First text
            text2: Second text
            method: Similarity method

        Returns:
            Tuple of (similarity_score, details_dict)
        """
        if method == "ratio":
            return self._ratio_similarity(text1, text2)
        elif method == "partial":
            return self._partial_similarity(text1, text2)
        elif method == "token":
            return self._token_similarity(text1, text2)
        elif method == "levenshtein":
            return self._levenshtein_similarity(text1, text2)
        elif method == "sequence":
            return self._sequence_similarity(text1, text2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")

    def _ratio_similarity(self, text1: str, text2: str) -> Tuple[float, dict]:
        """Calculate ratio-based similarity."""
        if FUZZYWUZZY_AVAILABLE:
            score = fuzz.ratio(text1, text2) / 100.0
            return score, {"method": "fuzzywuzzy_ratio"}
        else:
            # Fallback to sequence matcher
            return self._sequence_similarity(text1, text2)

    def _partial_similarity(self, text1: str, text2: str) -> Tuple[float, dict]:
        """Calculate partial string similarity (good for different lengths)."""
        if FUZZYWUZZY_AVAILABLE:
            score = fuzz.partial_ratio(text1, text2) / 100.0
            return score, {"method": "fuzzywuzzy_partial_ratio"}
        else:
            # Fallback: find longest common substring ratio
            longer = text1 if len(text1) > len(text2) else text2
            shorter = text2 if len(text1) > len(text2) else text1

            if not shorter:
                return 0.0, {"method": "partial_fallback", "reason": "empty_shorter"}

            # Find best substring match
            best_ratio = 0.0
            for i in range(len(longer) - len(shorter) + 1):
                substring = longer[i : i + len(shorter)]
                ratio = SequenceMatcher(None, shorter, substring).ratio()
                best_ratio = max(best_ratio, ratio)

            return best_ratio, {"method": "partial_fallback"}

    def _token_similarity(self, text1: str, text2: str) -> Tuple[float, dict]:
        """Calculate token-based similarity."""
        if FUZZYWUZZY_AVAILABLE:
            # Use both token_sort_ratio and token_set_ratio, take the higher
            sort_score = fuzz.token_sort_ratio(text1, text2) / 100.0
            set_score = fuzz.token_set_ratio(text1, text2) / 100.0
            score = max(sort_score, set_score)
            return score, {
                "method": "fuzzywuzzy_token",
                "sort_score": sort_score,
                "set_score": set_score,
            }
        else:
            # Fallback: Jaccard similarity
            tokens1 = set(text1.split())
            tokens2 = set(text2.split())

            if not tokens1 and not tokens2:
                return 1.0, {"method": "token_fallback", "reason": "both_empty"}
            if not tokens1 or not tokens2:
                return 0.0, {"method": "token_fallback", "reason": "one_empty"}

            intersection = tokens1.intersection(tokens2)
            union = tokens1.union(tokens2)

            score = len(intersection) / len(union)
            return score, {
                "method": "token_fallback",
                "intersection_size": len(intersection),
                "union_size": len(union),
            }

    def _levenshtein_similarity(self, text1: str, text2: str) -> Tuple[float, dict]:
        """Calculate Levenshtein distance-based similarity."""
        if LEVENSHTEIN_AVAILABLE:
            distance = Levenshtein.distance(text1, text2)
            max_len = max(len(text1), len(text2))

            if max_len == 0:
                score = 1.0
            else:
                score = 1.0 - (distance / max_len)

            return score, {
                "method": "levenshtein",
                "distance": distance,
                "max_length": max_len,
            }
        else:
            # Fallback to sequence matcher
            return self._sequence_similarity(text1, text2)

    def _sequence_similarity(self, text1: str, text2: str) -> Tuple[float, dict]:
        """Calculate sequence matcher-based similarity."""
        matcher = SequenceMatcher(None, text1, text2)
        score = matcher.ratio()

        # Get matching blocks for additional details
        matching_blocks = matcher.get_matching_blocks()
        total_matching_chars = sum(
            block.size for block in matching_blocks[:-1]
        )  # Exclude final dummy block

        return score, {
            "method": "sequence_matcher",
            "matching_blocks": len(matching_blocks) - 1,  # Exclude dummy block
            "matching_chars": total_matching_chars,
        }

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Simple interface to calculate similarity score.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0 to 1.0)
        """
        result = self.match_texts(text1, text2)
        return result.similarity

    def batch_match(
        self, original_texts: List[str], transcriptions: List[str], method: str = "auto"
    ) -> List[MatchResult]:
        """
        Match multiple text pairs in batch.

        Args:
            original_texts: List of original texts
            transcriptions: List of transcriptions
            method: Similarity method to use

        Returns:
            List of match results
        """
        if len(original_texts) != len(transcriptions):
            raise ValueError(
                "Number of original texts must match number of transcriptions"
            )

        results = []
        for i, (original, transcription) in enumerate(
            zip(original_texts, transcriptions)
        ):
            self.logger.debug(f"Matching text pair {i+1}/{len(original_texts)}")
            result = self.match_texts(original, transcription, method)
            results.append(result)

        return results
