from __future__ import annotations

import logging
import statistics
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class MOSProvider(ABC):
    """Abstract interface for MOS providers."""

    @abstractmethod
    def is_language_supported(self, language: str) -> bool:
        """
        Return True if the provider supports the given BCP-47 language code (e.g., 'de', 'en').
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, audio: torch.Tensor, sample_rate: int, language: str) -> Optional[float]:
        """
        Compute MOS for the given audio. Returns a float on the provider's native scale
        (usually 1..5) or None if scoring failed.
        """
        raise NotImplementedError

    def to_unit_score(self, mos_value: Optional[float], min_mos: float = 3.5) -> Optional[float]:
        """
        Map raw MOS (1..5) to 0..1 with threshold gating around min_mos.
        """
        if mos_value is None:
            return None
        mos_value = float(mos_value)
        mos_clipped = max(1.0, min(5.0, mos_value))
        if mos_clipped <= min_mos:
            return 0.0
        return max(0.0, min(1.0, (mos_clipped - min_mos) / (5.0 - min_mos)))

    def score_segmented(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        language: str,
        window_s: float = 12.0,
        hop_s: float = 10.0,
        aggregator: str = "median",
        min_segments: int = 1,
    ) -> Optional[float]:
        """Score long audio by splitting it into sliding windows.

        MOS models are typically trained on 3-10 second utterances and become
        unreliable on longer clips. For audio longer than ``window_s`` seconds
        this method produces per-window scores and aggregates them (median by
        default, falling back to mean/min).

        For shorter audio this is equivalent to :meth:`score`.
        """
        if audio is None or audio.numel() == 0 or sample_rate <= 0:
            return None
        try:
            total_samples = int(audio.shape[-1])
            duration_s = float(total_samples) / float(sample_rate)
        except Exception:
            return self.score(audio, sample_rate, language)

        window_samples = int(max(1, round(window_s * sample_rate)))
        hop_samples = int(max(1, round(max(0.1, hop_s) * sample_rate)))

        if duration_s <= window_s or total_samples <= window_samples:
            return self.score(audio, sample_rate, language)

        scores: List[float] = []
        failures: List[str] = []
        start = 0
        total_segments = 0
        while start < total_samples:
            end = min(total_samples, start + window_samples)
            segment = audio[..., start:end]
            try:
                if segment.numel() == 0 or (end - start) < int(sample_rate * 1.0):
                    break
                total_segments += 1
                val = self.score(segment, sample_rate, language)
                if val is not None:
                    scores.append(float(val))
                else:
                    failures.append(f"{start}..{end}: None")
            except Exception as e:
                failures.append(f"{start}..{end}: {e}")
            if end >= total_samples:
                break
            start += hop_samples

        # Aggregate per-chunk rather than per-segment so logs stay readable on
        # long VibeVoice outputs (see TESTRUN_FINDINGS R2).
        if failures:
            logger.debug(
                "MOS segment scoring: %d/%d segment(s) failed (first: %s)",
                len(failures), total_segments, failures[0],
            )

        if len(scores) < max(1, int(min_segments)):
            return self.score(audio, sample_rate, language)

        mode = (aggregator or "median").lower()
        try:
            if mode == "mean":
                return float(sum(scores) / len(scores))
            if mode == "min":
                return float(min(scores))
            return float(statistics.median(scores))
        except Exception:
            return float(sum(scores) / len(scores))


