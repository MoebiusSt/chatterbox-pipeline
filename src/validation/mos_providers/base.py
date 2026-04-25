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

    def _percentile(self, values: List[float], percentile: float) -> Optional[float]:
        """Return a linearly interpolated percentile for a non-empty score list."""
        if not values:
            return None
        pct = max(0.0, min(100.0, float(percentile)))
        ordered = sorted(float(v) for v in values)
        if len(ordered) == 1:
            return ordered[0]
        pos = (pct / 100.0) * (len(ordered) - 1)
        lower = int(pos)
        upper = min(lower + 1, len(ordered) - 1)
        weight = pos - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    def score_segmented(
        self,
        audio: torch.Tensor,
        sample_rate: int,
        language: str,
        window_s: float = 12.0,
        hop_s: float = 10.0,
        aggregator: str = "median",
        min_segments: int = 1,
        diagnostic_percentile: float = 10.0,
        export_window_scores: bool = False,
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
                aggregated = float(sum(scores) / len(scores))
            elif mode == "min":
                aggregated = float(min(scores))
            else:
                mode = "median"
                aggregated = float(statistics.median(scores))
        except Exception:
            mode = "mean"
            aggregated = float(sum(scores) / len(scores))

        stats: Dict[str, Any] = {
            "segmented": True,
            "aggregator": mode,
            "aggregated_mos": aggregated,
            "median": float(statistics.median(scores)),
            "mean": float(sum(scores) / len(scores)),
            "min": float(min(scores)),
            "low_percentile": float(self._percentile(scores, diagnostic_percentile) or 0.0),
            "low_percentile_p": float(max(0.0, min(100.0, diagnostic_percentile))),
            "num_segments": int(len(scores)),
            "num_failed_segments": int(len(failures)),
            "total_segments": int(total_segments),
            "window_s": float(window_s),
            "hop_s": float(hop_s),
        }
        if export_window_scores:
            stats["scores"] = [float(v) for v in scores]
        if failures:
            stats["first_failure"] = failures[0]

        self._last_details = {
            "provider": self.__class__.__name__,
            "language": language,
            "raw_mos": aggregated,
            "window_stats": stats,
        }
        return aggregated


