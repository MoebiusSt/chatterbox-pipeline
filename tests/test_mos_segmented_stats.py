from typing import Any, Optional, cast

import torch

from src.validation.mos_providers.base import MOSProvider


class _FakeMOSProvider(MOSProvider):
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls = 0
        self._last_details: Optional[dict[str, Any]] = None

    def is_language_supported(self, language: str) -> bool:
        return language in {"en", "de"}

    def score(self, audio: torch.Tensor, sample_rate: int, language: str) -> Optional[float]:
        if self.calls >= len(self.scores):
            return None
        score = self.scores[self.calls]
        self.calls += 1
        return score


def test_score_segmented_exports_compact_window_stats() -> None:
    provider = _FakeMOSProvider([1.0, 2.0, 4.0, 5.0])
    audio = torch.ones(40 * 10)

    mos = provider.score_segmented(
        audio,
        sample_rate=10,
        language="en",
        window_s=10.0,
        hop_s=10.0,
        aggregator="median",
        diagnostic_percentile=10,
    )

    assert mos == 3.0
    details = provider._last_details or {}
    stats = cast(dict[str, Any], details["window_stats"])
    assert stats["aggregator"] == "median"
    assert stats["median"] == 3.0
    assert stats["mean"] == 3.0
    assert stats["min"] == 1.0
    assert stats["low_percentile"] == 1.3
    assert stats["low_percentile_p"] == 10.0
    assert stats["num_segments"] == 4
    assert "scores" not in stats


def test_score_segmented_can_export_window_scores_when_enabled() -> None:
    provider = _FakeMOSProvider([1.0, 2.0, 4.0])
    audio = torch.ones(30 * 10)

    mos = provider.score_segmented(
        audio,
        sample_rate=10,
        language="en",
        window_s=10.0,
        hop_s=10.0,
        aggregator="min",
        export_window_scores=True,
    )

    assert mos == 1.0
    details = provider._last_details or {}
    stats = cast(dict[str, Any], details["window_stats"])
    assert stats["aggregator"] == "min"
    assert stats["scores"] == [1.0, 2.0, 4.0]
