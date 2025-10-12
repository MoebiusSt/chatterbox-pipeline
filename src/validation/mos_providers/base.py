from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


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


