from __future__ import annotations

from typing import Optional

import torch

from .base import MOSProvider


class CombinedMOSProvider(MOSProvider):
    """
    Combined MOS provider that queries providers in order and returns the first
    non-None MOS value. Applies a strict language gate: only 'de' and 'en'
    are allowed; for other languages MOS is skipped (returns None).
    """

    def __init__(self, providers: list[MOSProvider], enabled_languages: Optional[list[str]] = None):
        self.providers = providers
        langs = enabled_languages or ["de", "en"]
        self.enabled_languages = set(l.lower() for l in langs)
        self._last_details: dict | None = None

    def is_language_supported(self, language: str) -> bool:
        lang = (language or "").split("-")[0].lower()
        return lang in self.enabled_languages

    def score(self, audio: torch.Tensor, sample_rate: int, language: str) -> Optional[float]:
        if not self.is_language_supported(language):
            self._last_details = {
                "provider": "combined",
                "skipped": True,
                "reason": "language_not_supported",
                "language": language,
            }
            return None
        for provider in self.providers:
            try:
                mos = provider.score(audio, sample_rate, language)
                if mos is not None:
                    # Try to capture inner provider details if available
                    inner_details = getattr(provider, "_last_details", None)
                    self._last_details = {
                        "provider": "combined",
                        "selected": inner_details or {"name": provider.__class__.__name__},
                        "language": language,
                        "raw_mos": mos,
                    }
                    return mos
            except Exception:
                continue
        self._last_details = {
            "provider": "combined",
            "selected": None,
            "language": language,
            "raw_mos": None,
        }
        return None


