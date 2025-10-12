from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from .base import MOSProvider

logger = logging.getLogger(__name__)


class UTMOSProvider(MOSProvider):
    """
    UTMOS via VERSA integration. Expects `versa` to be installed and available.

    Notes:
    - Native MOS scale: 1..5
    - Multilingual coverage: de, en, fr, es, it, ja, zh (configurable)
    """

    def __init__(self, enabled_languages: Optional[list[str]] = None):
        self.enabled_languages = set(enabled_languages or ["de", "en", "fr", "es", "it", "ja", "zh"])
        self._versa = None
        self._model = None

    def _lazy_load(self):
        if self._versa is not None:
            return
        try:
            import versa  # type: ignore

            self._versa = versa
            # Try common ways to get a scorer from VERSA
            self._model = None
            try:
                if hasattr(self._versa, "load"):
                    self._model = self._versa.load("utmos_v2")
            except Exception:
                pass
            try:
                if self._model is None and hasattr(self._versa, "get_scorer"):
                    self._model = self._versa.get_scorer("utmos_v2")
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"VERSA/UTMOS not available: {e}")
            self._versa = None
            self._model = None

    def is_language_supported(self, language: str) -> bool:
        lang = (language or "").split("-")[0].lower()
        return lang in self.enabled_languages

    def score(self, audio: torch.Tensor, sample_rate: int, language: str) -> Optional[float]:
        self._lazy_load()
        if self._versa is None:
            return None
        try:
            # Ensure CPU numpy float32 mono waveform for typical metric APIs
            wav = audio.detach().cpu().float().unsqueeze(0).numpy()
            mos = None
            # 1) VERSA global score("utmos_v2", ...)
            if hasattr(self._versa, "score"):
                try:
                    res = self._versa.score("utmos_v2", wav, sample_rate=sample_rate, language=language)
                    if isinstance(res, dict):
                        mos = res.get("mos") or res.get("score")
                    else:
                        mos = float(res)
                except Exception:
                    mos = None

            # 2) Model object with predict()
            if mos is None and self._model is not None:
                if hasattr(self._model, "predict"):
                    try:
                        mos = float(self._model.predict(wav, sample_rate=sample_rate, language=language))
                    except Exception:
                        mos = None

            if mos is None:
                return None
            return float(mos)
        except Exception as e:
            logger.debug(f"UTMOS scoring failed: {e}")
            return None


