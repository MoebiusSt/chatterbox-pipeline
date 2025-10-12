from __future__ import annotations

import logging
from typing import Optional

import torch

from .base import MOSProvider

logger = logging.getLogger(__name__)


class NISQAProvider(MOSProvider):
    """
    Optional fallback MOS provider using NISQA pretrained models.
    Native scale: typically 1..5 (depending on checkpoint).
    """

    def __init__(self, enabled_languages: Optional[list[str]] = None):
        self.enabled_languages = set(enabled_languages or ["de", "en", "fr", "es", "it", "ja", "zh"])
        self._nisqa = None
        self._model = None

    def _lazy_load(self):
        if self._nisqa is not None:
            return
        try:
            import nisqa  # type: ignore

            self._nisqa = nisqa
            try:
                # Placeholder: load a general NISQA MOS model
                self._model = self._nisqa.load_model("nisqa_tts_mos")  # pseudo-API
            except Exception:
                self._model = None
        except Exception as e:
            logger.debug(f"NISQA not available: {e}")
            self._nisqa = None
            self._model = None

    def is_language_supported(self, language: str) -> bool:
        lang = (language or "").split("-")[0].lower()
        return lang in self.enabled_languages

    def score(self, audio: torch.Tensor, sample_rate: int, language: str) -> Optional[float]:
        self._lazy_load()
        if self._nisqa is None:
            return None
        try:
            wav = audio.detach().cpu().float().unsqueeze(0).numpy()
            if self._model is not None and hasattr(self._model, "predict"):
                mos = float(self._model.predict(wav, sample_rate))
            elif hasattr(self._nisqa, "score"):
                mos = float(self._nisqa.score(wav, sample_rate))
            else:
                return None
            return mos
        except Exception as e:
            logger.debug(f"NISQA scoring failed: {e}")
            return None


