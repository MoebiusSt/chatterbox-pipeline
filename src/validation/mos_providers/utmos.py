from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch

from .base import MOSProvider
from utils.silence import quiet_imports_and_warnings

logger = logging.getLogger(__name__)


class UTMOSProvider(MOSProvider):
    """
    UTMOS/UTMOSv2 via VERSA integration.

    Implementation notes:
    - Uses VERSA's pseudo_mos utilities:
      - setup: versa.utterance_metrics.pseudo_mos.pseudo_mos_setup
      - inference: versa.utterance_metrics.pseudo_mos.pseudo_mos_metric
    - Prefers 'utmosv2' if the optional package 'utmosv2' is installed; otherwise falls back to 'utmos'.
    - Native MOS scale: 1..5
    """

    def __init__(
        self,
        enabled_languages: Optional[list[str]] = None,
        use_gpu: bool = False,
        cache_dir: str = "versa_cache",
        prefer_utmosv2: bool = False,
    ):
        # Restrict to languages with acceptable reliability: English primary, German moderate
        self.enabled_languages = set(enabled_languages or ["de", "en"]) 
        self.use_gpu = bool(use_gpu)
        self.cache_dir = cache_dir
        self._pm: Optional[object] = None  # pseudo_mos module
        self._predictor_dict: Optional[dict] = None
        self._predictor_fs: Optional[dict] = None
        self._key: Optional[str] = None  # 'utmosv2' or 'utmos'
        self._last_details: dict | None = None
        self._prefer_utmosv2 = bool(prefer_utmosv2)

    def _lazy_setup(self) -> bool:
        if self._predictor_dict is not None and self._predictor_fs is not None and self._key is not None:
            return True
        try:
            # Import the pseudo_mos module from VERSA quietly in non-verbose mode
            with quiet_imports_and_warnings():
                from versa.utterance_metrics import pseudo_mos as pm  # type: ignore

                self._pm = pm

                # Decide which predictor to use
                predictor_types: list[str]
                if self._prefer_utmosv2:
                    # User prefers v2; fall back to v1 if missing
                    try:
                        import utmosv2  # type: ignore  # noqa: F401
                        predictor_types = ["utmosv2"]
                        self._key = "utmosv2"
                    except Exception:
                        predictor_types = ["utmos"]
                        self._key = "utmos"
                else:
                    # Prefer classic utmos unless v2 is explicitly requested
                    predictor_types = ["utmos"]
                    self._key = "utmos"

                # Setup predictors
                self._predictor_dict, self._predictor_fs = pm.pseudo_mos_setup(
                    predictor_types=predictor_types,
                    predictor_args={},
                    cache_dir=self.cache_dir,
                    use_gpu=self.use_gpu,
                )
            self._last_details = {
                "provider": self._key,
                "backend": "versa_pseudo_mos",
                "device": "cuda" if self.use_gpu else "cpu",
                "setup_ok": True,
            }
            return True
        except Exception as e:
            logger.debug(f"VERSA pseudo_mos setup failed: {e}")
            self._pm = None
            self._predictor_dict = None
            self._predictor_fs = None
            self._key = None
            self._last_details = {
                "provider": "utmos",
                "backend": "versa_pseudo_mos",
                "device": "cuda" if self.use_gpu else "cpu",
                "setup_ok": False,
                "error": str(e),
            }
            return False

    def is_language_supported(self, language: str) -> bool:
        lang = (language or "").split("-")[0].lower()
        return lang in self.enabled_languages

    def score(self, audio: torch.Tensor, sample_rate: int, language: str) -> Optional[float]:
        if not self.is_language_supported(language):
            self._last_details = {
                "provider": self._key or "utmos",
                "skipped": True,
                "reason": "language_not_supported",
                "language": language,
            }
            return None
        if not self._lazy_setup():
            return None
        try:
            if audio is None or not hasattr(audio, "numel") or audio.numel() == 0:
                return None
            # Convert to numpy mono float32 waveform
            if audio.dim() > 1:
                audio_np = audio.detach().mean(dim=0).cpu().numpy().astype("float32")
            else:
                audio_np = audio.detach().cpu().numpy().astype("float32")

            # Call VERSA pseudo_mos metric; it resamples internally as needed
            assert self._pm is not None
            # mypy: treat _pm as module with expected callable; runtime import guarantees attribute
            with quiet_imports_and_warnings():
                scores = getattr(self._pm, "pseudo_mos_metric")( 
                    pred=audio_np,
                    fs=int(sample_rate),
                    predictor_dict=self._predictor_dict or {},
                    predictor_fs=self._predictor_fs or {},
                    use_gpu=self.use_gpu,
                )

            # Retrieve the relevant key
            if not isinstance(scores, dict):
                return None
            if self._key in scores and scores[self._key] is not None:
                val = float(scores[self._key])
                self._last_details = {
                    "provider": self._key,
                    "device": "cuda" if self.use_gpu else "cpu",
                    "language": language,
                    "raw_mos": val,
                }
                return val
            # Fallback: some predictors may expose 'utmos' only
            if "utmos" in scores and scores["utmos"] is not None:
                val = float(scores["utmos"])
                self._last_details = {
                    "provider": "utmos",
                    "device": "cuda" if self.use_gpu else "cpu",
                    "language": language,
                    "raw_mos": val,
                }
                return val
            return None
        except Exception as e:
            logger.debug(f"UTMOS scoring failed: {e}")
            self._last_details = {
                "provider": self._key or "utmos",
                "error": str(e),
                "language": language,
            }
            return None


