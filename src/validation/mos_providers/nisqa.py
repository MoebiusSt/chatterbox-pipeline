from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .base import MOSProvider
from utils.silence import quiet_imports_and_warnings

logger = logging.getLogger(__name__)


class NISQAProvider(MOSProvider):
    """
    NISQA-TTS v1 (Naturalness) via VERSA integration.

    Notes:
    - Native MOS scale: 1..5
    - Supported languages: de, en (as agreed); other languages are skipped
    - Requires VERSA to be installed; uses `versa.utterance_metrics.nisqa`
    - Requires a checkpoint path for NISQA-TTS (e.g., NISQA/weights/nisqa_tts.tar)
    """

    def __init__(
        self,
        enabled_languages: Optional[list[str]] = None,
        weights_path: Optional[str] = None,
        use_gpu: bool = False,
    ):
        # Enforce language gate to {de,en} unless narrowed further via config
        langs = enabled_languages or ["de", "en"]
        self.enabled_languages = set(l.lower() for l in langs)
        self.weights_path = weights_path
        self.use_gpu = bool(use_gpu) and torch.cuda.is_available()
        self._versa_nisqa = None
        self._model = None
        self._last_details: dict | None = None

    def _resolve_default_weights(self) -> Optional[Path]:
        """Try to resolve a reasonable default path for nisqa_tts.tar."""
        # Prefer project-relative: ./NISQA/weights/nisqa_tts.tar
        candidates = [
            Path("NISQA/weights/nisqa_tts.tar").resolve(),
        ]
        try:
            # Also try repository root by traversing upward from this file (…/src/validation/mos_providers/nisqa.py)
            repo_root = Path(__file__).resolve().parents[4]
            candidates.append((repo_root / "NISQA/weights/nisqa_tts.tar").resolve())
        except Exception:
            pass
        for p in candidates:
            try:
                if p.exists():
                    return p
            except Exception:
                continue
        return None

    def _lazy_setup(self) -> bool:
        if self._model is not None:
            return True
        try:
            with quiet_imports_and_warnings():
                from versa.utterance_metrics import nisqa as ns  # type: ignore

                self._versa_nisqa = ns
                weights_path = self.weights_path
                if not weights_path:
                    resolved = self._resolve_default_weights()
                    if resolved is None:
                        logger.debug("NISQA weights not found; skipping setup")
                        return False
                    weights_path = str(resolved)

                # Setup underlying model via VERSA
                self._model = ns.nisqa_model_setup(
                    nisqa_model_path=weights_path,
                    use_gpu=self.use_gpu,
                )
            self._last_details = {
                "provider": "nisqa_tts",
                "backend": "versa",
                "weights_path": weights_path,
                "device": "cuda" if self.use_gpu else "cpu",
                "setup_ok": True,
            }
            return True
        except Exception as e:
            logger.debug(f"NISQA setup failed: {e}")
            self._versa_nisqa = None
            self._model = None
            self._last_details = {
                "provider": "nisqa_tts",
                "backend": "versa",
                "weights_path": self.weights_path,
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
                "provider": "nisqa_tts",
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
            # Convert to numpy mono float32 waveform (VERSA handles resampling)
            if audio.dim() > 1:
                wav = audio.detach().mean(dim=0).cpu().numpy().astype("float32")
            else:
                wav = audio.detach().cpu().numpy().astype("float32")

            # Use VERSA wrapper
            ns = self._versa_nisqa
            if ns is None or self._model is None:
                return None
            with quiet_imports_and_warnings():
                mos = ns.nisqa_metric(self._model, wav, int(sample_rate))
            if mos is None:
                return None
            self._last_details = {
                "provider": "nisqa_tts",
                "backend": "versa",
                "weights_path": self.weights_path or "resolved",
                "device": "cuda" if self.use_gpu else "cpu",
                "language": language,
                "raw_mos": float(mos),
            }
            return float(mos)
        except Exception as e:
            logger.debug(f"NISQA scoring failed: {e}")
            self._last_details = {
                "provider": "nisqa_tts",
                "error": str(e),
                "language": language,
            }
            return None



