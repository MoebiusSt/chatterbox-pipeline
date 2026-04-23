from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch

from .base import MOSProvider
from utils.silence import quiet_imports_and_warnings

logger = logging.getLogger(__name__)


class UTMOSProvider(MOSProvider):
    """
    UTMOS (v1) and UTMOSv2 MOS predictors (scale ~1..5).

    - UTMOS v1: VERSA ``pseudo_mos`` with torch.hub ``SpeechMOS`` (ftshijt/SpeechMOS).
    - UTMOSv2: **Native** `utmosv2` package (`create_model` + `predict`), because VERSA's
      integration targets an older UTMOSv2 API (`process_audio_only_versa`) that was removed
      in recent utmosv2 releases, which caused `import versa...pseudo_mos` to set
      `utmosv2` to None even when the package was installed.

    Install v2: ``pip install "git+https://github.com/sarulab-speech/UTMOSv2.git"``

    GPU: set ``validation.mos.use_gpu: true`` (default). UTMOSv2 uses ``cuda:0`` when available.
    ``utmosv2_num_repetitions`` (default 1): increase to 3–5 for more stable MOS at higher cost.
    """

    def __init__(
        self,
        enabled_languages: Optional[list[str]] = None,
        use_gpu: bool = True,
        cache_dir: str = "versa_cache",
        prefer_utmosv2: bool = False,
        utmosv2_num_repetitions: int = 1,
    ):
        # Restrict to languages with acceptable reliability: English primary, German moderate
        self.enabled_languages = set(enabled_languages or ["de", "en"])
        self.use_gpu = bool(use_gpu)
        self.cache_dir = cache_dir
        self.utmosv2_num_repetitions = max(1, int(utmosv2_num_repetitions))
        self._pm: Optional[object] = None  # pseudo_mos module
        self._predictor_dict: Optional[dict] = None
        self._predictor_fs: Optional[dict] = None
        self._key: Optional[str] = None  # 'utmosv2' or 'utmos'
        self._last_details: dict | None = None
        self._prefer_utmosv2 = bool(prefer_utmosv2)
        # Native UTMOSv2 (bypasses broken VERSA v2 path)
        self._utmosv2_model: Any = None
        self._utmosv2_device: str = "cpu"

    def _use_cuda(self) -> bool:
        return self.use_gpu and torch.cuda.is_available()

    def _setup_native_utmosv2(self) -> bool:
        if self._utmosv2_model is not None:
            return True
        try:
            from utmosv2 import create_model  # type: ignore[import-not-found]

            if self._use_cuda():
                self._utmosv2_device = "cuda:0"
            else:
                self._utmosv2_device = "cpu"
            device = self._utmosv2_device
            model = create_model(pretrained=True)
            if device != "cpu" and hasattr(model, "to"):
                model = model.to(device)
            self._utmosv2_model = model
            self._key = "utmosv2"
            self._last_details = {
                "provider": "utmosv2",
                "backend": "utmosv2.native",
                "device": device,
                "setup_ok": True,
            }
            return True
        except Exception as e:
            logger.debug(f"UTMOSv2 native setup failed: {e}")
            self._utmosv2_model = None
            return False

    def _lazy_setup(self) -> bool:
        if self._key == "utmosv2" and self._utmosv2_model is not None:
            return True
        if self._predictor_dict is not None and self._predictor_fs is not None and self._key == "utmos":
            return True
        if self._prefer_utmosv2:
            if self._setup_native_utmosv2():
                return True
            logger.info(
                "UTMOSv2 not available; falling back to UTMOS v1 (torch.hub). "
                'Install: pip install "git+https://github.com/sarulab-speech/UTMOSv2.git"'
            )
        # VERSA only supports v1 in practice; keep log noise from optional utmosv2 import low
        try:
            logging.getLogger("versa.utterance_metrics.pseudo_mos").setLevel(
                logging.WARNING
            )
            with quiet_imports_and_warnings():
                from versa.utterance_metrics import pseudo_mos as pm  # type: ignore

                self._pm = pm
                self._key = "utmos"
                self._predictor_dict, self._predictor_fs = pm.pseudo_mos_setup(
                    predictor_types=["utmos"],
                    predictor_args={},
                    cache_dir=self.cache_dir,
                    use_gpu=self._use_cuda(),
                )
            self._last_details = {
                "provider": "utmos",
                "backend": "versa_pseudo_mos",
                "device": "cuda:0" if self._use_cuda() else "cpu",
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
                "device": "cuda:0" if self._use_cuda() else "cpu",
                "setup_ok": False,
                "error": str(e),
            }
            return False

    def is_language_supported(self, language: str) -> bool:
        lang = (language or "").split("-")[0].lower()
        return lang in self.enabled_languages

    def _score_native_utmosv2(self, audio_np: np.ndarray, sample_rate: int) -> float:
        model = self._utmosv2_model
        assert model is not None
        with torch.no_grad():
            with quiet_imports_and_warnings():
                out = model.predict(  # type: ignore[union-attr]
                    data=audio_np,
                    sr=int(sample_rate),
                    device=self._utmosv2_device,
                    num_repetitions=self.utmosv2_num_repetitions,
                    verbose=False,
                )
        if isinstance(out, torch.Tensor):
            return float(out.flatten()[0].item())
        arr = np.asarray(out).flatten()
        return float(arr[0]) if arr.size else 0.0

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

            if self._key == "utmosv2" and self._utmosv2_model is not None:
                val = self._score_native_utmosv2(audio_np, sample_rate)
                self._last_details = {
                    "provider": "utmosv2",
                    "backend": "utmosv2.native",
                    "device": self._utmosv2_device,
                    "language": language,
                    "raw_mos": val,
                }
                return val

            assert self._pm is not None
            with quiet_imports_and_warnings():
                scores = getattr(self._pm, "pseudo_mos_metric")(
                    pred=audio_np,
                    fs=int(sample_rate),
                    predictor_dict=self._predictor_dict or {},
                    predictor_fs=self._predictor_fs or {},
                    use_gpu=self._use_cuda(),
                )

            if not isinstance(scores, dict):
                return None
            if "utmos" in scores and scores["utmos"] is not None:
                val = float(scores["utmos"])
                self._last_details = {
                    "provider": "utmos",
                    "device": "cuda:0" if self._use_cuda() else "cpu",
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


