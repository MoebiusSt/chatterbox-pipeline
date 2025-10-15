"""
Prosody scorer: computes prosody-related sub-scores and aggregates them.

Inputs:
- audio (torch.Tensor, mono)
- language (str)
- original_text (str)
- optional alignment (future: whisperx hook)

Outputs:
- dict with sub-scores (0..1) and overall prosody_score
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torchaudio
from utils.silence import quiet_imports_and_warnings

parselmouth = None
try:
    with quiet_imports_and_warnings():
        import parselmouth as _pm  # Praat bindings
        parselmouth = _pm
except Exception:
    pass

whisperx = None
try:
    with quiet_imports_and_warnings():
        import whisperx as _wx  # for alignment-based semantic score
        whisperx = _wx
except Exception:
    pass

from .mos_providers.base import MOSProvider

logger = logging.getLogger(__name__)


class ProsodyScorer:
    """
    Lightweight prosody scorer with configurable weights and simple features.
    Designed to be extended with whisperx alignment for punctuation-pause matching.
    """

    def __init__(self, config: Dict[str, Any], mos_provider: Optional[MOSProvider] = None, sample_rate: int = 24000):
        self.config = config or {}
        self.sample_rate = sample_rate
        self.mos_provider = mos_provider

        p_cfg = self.config.get("validation", {}).get("prosody", {})
        self.enabled = bool(p_cfg.get("enabled", True))
        weights = p_cfg.get("weights", {}) or {}
        self.w_semantic = float(weights.get("semantic_alignment", 0.20))
        self.w_flow = float(weights.get("flow", 0.25))
        self.w_liveliness = float(weights.get("liveliness", 0.25))
        self.w_intelligibility = float(weights.get("intelligibility", 0.10))
        self.w_mos = float(weights.get("mos", 0.20))

        # Targets
        tgt = p_cfg.get("targets", {}) or {}
        self.tgt_wpm_min = float(tgt.get("wpm_min", 130))
        self.tgt_wpm_max = float(tgt.get("wpm_max", 180))

        # Thresholds
        thr = p_cfg.get("thresholds", {}) or {}
        self.min_prosody_score = float(thr.get("min_prosody_score", 0.60))

        # MOS gating
        mos_cfg = self.config.get("validation", {}).get("mos", {})
        self.min_mos = float(mos_cfg.get("min_mos", 3.5))

        # WhisperX caching to avoid repeated model loads per candidate
        self._whisperx = whisperx
        self._whisperx_asr_models: Dict[str, Any] = {}
        self._whisperx_aligners: Dict[str, Tuple[Any, Any]] = {}
        # Reuse device preference from tail-trim if present; default CPU for stability
        tt_cfg = self.config.get("validation", {}).get("preprocessing", {}).get("tail_trim", {})
        self.whisperx_device: str = str(tt_cfg.get("whisperx_device", "cpu")).lower()

    def _estimate_wpm(self, audio_len_samples: int, word_count: int) -> float:
        if word_count <= 0:
            return 0.0
        duration_min = max(1e-6, audio_len_samples / float(self.sample_rate) / 60.0)
        return word_count / duration_min

    def _unit_score_range(self, value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.0
        if value <= low:
            return 0.0
        if value >= high:
            return 1.0
        return (value - low) / (high - low)

    def _energy_dynamics(self, audio: torch.Tensor) -> float:
        # Simple dynamic range proxy via percentile difference (rough)
        a = audio.detach().abs().flatten()
        if a.numel() == 0:
            return 0.0
        try:
            p95 = torch.quantile(a, 0.95).item()
            p50 = torch.quantile(a, 0.50).item()
        except Exception:
            a_sorted, _ = torch.sort(a)
            idx95 = int(0.95 * (a_sorted.numel() - 1))
            idx50 = int(0.50 * (a_sorted.numel() - 1))
            p95 = a_sorted[idx95].item()
            p50 = a_sorted[idx50].item()
        dyn = max(0.0, min(1.0, (p95 - p50) / max(1e-5, p95)))
        return dyn

    def _f0_stats_parselmouth(self, audio: torch.Tensor) -> Tuple[float, float]:
        if parselmouth is None:
            return 0.0, 0.0
        try:
            # parselmouth expects numpy float64 at real sample rate
            x = audio.detach().cpu().numpy().astype("float64")
            snd = parselmouth.Sound(x, self.sample_rate)
            pitch = snd.to_pitch()
            values = pitch.selected_array[0]
            values = values[values > 0]
            if values.size == 0:
                return 0.0, 0.0
            return float(values.mean()), float(values.std())
        except Exception:
            return 0.0, 0.0

    def _resample_to_16k(self, audio: torch.Tensor) -> torch.Tensor:
        if self.sample_rate == 16000:
            return audio
        try:
            resampler = torchaudio.transforms.Resample(orig_freq=self.sample_rate, new_freq=16000)
            return resampler(audio.unsqueeze(0)).squeeze(0)
        except Exception:
            ratio = 16000 / float(self.sample_rate)
            new_len = max(1, int(round(audio.shape[-1] * ratio)))
            idx = torch.linspace(0, audio.shape[-1] - 1, steps=new_len, device=audio.device)
            idx_floor = idx.long().clamp_(0, audio.shape[-1] - 1)
            return audio.index_select(-1, idx_floor)

    def _get_whisperx_asr_model(self, device: str) -> Optional[Any]:
        try:
            key = f"small:{device}"
            if key in self._whisperx_asr_models:
                return self._whisperx_asr_models[key]
            if self._whisperx is None:
                return None
            compute_type = "float16" if device == "cuda" else "int8"
            model = self._whisperx.load_model("small", device=device, compute_type=compute_type)
            self._whisperx_asr_models[key] = model
            return model
        except Exception:
            return None

    def _get_whisperx_aligner(self, language: str, device: str) -> Optional[Tuple[Any, Any]]:
        try:
            lang = (language or "en").lower()
            if lang in self._whisperx_aligners:
                return self._whisperx_aligners[lang]
            if self._whisperx is None:
                return None
            align_model, metadata = self._whisperx.load_align_model(language_code=lang, device=device)
            self._whisperx_aligners[lang] = (align_model, metadata)
            return self._whisperx_aligners[lang]
        except Exception:
            return None

    def score(self, audio: Optional[torch.Tensor], language: str, original_text: str, asr_transcription: Optional[str]) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "enabled": self.enabled,
            "subscores": {
                "semantic_alignment": 0.0,
                "flow": 0.0,
                "liveliness": 0.0,
                "intelligibility": 0.0,
                "mos": 0.0,
            },
            "prosody_score": 0.0,
        }
        if not self.enabled or audio is None or audio.numel() == 0:
            return result

        # Basic text stats
        word_count = len([w for w in (original_text or "").split() if w.strip()])

        # Flow: reading speed proximity to target band (triangular score)
        wpm = self._estimate_wpm(audio.shape[-1], word_count)
        # Clamp unrealistic extremes for stability on micro-chunks
        wpm_clamped = max(40.0, min(260.0, wpm))
        # Triangular score: 1.0 at band center, 0.0 at/ beyond band edges
        try:
            low = self.tgt_wpm_min
            high = self.tgt_wpm_max
            center = 0.5 * (low + high)
            half_range = max(1e-6, 0.5 * (high - low))
            flow_score = max(0.0, 1.0 - abs(wpm_clamped - center) / half_range)
        except Exception:
            flow_score = self._unit_score_range(wpm_clamped, self.tgt_wpm_min, self.tgt_wpm_max)
        # Micro-chunk neutralization: for very short segments, avoid penalizing flow
        try:
            duration_sec = max(0.0, float(audio.shape[-1]) / float(self.sample_rate))
        except Exception:
            duration_sec = 0.0
        if word_count < 5 or duration_sec < 2.0:
            # Use neutral contribution instead of 0 to avoid biasing selection against short chunks
            flow_score = 0.5

        # Liveliness: energy dynamics + F0 via parselmouth
        dyn_score = self._energy_dynamics(audio)
        f0_mean, f0_std = self._f0_stats_parselmouth(audio)
        # Map F0 std (variability) to [0..1] with soft bounds
        f0_score = max(0.0, min(1.0, f0_std / 35.0))
        liveliness = 0.5 * dyn_score + 0.5 * f0_score

        # Intelligibility: proxy from ASR presence/length
        asr_len = len(asr_transcription or "")
        intelligibility = 1.0 if asr_len > 0 else 0.0

        # Semantic alignment: if whisperx available, use alignment coverage; else ASR presence
        semantic_alignment = 1.0 if asr_len > 0 else 0.0
        try:
            # Language gate for alignment (reuse tail-trim language gate if configured)
            lang_gate = None
            try:
                tt_cfg = self.config.get("validation", {}).get("preprocessing", {}).get("tail_trim", {})
                lang_gate = set((tt_cfg.get("smart_match", {}) or {}).get("language_gate", []) or [])
            except Exception:
                lang_gate = None

            if self._whisperx is not None and asr_len > 0:
                device = self.whisperx_device if self.whisperx_device in {"cpu", "cuda"} else "cpu"
                asr_model = self._get_whisperx_asr_model(device)
                if asr_model is not None:
                    audio16 = self._resample_to_16k(audio)
                    # Guard against empty/None language
                    lang = language or "en"
                    # Optional: restrict alignment to configured languages for stability
                    if not lang_gate or (lang.split("-")[0].lower() in lang_gate):
                        with quiet_imports_and_warnings():
                            asr_result = asr_model.transcribe(audio16.cpu().numpy(), language=lang)
                        align_pair = self._get_whisperx_aligner(lang, device)
                        if align_pair is not None:
                            align_model, metadata = align_pair
                            with quiet_imports_and_warnings():
                                aligned = self._whisperx.align(
                                    asr_result.get("segments", []),
                                    align_model,
                                    metadata,
                                    audio16.cpu().numpy(),
                                    device=device,
                                    return_char_alignments=False,
                                )
                            words = aligned.get("word_segments") or []
                            if words:
                                covered = sum(1 for w in words if (w.get("start") is not None and w.get("end") is not None))
                                semantic_alignment = max(0.0, min(1.0, covered / max(1, len(words))))
        except Exception:
            pass

        # MOS
        mos_unit = 0.0
        raw_mos = None
        if self.mos_provider and self.mos_provider.is_language_supported(language):
            try:
                raw_mos = self.mos_provider.score(audio, self.sample_rate, language)
                mos_unit = self.mos_provider.to_unit_score(raw_mos, min_mos=self.min_mos) or 0.0
            except Exception as e:
                logger.debug(f"MOS provider failed: {e}")

        subs = result["subscores"]
        subs["semantic_alignment"] = float(semantic_alignment)
        subs["flow"] = float(flow_score)
        subs["liveliness"] = float(liveliness)
        subs["intelligibility"] = float(intelligibility)
        subs["mos"] = float(mos_unit)

        prosody_score = (
            self.w_semantic * subs["semantic_alignment"]
            + self.w_flow * subs["flow"]
            + self.w_liveliness * subs["liveliness"]
            + self.w_intelligibility * subs["intelligibility"]
            + self.w_mos * subs["mos"]
        )

        result["prosody_score"] = float(max(0.0, min(1.0, prosody_score)))
        result["raw_mos"] = raw_mos
        result["wpm"] = wpm
        return result


