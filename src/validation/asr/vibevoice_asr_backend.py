"""VibeVoice-ASR backend.

Wraps the ``microsoft/VibeVoice-ASR`` model for long-form speech recognition
with segment-level timestamps. Segment timestamps are expanded to per-word
estimates so ``TailTrimmer`` and ``ProsodyScorer`` can reuse them via the
``preloaded_words`` hook and skip their internal WhisperX calls.

The model code is vendored under ``src/third_party/vibevoice/`` (same pattern
as VibeVoice-TTS in :mod:`generation.model_cache`). The backend is loaded
lazily on first use. When the vendored tree is missing the backend raises
``ImportError`` from the factory so the pipeline falls back to Whisper.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .base import ASRBackend, ASRResult, ASRWord

logger = logging.getLogger(__name__)


def _vendored_vibevoice_path() -> Path:
    """Return the local vendored VibeVoice root (shared with TTS)."""
    return Path(__file__).resolve().parents[2] / "third_party" / "vibevoice"


class VibeVoiceASRBackend(ASRBackend):
    """Long-form ASR backend built on VibeVoice-ASR."""

    backend_name = "vibevoice_asr"
    supports_alignment = True

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        vv_cfg = (self.config.get("validation", {}) or {}).get("vibevoice_asr", {}) or {}
        self.model_id: str = str(vv_cfg.get("model_id", "microsoft/VibeVoice-ASR"))
        self.language_model: str = str(vv_cfg.get("language_model", "Qwen/Qwen2.5-7B"))
        self.device_pref: str = str(vv_cfg.get("device", "auto"))
        self.max_new_tokens: int = int(vv_cfg.get("max_new_tokens", 32768))
        # Upper bound decoding to ``audio_s * tokens_per_second`` (+ margin).
        # German ~3 tokens/word, ~150 WPM -> ~7.5 tok/s; 40 tok/s is ~5x safety
        # margin while still pruning pathological hangs on short clips.
        self.tokens_per_second: float = float(vv_cfg.get("tokens_per_second", 40.0))
        self.min_max_new_tokens: int = int(vv_cfg.get("min_max_new_tokens", 256))
        self.temperature: float = float(vv_cfg.get("temperature", 0.0))
        self.top_p: float = float(vv_cfg.get("top_p", 1.0))
        self.num_beams: int = int(vv_cfg.get("num_beams", 1))
        self.attn_impl: str = str(vv_cfg.get("attn_implementation", "auto"))
        self.target_sample_rate: int = int(vv_cfg.get("target_sample_rate", 24000))
        self.hotwords: List[str] = list(vv_cfg.get("hotwords", []) or [])

        self._model = None
        self._processor = None
        self._device: Optional[str] = None
        self._dtype = None

        # Eagerly verify the vendored VibeVoice tree (+ ASR modules) is present
        # so the factory can catch missing code and fall back to Whisper.
        vv_path = _vendored_vibevoice_path()
        asr_model = vv_path / "modular" / "modeling_vibevoice_asr.py"
        asr_proc = vv_path / "processor" / "vibevoice_asr_processor.py"
        if not (vv_path.exists() and asr_model.exists() and asr_proc.exists()):
            raise ImportError(
                "Vendored VibeVoice-ASR not found under "
                f"{vv_path}; expected modular/modeling_vibevoice_asr.py and "
                "processor/vibevoice_asr_processor.py"
            )

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        pref = (self.device_pref or "auto").lower()
        if pref == "auto":
            if torch.cuda.is_available():
                return "cuda"
            try:
                if torch.backends.mps.is_available():  # type: ignore[attr-defined]
                    return "mps"
            except Exception:
                pass
            return "cpu"
        return pref

    def _resolve_dtype(self, device: str):
        if device == "cuda":
            return torch.bfloat16
        return torch.float32

    def _resolve_attn_impl(self, device: str) -> str:
        if self.attn_impl != "auto":
            return self.attn_impl
        if device == "cuda":
            try:
                import flash_attn  # noqa: F401
                return "flash_attention_2"
            except Exception:
                return "sdpa"
        return "sdpa"

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        vv_path = _vendored_vibevoice_path()
        vv_path_str = str(vv_path)
        if vv_path_str not in sys.path:
            sys.path.insert(0, vv_path_str)

        # Install a top-level ``vibevoice`` package alias so absolute imports
        # inside vendored upstream code (e.g. ``from vibevoice.modular.X``)
        # resolve to the vendored tree.
        import importlib
        if "vibevoice" not in sys.modules:
            import types
            pkg = types.ModuleType("vibevoice")
            pkg.__path__ = [vv_path_str]  # type: ignore[attr-defined]
            sys.modules["vibevoice"] = pkg
        for sub in ("modular", "processor"):
            full = f"vibevoice.{sub}"
            if full not in sys.modules:
                sys.modules[full] = importlib.import_module(sub)

        from modular.modeling_vibevoice_asr import (  # type: ignore
            VibeVoiceASRForConditionalGeneration,
        )
        from processor.vibevoice_asr_processor import (  # type: ignore
            VibeVoiceASRProcessor,
        )

        device = self._resolve_device()
        dtype = self._resolve_dtype(device)
        attn_impl = self._resolve_attn_impl(device)

        logger.info(
            "🎙️  Loading VibeVoice-ASR (%s, device=%s, dtype=%s, attn=%s)",
            self.model_id, device, dtype, attn_impl,
        )
        # Heads-up for first-time loads: VibeVoice-ASR (~7B) and the language
        # model weights together are ~15 GB. If the Hugging Face cache is cold
        # the ``from_pretrained`` calls below can take 5-15 minutes on typical
        # consumer links. We emit a single info log line so users do not think
        # the pipeline has hung.
        try:
            from huggingface_hub import try_to_load_from_cache  # type: ignore

            cached = try_to_load_from_cache(self.model_id, "config.json")
            if cached is None:
                logger.info(
                    "   ↳ VibeVoice-ASR weights not in HF cache; initial download "
                    "(~%s + language model %s) may take several minutes.",
                    self.model_id, self.language_model,
                )
        except Exception:
            pass

        self._processor = VibeVoiceASRProcessor.from_pretrained(
            self.model_id,
            language_model_pretrained_name=self.language_model,
        )
        model = VibeVoiceASRForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            attn_implementation=attn_impl,
        )
        model = model.to(device)
        model.eval()
        self._model = model
        self._device = device
        self._dtype = dtype

    # ------------------------------------------------------------------
    # Audio preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _tensor_to_numpy(audio: torch.Tensor, sample_rate: int, target_sr: int) -> np.ndarray:
        t = audio
        if t.dim() == 2:
            t = t.mean(dim=0)
        t = t.detach().to(torch.float32).cpu()
        if sample_rate != target_sr and target_sr > 0:
            try:
                import torchaudio
                t = torchaudio.functional.resample(t, sample_rate, target_sr)
            except Exception:
                pass
        return t.numpy().astype(np.float32, copy=False)

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    _TIME_RE = re.compile(r"(\d+(?:\.\d+)?)")

    @staticmethod
    def _seg_time(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        m = VibeVoiceASRBackend._TIME_RE.search(str(value))
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    def _segments_to_words(self, segments: List[Dict[str, Any]]) -> List[ASRWord]:
        """Expand sentence segments into per-word estimates.

        VibeVoice-ASR produces segment-level (phrase) timestamps. We split each
        segment's text on whitespace and linearly interpolate timestamps so
        downstream consumers see a plausible word-level alignment.
        """
        words: List[ASRWord] = []
        for seg in segments or []:
            text = str(seg.get("text") or "").strip()
            if not text:
                continue
            tokens = [t for t in re.split(r"\s+", text) if t]
            if not tokens:
                continue
            t_start = self._seg_time(seg.get("start_time") or seg.get("start"))
            t_end = self._seg_time(seg.get("end_time") or seg.get("end"))
            if t_start is None and t_end is None:
                for tok in tokens:
                    words.append(ASRWord(word=tok, start=None, end=None))
                continue
            if t_start is None:
                t_start = max(0.0, float(t_end or 0.0) - 0.3 * len(tokens))
            if t_end is None or t_end <= t_start:
                t_end = float(t_start) + 0.3 * len(tokens)
            total = max(1e-3, float(t_end) - float(t_start))
            per_word = total / float(len(tokens))
            for i, tok in enumerate(tokens):
                ws = float(t_start) + i * per_word
                we = float(t_start) + (i + 1) * per_word
                words.append(ASRWord(word=tok, start=ws, end=we))
        return words

    def _parse_output(self, raw_text: str) -> List[Dict[str, Any]]:
        """Try processor's post_process_transcription first, otherwise parse
        ``[start-end] Speaker N: text`` style output as a fallback.
        """
        segments: List[Dict[str, Any]] = []
        proc = self._processor
        if proc is not None:
            try:
                segs = proc.post_process_transcription(raw_text) or []
                if isinstance(segs, list):
                    segments = list(segs)
            except Exception as e:
                logger.debug(f"post_process_transcription failed: {e}")

        if not segments:
            for line in (raw_text or "").splitlines():
                m = re.match(
                    r"\s*\[?\s*(?P<s>\d+(?:\.\d+)?)\s*[-–]\s*(?P<e>\d+(?:\.\d+)?)\s*\]?\s*"
                    r"(?:Speaker\s*(?P<sp>\d+)\s*:)?\s*(?P<t>.*)$",
                    line,
                )
                if m and (m.group("t") or "").strip():
                    segments.append({
                        "start_time": float(m.group("s")),
                        "end_time": float(m.group("e")),
                        "speaker_id": int(m.group("sp") or 0),
                        "text": m.group("t").strip(),
                    })
        return segments

    # ------------------------------------------------------------------
    # ASRBackend API
    # ------------------------------------------------------------------

    def transcribe_with_alignment(
        self,
        audio: torch.Tensor,
        language: str = "en",
        sample_rate: int = 24000,
    ) -> ASRResult:
        duration_s = 0.0
        try:
            if audio is not None and audio.numel() > 0:
                duration_s = float(audio.shape[-1]) / float(sample_rate)
        except Exception:
            duration_s = 0.0

        try:
            self._ensure_loaded()
        except Exception as e:
            logger.warning(f"VibeVoice-ASR load failed: {e}")
            return ASRResult(
                transcription="",
                language=language or "en",
                duration_s=duration_s,
                backend=self.backend_name,
                words=None,
                extra={"error": str(e)},
            )

        try:
            processor = self._processor
            model = self._model
            assert processor is not None and model is not None

            audio_np = self._tensor_to_numpy(audio, sample_rate, self.target_sample_rate)
            inputs = processor(
                audio=[audio_np],
                sampling_rate=self.target_sample_rate,
                return_tensors="pt",
                padding=True,
                add_generation_prompt=True,
            )
            device = self._device or "cpu"
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            dynamic_cap = max(
                self.min_max_new_tokens,
                int(duration_s * self.tokens_per_second),
            )
            effective_max_new_tokens = min(self.max_new_tokens, dynamic_cap)
            gen_cfg: Dict[str, Any] = {
                "max_new_tokens": effective_max_new_tokens,
                "pad_token_id": getattr(processor, "pad_id", None),
                "eos_token_id": processor.tokenizer.eos_token_id,
            }
            if self.num_beams > 1:
                gen_cfg["num_beams"] = self.num_beams
                gen_cfg["do_sample"] = False
            else:
                do_sample = self.temperature > 0.0
                gen_cfg["do_sample"] = do_sample
                if do_sample:
                    gen_cfg["temperature"] = self.temperature
                    gen_cfg["top_p"] = self.top_p

            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_cfg)

            input_length = inputs["input_ids"].shape[1]
            generated_ids = output_ids[0, input_length:]
            eos_id = processor.tokenizer.eos_token_id
            if eos_id is not None:
                eos_positions = (generated_ids == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    generated_ids = generated_ids[: eos_positions[0] + 1]
            raw_text = processor.decode(generated_ids, skip_special_tokens=True)

            segments = self._parse_output(raw_text)
            words = self._segments_to_words(segments)
            if segments:
                transcription = " ".join(
                    str(s.get("text") or "").strip() for s in segments
                ).strip()
            else:
                transcription = (raw_text or "").strip()

            return ASRResult(
                transcription=transcription,
                language=language or "en",
                duration_s=duration_s,
                backend=self.backend_name,
                words=words if words else None,
                extra={
                    "raw_text": raw_text,
                    "segments": segments,
                },
            )
        except Exception as e:
            logger.warning(f"VibeVoice-ASR inference failed: {e}")
            return ASRResult(
                transcription="",
                language=language or "en",
                duration_s=duration_s,
                backend=self.backend_name,
                words=None,
                extra={"error": str(e)},
            )
