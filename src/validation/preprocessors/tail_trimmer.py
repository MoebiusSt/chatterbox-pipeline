"""
Tail-end speech-aware trimming utility.

Trims trailing non-speech and hallucinations at the end of a candidate's audio
before validation and MOS/prosody scoring. Uses (in order of preference):
1) whisperx word-level alignment (if available and enabled) to find last spoken token end
2) WebRTC VAD backward scan on the last N seconds
3) Energy heuristic fallback

Configuration (validation.preprocessing.tail_trim):
  enabled: true
  lookback_seconds: 6.0
  post_speech_silence_ms: 200
  fade_out_ms: 120
  prefer_whisperx: true
  vad:
    aggressiveness: 2  # 0..3
    frame_ms: 20
    min_voiced_ms: 60
    hangover_ms: 120
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torchaudio

logger = logging.getLogger(__name__)


class TailTrimmer:
    """
    Speech-aware tail trimmer for audio candidates.
    """

    def __init__(self, config: Dict[str, Any], sample_rate: int = 24000):
        self.config = config or {}
        self.sample_rate = sample_rate

        # Defaults
        self.enabled = (
            self.config.get("validation", {})
            .get("preprocessing", {})
            .get("tail_trim", {})
            .get("enabled", True)
        )
        tt_cfg = (
            self.config.get("validation", {})
            .get("preprocessing", {})
            .get("tail_trim", {})
        )
        self.lookback_seconds: float = float(tt_cfg.get("lookback_seconds", 6.0))
        self.post_speech_silence_ms: int = int(tt_cfg.get("post_speech_silence_ms", 200))
        self.fade_out_ms: int = int(tt_cfg.get("fade_out_ms", 120))
        self.prefer_whisperx: bool = bool(tt_cfg.get("prefer_whisperx", True))
        # Force whisperx to CPU by default to avoid GPU/cuDNN dependency issues
        self.whisperx_device: str = str(tt_cfg.get("whisperx_device", "cpu")).lower()
        self.vad_cfg: Dict[str, Any] = tt_cfg.get("vad", {}) or {}

        # Lazy optional modules
        self._webrtcvad = None
        self._whisperx = None

    def _lazy_import_vad(self):
        if self._webrtcvad is not None:
            return
        try:
            import webrtcvad  # type: ignore

            self._webrtcvad = webrtcvad.Vad(int(self.vad_cfg.get("aggressiveness", 2)))
        except Exception as e:
            logger.debug(f"WebRTC VAD not available: {e}")
            self._webrtcvad = None

    def _lazy_import_whisperx(self):
        if self._whisperx is not None:
            return
        try:
            import whisperx  # type: ignore

            self._whisperx = whisperx
        except Exception as e:
            logger.debug(f"whisperx not available: {e}")
            self._whisperx = None

    def _apply_fade_out(self, audio: torch.Tensor, fade_out_ms: int) -> torch.Tensor:
        if audio is None or audio.numel() == 0 or fade_out_ms <= 0:
            return audio
        samples = int(self.sample_rate * (fade_out_ms / 1000.0))
        tail = min(samples, audio.shape[-1])
        if tail <= 1:
            return audio
        # Create linear fade from 1.0 → 0.0 over 'tail' samples at the end
        fade = torch.linspace(1.0, 0.0, steps=tail, device=audio.device, dtype=audio.dtype)
        audio[..., -tail:] = audio[..., -tail:] * fade
        return audio

    def _resample_to_16k(self, audio: torch.Tensor, from_sr: int) -> torch.Tensor:
        if from_sr == 16000:
            return audio
        try:
            resampler = torchaudio.transforms.Resample(orig_freq=from_sr, new_freq=16000)
            return resampler(audio.unsqueeze(0)).squeeze(0)
        except Exception:
            # Fallback nearest-neighbor style (unsafe but robust as last resort)
            ratio = 16000 / float(from_sr)
            new_len = max(1, int(round(audio.shape[-1] * ratio)))
            idx = torch.linspace(0, audio.shape[-1] - 1, steps=new_len, device=audio.device)
            idx_floor = idx.long().clamp_(0, audio.shape[-1] - 1)
            return audio.index_select(-1, idx_floor)

    def _trim_with_vad(self, audio: torch.Tensor, language: str) -> Optional[int]:
        self._lazy_import_vad()
        if self._webrtcvad is None:
            return None

        # Convert to 16 kHz mono PCM16 bytes
        audio16 = self._resample_to_16k(audio, self.sample_rate)
        # Normalize to int16
        audio16 = (audio16.clamp(-1.0, 1.0) * 32767.0).short().cpu()

        frame_ms = int(self.vad_cfg.get("frame_ms", 20))
        frame_len = int(16000 * frame_ms / 1000)
        hangover_ms = int(self.vad_cfg.get("hangover_ms", 120))
        hangover_frames = max(0, hangover_ms // frame_ms)
        min_voiced_ms = int(self.vad_cfg.get("min_voiced_ms", 60))
        min_voiced_frames = max(1, min_voiced_ms // frame_ms)

        num_frames = max(0, (audio16.shape[-1] // frame_len))
        if num_frames == 0:
            return None

        last_voiced_frame = None
        voiced_run = 0

        # Iterate from the end backwards to find last voiced region (with hangover)
        for i in range(num_frames - 1, -1, -1):
            start = i * frame_len
            end = start + frame_len
            frame = audio16[start:end]
            if frame.numel() < frame_len:
                # pad with zeros
                padded = torch.zeros(frame_len, dtype=torch.int16)
                padded[: frame.numel()] = frame
                frame = padded

            is_voiced = False
            try:
                is_voiced = bool(self._webrtcvad.is_speech(frame.tobytes(), sample_rate=16000))
            except Exception:
                is_voiced = False

            if is_voiced:
                if last_voiced_frame is None:
                    last_voiced_frame = i
                voiced_run += 1
                if voiced_run >= min_voiced_frames:
                    # Confirmed voiced region meets minimum length
                    last_voiced_frame = i
                    break
            else:
                # reset run if silence encountered again
                voiced_run = 0

        if last_voiced_frame is None:
            return None

        # Apply hangover forward from last_voiced_frame to include some trailing frames
        last_idx = min(num_frames - 1, last_voiced_frame + hangover_frames)
        cut_sample_16k = (last_idx + 1) * frame_len

        # Map back to original sample rate
        cut_ratio = cut_sample_16k / float(16000)
        cut_sample_orig = int(round(cut_ratio * self.sample_rate))
        return cut_sample_orig

    def _trim_with_energy(self, audio: torch.Tensor) -> Optional[int]:
        # Simple RMS-based heuristic over 20 ms windows on the last lookback_seconds
        window_ms = 20
        window_len = int(self.sample_rate * window_ms / 1000)
        tail_len = int(self.sample_rate * float(self.lookback_seconds))
        start_idx = max(0, audio.shape[-1] - tail_len)
        tail = audio[..., start_idx:]
        if tail.numel() == 0:
            return None

        # Compute RMS per window
        num_windows = max(1, tail.shape[-1] // window_len)
        threshold = 0.005  # conservative low-energy threshold
        last_energy_idx = None
        for w in range(num_windows - 1, -1, -1):
            s = w * window_len
            e = min(tail.shape[-1], s + window_len)
            seg = tail[..., s:e]
            rms = torch.sqrt(torch.clamp((seg ** 2).mean(), min=1e-12)).item()
            if rms > threshold:
                last_energy_idx = s + start_idx
                break

        return last_energy_idx

    def _trim_with_whisperx(self, audio: torch.Tensor, language: str, original_text: str) -> Optional[int]:
        if not self.prefer_whisperx:
            return None
        self._lazy_import_whisperx()
        if self._whisperx is None:
            return None
        try:
            # Minimal alignment: transcribe + align and use last word end time
            # Use configured device; default to CPU to avoid cuDNN runtime issues
            device = self.whisperx_device if self.whisperx_device in {"cpu", "cuda"} else "cpu"
            asr_model = self._whisperx.load_model("small", device=device, compute_type="int8")
            audio16 = self._resample_to_16k(audio, self.sample_rate)
            result = asr_model.transcribe(audio16.cpu().numpy(), language=language)
            # Load alignment model for given language
            align_model, metadata = self._whisperx.load_align_model(language_code=language, device=device)
            aligned = self._whisperx.align(result["segments"], align_model, metadata, audio16.cpu().numpy(), device=device, return_char_alignments=False)
            words = aligned.get("word_segments") or []
            if not words:
                return None
            last_end_s = max((w.get("end", 0.0) or 0.0) for w in words)
            cut_sample = int(round(last_end_s * self.sample_rate))
            return cut_sample
        except Exception as e:
            logger.debug(f"whisperx alignment failed; falling back to VAD: {e}")
            return None

    def trim(self, audio: Optional[torch.Tensor], language: str, original_text: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
        """
        Trim trailing non-speech from audio and apply a short fade-out.

        Returns trimmed audio and metadata (cut indices, method used).
        """
        meta: Dict[str, Any] = {
            "enabled": self.enabled,
            "method": None,
            "cut_sample": None,
            "kept_post_silence_ms": self.post_speech_silence_ms,
            "fade_out_ms": self.fade_out_ms,
        }

        if not self.enabled or audio is None or audio.numel() == 0:
            return audio, None, meta

        try:
            total_len = audio.shape[-1]
            if total_len <= 0:
                return audio, None, meta

            # Try whisperx alignment
            cut_idx = self._trim_with_whisperx(audio, language=language, original_text=original_text)
            method = "whisperx" if cut_idx is not None else None

            # Fallback: VAD
            if cut_idx is None:
                cut_idx = self._trim_with_vad(audio, language)
                method = "vad" if cut_idx is not None else None

            # Fallback: energy heuristic
            if cut_idx is None:
                cut_idx = self._trim_with_energy(audio)
                method = "energy" if cut_idx is not None else None

            if cut_idx is None:
                return audio, None, meta

            # Keep additional post-speech silence
            keep_extra = int(self.sample_rate * (self.post_speech_silence_ms / 1000.0))
            final_idx = min(total_len, max(0, cut_idx + keep_extra))

            trimmed = audio[..., :final_idx].clone()
            removed = audio[..., final_idx:].clone() if final_idx < total_len else None
            trimmed = self._apply_fade_out(trimmed, self.fade_out_ms)

            meta["method"] = method
            meta["cut_sample"] = int(final_idx)
            return trimmed, removed, meta
        except Exception as e:
            logger.debug(f"Tail trim failed: {e}")
            return audio, None, meta


