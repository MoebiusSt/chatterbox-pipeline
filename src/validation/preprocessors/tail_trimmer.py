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
  smart_match:
    last_n_words: 3
    max_trailing_asr_words: 10   # reject match if this many ASR tokens follow the span
    vad_reject_gap_seconds: 2.5  # reject content_cut when VAD hears speech longer after cut
    vad_reject_min_trailing_asr_words: 3  # ...only when enough ASR tokens still follow the span
  vad:
    aggressiveness: 2  # 0..3
    frame_ms: 20
    min_voiced_ms: 60
    hangover_ms: 120
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio

from validation.number_normalization import normalize_text_for_numbers
from utils.silence import quiet_imports_and_warnings
logger = logging.getLogger(__name__)


def _asr_words_to_dicts(words: Optional[List[Any]]) -> Optional[List[Dict[str, Any]]]:
    """Normalize preloaded ASR words (either dataclass ASRWord or dict) to
    the dict format used throughout this module (``{word, start, end}``).
    """
    if not words:
        return None
    out: List[Dict[str, Any]] = []
    for w in words:
        if isinstance(w, dict):
            out.append({
                "word": str(w.get("word") or w.get("text") or ""),
                "start": w.get("start"),
                "end": w.get("end"),
            })
        else:
            out.append({
                "word": str(getattr(w, "word", "") or ""),
                "start": getattr(w, "start", None),
                "end": getattr(w, "end", None),
            })
    return out


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
        # Caches for whisperx models/aligners to avoid repeated loads
        self._whisperx_asr_models: Dict[str, Any] = {}
        self._whisperx_aligners: Dict[str, Tuple[Any, Any]] = {}

    def _lazy_import_vad(self):
        if self._webrtcvad is not None:
            return
        try:
            with quiet_imports_and_warnings():
                import webrtcvad  # type: ignore

                self._webrtcvad = webrtcvad.Vad(int(self.vad_cfg.get("aggressiveness", 2)))
        except Exception as e:
            logger.debug(f"WebRTC VAD not available: {e}")
            self._webrtcvad = None

    def _lazy_import_whisperx(self):
        if self._whisperx is not None:
            return
        try:
            with quiet_imports_and_warnings():
                import whisperx  # type: ignore

                self._whisperx = whisperx
        except Exception as e:
            logger.debug(f"whisperx not available: {e}")
            self._whisperx = None

    # ------------------------
    # Smart-match helpers
    # ------------------------

    def _normalize_words(self, text: str, language: Optional[str] = None, numbers_mode: Optional[str] = None) -> list[str]:
        """
        Normalize a text into lowercased word tokens with optional number-aware normalization.

        If numbers_mode is provided (e.g., "placeholder"), apply the same normalization that
        the validation pipeline uses, so that number words and digits match consistently.
        """
        try:
            # Apply numbers normalization first to keep placeholder tokens consistent
            if numbers_mode:
                lang = (language or "en").lower()
                # placeholder maps digits and recognized number words to <NUM>
                text = normalize_text_for_numbers(text, lang, mode=str(numbers_mode))
                # Convert placeholder to a simple token that survives punctuation stripping
                text = text.replace("<NUM>", " NUM ")
        except Exception:
            # Best-effort fallback: continue with plain normalization
            pass

        t = text.lower()
        t = re.sub(r"[^\w\säöüß]", " ", t, flags=re.UNICODE)
        t = re.sub(r"\s+", " ", t).strip()
        return t.split() if t else []

    def _get_whisperx_asr_model(
        self, device: str, language: str = "en"
    ) -> Optional[Any]:
        try:
            lang = (language or "en").lower()
            # Include language in the cache key so we don't reuse a model
            # pinned to the wrong language for a different-language chunk.
            key = f"small:{device}:{lang}"
            if key in self._whisperx_asr_models:
                return self._whisperx_asr_models[key]
            if self._whisperx is None:
                return None
            compute_type = "float16" if device == "cuda" else "int8"
            # Passing ``language`` here silences WhisperX's
            # "No language specified, language will be first be detected …"
            # notice and skips per-file language detection (faster).
            with quiet_imports_and_warnings():
                model = self._whisperx.load_model(
                    "small",
                    device=device,
                    compute_type=compute_type,
                    language=lang,
                )
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
            with quiet_imports_and_warnings():
                align_model, metadata = self._whisperx.load_align_model(language_code=lang, device=device)
            self._whisperx_aligners[lang] = (align_model, metadata)
            return self._whisperx_aligners[lang]
        except Exception:
            return None

    def _trim_after_last_words_with_alignment(
        self,
        audio: torch.Tensor,
        language: str,
        original_text: str,
        last_n_words: int,
        fuzzy_ratio: float,
        search_window_words: int,
        preloaded_words: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Optional[int], Dict[str, Any]]:
        meta: Dict[str, Any] = {
            "match_type": None,
            "matched_words": None,
            "match_ratio": None,
            "language_gated": False,
        }

        # When no preloaded words are given we need WhisperX. With preloaded
        # words we can skip all WhisperX interactions entirely.
        if preloaded_words is None:
            if not self.prefer_whisperx:
                return None, meta
            self._lazy_import_whisperx()
            if self._whisperx is None:
                return None, meta

        try:
            sm_cfg = (
                self.config.get("validation", {})
                .get("preprocessing", {})
                .get("tail_trim", {})
                .get("smart_match", {})
            )
            gate = sm_cfg.get("language_gate", ["en", "de", "fr", "es", "it"]) or []
            if gate:
                gate_set = {str(g).lower() for g in gate}
                if (language or "").lower() not in gate_set:
                    meta["language_gated"] = True
                    return None, meta

            # Target words: last N content words of the original text (numbers-aware)
            # Use configured numbers normalization mode (may be: off|placeholder|digits|words)
            nm = (self.config.get("validation", {}) or {}).get("numbers_normalization_mode")
            norm_tokens = self._normalize_words(
                original_text,
                language=language,
                numbers_mode=nm,
            )
            # Keep numeric placeholders (num) and pure-digit tokens even if length < 2
            content_tokens = [t for t in norm_tokens if (len(t) >= 2) or (t == "num") or t.isdigit()]
            # Prefer content tokens, fallback to raw normalized if too few
            base_tokens = content_tokens if len(content_tokens) >= 1 else norm_tokens
            tgt_words = base_tokens[-int(max(1, last_n_words)) :]
            if not tgt_words:
                return None, meta

            if preloaded_words is not None:
                words = preloaded_words
            else:
                device = self.whisperx_device if self.whisperx_device in {"cpu", "cuda"} else "cpu"
                # Resample to 16k for ASR/alignment
                audio16 = self._resample_to_16k(audio, self.sample_rate)

                asr_model = self._get_whisperx_asr_model(device, language)
                if asr_model is None:
                    return None, meta
                with quiet_imports_and_warnings():
                    result = asr_model.transcribe(audio16.cpu().numpy(), language=language)

                align_pair = self._get_whisperx_aligner(language, device)
                if align_pair is None:
                    return None, meta
                align_model, metadata = align_pair
                whisperx_mod = self._whisperx
                if whisperx_mod is None:
                    return None, meta
                with quiet_imports_and_warnings():
                    aligned = whisperx_mod.align(
                        result.get("segments", []),
                        align_model,
                        metadata,
                        audio16.cpu().numpy(),
                        device=device,
                        return_char_alignments=False,
                    )
                words = aligned.get("word_segments") or []
            if not words:
                return None, meta

            # Build normalized recognized words list (index, token), numbers-aware
            rec_words: list[Tuple[int, str]] = []
            for i, w in enumerate(words):
                wtxt = str((w.get("word") or "")).strip()
                nws = self._normalize_words(wtxt, language=language, numbers_mode=nm)
                # Keep numeric placeholders and digit tokens; filter only obvious 1-letter shards
                nws = [t for t in nws if (len(t) >= 2) or (t == "num") or t.isdigit()]
                if not nws:
                    continue
                rec_words.append((i, nws[0]))
            if not rec_words:
                return None, meta

            # Focus search near the end
            window = max(1, len(tgt_words))
            k = max(window, int(search_window_words))
            start_j = max(0, len(rec_words) - k)

            tgt = " ".join(tgt_words)
            highest_ratio = 0.0
            best_qualifying_j = -1
            best_qualifying_ratio = 0.0
            best_span: Tuple[Optional[int], Optional[int]] = (None, None)
            best_span_words: list[str] = []

            # Prefer the rightmost span that meets the fuzzy threshold so repeated
            # phrases (e.g. a name in the URL and again at chunk end) cut at the
            # final occurrence, not an earlier one.
            for j in range(start_j, len(rec_words) - window + 1):
                span_tokens = [rec_words[m][1] for m in range(j, min(j + window, len(rec_words)))]
                if len(span_tokens) != window:
                    continue
                span = " ".join(span_tokens)
                ratio = SequenceMatcher(None, tgt, span).ratio()
                if ratio > highest_ratio:
                    highest_ratio = ratio
                if ratio >= float(fuzzy_ratio) and (
                    j > best_qualifying_j
                    or (j == best_qualifying_j and ratio > best_qualifying_ratio)
                ):
                    best_qualifying_j = j
                    best_qualifying_ratio = ratio
                    best_span = (rec_words[j][0], rec_words[j + window - 1][0])
                    best_span_words = span_tokens

            best_ratio = best_qualifying_ratio if best_qualifying_j >= 0 else highest_ratio
            meta["match_ratio"] = float(best_ratio)

            # If fuzzy threshold not met, try compound token heuristic (join target words)
            if best_qualifying_j < 0 or best_span[0] is None or best_span[1] is None:
                try:
                    tgt_compound = "".join(tgt_words)
                    # Search for exact compound match near the end
                    compound_idx = None
                    for j in range(len(rec_words) - 1, start_j - 1, -1):
                        if rec_words[j][1] == tgt_compound:
                            compound_idx = rec_words[j][0]
                            break
                    if compound_idx is not None:
                        compound_rec_j = next(
                            (idx for idx, pair in enumerate(rec_words) if pair[0] == compound_idx),
                            None,
                        )
                        if compound_rec_j is not None:
                            trailing = len(rec_words) - 1 - compound_rec_j
                            meta["trailing_asr_words"] = trailing
                            max_trailing = int(sm_cfg.get("max_trailing_asr_words", 10))
                            if trailing > max_trailing:
                                meta["match_rejected"] = "too_many_trailing_asr_words"
                                return None, meta
                        end_idx = int(compound_idx)
                        end_word = words[end_idx]
                        end_word_end_s = float(end_word.get("end") or 0.0)
                        meta["match_type"] = "content_cut"
                        meta["matched_words"] = tgt_words
                        meta["match_ratio"] = 1.0
                        cut_idx = int(round(end_word_end_s * self.sample_rate))
                        return cut_idx, meta
                except Exception:
                    pass

                return None, meta

            # Determine if match is at final alignment word
            end_idx = int(best_span[1])
            end_word = words[end_idx]
            last_end_s_all = max((float(w.get("end") or 0.0) for w in words), default=0.0)
            end_word_end_s = float(end_word.get("end") or 0.0)

            trailing_asr_words = len(rec_words) - 1 - best_qualifying_j - (window - 1)
            trailing_asr_words = max(0, trailing_asr_words)
            meta["trailing_asr_words"] = trailing_asr_words
            max_trailing = int(sm_cfg.get("max_trailing_asr_words", 10))
            if trailing_asr_words > max_trailing:
                meta["match_rejected"] = "too_many_trailing_asr_words"
                return None, meta

            potential_cut_idx = int(round(end_word_end_s * self.sample_rate))

            if abs(last_end_s_all - end_word_end_s) < 1e-3:
                meta["match_type"] = "exact_final"
            else:
                meta["match_type"] = "content_cut"
            meta["matched_words"] = best_span_words if best_span_words else None
            return potential_cut_idx, meta
        except Exception as e:
            logger.debug(f"smart-match alignment failed; skipping smart trim: {e}")
            return None, meta

    def _smart_match_cfg(self) -> Dict[str, Any]:
        return (
            self.config.get("validation", {})
            .get("preprocessing", {})
            .get("tail_trim", {})
            .get("smart_match", {})
            or {}
        )

    def _estimate_expected_duration_s(self, original_text: str) -> Optional[float]:
        """Estimate plausible speech duration from chunk word count and target WPM."""
        word_count = len([w for w in (original_text or "").split() if w.strip()])
        if word_count <= 0:
            return None
        p_cfg = self.config.get("validation", {}).get("prosody", {}) or {}
        targets = p_cfg.get("targets", {}) or {}
        wpm_min = float(targets.get("wpm_min", 115.0))
        wpm_max = float(targets.get("wpm_max", 155.0))
        target_wpm = max(1.0, 0.5 * (wpm_min + wpm_max))
        return (word_count / target_wpm) * 60.0

    def _is_cut_below_expected_minimum(
        self,
        cut_idx: int,
        original_text: str,
        sm_cfg: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[float], Optional[float]]:
        """Return (too_short, cut_s, expected_s) using text-implied minimum duration."""
        cfg = sm_cfg if sm_cfg is not None else self._smart_match_cfg()
        min_ratio = float(cfg.get("min_cut_vs_expected_ratio", 0.40))
        expected_s = self._estimate_expected_duration_s(original_text)
        if expected_s is None or expected_s <= 0:
            return False, None, None
        cut_s = cut_idx / float(self.sample_rate)
        return cut_s < expected_s * min_ratio, cut_s, expected_s

    def _should_reject_smart_match(
        self,
        sm_cut: int,
        audio: torch.Tensor,
        language: str,
        original_text: str,
        sm_meta: Dict[str, Any],
        sm_cfg: Dict[str, Any],
    ) -> Optional[str]:
        """Return rejection reason or None when a smart-match cut is safe to apply."""
        too_short, cut_s, expected_s = self._is_cut_below_expected_minimum(
            sm_cut, original_text, sm_cfg
        )
        if too_short:
            return "cut_below_expected_duration"

        vad_cut = self._trim_with_vad(audio, language)
        if not isinstance(vad_cut, int) or vad_cut <= sm_cut:
            return None

        gap_s = (vad_cut - sm_cut) / float(self.sample_rate)
        vad_reject_gap = float(sm_cfg.get("vad_reject_gap_seconds", 2.5))
        trailing_asr = int(sm_meta.get("trailing_asr_words") or 0)
        vad_reject_min_trailing = int(sm_cfg.get("vad_reject_min_trailing_asr_words", 3))

        if gap_s > vad_reject_gap and trailing_asr >= vad_reject_min_trailing:
            return "vad_speech_after_match"

        # Misaligned ASR on short clips: match sits at ASR end but VAD hears far more speech
        # while the cut is still well below the text-implied minimum duration.
        if (
            cut_s is not None
            and expected_s is not None
            and cut_s < expected_s * float(sm_cfg.get("vad_short_cut_expected_ratio", 0.75))
            and gap_s > float(sm_cfg.get("vad_short_cut_gap_seconds", 1.0))
        ):
            return "vad_speech_after_misaligned_match"

        return None

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
                # Convert to raw bytes for webrtcvad; ensure NumPy array backing
                is_voiced = bool(self._webrtcvad.is_speech(frame.numpy().tobytes(), sample_rate=16000))
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

    def _trim_with_whisperx(
        self,
        audio: torch.Tensor,
        language: str,
        original_text: str,
        preloaded_words: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[int]:
        # Fast path: if preloaded words are available, use their last end time
        # and skip the WhisperX ASR/alignment entirely.
        if preloaded_words:
            try:
                last_end_s = max((float(w.get("end") or 0.0) for w in preloaded_words), default=0.0)
                if last_end_s > 0:
                    return int(round(last_end_s * self.sample_rate))
            except Exception:
                return None
            return None

        if not self.prefer_whisperx:
            return None
        self._lazy_import_whisperx()
        if self._whisperx is None:
            return None
        try:
            # Minimal alignment: transcribe + align and use last word end time
            # Use configured device; default to CPU to avoid cuDNN runtime issues
            device = self.whisperx_device if self.whisperx_device in {"cpu", "cuda"} else "cpu"
            # Prefer cached model
            asr_model = self._get_whisperx_asr_model(device, language)
            audio16 = self._resample_to_16k(audio, self.sample_rate)
            if asr_model is None:
                return None
            result = asr_model.transcribe(audio16.cpu().numpy(), language=language)
            # Load alignment model for given language
            align_pair = self._get_whisperx_aligner(language, device)
            if align_pair is None:
                return None
            align_model, metadata = align_pair
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

    def trim(
        self,
        audio: Optional[torch.Tensor],
        language: str,
        original_text: str,
        preloaded_words: Optional[List[Any]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
        """
        Trim trailing non-speech from audio and apply a short fade-out.

        When ``preloaded_words`` is provided (list of ``ASRWord`` or dicts with
        ``word``/``start``/``end``), Smart-Match and the WhisperX fallback skip
        their internal WhisperX calls and use those alignments directly.

        Returns trimmed audio and metadata (cut indices, method used).
        """
        preloaded_dicts = _asr_words_to_dicts(preloaded_words)
        meta: Dict[str, Any] = {
            "enabled": self.enabled,
            "method": None,
            "cut_sample": None,
            "kept_post_silence_ms": self.post_speech_silence_ms,
            "fade_out_ms": self.fade_out_ms,
            "match_type": None,
            "matched_words": None,
            "match_ratio": None,
            "language_gated": False,
        }

        if not self.enabled or audio is None or audio.numel() == 0:
            return audio, None, meta

        try:
            total_len = audio.shape[-1]
            if total_len <= 0:
                return audio, None, meta

            cut_idx = None
            method = None
            sm_cfg = self._smart_match_cfg()

            # Smart-match (if enabled)
            try:
                if bool(sm_cfg.get("enabled", False)):
                    last_n = int(sm_cfg.get("last_n_words", 2))
                    fuzzy = float(sm_cfg.get("fuzzy_ratio", 0.85))
                    win_k = int(sm_cfg.get("search_window_words", 30))
                    sm_cut, sm_meta = self._trim_after_last_words_with_alignment(
                        audio,
                        language=language,
                        original_text=original_text,
                        last_n_words=last_n,
                        fuzzy_ratio=fuzzy,
                        search_window_words=win_k,
                        preloaded_words=preloaded_dicts,
                    )
                    # Merge meta
                    for k in (
                        "match_type",
                        "matched_words",
                        "match_ratio",
                        "language_gated",
                        "trailing_asr_words",
                        "match_rejected",
                    ):
                        meta[k] = sm_meta.get(k, meta.get(k))
                    if sm_cut is not None:
                        reject_reason = self._should_reject_smart_match(
                            sm_cut,
                            audio,
                            language,
                            original_text,
                            sm_meta,
                            sm_cfg,
                        )
                        if reject_reason:
                            meta["smart_match_rejected"] = reject_reason
                            too_short, cut_s, expected_s = self._is_cut_below_expected_minimum(
                                sm_cut, original_text, sm_cfg
                            )
                            if cut_s is not None:
                                meta["proposed_cut_s"] = cut_s
                            if expected_s is not None:
                                meta["expected_duration_s"] = expected_s
                            logger.debug(
                                "Smart-match rejected (%s): proposed cut %.2fs",
                                reject_reason,
                                cut_s if cut_s is not None else -1.0,
                            )
                        else:
                            cut_idx = sm_cut
                            method = "smart_words"
            except Exception as e:
                logger.debug(f"Smart-match failed: {e}")

            # Only expose Smart-Match words in meta if we actually cut via Smart-Match
            if cut_idx is not None and method == "smart_words":
                # keep as is
                pass
            else:
                # When Smart-Match failed or wasn't used, do not carry over its matched_words to avoid confusion
                meta["matched_words"] = None
                meta["match_type"] = None

            # Revised fallback order after Smart-Match failure:
            # 1) VAD (speech-aware, conservative)
            # 2) WhisperX last-word alignment (only if near the end)
            if cut_idx is None:
                vad_cut = self._trim_with_vad(audio, language)
                if isinstance(vad_cut, int):
                    cut_idx = vad_cut
                    method = "vad"

            if cut_idx is None:
                wx_cut = self._trim_with_whisperx(
                    audio,
                    language=language,
                    original_text=original_text,
                    preloaded_words=preloaded_dicts,
                )
                if isinstance(wx_cut, int):
                    # accept only if last-word end beyond 60% of duration
                    try:
                        if wx_cut >= int(0.50 * total_len):
                            cut_idx = wx_cut
                            method = "whisperx"
                    except Exception:
                        cut_idx = wx_cut
                        method = "whisperx"

            # Fallback: energy heuristic
            if cut_idx is None:
                cut_idx = self._trim_with_energy(audio)
                method = "energy" if cut_idx is not None else None

            if cut_idx is None:
                return audio, None, meta

            too_short, cut_s, expected_s = self._is_cut_below_expected_minimum(
                cut_idx, original_text, sm_cfg
            )
            if too_short:
                meta["trim_rejected"] = "cut_below_expected_duration"
                if cut_s is not None:
                    meta["proposed_cut_s"] = cut_s
                if expected_s is not None:
                    meta["expected_duration_s"] = expected_s
                logger.debug(
                    "Trim skipped: proposed cut %.2fs below expected minimum (expected %.2fs)",
                    cut_s if cut_s is not None else -1.0,
                    expected_s if expected_s is not None else -1.0,
                )
                return audio, None, meta

            # Keep additional post-speech silence – but never extend beyond detected last speech end
            keep_extra = int(self.sample_rate * (self.post_speech_silence_ms / 1000.0))
            # If we have whisperx words from smart-match, cap at last alignment word end (prevents cloning full file)
            cap_idx = None
            try:
                # best-effort: compute last speech end using VAD if method is not energy
                if method in {"smart_words", "whisperx", "vad"}:
                    # Compute a conservative last-voiced index using VAD to cap the kept region
                    vad_last = self._trim_with_vad(audio, language)
                    if isinstance(vad_last, int) and vad_last > 0:
                        cap_idx = max(0, min(total_len, vad_last))
            except Exception:
                cap_idx = None

            final_idx = max(0, cut_idx + keep_extra)
            if cap_idx is not None:
                final_idx = min(final_idx, cap_idx)
            final_idx = min(total_len, final_idx)

            trimmed = audio[..., :final_idx].clone()
            removed = audio[..., final_idx:].clone() if final_idx < total_len else None
            trimmed = self._apply_fade_out(trimmed, self.fade_out_ms)

            meta["method"] = method
            meta["cut_sample"] = int(final_idx)
            return trimmed, removed, meta
        except Exception as e:
            logger.debug(f"Tail trim failed: {e}")
            return audio, None, meta


