import logging
import warnings
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

from chunking.base_chunker import TextChunk
import torch
import re

# Suppress external package warnings early
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
    module="pkg_resources",
)

from generation.model_cache import ChatterboxModelCache

# Import the standardized AudioCandidate from file_manager
from utils.file_manager.io_handlers.candidate_io import AudioCandidate
from utils.qwen3_progress_streamer import Qwen3ProgressStreamer
from utils.language_registry import (
    MODEL_FEATURES,
    QWEN3_LANGUAGE_CODE_TO_NAME,
    VIBEVOICE_LANGUAGE_CODE_TO_NAME,
    VIBEVOICE_MODEL_TYPES,
    filter_params_for_model,
)

logger = logging.getLogger(__name__)

# Track languages we already warned about for VibeVoice so the log is not
# flooded when every chunk triggers the same warning.
_VIBEVOICE_LANGUAGE_WARNED: Set[str] = set()
_QWEN3_LANGUAGE_WARNED: Set[str] = set()


class TTSGenerator:
    """
    Simplified TTS Generator with direct model access.
    Uses direct model access without thread safety for sequential execution.
    Supports ChatterboxTTS (standard), ChatterboxMultilingualTTS (multilingual),
    ChatterboxTurboTTS (turbo), Qwen3TTSModel (qwen3), and VibeVoice Hub variants (vibevoice,
    vibevoice_1_5b, vibevoice_q4).
    """

    def __init__(self, config: Dict[str, Any], device: str = "auto", seed: Optional[int] = None):
        """
        Initializes the TTSGenerator.

        Args:
            config: Configuration dictionary with generation settings.
            device: The device to run inference on (cuda, mps, cpu).
            seed: Random seed for reproducibility. If None, uses global_seed from config.
        """
        self.config = config
        self.device = device if device != "auto" else self._detect_device()
        
        # Get seed from config if not provided
        generation_config = config.get("generation", {})
        if seed is None:
            self.global_seed = generation_config.get("global_seed", 0)
        else:
            self.global_seed = seed

        self.seed_fixed = bool(generation_config.get("seed_fixed", False))

        self.seed = self.global_seed

        # Set task-global seed once for consistent generation
        if self.seed > 0:
            torch.manual_seed(self.seed)
            logger.info("")
            logger.info(f"🎲 Task-global seed set to {self.seed}")
            logger.info("")
        else:
            logger.info("")
            logger.info("🎲 Using random seed per generation (global_seed = 0)")
            logger.info("")

        if self.seed_fixed:
            if self.global_seed > 0:
                logger.info(
                    "🎲 seed_fixed=true: per-candidate torch.manual_seed uses only the effective "
                    "base seed (no candidate index or text hash). See docs/SEEDING.md"
                )
            else:
                logger.warning(
                    "seed_fixed=true but global_seed is 0: set global_seed > 0 or "
                    "generation.speakers[].seed > 0 for a fixed torch seed; otherwise "
                    "behavior matches random-seed mode where applicable."
                )

        # Get model type from config
        self.model_type = generation_config.get("model_type", "standard")

        # Model type convenience flags
        self.is_turbo = (self.model_type == "turbo")
        self.is_qwen3 = (self.model_type == "qwen3")
        self.is_vibevoice = self.model_type in VIBEVOICE_MODEL_TYPES
        self.is_chatterbox = self.model_type in ("standard", "multilingual", "turbo")

        # All model types support speaker switching (different reference audio, tts_params, prosody).
        # Language-only restrictions (standard/turbo are EN-only) are enforced via
        # generation_handler._validate_languages_for_model() as soft warnings, not by
        # blocking speaker switches here.
        features = MODEL_FEATURES.get(self.model_type, {})
        self.supports_speaker_switch: bool = features.get("speaker_switch", True)

        # Use direct model access with model type
        self.model = ChatterboxModelCache.get_model(self.device, self.model_type, config=config)
        
        # Check if we actually got a multilingual model or fallback standard model
        self.is_multilingual = (
            self.model_type == "multilingual" and 
            hasattr(self.model, '__class__') and 
            'Multilingual' in self.model.__class__.__name__
        )

        # Speaker system attributes
        self.current_speaker_id = "default"
        self.speakers_config = config.get("generation", {}).get("speakers", [])

        # Qwen3: cache of voice clone prompts keyed by speaker_id
        self._qwen3_voice_prompts: Dict[str, Any] = {}
        # VibeVoice: cache of preprocessed reference audio keyed by speaker_id
        self._vibevoice_reference_audio: Dict[str, np.ndarray] = {}

        # Set of (model_type:param_key) combos already warned about to avoid log spam
        self._logged_unsupported_params: Set[str] = set()

        # Optional hook invoked after each successfully generated candidate
        # (see :meth:`set_candidate_ready_hook`). Used by the generation stage
        # to flush candidates to disk incrementally.
        self._candidate_ready_hook: Optional[Callable[[AudioCandidate], None]] = None

        # Determine display name for logging
        if self.is_qwen3:
            model_display = "Qwen3TTSModel (1.7B Base)"
        elif self.is_vibevoice:
            model_display = ChatterboxModelCache._MODEL_DISPLAY_NAMES.get(
                self.model_type, "VibeVoice"
            )
        elif self.is_turbo:
            model_display = "ChatterboxTurboTTS"
        elif self.is_multilingual:
            model_display = "ChatterboxMultilingualTTS"
        elif self.model_type == "multilingual":
            model_display = "ChatterboxTTS (multilingual fallback)"
        else:
            model_display = "ChatterboxTTS"
        
        logger.debug(
            f"TTSGenerator initialized on device: {self.device} with {len(self.speakers_config)} speakers, "
            f"using {model_display}"
        )

    # ------------------------------------------------------------------
    # Device and seed helpers
    # ------------------------------------------------------------------

    def get_speaker_seed(self, speaker_id: str) -> int:
        """Get the effective seed for a specific speaker."""
        speaker_config = None
        for speaker in self.speakers_config:
            if speaker.get("id") == speaker_id:
                speaker_config = speaker
                break
        
        if speaker_config is None:
            logger.warning(f"Speaker '{speaker_id}' not found in config, using global seed")
            return self.global_seed
            
        speaker_seed = speaker_config.get("seed")
        if speaker_seed is None:
            return self.global_seed
        elif speaker_seed == 0:
            return 0
        else:
            return speaker_seed

    def _torch_seed_per_candidate(self, base: int, candidate_idx: int, text: str) -> int:
        """Torch seed for one candidate row (chunk text + index)."""
        if self.seed_fixed and base > 0:
            return base
        return base + (candidate_idx * 1000) + hash(text) % 10000

    def _torch_seed_immediate_retry(self, base: int, attempt: int) -> int:
        """Torch seed inside Chatterbox immediate-retry loop."""
        if self.seed_fixed and base > 0:
            return int(base)
        return int(base + attempt * 9973)

    # ------------------------------------------------------------------
    # Candidate-ready hook (incremental save)
    # ------------------------------------------------------------------

    def release_model(self) -> None:
        """Drop the direct reference to the underlying TTS model so VRAM can
        be reclaimed at stage boundaries (generation → validation).

        ``ChatterboxModelCache.free_vram()`` removes the cache entry and calls
        ``.to("cpu")`` on the model, but this very ``TTSGenerator`` instance
        still holds a live reference via ``self.model``. As long as that
        reference exists, Python cannot garbage-collect the object and the
        CUDA allocator is unwilling to release the (now-"freed") memory to
        the OS. Nulling the attribute here closes the last handle so the next
        ``empty_cache()`` actually returns VRAM. ``_ensure_model()`` reloads
        on demand when generation is reentered (e.g. validation retry).
        """
        # Clear model-specific caches that may transitively hold GPU tensors
        # (e.g. Qwen3 voice prompts can retain speaker encoder outputs).
        try:
            self._qwen3_voice_prompts.clear()
        except Exception:
            pass
        try:
            self._vibevoice_reference_audio.clear()
        except Exception:
            pass
        self.model = None  # type: ignore[assignment]

    def _ensure_model(self) -> None:
        """Reload the TTS model if it was released via :meth:`release_model`.

        Called at the top of every generation entry point so retries after a
        VRAM eviction transparently re-fetch the model from the cache.
        """
        if getattr(self, "model", None) is None:
            logger.info(
                f"🔄 Reloading TTS model (model_type={self.model_type}) after VRAM eviction"
            )
            self.model = ChatterboxModelCache.get_model(
                self.device, self.model_type, config=self.config
            )

    def set_candidate_ready_hook(
        self, hook: Optional[Callable[[AudioCandidate], None]]
    ) -> None:
        """Register a callback invoked right after each AudioCandidate is
        rendered. Passing ``None`` clears a previously registered hook.

        The hook receives the freshly created candidate *before* it is added
        to any return list, so the caller can persist it immediately and
        mitigate data loss from crashes mid-batch. Exceptions raised inside
        the hook are logged and swallowed – generation never fails because of
        a failed side-effect.
        """
        self._candidate_ready_hook = hook

    def _emit_candidate_ready(self, candidate: AudioCandidate) -> None:
        """Fire the candidate-ready hook defensively."""
        hook = self._candidate_ready_hook
        if hook is None:
            return
        try:
            hook(candidate)
        except Exception as e:
            logger.warning(
                f"candidate_ready hook raised {type(e).__name__}: {e} — "
                f"candidate {candidate.candidate_idx+1} of chunk "
                f"{candidate.chunk_idx+1} not persisted incrementally"
            )

    def _detect_device(self) -> str:
        """Detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    # ------------------------------------------------------------------
    # Voice / conditionals preparation
    # ------------------------------------------------------------------

    def prepare_conditionals(self, wav_fpath: str):
        """
        Prepares model conditionals for voice cloning (Chatterbox models only).

        For Qwen3, voice prompts are built via _prepare_qwen3_voice_clone_prompt.
        For VibeVoice, this method is not used.
        calling this method for a Qwen3 model is a no-op with a warning.

        Args:
            wav_fpath: Path to reference audio file
        """
        if self.model is None:
            logger.warning("🚨 No model loaded - cannot prepare conditionals")
            return

        if self.is_qwen3 or self.is_vibevoice:
            logger.warning(
                "prepare_conditionals() called for non-Chatterbox model - "
                "use switch_speaker() to prepare model-specific speaker assets instead."
            )
            return

        logger.debug(f"🔄 Preparing conditionals for {Path(wav_fpath).name}")

        try:
            self.model.prepare_conditionals(wav_fpath=wav_fpath)
            logger.debug("✅ Conditionals prepared")
        except Exception as e:
            logger.error(f"🚨 Error preparing conditionals: {e}")
            raise

    def _load_qwen3_ref_text(
        self,
        speaker_id: str,
        audio_path: Path,
        speaker_config: Dict[str, Any],
    ) -> Optional[str]:
        """
        Load the reference text transcript for a Qwen3 speaker.

        Resolution order:
        1. Inline text in speaker_config["reference_text"] (if multi-line or > 100 chars)
        2. File path in speaker_config["reference_text"] (relative to reference_audio dir)
        3. Sidecar .txt file next to the .wav (same stem, .txt extension)

        Returns the transcript string, or None when not found.
        """
        ref_text_config = speaker_config.get("reference_text")
        if ref_text_config:
            text_val = str(ref_text_config).strip()
            # Treat as inline text if it looks like prose (newlines or long)
            if "\n" in text_val or len(text_val) > 100:
                logger.debug(f"Using inline reference_text for speaker '{speaker_id}'")
                return text_val
            # Treat as filename relative to the reference audio directory
            ref_text_path = audio_path.parent / text_val
            if ref_text_path.exists():
                content = ref_text_path.read_text(encoding="utf-8").strip()
                logger.info(
                    f"Loaded reference text for speaker '{speaker_id}' from config path: {ref_text_path.name}"
                )
                return content
            logger.warning(
                f"reference_text '{text_val}' for speaker '{speaker_id}' not found at {ref_text_path}"
            )

        # Fallback: sidecar .txt file alongside the reference audio
        ref_text_path = audio_path.with_suffix(".txt")
        if ref_text_path.exists():
            content = ref_text_path.read_text(encoding="utf-8").strip()
            logger.info(
                f"Loaded reference text for speaker '{speaker_id}' from sidecar: {ref_text_path.name}"
            )
            return content

        logger.info(
            f"No reference text found for speaker '{speaker_id}' (no .txt sidecar, no config field) - "
            "will use x_vector_only mode (reduced cloning quality)"
        )
        return None

    def _prepare_qwen3_voice_clone_prompt(
        self,
        speaker_id: str,
        audio_path: Path,
        ref_text: Optional[str],
    ) -> None:
        """
        Build and cache the Qwen3 voice clone prompt for a speaker.

        Performs a reference-audio duration check and warns if > 3 s.
        Falls back to x_vector_only_mode if no ref_text is available.
        Result is stored in self._qwen3_voice_prompts[speaker_id].
        """
        if self.model is None:
            logger.error("No Qwen3 model loaded - cannot build voice clone prompt")
            return

        # Duration check
        try:
            import torchaudio
            info = torchaudio.info(str(audio_path))
            duration = info.num_frames / info.sample_rate
            if duration > 3.0:
                logger.warning(
                    f"⚠️ Reference audio '{audio_path.name}' is {duration:.1f}s; "
                    "Qwen3 recommends <=3s for optimal voice cloning quality. "
                    "Consider trimming to the most expressive 3-second segment."
                )
        except Exception as e:
            logger.debug(f"Could not check reference audio duration: {e}")

        x_vector_only = ref_text is None
        if x_vector_only:
            logger.warning(
                f"⚠️ Speaker '{speaker_id}': no reference text available - "
                "using x_vector_only mode (lower cloning quality). "
                "Add a .txt sidecar file next to the reference .wav for better results."
            )

        try:
            prompt = self.model.create_voice_clone_prompt(
                ref_audio=str(audio_path),
                ref_text=ref_text,
                x_vector_only_mode=x_vector_only,
            )
            self._qwen3_voice_prompts[speaker_id] = prompt
            mode = "x_vector_only" if x_vector_only else "ICL"
            logger.info(
                f"✅ Qwen3 voice prompt cached for speaker '{speaker_id}' ({mode} mode)"
            )
        except Exception as e:
            logger.error(
                f"🚨 Failed to create Qwen3 voice clone prompt for speaker '{speaker_id}': {e}"
            )

    def _prepare_vibevoice_reference_audio(
        self,
        speaker_id: str,
        audio_path: Path,
        voice_speed_factor: float = 1.0,
    ) -> None:
        """
        Load and cache a VibeVoice reference waveform for a speaker.

        The waveform is resampled to 24kHz mono and optionally speed-adjusted by
        linear interpolation (factor range expected around 0.8..1.2).
        """
        try:
            import librosa

            audio_np, _ = librosa.load(str(audio_path), sr=24000, mono=True)
            audio_np = audio_np.astype(np.float32)

            if voice_speed_factor != 1.0:
                target_len = int(len(audio_np) / max(0.01, voice_speed_factor))
                audio_np = np.interp(
                    np.linspace(0, len(audio_np) - 1, target_len),
                    np.arange(len(audio_np)),
                    audio_np,
                ).astype(np.float32)

            self._vibevoice_reference_audio[speaker_id] = audio_np
            logger.info(
                f"✅ VibeVoice reference audio cached for speaker '{speaker_id}' "
                f"(speed_factor={voice_speed_factor:.2f})"
            )
        except Exception as e:
            logger.error(f"🚨 Failed to prepare VibeVoice reference audio: {e}")

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _filter_params_for_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter a tts_params dict to only include keys supported by self.model_type.

        Unsupported keys are dropped with a one-time info log.
        Internal deviation/control keys are always silently dropped.
        """
        return filter_params_for_model(
            self.model_type,
            params,
            logged_keys=self._logged_unsupported_params,
        )

    def _normalize_generation_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize generation kwargs to align with the underlying Chatterbox model.generate signature.

        This method is only used for Chatterbox models (standard/multilingual/turbo).
        For Qwen3, parameters are passed directly via generate_voice_clone().

        Behavior:
        - If model.generate has **kwargs, do NOT filter unknown keys (only alias-remap).
        - If model.generate does NOT have **kwargs, filter to accepted names to avoid TypeError.
        - Attempt common alias remaps (cfg_weight, temperature, exaggeration, top_p, min_p,
          language_id, repetition_penalty).
        """
        model_ref = self.model
        if model_ref is None or not hasattr(model_ref, "generate"):
            return params

        try:
            sig = inspect.signature(model_ref.generate)
            accepted_params = set(sig.parameters.keys())
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        except Exception:
            return params

        normalized: Dict[str, Any] = dict(params)

        def remap(src_key: str, candidates: List[str]):
            if src_key in normalized and src_key not in accepted_params:
                for cand in candidates:
                    if cand in accepted_params:
                        normalized[cand] = normalized.pop(src_key)
                        return

        remap("cfg_weight", ["cfg_weight", "cfg", "guidance_scale", "classifier_free_guidance_weight"])
        remap("temperature", ["temperature", "temp"])
        remap("exaggeration", ["exaggeration", "style_exaggeration", "expressiveness"])
        remap("top_p", ["top_p", "nucleus_p"])
        remap("min_p", ["min_p", "p_min"])
        remap("language_id", ["language_id", "language", "lang"])
        remap("repetition_penalty", ["repetition_penalty", "repeat_penalty"])

        if has_var_kw:
            logger.debug(
                f"Model.generate accepts **kwargs; passing params without filtering. Keys: {sorted(normalized.keys())}"
            )
            return normalized

        filtered = {k: v for k, v in normalized.items() if k in accepted_params}
        dropped = [k for k in normalized.keys() if k not in accepted_params]
        if dropped:
            logger.debug(f"Dropping unsupported TTS params for model.generate (no **kwargs): {dropped}")
        logger.debug(f"Params passed to model.generate: {sorted(filtered.keys())}")
        return filtered

    def _calculate_ramped_params(
        self,
        base_params: Dict[str, Any],
        i: int,
        num_expressive_minus_one: int,
    ) -> Dict[str, Any]:
        """
        Calculate ramped TTS parameters for candidate variation.

        For Chatterbox models (standard/multilingual/turbo):
          - exaggeration, cfg_weight, temperature:  base + (dev * ramp_position)
            Use negative *dev* for a downward ramp (e.g. exaggeration).

        For Qwen3:
          - Only temperature is ramped; exaggeration/cfg_weight are not applicable.
        """
        if num_expressive_minus_one <= 0:
            return dict(base_params)

        ramp_position = i / max(1, num_expressive_minus_one)
        params = dict(base_params)

        base_temp = base_params.get("temperature", 1.0)
        temp_dev = base_params.get("temperature_max_deviation", 0.2)
        params["temperature"] = base_temp + (temp_dev * ramp_position)

        if not self.is_qwen3:
            base_exag = base_params.get("exaggeration", 0.6)
            exag_dev = base_params.get("exaggeration_max_deviation", 0.15)
            base_cfg = base_params.get("cfg_weight", 0.7)
            cfg_dev = base_params.get("cfg_weight_max_deviation", 0.15)
            params["exaggeration"] = base_exag + (exag_dev * ramp_position)
            params["cfg_weight"] = base_cfg + (cfg_dev * ramp_position)

        return params

    # ------------------------------------------------------------------
    # Qwen3-specific generation path
    # ------------------------------------------------------------------

    def _generate_qwen3_single(
        self,
        text: str,
        language_name: str,
        params: Dict[str, Any],
        voice_prompt: Any,
        speaker_id: str,
        attempt_seed: int,
    ) -> torch.Tensor:
        """
        Generate a single audio sample using the Qwen3 voice clone API.

        Args:
            text: Text to synthesize.
            language_name: Full language name, e.g. "English", "German".
            params: Filtered tts_params (only Qwen3-supported keys).
            voice_prompt: Pre-built voice clone prompt from create_voice_clone_prompt().
            speaker_id: Speaker ID (used only for logging).
            attempt_seed: Seed for reproducibility via torch.manual_seed.

        Returns:
            1D audio tensor.
        """
        if self.model is None:
            logger.warning("🚨 No Qwen3 model loaded - generating silence")
            return torch.zeros(24000, device=self.device)

        if voice_prompt is None:
            logger.warning(
                f"🚨 No voice prompt for speaker '{speaker_id}' - generating silence"
            )
            return torch.zeros(24000, device=self.device)

        # Apply seed for reproducibility (Qwen3 has no native seed param)
        if attempt_seed > 0:
            try:
                torch.manual_seed(attempt_seed)
            except Exception:
                pass

        # Build generation kwargs from filtered params
        gen_kwargs: Dict[str, Any] = {}
        for key in (
            "top_k", "top_p", "temperature", "repetition_penalty",
            "max_new_tokens", "do_sample",
            "subtalker_dosample", "subtalker_top_k", "subtalker_top_p",
            "subtalker_temperature",
        ):
            if key in params:
                gen_kwargs[key] = params[key]

        # Token-level progress on the talker LM (same hook as chatterbox_tester).
        talker = getattr(getattr(self.model, "model", None), "talker", None)
        total_tok = int(gen_kwargs.get("max_new_tokens") or 2048)
        streamer = Qwen3ProgressStreamer(total=total_tok, label="Qwen3")
        orig_talker_generate = None
        if talker is not None and hasattr(talker, "generate"):
            orig_talker_generate = talker.generate

            def _patched_generate(*a, _orig=orig_talker_generate, _s=streamer, **kw):
                kw.setdefault("streamer", _s)
                return _orig(*a, **kw)

            talker.generate = _patched_generate  # type: ignore[assignment]

        qwen3_text = re.sub(r"\n{2,}", " ", text).strip()

        try:
            try:
                wavs, _sr = self.model.generate_voice_clone(
                    text=qwen3_text,
                    language=language_name,
                    voice_clone_prompt=voice_prompt,
                    **gen_kwargs,
                )
            finally:
                if talker is not None and orig_talker_generate is not None:
                    talker.generate = orig_talker_generate  # type: ignore[assignment]
                streamer.end()
        except Exception as e:
            logger.error(f"Qwen3 generate_voice_clone failed: {e}")
            return torch.zeros(24000, device=self.device)

        audio = torch.tensor(wavs[0], dtype=torch.float32)
        if audio.ndim == 2:
            audio = audio.squeeze(0)
        return audio.to(self.device)

    def _generate_qwen3_candidates(
        self,
        text: str,
        speaker_id: str,
        speaker_config: Dict[str, Any],
        num_candidates: int,
        language_id: str,
    ) -> List[AudioCandidate]:
        """
        Generate multiple Qwen3 voice clone candidates with temperature ramping.
        """
        language_name = QWEN3_LANGUAGE_CODE_TO_NAME.get(language_id, "English")
        if language_id not in QWEN3_LANGUAGE_CODE_TO_NAME and language_id not in _QWEN3_LANGUAGE_WARNED:
            _QWEN3_LANGUAGE_WARNED.add(language_id)
            logger.warning(
                f"Unknown language code '{language_id}' for Qwen3 - defaulting to 'English'"
            )

        raw_tts_params = speaker_config.get("tts_params", {})
        raw_conservative = speaker_config.get("conservative_candidate", {})
        conservative_enabled = raw_conservative.get("enabled", False)

        # Filter to Qwen3-supported parameters only
        filtered_base = self._filter_params_for_model(raw_tts_params)
        filtered_conservative = (
            self._filter_params_for_model(raw_conservative)
            if conservative_enabled
            else {}
        )

        base_temperature = filtered_base.get("temperature", 0.9)
        temp_deviation = raw_tts_params.get("temperature_max_deviation", 0.2)

        # Qwen3 ramped params – read deviations from raw (unfiltered) params
        top_k_dev = raw_tts_params.get("top_k_max_deviation", 0)
        subtalker_temp_dev = raw_tts_params.get("subtalker_temperature_max_deviation", 0)
        subtalker_top_k_dev = raw_tts_params.get("subtalker_top_k_max_deviation", 0)
        top_k_base = filtered_base.get("top_k")
        subtalker_temp_base = filtered_base.get("subtalker_temperature")
        subtalker_top_k_base = filtered_base.get("subtalker_top_k")

        num_expressive = num_candidates - 1 if conservative_enabled else num_candidates
        voice_prompt = self._qwen3_voice_prompts.get(speaker_id)

        if voice_prompt is None:
            logger.warning(
                f"⚠️ Qwen3 voice prompt for speaker '{speaker_id}' is not cached - "
                "generation will produce silence. Ensure switch_speaker() was called first."
            )

        base_seed = self.get_speaker_seed(speaker_id)
        candidates: List[AudioCandidate] = []

        for i in range(num_candidates):
            is_conservative = conservative_enabled and (i + 1) == num_candidates
            candidate_seed = self._torch_seed_per_candidate(base_seed, i, text)

            if is_conservative:
                params = dict(filtered_base)
                params.update(filtered_conservative)
                # Conservative: use base temperature * 0.85 if not explicitly set
                if "temperature" not in filtered_conservative:
                    params["temperature"] = base_temperature * 0.85
                candidate_type = "CONSERVATIVE"
            else:
                params = dict(filtered_base)

                # subtalker_temperature: from base, ramp via base + (dev * ramp_pos)
                # (negative dev = RAMP-DOWN, positive dev = RAMP-UP; same as other params)
                if subtalker_temp_base is not None and subtalker_temp_dev != 0:
                    params["subtalker_temperature"] = subtalker_temp_base

                if i > 0 and num_expressive > 1:
                    ramp_pos = i / max(1, num_expressive - 1)
                    # temperature: RAMP-UP
                    params["temperature"] = base_temperature + (temp_deviation * ramp_pos)
                    # top_k: RAMP-UP
                    if top_k_base is not None and top_k_dev != 0:
                        params["top_k"] = int(round(top_k_base + top_k_dev * ramp_pos))
                    # subtalker_temperature: base + (dev * ramp_pos)
                    if subtalker_temp_base is not None and subtalker_temp_dev != 0:
                        params["subtalker_temperature"] = (
                            subtalker_temp_base + subtalker_temp_dev * ramp_pos
                        )
                    # subtalker_top_k: RAMP-UP
                    if subtalker_top_k_base is not None and subtalker_top_k_dev != 0:
                        params["subtalker_top_k"] = int(round(subtalker_top_k_base + subtalker_top_k_dev * ramp_pos))

                candidate_type = "EXPRESSIVE"

            try:
                log_temp = params.get("temperature", base_temperature)
                log_sub_temp = params.get("subtalker_temperature")
                log_sub_top_k = params.get("subtalker_top_k")
                logger.info(
                    f"QWEN3 CAND {i+1}/{num_candidates} (lang={language_name}) "
                    f"({'CONS' if is_conservative else 'EXP'}): "
                    f"temp={log_temp:.3f}, top_k={params.get('top_k', '-')}, "
                    f"top_p={params.get('top_p', '-')}, rep_pen={params.get('repetition_penalty', '-')}, "
                    f"sub_temp={log_sub_temp if log_sub_temp is None else f'{log_sub_temp:.3f}'}, "
                    f"sub_top_k={log_sub_top_k if log_sub_top_k is not None else '-'}"
                )
                audio = self._generate_qwen3_single(
                    text=text,
                    language_name=language_name,
                    params=params,
                    voice_prompt=voice_prompt,
                    speaker_id=speaker_id,
                    attempt_seed=candidate_seed,
                )
                candidate = AudioCandidate(
                    chunk_idx=0,
                    candidate_idx=i,
                    audio_path=Path(),
                    audio_tensor=audio,
                    generation_params={
                        **params,
                        "language_id": language_id,
                        "type": candidate_type,
                        "seed": candidate_seed,
                    },
                )
                candidates.append(candidate)
                self._emit_candidate_ready(candidate)
                logger.debug(
                    f"Qwen3 candidate {i+1}: duration={audio.shape[-1]/24000:.2f}s"
                )
            except Exception as e:
                logger.error(f"Failed to generate Qwen3 candidate {i+1}: {e}")
                continue

        return candidates

    # ------------------------------------------------------------------
    # VibeVoice-specific generation path
    # ------------------------------------------------------------------

    def _generate_vibevoice_single(
        self,
        text: str,
        params: Dict[str, Any],
        reference_audio: np.ndarray,
        attempt_seed: int,
    ) -> torch.Tensor:
        """Generate one VibeVoice sample from text + reference audio."""
        if self.model is None:
            logger.warning("🚨 No VibeVoice model loaded - generating silence")
            return torch.zeros(24000, device=self.device)
        if not hasattr(self.model, "_vv_processor"):
            logger.error("🚨 VibeVoice processor missing on model - generating silence")
            return torch.zeros(24000, device=self.device)

        processor = self.model._vv_processor
        if attempt_seed > 0:
            try:
                torch.manual_seed(attempt_seed)
            except Exception:
                pass

        formatted_text = f"Speaker 1: {' '.join(text.split())}"
        try:
            inputs = processor(
                [formatted_text],
                voice_samples=[[reference_audio]],
                return_tensors="pt",
                return_attention_mask=True,
            )
            model_device = next(self.model.parameters()).device
            inputs = {
                k: v.to(model_device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
        except Exception as e:
            logger.error(f"VibeVoice input preparation failed: {e}")
            return torch.zeros(24000, device=self.device)

        diffusion_steps = int(params.get("diffusion_steps", 20))
        try:
            self.model.set_ddpm_inference_steps(diffusion_steps)
        except Exception:
            pass

        # temperature/top_p are only forwarded when do_sample=True; with use_sampling=false the LM is greedy/deterministic.
        use_sampling = bool(params.get("use_sampling", False))
        gen_kwargs: Dict[str, Any] = {
            "tokenizer": processor.tokenizer,
            "cfg_scale": float(params.get("cfg_scale", 1.3)),
            "max_new_tokens": None,
            "do_sample": use_sampling,
        }
        if use_sampling:
            gen_kwargs["temperature"] = float(params.get("temperature", 0.95))
            gen_kwargs["top_p"] = float(params.get("top_p", 0.95))

        try:
            with torch.no_grad():
                output = self.model.generate(**inputs, **gen_kwargs)
            if not hasattr(output, "speech_outputs") or not output.speech_outputs:
                raise RuntimeError("missing speech_outputs")
            speech_outputs = output.speech_outputs
            audio_tensor = (
                torch.cat(speech_outputs, dim=-1)
                if isinstance(speech_outputs, list)
                else speech_outputs
            )
            audio_tensor = audio_tensor.float().squeeze()
            return audio_tensor.to(self.device)
        except Exception as e:
            logger.error(f"VibeVoice generation failed: {e}")
            return torch.zeros(24000, device=self.device)

    def _generate_vibevoice_candidates(
        self,
        text: str,
        speaker_id: str,
        speaker_config: Dict[str, Any],
        num_candidates: int,
        language_id: str,
    ) -> List[AudioCandidate]:
        """Generate VibeVoice candidates with optional ramping and conservative tail.

        Across expressive candidates, ``temperature`` ramps UP from configured base caml values (higher values make prosody increasingly lively). ``cfg_scale`` ramps UP from the configured base. beginning with lower values for more lively prosody, higher values have a calming effect. = The crossover effect avoids both knobs pushing toward wilder output at once.
        """
        language_name = VIBEVOICE_LANGUAGE_CODE_TO_NAME.get(language_id, "English")
        if language_id not in VIBEVOICE_LANGUAGE_CODE_TO_NAME:
            vv_cfg = (self.config.get("generation", {}) or {}).get("vibevoice", {}) or {}
            language_strict = bool(vv_cfg.get("language_strict", False))
            if language_strict:
                raise ValueError(
                    f"VibeVoice language_strict=true and language '{language_id}' is not supported. "
                    f"Supported: {sorted(VIBEVOICE_LANGUAGE_CODE_TO_NAME.keys())}"
                )
            if language_id not in _VIBEVOICE_LANGUAGE_WARNED:
                _VIBEVOICE_LANGUAGE_WARNED.add(language_id)
                logger.warning(
                    f"Unknown language code '{language_id}' for VibeVoice - defaulting to English "
                    f"(officially supported: {sorted(VIBEVOICE_LANGUAGE_CODE_TO_NAME.keys())}, but many more language are in the training set, so it's likely to work anyway). "
                    "Set generation.vibevoice.language_strict=true to fail instead."
                )

        raw_tts_params = speaker_config.get("tts_params", {})
        raw_conservative = speaker_config.get("conservative_candidate", {})
        conservative_enabled = raw_conservative.get("enabled", False)

        filtered_base = self._filter_params_for_model(raw_tts_params)
        filtered_conservative = (
            self._filter_params_for_model(raw_conservative)
            if conservative_enabled
            else {}
        )

        ref_audio = self._vibevoice_reference_audio.get(speaker_id)
        if ref_audio is None:
            logger.warning(
                f"⚠️ VibeVoice reference audio for speaker '{speaker_id}' is not cached - generating silence"
            )
            ref_audio = np.zeros(24000, dtype=np.float32)

        base_cfg = float(filtered_base.get("cfg_scale", 1.3))
        cfg_dev = float(raw_tts_params.get("cfg_scale_max_deviation", 0.0))
        base_temp = float(filtered_base.get("temperature", 0.95))
        temp_dev = float(raw_tts_params.get("temperature_max_deviation", 0.0))
        base_steps = int(filtered_base.get("diffusion_steps", 20))
        steps_dev = float(raw_tts_params.get("diffusion_steps_max_deviation", 0.0))
        num_expressive = num_candidates - 1 if conservative_enabled else num_candidates
        base_seed = self.get_speaker_seed(speaker_id)
        candidates: List[AudioCandidate] = []

        for i in range(num_candidates):
            is_conservative = conservative_enabled and (i + 1) == num_candidates
            params = dict(filtered_base)
            if is_conservative:
                params.update(filtered_conservative)
                candidate_type = "CONSERVATIVE"
            else:
                if i > 0 and num_expressive > 1:
                    ramp_pos = i / max(1, num_expressive - 1)
                    # cfg_scale: ramp UP
                    params["cfg_scale"] = base_cfg + (cfg_dev * ramp_pos)
                    # temperature: ramp UP
                    params["temperature"] = base_temp + (temp_dev * ramp_pos)
                    # diffusion_steps: ramp UP (same ramp_pos; offsets cfg_scale calming vs temperature liveliness)
                    if steps_dev != 0.0:
                        params["diffusion_steps"] = max(
                            5,
                            min(60, int(round(base_steps + steps_dev * ramp_pos))),
                        )
                candidate_type = "EXPRESSIVE"

            candidate_seed = self._torch_seed_per_candidate(base_seed, i, text)
            logger.info(
                f"VIBEVOICE CAND {i+1}/{num_candidates} (lang={language_name}) "
                f"({'CONS' if is_conservative else 'EXP'}): "
                f"cfg_scale={params.get('cfg_scale', 1.3):.3f}, "
                f"temp={params.get('temperature', 0.95):.3f}, "
                f"top_p={params.get('top_p', 0.95):.3f}, "
                f"steps={int(params.get('diffusion_steps', 20))}"
                f"{f', steps_dev={steps_dev:.2f}' if steps_dev != 0.0 else ''}, "
                f"use_sampling={bool(params.get('use_sampling', False))}"
            )
            audio = self._generate_vibevoice_single(
                text=text,
                params=params,
                reference_audio=ref_audio,
                attempt_seed=candidate_seed,
            )
            candidate = AudioCandidate(
                chunk_idx=0,
                candidate_idx=i,
                audio_path=Path(),
                audio_tensor=audio,
                generation_params={
                    **params,
                    "language_id": language_id,
                    "type": candidate_type,
                    "seed": candidate_seed,
                },
            )
            candidates.append(candidate)
            self._emit_candidate_ready(candidate)

        return candidates

    def generate_validation_retry_candidates(
        self,
        chunk: TextChunk,
        speaker_config: Dict[str, Any],
        language_id: str,
        max_retries: int,
        start_candidate_idx: int,
    ) -> List[AudioCandidate]:
        """
        Post-validation retry candidates for VibeVoice and Qwen3.

        Merges ``tts_params`` with ``conservative_candidate`` (filtered, same as the
        main conservative slot). When ``conservative_candidate.enabled`` is false,
        uses base ``tts_params`` only (with a warning). Each retry applies the same
        small ``variation_factor`` sweep as Chatterbox retries in ``RetryLogic``.
        """
        if max_retries <= 0:
            return []

        def _retry_variation_factor(i: int, n_retry: int) -> float:
            if i == 0:
                return 0.0
            vf = ((i - 1) / max(1, n_retry - 2)) * 2.0 - 1.0
            return vf * 0.05

        speaker_id = speaker_config.get("id") or getattr(chunk, "speaker_id", "default")
        text = chunk.text
        raw_tts = speaker_config.get("tts_params", {}) or {}
        raw_cons = dict(speaker_config.get("conservative_candidate", {}) or {})
        cons_enabled = bool(raw_cons.pop("enabled", False))

        filtered_base = self._filter_params_for_model(raw_tts)
        merged: Dict[str, Any] = dict(filtered_base)
        if cons_enabled:
            merged.update(self._filter_params_for_model(raw_cons))
        else:
            logger.warning(
                "⚠️ conservative_candidate not enabled for speaker '%s' — validation "
                "retries use base tts_params only",
                speaker_id,
            )

        retry_candidates: List[AudioCandidate] = []

        if self.is_vibevoice:
            ref_audio = self._vibevoice_reference_audio.get(speaker_id)
            if ref_audio is None:
                logger.warning(
                    "⚠️ VibeVoice reference audio for speaker '%s' not cached — silence",
                    speaker_id,
                )
                ref_audio = np.zeros(24000, dtype=np.float32)

            steps_dev_retry = float(raw_tts.get("diffusion_steps_max_deviation", 0.0))
            base_seed = self.get_speaker_seed(speaker_id)
            for i in range(max_retries):
                vf = _retry_variation_factor(i, max_retries)
                params = dict(merged)
                if "cfg_scale" in params:
                    params["cfg_scale"] = float(
                        max(1.0, min(2.5, float(params["cfg_scale"]) + vf * 0.12))
                    )
                if "temperature" in params:
                    params["temperature"] = float(
                        max(0.05, min(2.0, float(params["temperature"]) + vf * 0.06))
                    )
                if "top_p" in params:
                    params["top_p"] = float(
                        max(0.0, min(1.0, float(params["top_p"]) + vf * 0.04))
                    )
                # Keep merged diffusion_steps when deviation is 0; else small nudge scaled by deviation
                if "diffusion_steps" in params and steps_dev_retry != 0.0:
                    merged_steps = float(params["diffusion_steps"])
                    params["diffusion_steps"] = int(
                        max(
                            5,
                            min(
                                60,
                                int(round(merged_steps + vf * steps_dev_retry * 0.2)),
                            ),
                        )
                    )

                candidate_idx = start_candidate_idx + i
                candidate_seed = self._torch_seed_per_candidate(base_seed, candidate_idx, text)

                logger.info(
                    f"VIBEVOICE RETRY {i+1}/{max_retries}: cfg_scale={params.get('cfg_scale', '-')}, "
                    f"temp={params.get('temperature', '-')}, top_p={params.get('top_p', '-')}, "
                    f"steps={int(params.get('diffusion_steps', 20))}, vf={vf:.4f}"
                )
                audio = self._generate_vibevoice_single(
                    text=text,
                    params=params,
                    reference_audio=ref_audio,
                    attempt_seed=candidate_seed,
                )
                retry_candidates.append(
                    AudioCandidate(
                        chunk_idx=chunk.idx,
                        candidate_idx=candidate_idx,
                        audio_path=Path(),
                        audio_tensor=audio,
                        generation_params={
                            **params,
                            "language_id": language_id,
                            "type": "RETRY_CONSERVATIVE",
                            "variation_factor": vf,
                            "retry_attempt": i + 1,
                            "seed": candidate_seed,
                        },
                        chunk_text=text,
                    )
                )
                self._emit_candidate_ready(retry_candidates[-1])

            return retry_candidates

        if self.is_qwen3:
            language_name = QWEN3_LANGUAGE_CODE_TO_NAME.get(language_id, "English")
            voice_prompt = self._qwen3_voice_prompts.get(speaker_id)
            if voice_prompt is None:
                logger.warning(
                    "⚠️ Qwen3 voice prompt for speaker '%s' not cached — silence",
                    speaker_id,
                )

            base_seed = self.get_speaker_seed(speaker_id)
            for i in range(max_retries):
                vf = _retry_variation_factor(i, max_retries)
                params = dict(merged)
                if "temperature" in params:
                    params["temperature"] = float(
                        max(0.05, min(2.0, float(params["temperature"]) + vf * 0.08))
                    )
                if params.get("top_k") is not None:
                    params["top_k"] = int(
                        max(1, min(512, round(float(params["top_k"]) + vf * 8.0)))
                    )
                if params.get("subtalker_temperature") is not None:
                    params["subtalker_temperature"] = float(
                        max(
                            0.0,
                            min(2.0, float(params["subtalker_temperature"]) + vf * 0.06),
                        )
                    )
                if params.get("subtalker_top_k") is not None:
                    params["subtalker_top_k"] = int(
                        max(
                            1,
                            min(512, round(float(params["subtalker_top_k"]) + vf * 6.0)),
                        )
                    )

                candidate_idx = start_candidate_idx + i
                candidate_seed = self._torch_seed_per_candidate(base_seed, candidate_idx, text)

                logger.info(
                    f"QWEN3 RETRY {i+1}/{max_retries}: temp={params.get('temperature', '-')}, "
                    f"top_k={params.get('top_k', '-')}, sub_temp={params.get('subtalker_temperature', '-')}, vf={vf:.4f}"
                )
                audio = self._generate_qwen3_single(
                    text=text,
                    language_name=language_name,
                    params=params,
                    voice_prompt=voice_prompt,
                    speaker_id=speaker_id,
                    attempt_seed=candidate_seed,
                )
                retry_candidates.append(
                    AudioCandidate(
                        chunk_idx=chunk.idx,
                        candidate_idx=candidate_idx,
                        audio_path=Path(),
                        audio_tensor=audio,
                        generation_params={
                            **params,
                            "language_id": language_id,
                            "type": "RETRY_CONSERVATIVE",
                            "variation_factor": vf,
                            "retry_attempt": i + 1,
                            "seed": candidate_seed,
                        },
                        chunk_text=text,
                    )
                )
                self._emit_candidate_ready(retry_candidates[-1])

            return retry_candidates

        logger.error(
            "generate_validation_retry_candidates called for unsupported model_type=%s",
            self.model_type,
        )
        return []

    # ------------------------------------------------------------------
    # Chatterbox generation path
    # ------------------------------------------------------------------

    def generate_single(
        self,
        text: str,
        exaggeration: float = 0.6,
        cfg_weight: float = 0.7,
        temperature: float = 1.0,
        reference_audio_path: Optional[str] = None,
        language_id: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate single audio using direct Chatterbox model access.

        Not used for Qwen3 (see _generate_qwen3_single).
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for generation")
            return torch.zeros(1000, device=self.device)

        self._ensure_model()

        if self.model is None:
            logger.warning("🚨 No model loaded - generating silence")
            return torch.zeros(48000, device=self.device)

        logger.debug("Starting TTS generation")

        if not hasattr(self.model, "conds") or self.model.conds is None:
            logger.warning("🚨 No conditionals loaded - this should not happen with speaker system")
            return torch.zeros(48000, device=self.device)
        
        logger.debug("Using loaded conditionals (speaker-specific)")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*torch.backends.cuda.sdp_kernel.*", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*LlamaModel is using LlamaSdpaAttention.*")
            warnings.filterwarnings("ignore", message=".*torch.nn.functional.scaled_dot_product_attention.*does not support.*output_attentions=True.*")
            warnings.filterwarnings("ignore", message=".*Falling back to the manual attention implementation.*")
            warnings.filterwarnings("ignore", message=".*specifying the manual implementation will be required from Transformers version v5.0.0.*")
            warnings.filterwarnings("ignore", message=".*This warning can be removed using the argument.*attn_implementation.*eager.*")
            warnings.filterwarnings("ignore", message=".*does not support `output_attentions=True`.*")
            warnings.filterwarnings("ignore", message=".*return_dict_in_generate.*is NOT set to.*True.*but.*output_attentions.*is.*")
            warnings.filterwarnings("ignore", message=".*past_key_values.*tuple of tuples.*", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*detected that you are passing.*past_key_values.*as a tuple of tuples.*", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*convert your cache or use an appropriate.*Cache.*class.*", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*attn_implementation.*", category=FutureWarning)

            logger.debug(f"Generating audio for text (len={len(text)}): '{text[:50]}...'")

            generate_params = {
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "temperature": temperature,
                **kwargs,
            }
            
            if self.is_multilingual and language_id is not None:
                generate_params["language_id"] = language_id
                logger.debug(f"Using language_id: {language_id} for multilingual model")
            elif self.model_type == "multilingual" and not self.is_multilingual:
                logger.debug("Multilingual model requested but not available, using standard model without language_id")
            
            try:
                generate_params = self._normalize_generation_params(generate_params)
            except Exception as e:
                logger.debug(f"Param normalization failed (non-fatal): {e}")

            audio = self.model.generate(text, **generate_params)

        if audio.ndim == 2:
            audio = audio.squeeze(0)
        audio = audio.to(self.device)

        logger.debug(f"Generated audio with shape: {audio.shape}")
        return audio

    def _generate_single_with_immediate_retries(
        self,
        *,
        text: str,
        exaggeration: float,
        cfg_weight: float,
        temperature: float,
        base_seed: int,
        language_id: Optional[str],
        additional_params: Dict[str, Any],
        reference_audio_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate Chatterbox audio with immediate retries on artifact warnings.

        Strategy per retry (attempt >= 1):
        - Change seed deterministically based on base_seed and attempt
        - Increase min_p by +0.05 per attempt (clamped to [0.0, 1.0])
        - Decrease top_p by -0.05 per attempt (clamped to [0.0, 1.0])
        - Decrease temperature by -0.05 per attempt (>= 0.05)

        Returns:
            {
              "audio": torch.Tensor,
              "used_params": Dict[str, Any],
              "attempts": int,
              "flags": Dict[str, bool]
            }
        """
        self._ensure_model()

        generation_config = self.config.get("generation", {})
        max_retries: int = int(generation_config.get("max_retries", 0))

        base_min_p = float(additional_params.get("min_p", 0.05))
        base_top_p = float(additional_params.get("top_p", 0.95))
        base_temperature = float(temperature)

        static_params: Dict[str, Any] = {
            k: v
            for k, v in additional_params.items()
            if k not in {"min_p", "top_p"}
        }

        try:
            model_ref = self.model
            if (
                model_ref is not None
                and (not hasattr(model_ref, "conds") or getattr(model_ref, "conds", None) is None)
                and reference_audio_path
            ):
                logger.debug(
                    f"Preparing missing conditionals from reference audio: {Path(reference_audio_path).name}"
                )
                self.prepare_conditionals(reference_audio_path)
        except Exception as e:
            logger.error(f"Failed to prepare conditionals from reference audio: {e}")

        def _clamp(v: float, lo: float, hi: float) -> float:
            return max(lo, min(hi, v))

        def _parse_generation_logs(log_text: str) -> Dict[str, Any]:
            flags = {
                "long_tail": False,
                "alignment_repetition": False,
                "token_repetition": False,
                "eos_success": False,
                "forcing_eos": False,
            }
            if not log_text:
                return flags

            last_forcing_line: Optional[str] = None
            eos_ok: bool = False

            for line in log_text.split("\n"):
                if "EOS token detected" in line:
                    eos_ok = True
                if "forcing EOS token" in line:
                    last_forcing_line = line

            def _extract_bool(name: str, text: str) -> Optional[bool]:
                m = re.search(rf"{name}\\s*=\\s*(tensor\\((True|False)\\)|True|False)", text)
                if not m:
                    return None
                raw = m.group(1)
                return "True" in raw

            if last_forcing_line is not None:
                lt = _extract_bool("long_tail", last_forcing_line)
                ar = _extract_bool("alignment_repetition", last_forcing_line)
                tr = _extract_bool("token_repetition", last_forcing_line)
                flags["long_tail"] = bool(lt) if lt is not None else False
                flags["alignment_repetition"] = bool(ar) if ar is not None else False
                flags["token_repetition"] = bool(tr) if tr is not None else False
                flags["forcing_eos"] = True

            flags["eos_success"] = eos_ok
            return flags

        def _should_retry(flags: Dict[str, Any]) -> bool:
            return bool(flags.get("long_tail"))

        attempts = 0
        final_flags: Dict[str, Any] = {}
        audio_tensor: Optional[torch.Tensor] = None
        used_params: Dict[str, Any] = {}

        while True:
            min_p = _clamp(base_min_p + 0.05 * attempts, 0.0, 1.0)
            top_p = _clamp(base_top_p - 0.05 * attempts, 0.0, 1.0)
            temp = _clamp(base_temperature - 0.05 * attempts, 0.05, 2.0)
            attempt_seed = self._torch_seed_immediate_retry(base_seed, attempts)

            try:
                torch.manual_seed(attempt_seed)
            except Exception:
                pass

            gen_params = {
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "temperature": temp,
                "min_p": min_p,
                "top_p": top_p,
                **static_params,
            }

            if self.is_multilingual and language_id is not None:
                gen_params["language_id"] = language_id

            captured_messages: List[str] = []

            class _MemoryHandler(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:
                    try:
                        msg = record.getMessage()
                        if (
                            "chatterbox" in record.name
                            or "EOS token" in msg
                            or "forcing EOS token" in msg
                        ):
                            captured_messages.append(msg)
                    except Exception:
                        pass

            root_logger = logging.getLogger()
            mem_handler = _MemoryHandler(level=logging.DEBUG)
            root_logger.addHandler(mem_handler)
            try:
                gen_params = self._normalize_generation_params(gen_params)
            except Exception as e:
                logger.debug(f"Param normalization failed (non-fatal): {e}")

            try:
                audio_tensor = self.model.generate(text, **gen_params)
            except Exception as e:
                logger.error(f"Immediate retry attempt {attempts+1} failed with exception: {e}")
                final_flags = {"exception": True}
            finally:
                root_logger.removeHandler(mem_handler)

            if isinstance(audio_tensor, torch.Tensor):
                if audio_tensor.ndim == 2:
                    audio_tensor = audio_tensor.squeeze(0)
                audio_tensor = audio_tensor.to(self.device)

            log_text = "\n".join(captured_messages)
            flags = _parse_generation_logs(log_text)
            final_flags = flags

            if not _should_retry(flags):
                used_params = {
                    "exaggeration": exaggeration,
                    "cfg_weight": cfg_weight,
                    "temperature": temp,
                    "min_p": min_p,
                    "top_p": top_p,
                    "seed": attempt_seed,
                }
                if attempts > 0 and max_retries > 0:
                    logger.info("Retry successful")
                break

            if attempts >= max_retries:
                if max_retries > 0:
                    logger.warning(
                        "Exceeded immediate retries; returning last attempt despite warnings"
                    )
                used_params = {
                    "exaggeration": exaggeration,
                    "cfg_weight": cfg_weight,
                    "temperature": temp,
                    "min_p": min_p,
                    "top_p": top_p,
                    "seed": attempt_seed,
                }
                break

            attempts += 1

        return {
            "audio": audio_tensor if isinstance(audio_tensor, torch.Tensor) else torch.zeros(48000, device=self.device),
            "used_params": used_params,
            "attempts": attempts + 1,
            "flags": final_flags,
        }

    def generate_candidates(
        self,
        text: str,
        num_candidates: int = 3,
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        conservative_config: Optional[Dict[str, Any]] = None,
        tts_params: Optional[Dict[str, Any]] = None,
        reference_audio_path: Optional[str] = None,
        language_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
        **kwargs,
    ) -> List[AudioCandidate]:
        """
        Generates multiple audio candidates for the same text input with parameter variation.

        PARAMETER SEMANTICS (Chatterbox models):
        - exaggeration: Config value = MAX, ramps DOWN to (config - max_deviation)
        - cfg_weight: Config value = MIN, ramps UP to (config + max_deviation)
        - temperature: Config value = MIN, ramps UP to (config + max_deviation)

        For Qwen3: only temperature is ramped.
        Unsupported parameters for the current model are gracefully skipped.
        """
        self._ensure_model()
        candidates = []
        
        effective_seed = self.seed
        if speaker_id is not None:
            effective_seed = self.get_speaker_seed(speaker_id)
            if effective_seed != self.seed:
                logger.debug(f"Using speaker-specific seed {effective_seed} for speaker '{speaker_id}'")
        elif effective_seed == 0:
            logger.debug("Using random seed per generation (seed = 0)")

        if tts_params is None:
            generation_config = self.config.get("generation", {})
            tts_params = generation_config.get("tts_params", {})

        resolved_tts_params: Dict[str, Any] = tts_params or {}

        # Filter params to supported set for current model (warn once per unsupported key)
        filtered_tts_params = self._filter_params_for_model(resolved_tts_params)

        base_exaggeration = (
            exaggeration if exaggeration is not None
            else resolved_tts_params.get("exaggeration", 0.6)
        )
        base_cfg_weight = (
            cfg_weight if cfg_weight is not None
            else resolved_tts_params.get("cfg_weight", 0.7)
        )
        base_temperature = (
            temperature if temperature is not None
            else resolved_tts_params.get("temperature", 1.0)
        )

        exag_max_deviation = resolved_tts_params.get("exaggeration_max_deviation", 0.15)
        cfg_max_deviation = resolved_tts_params.get("cfg_weight_max_deviation", 0.15)
        temp_max_deviation = resolved_tts_params.get("temperature_max_deviation", 0.2)

        logger.info(f"Generating {num_candidates} diverse candidates for text (len={len(text)})")

        # Normalise once so mypy (and downstream code) can rely on a dict.
        cc: Dict[str, Any] = conservative_config or {}
        is_conservative_enabled = bool(cc.get("enabled", False))
        num_expressive = num_candidates - 1 if is_conservative_enabled else num_candidates

        # Special case: 1 candidate + conservative enabled = only conservative
        if num_candidates == 1 and is_conservative_enabled:
            logger.debug("Single candidate mode with conservative enabled - generating only conservative candidate")
            try:
                candidate_seed = self._torch_seed_per_candidate(effective_seed, 0, text)
                var_exaggeration = cc.get("exaggeration", 0.4)
                var_cfg_weight = cc.get("cfg_weight", 0.3)
                var_temperature = cc.get("temperature", 0.5)
                var_min_p = cc.get("min_p", resolved_tts_params.get("min_p", 0.1))
                var_top_p = cc.get("top_p", resolved_tts_params.get("top_p", 0.8))

                additional_params = {k: v for k, v in filtered_tts_params.items()
                                     if k not in {"exaggeration", "cfg_weight", "temperature", "min_p", "top_p"}}
                additional_params["min_p"] = var_min_p
                additional_params["top_p"] = var_top_p

                generation_params = {
                    "exaggeration": var_exaggeration,
                    "cfg_weight": var_cfg_weight,
                    "temperature": var_temperature,
                    "seed": candidate_seed,
                    "type": "CONSERVATIVE",
                    **additional_params,
                    **kwargs,
                }

                result = self._generate_single_with_immediate_retries(
                    text=text, exaggeration=var_exaggeration, cfg_weight=var_cfg_weight,
                    temperature=var_temperature, base_seed=candidate_seed,
                    language_id=language_id, additional_params=additional_params,
                    reference_audio_path=reference_audio_path,
                )
                audio = result["audio"]

                candidate = AudioCandidate(
                    chunk_idx=0, candidate_idx=0, audio_path=Path(), audio_tensor=audio,
                    generation_params={**generation_params, **result.get("used_params", {})},
                )
                candidates.append(candidate)
                self._emit_candidate_ready(candidate)
                logger.debug(f"Generated candidate 1/1: duration={audio.shape[-1]/24000:.2f}s")
            except Exception as e:
                logger.error(f"Failed to generate conservative candidate: {e}")
            return candidates

        for i in range(num_candidates):
            try:
                candidate_seed = self._torch_seed_per_candidate(effective_seed, i, text)
                is_conservative = is_conservative_enabled and (i + 1) == num_candidates

                if is_conservative:
                    var_exaggeration = cc.get("exaggeration", 0.4)
                    var_cfg_weight = cc.get("cfg_weight", 0.3)
                    var_temperature = cc.get("temperature", 0.5)
                    var_min_p = cc.get("min_p", resolved_tts_params.get("min_p", 0.1))
                    var_top_p = cc.get("top_p", resolved_tts_params.get("top_p", 0.8))
                    candidate_type = "CONSERVATIVE"
                else:
                    if num_expressive == 1 or i == 0:
                        var_exaggeration = base_exaggeration
                        var_cfg_weight = base_cfg_weight
                        var_temperature = base_temperature
                    else:
                        ramp_position = i / (num_expressive - 1)
                        var_exaggeration = base_exaggeration + (exag_max_deviation * ramp_position)
                        var_cfg_weight = base_cfg_weight + (cfg_max_deviation * ramp_position)
                        var_temperature = base_temperature + (temp_max_deviation * ramp_position)

                    var_min_p = resolved_tts_params.get("min_p", 0.05)
                    var_top_p = resolved_tts_params.get("top_p", 0.95)
                    candidate_type = "EXPRESSIVE"

                additional_params = {k: v for k, v in filtered_tts_params.items()
                                     if k not in {"exaggeration", "cfg_weight", "temperature", "min_p", "top_p"}}
                additional_params["min_p"] = var_min_p
                additional_params["top_p"] = var_top_p

                logger.info(
                    f"CANDIDATE {i+1} ({candidate_type}): exag={var_exaggeration:.2f}, "
                    f"cfg={var_cfg_weight:.2f}, temp={var_temperature:.2f}, "
                    f"min_p={var_min_p:.2f}, top_p={var_top_p:.2f}"
                )

                generation_params = {
                    "exaggeration": var_exaggeration,
                    "cfg_weight": var_cfg_weight,
                    "temperature": var_temperature,
                    "seed": candidate_seed,
                    "type": candidate_type,
                    **additional_params,
                    **kwargs,
                }

                result = self._generate_single_with_immediate_retries(
                    text=text, exaggeration=var_exaggeration, cfg_weight=var_cfg_weight,
                    temperature=var_temperature, base_seed=candidate_seed,
                    language_id=language_id, additional_params=additional_params,
                    reference_audio_path=reference_audio_path,
                )
                audio = result["audio"]

                candidate = AudioCandidate(
                    chunk_idx=0, candidate_idx=i, audio_path=Path(), audio_tensor=audio,
                    generation_params={**generation_params, **result.get("used_params", {})},
                )
                candidates.append(candidate)
                self._emit_candidate_ready(candidate)
                logger.debug(f"Generated: duration={audio.shape[-1]/24000:.2f}s, seed={candidate_seed}")

            except Exception as e:
                logger.error(f"Failed to generate candidate {i+1}/{num_candidates}: {e}")
                continue

        logger.debug(f"Successfully generated {len(candidates)}/{num_candidates} diverse candidates")
        return candidates

    def generate_specific_candidates(
        self,
        text: str,
        candidate_indices: List[int],
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        conservative_config: Optional[Dict[str, Any]] = None,
        tts_params: Optional[Dict[str, Any]] = None,
        total_candidates: int = 5,
        reference_audio_path: Optional[str] = None,
        language_id: Optional[str] = None,
        speaker_id: Optional[str] = None,
        **kwargs,
    ) -> List[AudioCandidate]:
        """
        Generates specific audio candidates for recovery or targeted testing.
        Unsupported parameters for the current model are gracefully skipped.
        """
        if not candidate_indices:
            logger.warning("No candidate indices provided for specific generation")
            return []

        logger.info(
            f"Generating candidates {candidate_indices} for text (len={len(text)}) "
            f"from total set of {total_candidates}\n"
        )

        candidates: List[AudioCandidate] = []
        
        effective_seed = self.seed
        if speaker_id is not None:
            effective_seed = self.get_speaker_seed(speaker_id)
        elif effective_seed == 0:
            logger.debug("Using random seed per generation (seed = 0)")

        if tts_params is None:
            generation_config = self.config.get("generation", {})
            tts_params = generation_config.get("tts_params", {})

        resolved_tts_params: Dict[str, Any] = tts_params or {}
        filtered_tts_params = self._filter_params_for_model(resolved_tts_params)

        base_exaggeration = exaggeration if exaggeration is not None else resolved_tts_params.get("exaggeration", 0.6)
        base_cfg_weight = cfg_weight if cfg_weight is not None else resolved_tts_params.get("cfg_weight", 0.7)
        base_temperature = temperature if temperature is not None else resolved_tts_params.get("temperature", 1.0)

        exag_max_deviation = resolved_tts_params.get("exaggeration_max_deviation", 0.15)
        cfg_max_deviation = resolved_tts_params.get("cfg_weight_max_deviation", 0.15)
        temp_max_deviation = resolved_tts_params.get("temperature_max_deviation", 0.2)

        cc2: Dict[str, Any] = conservative_config or {}
        is_conservative_enabled = bool(cc2.get("enabled", False))
        num_expressive = total_candidates - 1 if is_conservative_enabled else total_candidates

        for i in candidate_indices:
            try:
                candidate_seed = self._torch_seed_per_candidate(effective_seed, i, text)
                is_conservative = is_conservative_enabled and (i + 1) == total_candidates

                if is_conservative:
                    var_exaggeration = cc2.get("exaggeration", 0.4)
                    var_cfg_weight = cc2.get("cfg_weight", 0.3)
                    var_temperature = cc2.get("temperature", 0.5)
                    var_min_p = cc2.get("min_p", resolved_tts_params.get("min_p", 0.1))
                    var_top_p = cc2.get("top_p", resolved_tts_params.get("top_p", 0.8))
                    candidate_type = "CONSERVATIVE"
                else:
                    if num_expressive == 1 or i == 0:
                        var_exaggeration = base_exaggeration
                        var_cfg_weight = base_cfg_weight
                        var_temperature = base_temperature
                    else:
                        ramp_position = i / (num_expressive - 1)
                        var_exaggeration = base_exaggeration + (exag_max_deviation * ramp_position)
                        var_cfg_weight = base_cfg_weight + (cfg_max_deviation * ramp_position)
                        var_temperature = base_temperature + (temp_max_deviation * ramp_position)

                    var_min_p = resolved_tts_params.get("min_p", 0.05)
                    var_top_p = resolved_tts_params.get("top_p", 0.95)
                    candidate_type = "EXPRESSIVE"

                additional_params = {k: v for k, v in filtered_tts_params.items()
                                     if k not in {"exaggeration", "cfg_weight", "temperature", "min_p", "top_p"}}
                additional_params["min_p"] = var_min_p
                additional_params["top_p"] = var_top_p

                logger.debug(
                    f"Candidate {i+1} ({candidate_type}): exag={var_exaggeration:.2f}, "
                    f"cfg={var_cfg_weight:.2f}, temp={var_temperature:.2f}, "
                    f"min_p={var_min_p:.2f}, top_p={var_top_p:.2f}, seed={candidate_seed}"
                )

                generation_params = {
                    "exaggeration": var_exaggeration,
                    "cfg_weight": var_cfg_weight,
                    "temperature": var_temperature,
                    "seed": candidate_seed,
                    "type": candidate_type,
                    **additional_params,
                    **kwargs,
                }

                result = self._generate_single_with_immediate_retries(
                    text=text, exaggeration=var_exaggeration, cfg_weight=var_cfg_weight,
                    temperature=var_temperature, base_seed=candidate_seed,
                    language_id=language_id, additional_params=additional_params,
                    reference_audio_path=reference_audio_path,
                )
                audio = result["audio"]

                candidate = AudioCandidate(
                    chunk_idx=0, candidate_idx=i, audio_path=Path(), audio_tensor=audio,
                    generation_params={**generation_params, **result.get("used_params", {})},
                )
                candidates.append(candidate)
                self._emit_candidate_ready(candidate)
                logger.debug(f"Generated candidate {i+1}: duration={audio.shape[-1]/24000:.2f}s")

            except Exception as e:
                logger.error(f"Failed to generate candidate {i+1}/{total_candidates}: {e}")
                continue

        logger.debug(f"Successfully generated {len(candidates)} specific candidates")
        return candidates

    def get_current_params(self) -> Dict[str, Any]:
        """Get current TTS generation parameters."""
        generation_config = self.config.get("generation", {})
        tts_params = generation_config.get("tts_params", {})
        return {
            "device": self.device,
            "seed": self.seed,
            "model_type": self.model_type,
            "tts_params": tts_params,
            "model_loaded": self.model is not None,
        }

    # ------------------------------------------------------------------
    # Speaker system methods
    # ------------------------------------------------------------------

    def switch_speaker(self, speaker_id: str, config_manager=None):
        """
        Switch to a different speaker with the appropriate voice loading mechanism.

        - Chatterbox (standard/multilingual/turbo): calls prepare_conditionals()
        - Qwen3: builds and caches a voice clone prompt via _prepare_qwen3_voice_clone_prompt()
        - VibeVoice: loads and caches reference waveform for voice cloning

        Args:
            speaker_id: Target speaker ID
            config_manager: FileManager or ConfigManager for file access
        """
        actual_speaker_id = self._resolve_speaker_id(speaker_id, config_manager)
        
        if self.current_speaker_id == actual_speaker_id:
            logger.debug(f"Speaker '{actual_speaker_id}' already active, skipping switch")
            return

        speaker_config = self._get_speaker_config(actual_speaker_id)
        if not speaker_config:
            logger.error(f"Resolved speaker '{actual_speaker_id}' not found in configuration")
            return

        reference_audio = speaker_config.get("reference_audio")
        if not (reference_audio and config_manager):
            logger.warning(f"No reference_audio or config_manager for speaker '{actual_speaker_id}'")
            return

        try:
            audio_path = config_manager.get_reference_audio_for_speaker(actual_speaker_id)
            logger.info(f"🎭 Switching to speaker '{actual_speaker_id}' with voice: {audio_path.name}")

            if self.is_qwen3:
                ref_text = self._load_qwen3_ref_text(actual_speaker_id, audio_path, speaker_config)
                self._prepare_qwen3_voice_clone_prompt(actual_speaker_id, audio_path, ref_text)
                if actual_speaker_id in self._qwen3_voice_prompts:
                    self.current_speaker_id = actual_speaker_id
                    logger.debug(f"✅ Qwen3 voice prompt ready for speaker '{actual_speaker_id}'")
                else:
                    logger.error(f"❌ Failed to build Qwen3 voice prompt for speaker '{actual_speaker_id}'")
            elif self.is_vibevoice:
                voice_speed_factor = float(
                    speaker_config.get("tts_params", {}).get("voice_speed_factor", 1.0)
                )
                self._prepare_vibevoice_reference_audio(
                    actual_speaker_id, audio_path, voice_speed_factor
                )
                if actual_speaker_id in self._vibevoice_reference_audio:
                    self.current_speaker_id = actual_speaker_id
                    logger.debug(
                        f"✅ VibeVoice reference audio ready for speaker '{actual_speaker_id}'"
                    )
                else:
                    logger.error(
                        f"❌ Failed to prepare VibeVoice reference audio for speaker '{actual_speaker_id}'"
                    )
            else:
                self.prepare_conditionals(str(audio_path))
                model_ref = self.model
                if model_ref is not None and getattr(model_ref, "conds", None) is not None:
                    logger.debug(f"✅ Conditionals successfully loaded for speaker '{actual_speaker_id}'")
                    self.current_speaker_id = actual_speaker_id
                else:
                    logger.error(f"❌ Failed to load conditionals for speaker '{actual_speaker_id}'")

        except Exception as e:
            logger.error(f"Failed to switch to speaker '{actual_speaker_id}': {e}")

    def _resolve_speaker_id(self, speaker_id: str, config_manager=None) -> str:
        """Resolve speaker ID with fallback logic."""
        for speaker in self.speakers_config:
            if speaker.get("id") == speaker_id:
                return speaker_id

        if config_manager and hasattr(config_manager, "get_default_speaker_id"):
            try:
                fallback_speaker_id = config_manager.get_default_speaker_id()
                if speaker_id != fallback_speaker_id:
                    logger.warning(
                        f"Speaker '{speaker_id}' not found, using default speaker '{fallback_speaker_id}'"
                    )
                return fallback_speaker_id
            except Exception as e:
                logger.debug(f"Could not get default speaker from config_manager: {e}")

        if self.speakers_config:
            fallback_speaker_id = self.speakers_config[0].get("id", "default")
            logger.warning(f"Using first speaker as fallback: '{fallback_speaker_id}'")
            return fallback_speaker_id

        return "default"

    def generate_candidates_with_speaker(
        self, text: str, speaker_id: str, num_candidates: int, config_manager
    ) -> List[AudioCandidate]:
        """
        Generate candidates for given text using the specified speaker.

        Speaker switching (different reference audio, tts_params, prosody) works for all
        model types.  Language restrictions (standard/turbo are EN-only) are already
        enforced as soft warnings in generation_handler._validate_languages_for_model()
        and are NOT enforced here.

        For Qwen3, dispatches to _generate_qwen3_candidates().
        """
        self._ensure_model()
        speaker_config = self._get_speaker_config(speaker_id)
        reference_audio_path = config_manager.get_reference_audio_for_speaker(speaker_id)

        try:
            self.switch_speaker(speaker_id, config_manager)
        except Exception as e:
            logger.error(f"Failed to switch to speaker '{speaker_id}': {e}")

        # -- Language resolution --
        language_id = speaker_config.get("language")
        if not language_id:
            default_speaker_id = (
                config_manager.get_default_speaker_id()
                if hasattr(config_manager, "get_default_speaker_id")
                else speaker_id
            )
            default_speaker_config = self._get_speaker_config(default_speaker_id)
            language_id = default_speaker_config.get("language")
            if not language_id:
                try:
                    language_id = self.config.get("generation", {}).get("default_language")
                except Exception:
                    language_id = None
            if not language_id:
                language_id = "en"
            logger.warning(
                f"No language defined for speaker '{speaker_id}', using fallback: {language_id}"
            )

        # -- Dispatch to model-specific paths --
        if self.is_qwen3:
            return self._generate_qwen3_candidates(
                text=text,
                speaker_id=speaker_id,
                speaker_config=speaker_config,
                num_candidates=num_candidates,
                language_id=language_id,
            )
        if self.is_vibevoice:
            return self._generate_vibevoice_candidates(
                text=text,
                speaker_id=speaker_id,
                speaker_config=speaker_config,
                num_candidates=num_candidates,
                language_id=language_id,
            )

        # -- Chatterbox path --
        base_params = speaker_config["tts_params"]
        conservative_params = speaker_config.get("conservative_candidate", {})
        conservative_enabled = conservative_params.get("enabled", False)

        # Filter params to supported set
        filtered_base = self._filter_params_for_model(base_params)
        filtered_conservative = (
            self._filter_params_for_model(conservative_params)
            if conservative_enabled
            else {}
        )

        num_expressive = num_candidates - 1 if conservative_enabled else num_candidates
        candidates: List[AudioCandidate] = []

        for i in range(num_candidates):
            if i == num_candidates - 1 and conservative_enabled:
                params = dict(filtered_base)
                params.update(filtered_conservative)
                is_conservative_this = True
            elif i == 0:
                params = dict(filtered_base)
                is_conservative_this = False
            else:
                params = self._calculate_ramped_params(filtered_base, i, num_expressive - 1)
                is_conservative_this = False

            effective_params = dict(filtered_base)
            effective_params.update(params)

            core_exaggeration = effective_params.get("exaggeration", 0.6)
            core_cfg_weight = effective_params.get("cfg_weight", 0.7)
            core_temperature = effective_params.get("temperature", 1.0)

            additional_params = {
                k: v for k, v in effective_params.items()
                if k not in {
                    "exaggeration", "cfg_weight", "temperature",
                    "exaggeration_max_deviation", "cfg_weight_max_deviation",
                    "temperature_max_deviation", "enabled",
                }
            }

            try:
                log_min_p = additional_params.get("min_p")
                log_top_p = additional_params.get("top_p")
                log_rep = additional_params.get("repetition_penalty")
                logger.info(
                    f"CAND {i+1}/{num_candidates} (lang={language_id}) "
                    f"({'CONS' if is_conservative_this else 'EXP'}): "
                    f"exag={core_exaggeration:.2f}, cfg={core_cfg_weight:.2f}, temp={core_temperature:.2f}, "
                    f"min_p={log_min_p if log_min_p is not None else '-'}, "
                    f"top_p={log_top_p if log_top_p is not None else '-'}, "
                    f"rep_pen={log_rep if log_rep is not None else '-'}"
                )
            except Exception:
                pass

            result = self._generate_single_with_immediate_retries(
                text=text,
                exaggeration=core_exaggeration,
                cfg_weight=core_cfg_weight,
                temperature=core_temperature,
                base_seed=self._torch_seed_per_candidate(
                    self.get_speaker_seed(speaker_id), i, text
                ),
                language_id=language_id,
                additional_params=additional_params,
                reference_audio_path=str(reference_audio_path),
            )
            audio = result["audio"]

            candidate = AudioCandidate(
                chunk_idx=0,
                candidate_idx=i,
                audio_path=Path(),
                audio_tensor=audio,
                generation_params={
                    **params,
                    "language_id": language_id,
                    **result.get("used_params", {}),
                },
            )
            candidates.append(candidate)
            self._emit_candidate_ready(candidate)

        return candidates

    def _get_speaker_config(self, speaker_id: str) -> Dict[str, Any]:
        """Get configuration for a specific speaker."""
        actual_speaker_id = self._resolve_speaker_id(speaker_id)
        
        for speaker in self.speakers_config:
            if speaker.get("id") == actual_speaker_id:
                return speaker

        logger.error(f"Could not find configuration for resolved speaker '{actual_speaker_id}'")
        return {}
