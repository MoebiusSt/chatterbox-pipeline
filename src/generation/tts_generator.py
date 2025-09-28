import logging
import warnings
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional

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

logger = logging.getLogger(__name__)


class TTSGenerator:
    """
    Simplified TTS Generator with direct model access.
    Uses direct model access without thread safety for sequential execution.
    """

    def __init__(self, config: Dict[str, Any], device: str = "auto", seed: int = 12345):
        """
        Initializes the TTSGenerator.

        Args:
            config: Configuration dictionary with generation settings.
            device: The device to run inference on (cuda, mps, cpu).
            seed: Random seed for reproducibility.
        """
        self.config = config
        self.device = device if device != "auto" else self._detect_device()
        self.seed = seed

        # Set task-global seed once for consistent generation
        torch.manual_seed(seed)
        logger.info("")
        logger.info(f"🎲 Task-global seed set to {seed}")
        logger.info("")
        # Get model type and default language from config
        generation_config = config.get("generation", {})
        self.model_type = generation_config.get("model_type", "standard")
        
        # Use direct model access with model type
        self.model = ChatterboxModelCache.get_model(self.device, self.model_type)
        
        # Check if we actually got a multilingual model or fallback standard model
        self.is_multilingual = (
            self.model_type == "multilingual" and 
            hasattr(self.model, '__class__') and 
            'Multilingual' in self.model.__class__.__name__
        )

        # Speaker system attributes
        self.current_speaker_id = "default"
        self.speakers_config = config.get("generation", {}).get("speakers", [])

        if self.is_multilingual:
            model_name = "ChatterboxMultilingualTTS"
        elif self.model_type == "multilingual":
            model_name = "ChatterboxTTS (multilingual fallback)"
        else:
            model_name = "ChatterboxTTS"
        
        logger.debug(
            f"TTSGenerator initialized on device: {self.device} with {len(self.speakers_config)} speakers, using {model_name}"
        )

    def _detect_device(self) -> str:
        """Detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def prepare_conditionals(self, wav_fpath: str):
        """
        Prepares model conditionals for voice cloning.

        Args:
            wav_fpath: Path to reference audio file
        """
        if self.model is None:
            logger.warning("🚨 No model loaded - cannot prepare conditionals")
            return

        logger.debug(f"🔄 Preparing conditionals for {Path(wav_fpath).name}")

        try:
            # Direct model access - no thread safety needed
            self.model.prepare_conditionals(wav_fpath=wav_fpath)
            logger.debug("✅ Conditionals prepared")
        except Exception as e:
            logger.error(f"🚨 Error preparing conditionals: {e}")
            raise

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
        Generate single audio using direct model access.

        Args:
            text: Text to synthesize
            exaggeration: Voice exaggeration parameter (0.0-1.0)
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature for diversity
            reference_audio_path: Path to reference audio (only used if no conditionals loaded)
            **kwargs: Additional arguments passed to the TTS model

        Returns:
            Generated audio tensor (1D)
        """
        # Validate inputs
        if not text or not text.strip():
            logger.warning("Empty text provided for generation")
            return torch.zeros(1000, device=self.device)

        if self.model is None:
            logger.warning("🚨 No model loaded - generating silence")
            return torch.zeros(48000, device=self.device)

        logger.debug("Starting TTS generation")

        # Conditionals should always be loaded through speaker system
        # Removed redundant safety checks to avoid unnecessary prepare_conditionals calls
        if not hasattr(self.model, "conds") or self.model.conds is None:
            logger.warning("🚨 No conditionals loaded - this should not happen with speaker system")
            return torch.zeros(48000, device=self.device)
        
        logger.debug("Using loaded conditionals (speaker-specific)")

        # Suppress PyTorch and Transformers warnings during model generation
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*torch.backends.cuda.sdp_kernel.*",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore", message=".*LlamaModel is using LlamaSdpaAttention.*"
            )
            warnings.filterwarnings(
                "ignore", 
                message=".*torch.nn.functional.scaled_dot_product_attention.*does not support.*output_attentions=True.*"
            )
            warnings.filterwarnings(
                "ignore", 
                message=".*Falling back to the manual attention implementation.*"
            )
            warnings.filterwarnings(
                "ignore", 
                message=".*specifying the manual implementation will be required from Transformers version v5.0.0.*"
            )
            warnings.filterwarnings(
                "ignore", 
                message=".*This warning can be removed using the argument.*attn_implementation.*eager.*"
            )
            warnings.filterwarnings(
                "ignore", message=".*does not support `output_attentions=True`.*"
            )
            warnings.filterwarnings(
                "ignore", 
                message=".*return_dict_in_generate.*is NOT set to.*True.*but.*output_attentions.*is.*"
            )
            warnings.filterwarnings(
                "ignore",
                message=".*past_key_values.*tuple of tuples.*",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*detected that you are passing.*past_key_values.*as a tuple of tuples.*",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message=".*convert your cache or use an appropriate.*Cache.*class.*",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore", message=".*attn_implementation.*", category=FutureWarning
            )

            logger.debug(
                f"Generating audio for text (len={len(text)}): '{text[:50]}...'"
            )

            # Generate audio using the model directly
            generate_params = {
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "temperature": temperature,
                **kwargs,
            }
            
            # Add language_id for multilingual model (only if actually multilingual)
            if self.is_multilingual and language_id is not None:
                generate_params["language_id"] = language_id
                logger.debug(f"Using language_id: {language_id} for multilingual model")
            elif self.model_type == "multilingual" and not self.is_multilingual:
                logger.debug("Multilingual model requested but not available, using standard model without language_id")
            
            # Normalize parameters to what the model accepts
            try:
                generate_params = self._normalize_generation_params(generate_params)
            except Exception as e:
                logger.debug(f"Param normalization failed (non-fatal): {e}")

            audio = self.model.generate(text, **generate_params)

        # ChatterboxTTS returns 1D tensor - ensure consistency
        if audio.ndim == 2:
            audio = audio.squeeze(0)  # Remove batch dimension if present
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
        Generate audio with immediate retries if Chatterbox reports artifact-related warnings.

        Strategy per retry (attempt >= 1):
        - Change seed deterministically based on base_seed and attempt
        - Increase min_p by +0.05 per attempt (clamped to [0.0, 1.0])
        - Decrease top_p by -0.05 per attempt (clamped to [0.0, 1.0])
        - Decrease temperature by -0.05 per attempt (>= 0.05)

        Retries stop early if logs contain a clean EOS without repetition/long_tail flags.

        Returns:
            {
              "audio": torch.Tensor,
              "used_params": Dict[str, Any],  # actual params used on final attempt
              "attempts": int,                # number of attempts performed
              "flags": Dict[str, bool]        # detected flags from logs on final attempt
            }
        """
        generation_config = self.config.get("generation", {})
        max_retries: int = int(generation_config.get("max_retries", 0))

        # Extract and prepare adjustable parameters
        base_min_p = float(additional_params.get("min_p", 0.05))
        base_top_p = float(additional_params.get("top_p", 0.95))
        base_temperature = float(temperature)

        # Non-adjustable passthrough params (copy to avoid side effects)
        static_params: Dict[str, Any] = {
            k: v
            for k, v in additional_params.items()
            if k not in {"min_p", "top_p"}
        }

        # Ensure speaker conditionals are loaded; if missing and a reference audio is provided,
        # prepare conditionals now as a safety net for speaker-aware generation paths.
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
            # Default flags
            flags = {
                "long_tail": False,
                "alignment_repetition": False,
                "token_repetition": False,
                "eos_success": False,
                "forcing_eos": False,
            }

            if not log_text:
                return flags

            # Evaluate only the LAST occurrence of a "forcing EOS token" line within this attempt.
            # Earlier we searched the entire log blob which could contain both True and False,
            # and any True would force a retry. Here we restrict to the last forcing line.
            last_forcing_line: Optional[str] = None
            eos_ok: bool = False

            for line in log_text.split("\n"):
                if "EOS token detected" in line:
                    eos_ok = True
                if "forcing EOS token" in line:
                    last_forcing_line = line

            def _extract_bool(name: str, text: str) -> Optional[bool]:
                # Matches: name=tensor(True|False) or name=True|False
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
            # Retry only if long_tail is True (removed other conditions from retry logic)
            return bool(flags.get("long_tail"))

        attempts = 0
        final_flags: Dict[str, Any] = {}
        audio_tensor: Optional[torch.Tensor] = None
        used_params: Dict[str, Any] = {}

        while True:
            # Compute attempt-adjusted parameters
            min_p = _clamp(base_min_p + 0.05 * attempts, 0.0, 1.0)
            top_p = _clamp(base_top_p - 0.05 * attempts, 0.0, 1.0)
            temp = _clamp(base_temperature - 0.05 * attempts, 0.05, 2.0)
            attempt_seed = int(base_seed + attempts * 9973)

            # Re-seed for this attempt
            try:
                torch.manual_seed(attempt_seed)
            except Exception:
                pass

            # Build parameter set for this attempt
            gen_params = {
                "exaggeration": exaggeration,
                "cfg_weight": cfg_weight,
                "temperature": temp,
                "min_p": min_p,
                "top_p": top_p,
                **static_params,
            }

            # Add language_id only for true multilingual
            if self.is_multilingual and language_id is not None:
                gen_params["language_id"] = language_id

            # Capture logs during generation
            captured_messages: List[str] = []

            class _MemoryHandler(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
                    try:
                        msg = record.getMessage()
                        # Only capture chatterbox-related or EOS messages
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
            # Normalize parameters to what the model accepts
            try:
                gen_params = self._normalize_generation_params(gen_params)
            except Exception as e:
                logger.debug(f"Param normalization failed (non-fatal): {e}")

            try:
                audio_tensor = self.model.generate(text, **gen_params)
            except Exception as e:
                # On hard error, decide to retry if possible
                logger.error(f"Immediate retry attempt {attempts+1} failed with exception: {e}")
                final_flags = {"exception": True}
            finally:
                root_logger.removeHandler(mem_handler)

            # Normalize audio tensor shape/device if we have a result
            if isinstance(audio_tensor, torch.Tensor):
                if audio_tensor.ndim == 2:
                    audio_tensor = audio_tensor.squeeze(0)
                audio_tensor = audio_tensor.to(self.device)

            # Analyze logs
            log_text = "\n".join(captured_messages)
            flags = _parse_generation_logs(log_text)
            final_flags = flags

            # Decide to stop or retry
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
                    logger.info("Retry succesful")
                break

            if attempts >= max_retries:
                # Give up and return the latest attempt (even if flagged)
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
        **kwargs,
    ) -> List[AudioCandidate]:
        """
        Generates multiple audio candidates for the same text input with parameter variation.

        PARAMETER SEMANTICS (as specified by user requirements):
        - exaggeration: Config value = MAX, ramps DOWN to (config - max_deviation)
        - cfg_weight: Config value = MIN, ramps UP to (config + max_deviation)
        - temperature: Config value = MIN, ramps UP to (config + max_deviation)

        CANDIDATE LOGIC:
        - num_candidates=1 + conservative_enabled=true  → 1 conservative candidate
        - num_candidates=1 + conservative_enabled=false → 1 expressive candidate (exact config)
        - num_candidates>1 + conservative_enabled=true  → N-1 expressive + 1 conservative (last)
        - num_candidates>1 + conservative_enabled=false → N expressive (1 exact + N-1 ramped)
        """
        candidates = []

        # Get parameters from config if not provided
        if tts_params is None:
            generation_config = self.config.get("generation", {})
            tts_params = generation_config.get("tts_params", {})

        # Ensure non-None dict for downstream access
        resolved_tts_params: Dict[str, Any] = tts_params or {}

        # Use config values as defaults - these are now the starting points for ramping
        base_exaggeration = (
            exaggeration
            if exaggeration is not None
            else resolved_tts_params.get("exaggeration", 0.6)
        )  # MAX value
        base_cfg_weight = (
            cfg_weight if cfg_weight is not None else resolved_tts_params.get("cfg_weight", 0.7)
        )  # MIN value
        base_temperature = (
            temperature
            if temperature is not None
            else resolved_tts_params.get("temperature", 1.0)
        )  # MIN value

        # Get deviation ranges from config
        exag_max_deviation = resolved_tts_params.get("exaggeration_max_deviation", 0.15)
        cfg_max_deviation = resolved_tts_params.get("cfg_weight_max_deviation", 0.15)
        temp_max_deviation = resolved_tts_params.get("temperature_max_deviation", 0.2)

        logger.info(
            f"Generating {num_candidates} diverse candidates for text (len={len(text)})"
        )
        logger.debug(
            f"Expressive ranges: exag=[{base_exaggeration-exag_max_deviation:.2f}, {base_exaggeration:.2f}], "
            f"cfg=[{base_cfg_weight:.2f}, {base_cfg_weight+cfg_max_deviation:.2f}], "
            f"temp=[{base_temperature:.2f}, {base_temperature+temp_max_deviation:.2f}]"
        )

        # Special case: 1 candidate + conservative enabled = only conservative
        if (
            num_candidates == 1
            and conservative_config
            and conservative_config.get("enabled", False)
        ):
            logger.debug(
                "Single candidate mode with conservative enabled - generating only conservative candidate"
            )
            try:
                candidate_seed = self.seed + hash(text) % 10000

                # Use conservative parameters
                var_exaggeration = conservative_config.get("exaggeration", 0.4)
                var_cfg_weight = conservative_config.get("cfg_weight", 0.3)
                var_temperature = conservative_config.get("temperature", 0.5)
                # Use conservative min_p and top_p with fallback to regular tts_params
                var_min_p = conservative_config.get("min_p", resolved_tts_params.get("min_p", 0.1))
                var_top_p = conservative_config.get("top_p", resolved_tts_params.get("top_p", 0.8))
                candidate_type = "CONSERVATIVE"

                # Debug: Log tts_params before extracting additional_params
                logger.info(f"🔍 tts_params for candidate 1: {resolved_tts_params}")
                # Extract additional TTS parameters from tts_params (excluding the ones we handle explicitly)
                additional_params = {
                    k: v
                    for k, v in resolved_tts_params.items()
                    if k
                    not in [
                        "exaggeration",
                        "cfg_weight",
                        "temperature",
                        "min_p",
                        "top_p",
                        "exaggeration_max_deviation",
                        "cfg_weight_max_deviation",
                        "temperature_max_deviation",
                    ]
                }

                # Add the candidate-specific min_p and top_p to additional_params
                additional_params["min_p"] = var_min_p
                additional_params["top_p"] = var_top_p

                generation_params = {
                    "exaggeration": var_exaggeration,
                    "cfg_weight": var_cfg_weight,
                    "temperature": var_temperature,
                    "seed": candidate_seed,
                    "type": candidate_type,
                    **additional_params,  # Include repetition_penalty and other TTS params
                    **kwargs,
                }

                logger.debug(
                    f"Candidate 1 ({candidate_type}): exag={var_exaggeration:.2f}, cfg={var_cfg_weight:.2f}, temp={var_temperature:.2f}, min_p={var_min_p:.2f}, top_p={var_top_p:.2f}, seed={candidate_seed}"
                )

                result = self._generate_single_with_immediate_retries(
                    text=text,
                    exaggeration=var_exaggeration,
                    cfg_weight=var_cfg_weight,
                    temperature=var_temperature,
                    base_seed=candidate_seed,
                    language_id=language_id,
                    additional_params=additional_params,
                    reference_audio_path=reference_audio_path,
                )
                audio = result["audio"]

                candidate = AudioCandidate(
                    chunk_idx=0,  # Will be set by caller
                    candidate_idx=0,
                    audio_path=Path(),  # Will be set when saving
                    audio_tensor=audio,
                    generation_params={
                        **generation_params,
                        **{k: v for k, v in result.get("used_params", {}).items()},
                    },
                )

                candidates.append(candidate)
                logger.debug(
                    f"Generated candidate 1/{num_candidates}: duration={audio.shape[-1]/24000:.2f}s\n"
                )

            except Exception as e:
                logger.error(f"Failed to generate conservative candidate: {e}")

            logger.debug(
                f"Successfully generated {len(candidates)}/1 conservative candidate"
            )
            return candidates

        # Multi-candidate mode or single expressive mode
        is_conservative_enabled = conservative_config and conservative_config.get(
            "enabled", False
        )
        num_expressive = (
            num_candidates - 1 if is_conservative_enabled else num_candidates
        )

        for i in range(num_candidates):
            try:
                # Set unique seed for this candidate
                candidate_seed = self.seed + (i * 1000) + hash(text) % 10000

                # Check if this should be a conservative candidate (always last)
                is_conservative = is_conservative_enabled and (i + 1) == num_candidates

                if is_conservative:
                    # Use conservative parameters for guaranteed correctness
                    logger.debug(
                        f"Applying conservative parameters for candidate {i+1}"
                    )
                    if conservative_config is None:
                        raise RuntimeError(
                            "Conservative config is None but conservative mode is enabled"
                        )
                    var_exaggeration = conservative_config.get("exaggeration", 0.4)
                    var_cfg_weight = conservative_config.get("cfg_weight", 0.3)
                    var_temperature = conservative_config.get("temperature", 0.5)
                    # Use conservative min_p and top_p with fallback to regular tts_params
                    var_min_p = conservative_config.get("min_p", resolved_tts_params.get("min_p", 0.1))
                    var_top_p = conservative_config.get("top_p", resolved_tts_params.get("top_p", 0.8))
                    candidate_type = "CONSERVATIVE"
                else:
                    # Expressive candidate logic
                    if num_expressive == 1:
                        # Only one expressive candidate: use exact config values
                        var_exaggeration = base_exaggeration
                        var_cfg_weight = base_cfg_weight
                        var_temperature = base_temperature
                    elif i == 0:
                        # First expressive candidate: always use exact config values
                        var_exaggeration = base_exaggeration
                        var_cfg_weight = base_cfg_weight
                        var_temperature = base_temperature
                    else:
                        # Subsequent expressive candidates: apply RAMP strategy
                        # Calculate ramp position: candidate 2 = 0.25, candidate 3 = 0.5, ..., last = 1.0
                        ramp_position = i / (
                            num_expressive - 1
                        )  # i=1 → 1/4=0.25, i=2 → 2/4=0.5, etc.

                        # Apply user-specified ramp directions:
                        # exaggeration: RAMP-DOWN from MAX (config) to MIN (config - deviation)
                        var_exaggeration = base_exaggeration - (
                            exag_max_deviation * ramp_position
                        )

                        # cfg_weight: RAMP-UP from MIN (config) to MAX (config + deviation)
                        var_cfg_weight = base_cfg_weight + (
                            cfg_max_deviation * ramp_position
                        )

                        # temperature: RAMP-UP from MIN (config) to MAX (config + deviation)
                        var_temperature = base_temperature + (
                            temp_max_deviation * ramp_position
                        )

                    # Use regular min_p and top_p for expressive candidates
                    var_min_p = resolved_tts_params.get("min_p", 0.05)
                    var_top_p = resolved_tts_params.get("top_p", 0.95)
                    candidate_type = "EXPRESSIVE"

                # Extract additional TTS parameters from tts_params (excluding the ones we handle explicitly)
                additional_params = {
                    k: v
                    for k, v in resolved_tts_params.items()
                    if k
                    not in [
                        "exaggeration",
                        "cfg_weight",
                        "temperature",
                        "min_p",
                        "top_p",
                        "exaggeration_max_deviation",
                        "cfg_weight_max_deviation",
                        "temperature_max_deviation",
                    ]
                }

                # Add the candidate-specific min_p and top_p to additional_params
                additional_params["min_p"] = var_min_p
                additional_params["top_p"] = var_top_p

                # Debug: Log tts_params with all parameters
                logger.info(
                    f"CANDIDATE {i+1} ({candidate_type}): exag={var_exaggeration:.2f}, cfg={var_cfg_weight:.2f}, temp={var_temperature:.2f}, min_p={var_min_p:.2f}, top_p={var_top_p:.2f}"
                )

                generation_params = {
                    "exaggeration": var_exaggeration,
                    "cfg_weight": var_cfg_weight,
                    "temperature": var_temperature,
                    "seed": candidate_seed,
                    "type": candidate_type,
                    **additional_params,  # Include repetition_penalty and other TTS params
                    **kwargs,
                }

                result = self._generate_single_with_immediate_retries(
                    text=text,
                    exaggeration=var_exaggeration,
                    cfg_weight=var_cfg_weight,
                    temperature=var_temperature,
                    base_seed=candidate_seed,
                    language_id=language_id,
                    additional_params=additional_params,
                    reference_audio_path=reference_audio_path,
                )
                audio = result["audio"]

                candidate = AudioCandidate(
                    chunk_idx=0,  # Will be set by caller
                    candidate_idx=i,
                    audio_path=Path(),  # Will be set when saving
                    audio_tensor=audio,
                    generation_params={
                        **generation_params,
                        **{k: v for k, v in result.get("used_params", {}).items()},
                    },
                )

                candidates.append(candidate)
                # NOTE: Using ChatterboxTTS native sample rate (24kHz) for duration calculation
                logger.debug(
                    f"Generated: duration={audio.shape[-1]/24000:.2f}s, seed={candidate_seed}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to generate candidate {i+1}/{num_candidates}: {e}"
                )
                # Continue with remaining candidates
                continue

        logger.debug(
            f"Successfully generated {len(candidates)}/{num_candidates} diverse candidates"
        )
        return candidates

    def _normalize_generation_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize generation kwargs to align with the underlying model.generate signature.

        Behavior:
        - If model.generate has **kwargs (VAR_KEYWORD), do NOT filter unknown keys (only alias-remap).
        - If model.generate does NOT have **kwargs, filter to accepted names to avoid TypeError.
        - Attempt common alias remaps (cfg_weight, temperature, exaggeration, top_p, min_p, language_id, repetition_penalty).
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
                # keep original; it may be filtered below

        # Attempt common alias remaps
        remap("cfg_weight", ["cfg_weight", "cfg", "guidance_scale", "classifier_free_guidance_weight"]) 
        remap("temperature", ["temperature", "temp"]) 
        remap("exaggeration", ["exaggeration", "style_exaggeration", "expressiveness"]) 
        remap("top_p", ["top_p", "nucleus_p"]) 
        remap("min_p", ["min_p", "p_min"]) 
        remap("language_id", ["language_id", "language", "lang"]) 
        remap("repetition_penalty", ["repetition_penalty", "repeat_penalty"]) 

        # If model accepts **kwargs, retain all keys (no filtering) to allow the model to consume kwargs.
        if has_var_kw:
            try:
                logger.debug(
                    f"Model.generate accepts **kwargs; passing params without filtering. Keys: {sorted(normalized.keys())}"
                )
            except Exception:
                pass
            return normalized

        # Otherwise, filter to accepted names to avoid TypeError
        filtered = {k: v for k, v in normalized.items() if k in accepted_params}
        dropped = [k for k in normalized.keys() if k not in accepted_params]
        if dropped:
            logger.debug(f"Dropping unsupported TTS params for model.generate (no **kwargs): {dropped}")
        try:
            logger.debug(f"Params passed to model.generate: {sorted(filtered.keys())}")
        except Exception:
            pass
        return filtered

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
        **kwargs,
    ) -> List[AudioCandidate]:
        """
        Generates specific audio candidates for the same text input with parameter variation.
        This method is designed to regenerate specific candidates for recovery or targeted testing.

        PARAMETER SEMANTICS (as specified by user requirements):
        - exaggeration: Config value = MAX, ramps DOWN to (config - max_deviation)
        - cfg_weight: Config value = MIN, ramps UP to (config + max_deviation)
        - temperature: Config value = MIN, ramps UP to (config + max_deviation)

        CANDIDATE LOGIC (for the specified candidates):
        - The provided `candidate_indices` will be generated using parameters determined
          by their position in the full `num_candidates` range, respecting the ramping logic.
        - If a conservative candidate is requested, its parameters will be used.
        """
        if not candidate_indices:
            logger.warning("No candidate indices provided for specific generation")
            return []

        logger.info(
            f"Generating candidates {candidate_indices} for text (len={len(text)}) "
            f"from total set of {total_candidates}\n"
        )

        candidates: List[AudioCandidate] = []

        # Get parameters from config if not provided
        if tts_params is None:
            generation_config = self.config.get("generation", {})
            tts_params = generation_config.get("tts_params", {})

        # Ensure non-None dict for downstream access
        resolved_tts_params: Dict[str, Any] = tts_params or {}

        # Use config values as defaults
        base_exaggeration = (
            exaggeration
            if exaggeration is not None
            else resolved_tts_params.get("exaggeration", 0.6)
        )
        base_cfg_weight = (
            cfg_weight if cfg_weight is not None else resolved_tts_params.get("cfg_weight", 0.7)
        )
        base_temperature = (
            temperature
            if temperature is not None
            else resolved_tts_params.get("temperature", 1.0)
        )

        # Get deviation ranges from config
        exag_max_deviation = resolved_tts_params.get("exaggeration_max_deviation", 0.15)
        cfg_max_deviation = resolved_tts_params.get("cfg_weight_max_deviation", 0.15)
        temp_max_deviation = resolved_tts_params.get("temperature_max_deviation", 0.2)

        is_conservative_enabled = conservative_config and conservative_config.get(
            "enabled", False
        )
        num_expressive = (
            total_candidates - 1 if is_conservative_enabled else total_candidates
        )

        for i in candidate_indices:
            try:
                # Set unique seed for this candidate
                candidate_seed = self.seed + (i * 1000) + hash(text) % 10000

                # Check if this should be a conservative candidate (always last)
                is_conservative = (
                    is_conservative_enabled and (i + 1) == total_candidates
                )

                if is_conservative:
                    # Use conservative parameters
                    logger.debug(
                        f"Applying conservative parameters for candidate {i+1}"
                    )
                    if conservative_config is None:
                        raise RuntimeError(
                            "Conservative config is None but conservative mode is enabled"
                        )
                    var_exaggeration = conservative_config.get("exaggeration", 0.4)
                    var_cfg_weight = conservative_config.get("cfg_weight", 0.3)
                    var_temperature = conservative_config.get("temperature", 0.5)
                    # Use conservative min_p and top_p with fallback to regular tts_params
                    var_min_p = conservative_config.get("min_p", resolved_tts_params.get("min_p", 0.1))
                    var_top_p = conservative_config.get("top_p", resolved_tts_params.get("top_p", 0.8))
                    candidate_type = "CONSERVATIVE"
                else:
                    # Expressive candidate logic - same as in generate_candidates
                    if num_expressive == 1:
                        var_exaggeration = base_exaggeration
                        var_cfg_weight = base_cfg_weight
                        var_temperature = base_temperature
                    elif i == 0:
                        var_exaggeration = base_exaggeration
                        var_cfg_weight = base_cfg_weight
                        var_temperature = base_temperature
                    else:
                        ramp_position = i / (num_expressive - 1)
                        var_exaggeration = base_exaggeration - (
                            exag_max_deviation * ramp_position
                        )
                        var_cfg_weight = base_cfg_weight + (
                            cfg_max_deviation * ramp_position
                        )
                        var_temperature = base_temperature + (
                            temp_max_deviation * ramp_position
                        )

                    # Use regular min_p and top_p for expressive candidates
                    var_min_p = resolved_tts_params.get("min_p", 0.05)
                    var_top_p = resolved_tts_params.get("top_p", 0.95)
                    candidate_type = "EXPRESSIVE"

                # Extract additional TTS parameters (excluding the ones we handle explicitly)
                additional_params = {
                    k: v
                    for k, v in resolved_tts_params.items()
                    if k
                    not in [
                        "exaggeration",
                        "cfg_weight",
                        "temperature",
                        "min_p",
                        "top_p",
                        "exaggeration_max_deviation",
                        "cfg_weight_max_deviation",
                        "temperature_max_deviation",
                    ]
                }

                # Add the candidate-specific min_p and top_p to additional_params
                additional_params["min_p"] = var_min_p
                additional_params["top_p"] = var_top_p

                generation_params = {
                    "exaggeration": var_exaggeration,
                    "cfg_weight": var_cfg_weight,
                    "temperature": var_temperature,
                    "seed": candidate_seed,
                    "type": candidate_type,
                    **additional_params,
                    **kwargs,
                }

                logger.debug(
                    f"Candidate {i+1} ({candidate_type}): exag={var_exaggeration:.2f}, cfg={var_cfg_weight:.2f}, temp={var_temperature:.2f}, min_p={var_min_p:.2f}, top_p={var_top_p:.2f}, seed={candidate_seed}"
                )

                result = self._generate_single_with_immediate_retries(
                    text=text,
                    exaggeration=var_exaggeration,
                    cfg_weight=var_cfg_weight,
                    temperature=var_temperature,
                    base_seed=candidate_seed,
                    language_id=language_id,
                    additional_params=additional_params,
                    reference_audio_path=reference_audio_path,
                )
                audio = result["audio"]

                candidate = AudioCandidate(
                    chunk_idx=0,  # Will be set by caller
                    candidate_idx=i,
                    audio_path=Path(),  # Will be set when saving
                    audio_tensor=audio,
                    generation_params={
                        **generation_params,
                        **{k: v for k, v in result.get("used_params", {}).items()},
                    },
                )

                candidates.append(candidate)
                logger.debug(
                    f"Generated candidate {i+1}: duration={audio.shape[-1]/24000:.2f}s"
                )

            except Exception as e:
                logger.error(
                    f"Failed to generate candidate {i+1}/{total_candidates}: {e}"
                )
                continue

        logger.debug(f"Successfully generated {len(candidates)} specific candidates")
        return candidates



    def get_current_params(self) -> Dict[str, Any]:
        """
        Get current TTS generation parameters.

        Returns:
            Dictionary of current generation parameters
        """
        generation_config = self.config.get("generation", {})
        tts_params = generation_config.get("tts_params", {})

        return {
            "device": self.device,
            "seed": self.seed,
            "tts_params": tts_params,
            "model_loaded": self.model is not None,
        }

    # Speaker system methods
    def switch_speaker(self, speaker_id: str, config_manager=None):
        """
        Switch to different speaker with new reference_audio.

        Args:
            speaker_id: Target speaker ID
            config_manager: Optional ConfigManager for file access
        """
        # Resolve actual speaker ID first (handle fallbacks before Guard check)
        actual_speaker_id = self._resolve_speaker_id(speaker_id, config_manager)
        
        # Check if we're already using the resolved speaker
        if self.current_speaker_id == actual_speaker_id:
            logger.debug(f"Speaker '{actual_speaker_id}' already active, skipping switch")
            return

        # Get speaker configuration for the resolved speaker
        speaker_config = None
        for speaker in self.speakers_config:
            if speaker.get("id") == actual_speaker_id:
                speaker_config = speaker
                break

        if not speaker_config:
            logger.error(f"Resolved speaker '{actual_speaker_id}' not found in configuration")
            return

        # Load new reference_audio
        reference_audio = speaker_config.get("reference_audio")
        if reference_audio and config_manager:
            try:
                audio_path = config_manager.get_reference_audio_for_speaker(actual_speaker_id)
                logger.info(
                    f"🎭 Switching to speaker '{actual_speaker_id}' with voice: {audio_path.name}"
                )
                self.prepare_conditionals(str(audio_path))

                # Verify conditionals are loaded
                model_ref = self.model
                if model_ref is not None and getattr(model_ref, "conds", None) is not None:
                    logger.debug(
                        f"✅ Conditionals successfully loaded for speaker '{actual_speaker_id}'"
                    )
                    self.current_speaker_id = actual_speaker_id
                else:
                    logger.error(
                        f"❌ Failed to load conditionals for speaker '{actual_speaker_id}'"
                    )

            except Exception as e:
                logger.error(f"Failed to switch to speaker '{actual_speaker_id}': {e}")
        else:
            logger.warning(
                f"No reference_audio or config_manager for speaker '{actual_speaker_id}'"
            )

    def _resolve_speaker_id(self, speaker_id: str, config_manager=None) -> str:
        """
        Resolve speaker ID with fallback logic, but don't log warnings here.
        
        Args:
            speaker_id: Requested speaker ID
            config_manager: Optional ConfigManager for file access
            
        Returns:
            Actual speaker ID to use
        """
        # Check if requested speaker exists
        for speaker in self.speakers_config:
            if speaker.get("id") == speaker_id:
                return speaker_id

        # Speaker not found - use cascading fallback logic
        if config_manager and hasattr(config_manager, 'get_default_speaker_id'):
            try:
                fallback_speaker_id = config_manager.get_default_speaker_id()
                if speaker_id != fallback_speaker_id:  # Only log if there was actually a fallback
                    logger.warning(
                        f"Speaker '{speaker_id}' not found, using default speaker '{fallback_speaker_id}'"
                    )
                return fallback_speaker_id
            except Exception as e:
                logger.debug(f"Could not get default speaker from config_manager: {e}")

        # Final fallback to first speaker
        if self.speakers_config:
            fallback_speaker_id = self.speakers_config[0].get("id", "default")
            logger.warning(f"Using first speaker as fallback: '{fallback_speaker_id}'")
            return fallback_speaker_id

        return "default"

    def generate_candidates_with_speaker(
        self, text: str, speaker_id: str, num_candidates: int, config_manager
    ) -> List[AudioCandidate]:
        """
        Generate candidates for given text using specified speaker.
        """
        # Get speaker config from internal config and resolve reference audio via file/config manager
        speaker_config = self._get_speaker_config(speaker_id)
        reference_audio_path = config_manager.get_reference_audio_for_speaker(speaker_id)
 
        # Activate the requested speaker to load/update conditionals before generation.
        # This is idempotent if the speaker is already active.
        try:
            self.switch_speaker(speaker_id, config_manager)
        except Exception as e:
            logger.error(f"Failed to switch to speaker '{speaker_id}': {e}")

        # Prepare generation parameters
        base_params = speaker_config["tts_params"]
        conservative_params = speaker_config.get("conservative_candidate", {})
        conservative_enabled = conservative_params.get("enabled", False)
 
        # Language from speaker
        language_id = speaker_config.get("language")
        if not language_id:
            # Fallback to default speaker's language via internal config
            default_speaker_id = (
                config_manager.get_default_speaker_id()
                if hasattr(config_manager, "get_default_speaker_id")
                else speaker_id
            )
            default_speaker_config = self._get_speaker_config(default_speaker_id)
            language_id = default_speaker_config.get("language")
            # Next fallback: generation.default_language
            if not language_id:
                try:
                    generation_cfg = self.config.get("generation", {})
                    language_id = generation_cfg.get("default_language")
                except Exception:
                    language_id = None
            # Final fallback to English
            if not language_id:
                language_id = "en"
            logger.warning(
                f"No language defined for speaker '{speaker_id}', using fallback language: {language_id}"
            )
 
        candidates = []
        for i in range(num_candidates):
            if i == num_candidates - 1 and conservative_enabled:
                # Last candidate: Use conservative parameters
                params = conservative_params
            elif i == 0:
                # First candidate: Exact base parameters
                params = base_params
            else:
                # Intermediate candidates: Ramped parameters
                params = self._calculate_ramped_params(base_params, i, num_candidates - 1)
 
            # Merge with base tts_params to ensure fallbacks (e.g., repetition_penalty)
            # Conservative params override base; for ramped/base, params already contains core values
            effective_params = dict(base_params)
            effective_params.update(params)

            # Generate with speaker and language
            # Split core params to avoid duplicate kwargs
            core_exaggeration = effective_params.get("exaggeration", 0.6)
            core_cfg_weight = effective_params.get("cfg_weight", 0.7)
            core_temperature = effective_params.get("temperature", 1.0)
            additional_params = {
                k: v
                for k, v in effective_params.items()
                if k
                not in [
                    "exaggeration",
                    "cfg_weight",
                    "temperature",
                    "exaggeration_max_deviation",
                    "cfg_weight_max_deviation",
                    "temperature_max_deviation",
                    "enabled",  # do not pass control flag into model.generate
                ]
            }
 
            # Compact per-candidate parameter log for runtime verification
            try:
                log_min_p = additional_params.get("min_p")
                log_top_p = additional_params.get("top_p")
                log_rep = additional_params.get("repetition_penalty")
                logger.info(
                    f"CAND {i+1}/{num_candidates} (lang={language_id}) "
                    f"({'CONS' if (i == num_candidates - 1 and conservative_enabled) else 'EXP'}): "
                    f"exag={core_exaggeration:.2f}, cfg={core_cfg_weight:.2f}, temp={core_temperature:.2f}, "
                    f"min_p={log_min_p if log_min_p is not None else '-'}, top_p={log_top_p if log_top_p is not None else '-'}, rep_pen={log_rep if log_rep is not None else '-'}"
                )
            except Exception:
                # Never fail generation due to logging
                pass
 
            result = self._generate_single_with_immediate_retries(
                text=text,
                exaggeration=core_exaggeration,
                cfg_weight=core_cfg_weight,
                temperature=core_temperature,
                base_seed=self.seed + (i * 1000) + hash(text) % 10000,
                language_id=language_id,
                additional_params=additional_params,
                reference_audio_path=str(reference_audio_path),
            )
            audio = result["audio"]
 
            candidate = AudioCandidate(
                chunk_idx=0,  # Will be set by caller
                candidate_idx=i,
                audio_path=Path(),  # Will be set by saver
                audio_tensor=audio,
                generation_params={
                    **params,
                    "language_id": language_id,
                    **{k: v for k, v in result.get("used_params", {}).items()},
                },
            )
            candidates.append(candidate)
 
        return candidates

    def _get_speaker_config(self, speaker_id: str) -> Dict[str, Any]:
        """
        Get configuration for specific speaker.

        Args:
            speaker_id: Speaker ID

        Returns:
            Speaker configuration or default speaker
        """
        # Use the same resolution logic as switch_speaker
        actual_speaker_id = self._resolve_speaker_id(speaker_id)
        
        for speaker in self.speakers_config:
            if speaker.get("id") == actual_speaker_id:
                return speaker

        # This should not happen if _resolve_speaker_id works correctly
        logger.error(f"Could not find configuration for resolved speaker '{actual_speaker_id}'")
        return {}

    def _calculate_ramped_params(self, base_params: Dict[str, Any], i: int, num_expressive_minus_one: int) -> Dict[str, Any]:
        """
        Calculate ramped TTS parameters for candidate variation based on base_params.

        Ramping semantics:
        - exaggeration: ramp DOWN from base to (base - max_deviation)
        - cfg_weight:  ramp UP   from base to (base + max_deviation)
        - temperature: ramp UP   from base to (base + max_deviation)
        """
        if num_expressive_minus_one <= 0:
            return dict(base_params)

        ramp_position = i / max(1, num_expressive_minus_one)

        base_exag = base_params.get("exaggeration", 0.6)
        exag_dev = base_params.get("exaggeration_max_deviation", 0.15)
        base_cfg = base_params.get("cfg_weight", 0.7)
        cfg_dev = base_params.get("cfg_weight_max_deviation", 0.15)
        base_temp = base_params.get("temperature", 1.0)
        temp_dev = base_params.get("temperature_max_deviation", 0.2)

        params = dict(base_params)
        params["exaggeration"] = base_exag - (exag_dev * ramp_position)
        params["cfg_weight"] = base_cfg + (cfg_dev * ramp_position)
        params["temperature"] = base_temp + (temp_dev * ramp_position)
        return params
