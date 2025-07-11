import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

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

        # Use direct model access
        self.model = ChatterboxModelCache.get_model(self.device)

        # Speaker system attributes
        self.current_speaker_id = "default"
        self.speakers_config = config.get("generation", {}).get("speakers", [])

        logger.debug(
            f"TTSGenerator initialized on device: {self.device} with {len(self.speakers_config)} speakers"
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

        # Only prepare conditionals if none are loaded yet
        # This prevents overwriting speaker-specific conditionals
        if not hasattr(self.model, "conds") or self.model.conds is None:
            if not reference_audio_path:
                logger.error(
                    "🚨 No conditionals loaded and no reference_audio_path provided"
                )
                return torch.zeros(48000, device=self.device)
            logger.debug("No conditionals loaded - preparing from reference_audio_path")
            self.prepare_conditionals(reference_audio_path)
        else:
            logger.debug("Using existing conditionals (speaker-specific)")

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
                "ignore", message=".*does not support `output_attentions=True`.*"
            )
            warnings.filterwarnings(
                "ignore",
                message=".*past_key_values.*tuple of tuples.*",
                category=FutureWarning,
            )
            warnings.filterwarnings(
                "ignore", message=".*attn_implementation.*", category=FutureWarning
            )

            logger.debug(
                f"Generating audio for text (len={len(text)}): '{text[:50]}...'"
            )

            # Generate audio using the ChatterboxTTS model directly
            audio = self.model.generate(
                text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                **kwargs,
            )

        # ChatterboxTTS returns 1D tensor - ensure consistency
        if audio.ndim == 2:
            audio = audio.squeeze(0)  # Remove batch dimension if present
        audio = audio.to(self.device)

        logger.debug(f"Generated audio with shape: {audio.shape}")
        return audio

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

        # Use config values as defaults - these are now the starting points for ramping
        base_exaggeration = (
            exaggeration
            if exaggeration is not None
            else tts_params.get("exaggeration", 0.6)
        )  # MAX value
        base_cfg_weight = (
            cfg_weight if cfg_weight is not None else tts_params.get("cfg_weight", 0.7)
        )  # MIN value
        base_temperature = (
            temperature
            if temperature is not None
            else tts_params.get("temperature", 1.0)
        )  # MIN value

        # Get deviation ranges from config
        exag_max_deviation = tts_params.get("exaggeration_max_deviation", 0.15)
        cfg_max_deviation = tts_params.get("cfg_weight_max_deviation", 0.15)
        temp_max_deviation = tts_params.get("temperature_max_deviation", 0.2)

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
                torch.manual_seed(candidate_seed)

                # Use conservative parameters
                var_exaggeration = conservative_config.get("exaggeration", 0.4)
                var_cfg_weight = conservative_config.get("cfg_weight", 0.3)
                var_temperature = conservative_config.get("temperature", 0.5)
                # Use conservative min_p and top_p with fallback to regular tts_params
                var_min_p = conservative_config.get("min_p", tts_params.get("min_p", 0.1))
                var_top_p = conservative_config.get("top_p", tts_params.get("top_p", 0.8))
                candidate_type = "CONSERVATIVE"

                # Debug: Log tts_params before extracting additional_params
                logger.info(f"🔍 tts_params for candidate 1: {tts_params}")
                # Extract additional TTS parameters from tts_params (excluding the ones we handle explicitly)
                additional_params = {
                    k: v
                    for k, v in tts_params.items()
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

                audio = self.generate_single(
                    text,
                    exaggeration=var_exaggeration,
                    cfg_weight=var_cfg_weight,
                    temperature=var_temperature,
                    reference_audio_path=reference_audio_path,
                    **additional_params,  # Pass repetition_penalty to renderer
                    **kwargs,
                )

                candidate = AudioCandidate(
                    chunk_idx=0,  # Will be set by caller
                    candidate_idx=0,
                    audio_path=Path(),  # Will be set when saving
                    audio_tensor=audio,
                    generation_params=generation_params.copy(),
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
                torch.manual_seed(candidate_seed)

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
                    var_min_p = conservative_config.get("min_p", tts_params.get("min_p", 0.1))
                    var_top_p = conservative_config.get("top_p", tts_params.get("top_p", 0.8))
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
                    var_min_p = tts_params.get("min_p", 0.05)
                    var_top_p = tts_params.get("top_p", 0.95)
                    candidate_type = "EXPRESSIVE"

                # Extract additional TTS parameters from tts_params (excluding the ones we handle explicitly)
                additional_params = {
                    k: v
                    for k, v in tts_params.items()
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

                audio = self.generate_single(
                    text,
                    exaggeration=var_exaggeration,
                    cfg_weight=var_cfg_weight,
                    temperature=var_temperature,
                    reference_audio_path=reference_audio_path,
                    **additional_params,  # Pass repetition_penalty to renderer
                    **kwargs,
                )

                candidate = AudioCandidate(
                    chunk_idx=0,  # Will be set by caller
                    candidate_idx=i,
                    audio_path=Path(),  # Will be set when saving
                    audio_tensor=audio,
                    generation_params=generation_params.copy(),
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

        # Use config values as defaults
        base_exaggeration = (
            exaggeration
            if exaggeration is not None
            else tts_params.get("exaggeration", 0.6)
        )
        base_cfg_weight = (
            cfg_weight if cfg_weight is not None else tts_params.get("cfg_weight", 0.7)
        )
        base_temperature = (
            temperature
            if temperature is not None
            else tts_params.get("temperature", 1.0)
        )

        # Get deviation ranges from config
        exag_max_deviation = tts_params.get("exaggeration_max_deviation", 0.15)
        cfg_max_deviation = tts_params.get("cfg_weight_max_deviation", 0.15)
        temp_max_deviation = tts_params.get("temperature_max_deviation", 0.2)

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
                torch.manual_seed(candidate_seed)

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
                    var_min_p = conservative_config.get("min_p", tts_params.get("min_p", 0.1))
                    var_top_p = conservative_config.get("top_p", tts_params.get("top_p", 0.8))
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
                    var_min_p = tts_params.get("min_p", 0.05)
                    var_top_p = tts_params.get("top_p", 0.95)
                    candidate_type = "EXPRESSIVE"

                # Extract additional TTS parameters (excluding the ones we handle explicitly)
                additional_params = {
                    k: v
                    for k, v in tts_params.items()
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

                audio = self.generate_single(
                    text,
                    exaggeration=var_exaggeration,
                    cfg_weight=var_cfg_weight,
                    temperature=var_temperature,
                    reference_audio_path=reference_audio_path,
                    **additional_params,
                    **kwargs,
                )

                candidate = AudioCandidate(
                    chunk_idx=0,  # Will be set by caller
                    candidate_idx=i,
                    audio_path=Path(),  # Will be set when saving
                    audio_tensor=audio,
                    generation_params=generation_params.copy(),
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
        if self.current_speaker_id == speaker_id:
            logger.debug(f"Speaker '{speaker_id}' already active, skipping switch")
            return

        # Get speaker configuration
        speaker_config = None
        for speaker in self.speakers_config:
            if speaker.get("id") == speaker_id:
                speaker_config = speaker
                break

        if not speaker_config:
            # Try to use explicit default_speaker from config
            default_speaker = self.config.get("generation", {}).get("default_speaker")
            if default_speaker and self.speakers_config:
                logger.warning(
                    f"Speaker '{speaker_id}' not found, using default speaker '{default_speaker}'"
                )
                for speaker in self.speakers_config:
                    if speaker.get("id") == default_speaker:
                        speaker_config = speaker
                        speaker_id = default_speaker
                        break

            # Final fallback to first speaker
            if not speaker_config:
                logger.warning("Default speaker not found, using first speaker")
                speaker_config = self.speakers_config[0] if self.speakers_config else {}
                speaker_id = speaker_config.get("id", "default")

        # Load new reference_audio
        reference_audio = speaker_config.get("reference_audio")
        if reference_audio and config_manager:
            try:
                audio_path = config_manager.get_reference_audio_for_speaker(speaker_id)
                logger.info(
                    f"🎭 Switching to speaker '{speaker_id}' with voice: {audio_path.name}"
                )
                self.prepare_conditionals(str(audio_path))

                # Verify conditionals are loaded
                if hasattr(self.model, "conds") and self.model.conds is not None:
                    logger.debug(
                        f"✅ Conditionals successfully loaded for speaker '{speaker_id}'"
                    )
                    self.current_speaker_id = speaker_id
                else:
                    logger.error(
                        f"❌ Failed to load conditionals for speaker '{speaker_id}'"
                    )

            except Exception as e:
                logger.error(f"Failed to switch to speaker '{speaker_id}': {e}")
        else:
            logger.warning(
                f"No reference_audio or config_manager for speaker '{speaker_id}'"
            )

    def generate_candidates_with_speaker(
        self,
        text: str,
        speaker_id: str = "default",
        num_candidates: int = 3,
        config_manager=None,
        **kwargs,
    ) -> List[AudioCandidate]:
        """
        Generate candidates with specific speaker.

        Args:
            text: Text for generation
            speaker_id: ID of speaker to use
            num_candidates: Number of candidates
            config_manager: ConfigManager for file access
            **kwargs: Additional parameters

        Returns:
            List of AudioCandidate objects
        """

        # 1. Switch to speaker
        self.switch_speaker(speaker_id, config_manager)

        # 2. Get speaker-specific parameters
        speaker_config = self._get_speaker_config(speaker_id)
        tts_params = speaker_config.get("tts_params", {})
        conservative_config = speaker_config.get("conservative_candidate", {})

        # 3. Get reference_audio for this speaker
        reference_audio_path = None
        if config_manager:
            try:
                audio_path = config_manager.get_reference_audio_for_speaker(speaker_id)
                reference_audio_path = str(audio_path)
            except Exception as e:
                logger.error(
                    f"Could not get reference audio for speaker '{speaker_id}': {e}"
                )

        # 4. Generate with speaker parameters
        candidates = self.generate_candidates(
            text=text,
            num_candidates=num_candidates,
            tts_params=tts_params,
            conservative_config=conservative_config,
            reference_audio_path=reference_audio_path,
            **kwargs,
        )

        # 5. Set speaker ID in candidate metadata
        for candidate in candidates:
            if hasattr(candidate, "generation_params"):
                candidate.generation_params["speaker_id"] = speaker_id

        return candidates

    def _get_speaker_config(self, speaker_id: str) -> Dict[str, Any]:
        """
        Get configuration for specific speaker.

        Args:
            speaker_id: Speaker ID

        Returns:
            Speaker configuration or default speaker
        """
        for speaker in self.speakers_config:
            if speaker.get("id") == speaker_id:
                return speaker

        # Fallback to default speaker
        default_speaker = self.config.get("generation", {}).get("default_speaker")
        if default_speaker and self.speakers_config:
            logger.debug(
                f"Speaker '{speaker_id}' not found, using default speaker '{default_speaker}'"
            )
            for speaker in self.speakers_config:
                if speaker.get("id") == default_speaker:
                    return speaker

        # Final fallback to first speaker
        if self.speakers_config:
            logger.debug("Default speaker not found, using first speaker")
            return self.speakers_config[0]

        return {}
