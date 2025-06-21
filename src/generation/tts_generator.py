import logging
import uuid
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from generation.model_cache import ChatterboxModelCache, ConditionalCache

# Import the standardized AudioCandidate from file_manager
from utils.file_manager.io_handlers.candidate_io import AudioCandidate

logger = logging.getLogger(__name__)


class TTSGenerator:
    """
    Wrapper for ChatterboxTTS model that provides generation capabilities
    with candidate management and retry logic.
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

        # Use cached model instead of loading fresh
        self.model = ChatterboxModelCache.get_model(self.device)
        self.conditional_cache = ConditionalCache(self.model)

        logger.debug(
            f"TTSGenerator initialized on device: {self.device} (using cached model)"
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
        Prepares the model conditionals using reference audio.
        Uses conditional cache to avoid redundant preparation calls.
        """
        try:
            was_prepared = self.conditional_cache.ensure_conditionals(wav_fpath)
            if was_prepared:
                logger.debug("Conditionals prepared successfully (fresh)")
            else:
                logger.debug("Conditionals were already prepared (cached)")
        except Exception as e:
            logger.error(f"Error preparing conditionals: {e}")
            raise

    def generate_single(
        self,
        text: str,
        exaggeration: float = 0.6,
        cfg_weight: float = 0.7,
        temperature: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generates a single audio output for the given text.

        Returns:
            Audio tensor containing the generated speech.
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for generation")
            return torch.zeros((1, 1000), device=self.device)

        try:
            logger.debug(
                f"Generating audio for text (len={len(text)}): '{text[:50]}...'"
            )

            # Check if model is loaded
            if self.model is None:
                logger.warning("Model not loaded - generating mock audio for testing")
                # Return mock audio with length proportional to text length (24kHz sample rate)
                mock_duration = len(text) * 0.05  # ~50ms per character
                mock_samples = int(mock_duration * 24000)
                return torch.randn(mock_samples, device=self.device) * 0.1  # 1D tensor

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

                # Generate audio using the ChatterboxTTS model
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

        except Exception as e:
            logger.error(f"Error generating audio for text '{text[:50]}...': {e}")
            # Return silence as fallback
            return torch.zeros((1, 1000), device=self.device)

    def generate_candidates(
        self,
        text: str,
        num_candidates: int = 3,
        exaggeration: Optional[float] = None,
        cfg_weight: Optional[float] = None,
        temperature: Optional[float] = None,
        conservative_config: Optional[Dict[str, Any]] = None,
        tts_params: Optional[Dict[str, Any]] = None,
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
        # Use passed tts_params if available, otherwise fall back to self.config
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
                candidate_type = "CONSERVATIVE"

                generation_params = {
                    "exaggeration": var_exaggeration,
                    "cfg_weight": var_cfg_weight,
                    "temperature": var_temperature,
                    "seed": candidate_seed,
                    "type": candidate_type,
                    **kwargs,
                }

                logger.debug(
                    f"Candidate 1 ({candidate_type}): exag={var_exaggeration:.2f}, cfg={var_cfg_weight:.2f}, temp={var_temperature:.2f}, seed={candidate_seed}"
                )

                audio = self.generate_single(
                    text,
                    exaggeration=var_exaggeration,
                    cfg_weight=var_cfg_weight,
                    temperature=var_temperature,
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

                    candidate_type = "EXPRESSIVE"

                generation_params = {
                    "exaggeration": var_exaggeration,
                    "cfg_weight": var_cfg_weight,
                    "temperature": var_temperature,
                    "seed": candidate_seed,
                    "type": candidate_type,
                    **kwargs,
                }

                logger.debug(f"Candidate {i+1} ({candidate_type}):")

                audio = self.generate_single(
                    text,
                    exaggeration=var_exaggeration,
                    cfg_weight=var_cfg_weight,
                    temperature=var_temperature,
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
                    f"Generated candidate {i+1}/{num_candidates}: duration={audio.shape[-1]/24000:.2f}s, idx={candidate.candidate_idx}"
                    + f" exag={var_exaggeration:.2f}, cfg={var_cfg_weight:.2f}, temp={var_temperature:.2f}, seed={candidate_seed}\n"
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
        candidates = []

        if not candidate_indices:
            logger.warning("No candidate indices provided for specific generation")
            return []

        logger.info(
            f"Generating specific candidates {candidate_indices} for text (len={len(text)}) "
            f"from total set of {total_candidates}\n"
        )

        candidates = []

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

        # Calculate expressive candidate parameters (same logic as generate_candidates)
        is_conservative_enabled = conservative_config and conservative_config.get(
            "enabled", False
        )
        num_expressive = (
            total_candidates - 1 if is_conservative_enabled else total_candidates
        )

        for candidate_idx in candidate_indices:
            try:
                # Set unique seed for this candidate (same logic as generate_candidates)
                candidate_seed = self.seed + (candidate_idx * 1000) + hash(text) % 10000
                torch.manual_seed(candidate_seed)

                # Check if this should be a conservative candidate (always last in total set)
                is_conservative = (
                    is_conservative_enabled and (candidate_idx + 1) == total_candidates
                )

                if is_conservative:
                    # Use conservative parameters
                    logger.debug(
                        f"Applying conservative parameters for candidate {candidate_idx+1}"
                    )
                    if conservative_config is None:
                        raise RuntimeError(
                            "Conservative config is None but conservative mode is enabled"
                        )
                    var_exaggeration = conservative_config.get("exaggeration", 0.4)
                    var_cfg_weight = conservative_config.get("cfg_weight", 0.3)
                    var_temperature = conservative_config.get("temperature", 0.5)
                    candidate_type = "CONSERVATIVE"
                else:
                    # Expressive candidate logic (same as generate_candidates)
                    if num_expressive == 1:
                        # Only one expressive candidate: use exact config values
                        var_exaggeration = base_exaggeration
                        var_cfg_weight = base_cfg_weight
                        var_temperature = base_temperature
                    elif candidate_idx == 0:
                        # First expressive candidate: always use exact config values
                        var_exaggeration = base_exaggeration
                        var_cfg_weight = base_cfg_weight
                        var_temperature = base_temperature
                    else:
                        # Subsequent expressive candidates: apply RAMP strategy
                        ramp_position = candidate_idx / (num_expressive - 1)

                        # Apply ramp directions:
                        var_exaggeration = base_exaggeration - (
                            exag_max_deviation * ramp_position
                        )
                        var_cfg_weight = base_cfg_weight + (
                            cfg_max_deviation * ramp_position
                        )
                        var_temperature = base_temperature + (
                            temp_max_deviation * ramp_position
                        )

                    candidate_type = "EXPRESSIVE"

                generation_params = {
                    "exaggeration": var_exaggeration,
                    "cfg_weight": var_cfg_weight,
                    "temperature": var_temperature,
                    "seed": candidate_seed,
                    "type": candidate_type,
                    **kwargs,
                }

                logger.info(
                    f"▶️ Candidate {candidate_idx+1} ({candidate_type}): "
                    f"exag={var_exaggeration:.2f}, cfg={var_cfg_weight:.2f}, temp={var_temperature:.2f}"
                )

                audio = self.generate_single(
                    text,
                    exaggeration=var_exaggeration,
                    cfg_weight=var_cfg_weight,
                    temperature=var_temperature,
                    **kwargs,
                )

                candidate = AudioCandidate(
                    chunk_idx=0,  # Will be set by caller
                    candidate_idx=candidate_idx,
                    audio_path=Path(),  # Will be set when saving
                    audio_tensor=audio,
                    generation_params=generation_params.copy(),
                )

                candidates.append(candidate)

                logger.info(
                    f"✅ Generated candidate {candidate_idx+1}: "
                    f"duration={audio.shape[-1]/24000:.2f}s\n"
                )

            except Exception as e:
                logger.error(f"Failed to generate candidate {candidate_idx+1}: {e}")
                continue

        logger.debug(
            f"Successfully generated {len(candidates)}/{len(candidate_indices)} candidates"
        )
        return candidates

    def load_reference_audio(self, wav_fpath: str):
        """
        Load reference audio for TTS generation.
        Alias for prepare_conditionals for compatibility.

        Args:
            wav_fpath: Path to the reference audio file.
        """
        self.prepare_conditionals(wav_fpath)

    def get_current_params(self) -> Dict[str, Any]:
        """
        Get current generation parameters including cache state.

        Returns:
            Dictionary of current TTS parameters and cache information
        """
        cache_info = ChatterboxModelCache.get_cache_info()
        conditional_state = (
            self.conditional_cache.get_current_state() if self.conditional_cache else {}
        )

        return {
            "device": self.device,
            "seed": self.seed,
            "model_type": "ChatterboxTTS",
            "model_cache": cache_info,
            "conditional_cache": conditional_state,
        }
