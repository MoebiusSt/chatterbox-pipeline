import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
import threading

import torch

from generation.model_cache import ChatterboxModelCache, SerializedModelAccess

# Import the standardized AudioCandidate from file_manager
from utils.file_manager.io_handlers.candidate_io import AudioCandidate

logger = logging.getLogger(__name__)

# Thread-lokaler Context für Chunk/Candidate-Information
_thread_local = threading.local()

def set_generation_context(task_name: str = "", chunk_num: int = 0, candidate_num: int = 0, total_chunks: int = 0):
    """Setzt den aktuellen Generierungskontext für Thread-lokales Logging."""
    _thread_local.task_name = task_name
    _thread_local.chunk_num = chunk_num
    _thread_local.candidate_num = candidate_num 
    _thread_local.total_chunks = total_chunks

def get_generation_context() -> tuple[str, int, int, int]:
    """Gibt den aktuellen Generierungskontext zurück."""
    task_name = getattr(_thread_local, 'task_name', 'unknown')
    chunk_num = getattr(_thread_local, 'chunk_num', 0)
    candidate_num = getattr(_thread_local, 'candidate_num', 0)
    total_chunks = getattr(_thread_local, 'total_chunks', 0)
    return task_name, chunk_num, candidate_num, total_chunks

def get_context_prefix() -> str:
    """Erstellt einen Kontext-Prefix für Logging-Nachrichten."""
    task_name, chunk_num, candidate_num, total_chunks = get_generation_context()
    
    if task_name and task_name != 'unknown':
        # Kürze den Task-Namen für bessere Lesbarkeit
        short_task = task_name.split('_')[0] if '_' in task_name else task_name[:10]
        if chunk_num > 0 and candidate_num > 0:
            return f"[{short_task}-C{chunk_num:02d}-{candidate_num}]"
        elif chunk_num > 0:
            return f"[{short_task}-C{chunk_num:02d}]"
        else:
            return f"[{short_task}]"
    else:
        return "[?]"



class TTSGenerator:
    """
    THREAD-SAFE TTS Generator with complete model serialization.
    Uses SerializedModelAccess to prevent all race conditions in parallel execution.
    """

    def __init__(self, config: Dict[str, Any], device: str = "auto", seed: int = 12345):
        """
        Initializes the THREAD-SAFE TTSGenerator.

        Args:
            config: Configuration dictionary with generation settings.
            device: The device to run inference on (cuda, mps, cpu).
            seed: Random seed for reproducibility.
        """
        self.config = config
        self.device = device if device != "auto" else self._detect_device()
        self.seed = seed

        # Use SERIALIZED model access instead of direct model reference
        self.model = ChatterboxModelCache.get_model(self.device)
        
        # NOTE: No longer using ConditionalCache - using SerializedModelAccess directly
        logger.debug(
            f"THREAD-SAFE TTSGenerator initialized on device: {self.device} (using SERIALIZED model access)"
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
        THREAD-SAFE: Prepares model conditionals using COMPLETE MODEL SERIALIZATION.
        
        This method now uses SerializedModelAccess to ensure only one thread
        can access the model at a time, preventing all race conditions.
        """
        if self.model is None:
            logger.warning("🚨 No model loaded - cannot prepare conditionals")
            return
            
        thread_id = threading.current_thread().ident
        logger.debug(f"🔄 Thread {thread_id}: Preparing conditionals for {Path(wav_fpath).name}")
        
        try:
            # Use SERIALIZED model access - completely thread-safe
            with SerializedModelAccess.with_exclusive_model_access(self.model, wav_fpath) as model:
                # Model is now configured with correct conditionals
                # The context manager handles all the thread-safety
                logger.debug(f"✅ Thread {thread_id}: Conditionals prepared using SERIALIZED access")
        except Exception as e:
            logger.error(f"🚨 Thread {thread_id}: Error preparing conditionals: {e}")
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
        THREAD-SAFE: Generate single audio using COMPLETE MODEL SERIALIZATION.
        
        This method now uses SerializedModelAccess to ensure only one thread
        can access the ChatterboxTTS model at a time, preventing all race conditions.

        Args:
            text: Text to synthesize
            exaggeration: Voice exaggeration parameter (0.0-1.0)
            cfg_weight: Classifier-free guidance weight
            temperature: Sampling temperature for diversity
            reference_audio_path: Path to reference audio (required for SERIALIZED access)
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
            
        if not reference_audio_path:
            logger.error("🚨 reference_audio_path is required for SERIALIZED model access")
            return torch.zeros(48000, device=self.device)

        # Get context for logging
        thread_id = threading.current_thread().ident
        logger.debug(f"Thread {thread_id}: Starting SERIALIZED TTS generation")

        # Use COMPLETE MODEL SERIALIZATION for thread safety
        with SerializedModelAccess.with_exclusive_model_access(self.model, reference_audio_path) as model:
            logger.debug(f"Thread {thread_id}: Acquired EXCLUSIVE model access")
            
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

                logger.debug(f"Generating audio for text (len={len(text)}): '{text[:50]}...'")
                
                # Generate audio using the ChatterboxTTS model (COMPLETELY SERIALIZED)
                audio = model.generate(
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
            logger.debug(f"Thread {thread_id}: Released EXCLUSIVE model access")

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
                candidate_type = "CONSERVATIVE"

                # Debug: Log tts_params before extracting additional_params
                logger.info(f"🔍 tts_params for candidate 1: {tts_params}")
                # Extract additional TTS parameters from tts_params
                additional_params = {k: v for k, v in tts_params.items() 
                                   if k not in ["exaggeration", "cfg_weight", "temperature", 
                                               "exaggeration_max_deviation", "cfg_weight_max_deviation", "temperature_max_deviation"]}
                
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
                    f"Candidate 1 ({candidate_type}): exag={var_exaggeration:.2f}, cfg={var_cfg_weight:.2f}, temp={var_temperature:.2f}, seed={candidate_seed}"
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

                # Debug: Log tts_params before extracting additional_params
                logger.info(f"CANDIDATE {i+1} ({candidate_type}): exag={var_exaggeration:.2f}, cfg={var_cfg_weight:.2f}, temp={var_temperature:.2f}")

                # Extract additional TTS parameters from tts_params
                additional_params = {k: v for k, v in tts_params.items() 
                                   if k not in ["exaggeration", "cfg_weight", "temperature", 
                                               "exaggeration_max_deviation", "cfg_weight_max_deviation", "temperature_max_deviation"]}
                
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

                # Extract additional TTS parameters from tts_params
                additional_params = {k: v for k, v in tts_params.items() 
                                   if k not in ["exaggeration", "cfg_weight", "temperature", 
                                               "exaggeration_max_deviation", "cfg_weight_max_deviation", "temperature_max_deviation"]}
                
                generation_params = {
                    "exaggeration": var_exaggeration,
                    "cfg_weight": var_cfg_weight,
                    "temperature": var_temperature,
                    "seed": candidate_seed,
                    "type": candidate_type,
                    **additional_params,  # Include repetition_penalty and other TTS params
                    **kwargs,
                }

                logger.info(
                    f"CANDIDATE {candidate_idx+1} ({candidate_type}): "
                    f"exag={var_exaggeration:.2f}, cfg={var_cfg_weight:.2f}, temp={var_temperature:.2f}"
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

        return {
            "device": self.device,
            "seed": self.seed,
            "model_type": "ChatterboxTTS",
            "model_cache": cache_info,
            "serialized_access": "SerializedModelAccess enabled for thread-safety",
        }
