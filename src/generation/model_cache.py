"""
Model cache system for ChatterboxTTS to avoid repeated model loading.
Implements singleton pattern with device-specific caching and conditional state management.
"""

import hashlib
import logging
import threading
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from utils.logging_config import get_logger

logger = get_logger(__name__, verbose=True)


class ChatterboxModelCache:
    """
    Singleton cache for ChatterboxTTS models.
    Caches models per device to avoid repeated loading.
    """

    _instance = None
    _lock = threading.Lock()
    _model_cache: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ChatterboxModelCache, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_model(cls, device: str = "auto"):
        """
        Get cached ChatterboxTTS model for the specified device.

        Args:
            device: Target device ("auto", "cuda", "mps", "cpu")

        Returns:
            ChatterboxTTS model instance
        """
        # Resolve auto device
        actual_device = cls._detect_device() if device == "auto" else device
        cache_key = actual_device

        if cache_key not in cls._model_cache:
            logger.primary(
                f"🔄 Loading ChatterboxTTS model for device: {cache_key} (cache miss)"
            )
            cls._model_cache[cache_key] = cls._load_fresh_model(cache_key)
        else:
            logger.primary(
                f"♻️ Using cached ChatterboxTTS model for device: {cache_key} (cache hit)"
            )

        return cls._model_cache[cache_key]

    @classmethod
    def _detect_device(cls) -> str:
        """Detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @classmethod
    def _load_fresh_model(cls, device: str):
        """Load a fresh ChatterboxTTS model instance."""
        try:
            # Suppress PyTorch and Transformers warnings during model loading
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*torch.backends.cuda.sdp_kernel.*", category=FutureWarning)
                warnings.filterwarnings("ignore", message=".*LlamaModel is using LlamaSdpaAttention.*")
                warnings.filterwarnings("ignore", message=".*does not support `output_attentions=True`.*")
                warnings.filterwarnings("ignore", message=".*attn_implementation.*", category=FutureWarning)
                
                # Import ChatterboxTTS
                from chatterbox.tts import ChatterboxTTS

                # Try to pass attn_implementation – fallback gracefully if the
                # ChatterboxTTS signature does not yet support the kwarg.
                try:
                    # Use "eager" attention implementation to silence the warning
                    model = ChatterboxTTS.from_pretrained(
                        device=device, attn_implementation="eager"
                    )
                except TypeError:
                    # Older library version – ignore kwarg and log info
                    logger.verbose(
                        "ChatterboxTTS.from_pretrained() does not accept attn_implementation – falling back without it"
                    )
                    model = ChatterboxTTS.from_pretrained(device=device)

            logger.verbose(
                f"ChatterboxTTS model loaded successfully for device: {device}"
            )
            return model

        except Exception as e:
            logger.error(f"Failed to load ChatterboxTTS model for device {device}: {e}")
            logger.primary("Returning None - will use mock mode for testing")
            return None

    @classmethod
    def clear_cache(cls):
        """Clear all cached models (useful for testing)."""
        logger.verbose("🗑️ Clearing ChatterboxTTS model cache")
        cls._model_cache.clear()

    @classmethod
    def get_cache_info(cls) -> Dict[str, Any]:
        """Get information about current cache state."""
        return {
            "cached_devices": list(cls._model_cache.keys()),
            "cache_size": len(cls._model_cache),
            "models_loaded": {
                device: model is not None for device, model in cls._model_cache.items()
            },
        }


class ConditionalCache:
    """
    Manages conditional state for a ChatterboxTTS model.
    Tracks which reference audio was last used to avoid redundant prepare_conditionals calls.
    """

    def __init__(self, model):
        """
        Initialize conditional cache for a model.

        Args:
            model: ChatterboxTTS model instance
        """
        self.model = model
        self.current_reference_audio = None
        self.current_reference_hash = None

    def ensure_conditionals(self, reference_audio_path: str) -> bool:
        """
        Ensure conditionals are prepared for the given reference audio.
        Only calls prepare_conditionals if the reference audio has changed.

        Args:
            reference_audio_path: Path to reference audio file

        Returns:
            True if conditionals were prepared, False if already cached
        """
        if self.model is None:
            logger.warning("Model not loaded - skipping conditional preparation")
            return False

        # Calculate hash of reference audio path for comparison
        reference_hash = self._calculate_file_hash(reference_audio_path)

        if (
            self.current_reference_audio != reference_audio_path
            or self.current_reference_hash != reference_hash
        ):

            logger.verbose(
                f"Preparing conditionals for: {Path(reference_audio_path).name}"
            )
            try:
                self.model.prepare_conditionals(wav_fpath=reference_audio_path)
                self.current_reference_audio = reference_audio_path
                self.current_reference_hash = reference_hash
                logger.verbose("Conditionals prepared successfully")
                return True
            except Exception as e:
                logger.error(f"Error preparing conditionals: {e}")
                raise
        else:
            logger.verbose(
                f"Conditionals already prepared for: {Path(reference_audio_path).name}"
            )
            return False

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of file for change detection."""
        try:
            path = Path(file_path)
            if not path.exists():
                return f"missing_{file_path}"

            # Use file size and modification time for quick hash
            stat = path.stat()
            content = f"{path.name}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(content.encode()).hexdigest()[:8]
        except Exception as e:
            logger.warning(f"Could not calculate hash for {file_path}: {e}")
            return f"error_{file_path}"

    def get_current_state(self) -> Dict[str, Any]:
        """Get current conditional state information."""
        return {
            "reference_audio": self.current_reference_audio,
            "reference_hash": self.current_reference_hash,
            "has_conditionals": self.current_reference_audio is not None,
        }

    def reset(self):
        """Reset conditional cache state."""
        logger.verbose("🔄 Resetting conditional cache state")
        self.current_reference_audio = None
        self.current_reference_hash = None
