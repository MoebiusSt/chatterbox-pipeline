"""
Model cache system for ChatterboxTTS to avoid repeated model loading.
Implements singleton pattern with device-specific caching.
"""

import logging
import warnings
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


class ChatterboxModelCache:
    """
    Singleton cache for ChatterboxTTS models.
    Caches models per device to avoid repeated loading.
    """

    _instance = None
    _model_cache: Dict[str, Any] = {}

    def __new__(cls):
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
            logger.info(
                f"🔄 Loading ChatterboxTTS model for device: {cache_key} (cache miss)"
            )
            cls._model_cache[cache_key] = cls._load_fresh_model(cache_key)
        else:
            logger.info(
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
                    "ignore", message=".*attn_implementation.*", category=FutureWarning
                )

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
                    logger.debug(
                        "ChatterboxTTS.from_pretrained() does not accept attn_implementation – falling back without it"
                    )
                    model = ChatterboxTTS.from_pretrained(device=device)

            logger.debug(
                f"ChatterboxTTS model loaded successfully for device: {device}"
            )
            return model

        except Exception as e:
            logger.error(
                f"🚨 CRITICAL: Failed to load ChatterboxTTS model for device {device}: {e}"
            )
            logger.error("=" * 80)
            logger.error("⚠️  WARNING: TTS MODEL LOADING FAILED!")
            logger.error(
                "⚠️  The system will run in MOCK MODE and generate only NOISE/SILENCE!"
            )
            logger.error("⚠️  Your final audio output will contain NO SPEECH!")
            logger.error("=" * 80)
            logger.error("💡 To fix this issue:")
            logger.error(
                "   1. Check ChatterboxTTS installation: pip install chatterbox-tts"
            )
            logger.error("   2. Check perth dependency: pip install perth")
            logger.error(
                "   3. Update dependencies: pip install --upgrade chatterbox-tts perth"
            )
            logger.error("   4. If issue persists, check GPU/CUDA compatibility")
            logger.error("=" * 80)
            logger.info("Returning None - will use mock mode for testing")
            return None

    @classmethod
    def clear_cache(cls):
        """Clear all cached models (useful for testing)."""
        logger.debug("🗑️ Clearing ChatterboxTTS model cache")
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
