"""
Model cache system for ChatterboxTTS to avoid repeated model loading.
Implements singleton pattern with device-specific caching.

Note: Cache miss on new process start is normal behavior. 
The cache only works within a single program run.
"""

import logging
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class ChatterboxModelCache:
    """
    Singleton cache for ChatterboxTTS and ChatterboxMultilingualTTS models.
    Caches models per device and model type to avoid repeated loading within a single process.
    
    Important: Cache miss on new program start is NORMAL behavior.
    The cache only persists during a single program execution.
    """

    _instance = None
    _model_cache: Dict[str, Any] = {}
    _load_times: Dict[str, float] = {}  # Track loading times

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ChatterboxModelCache, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_model(cls, device: str = "auto", model_type: str = "standard"):
        """
        Get cached ChatterboxTTS or ChatterboxMultilingualTTS model for the specified device.

        Args:
            device: Target device ("auto", "cuda", "mps", "cpu")
            model_type: Model type ("standard" for ChatterboxTTS or "multilingual" for ChatterboxMultilingualTTS)

        Returns:
            ChatterboxTTS or ChatterboxMultilingualTTS model instance
        """
        # Resolve auto device
        actual_device = cls._detect_device() if device == "auto" else device
        cache_key = f"{actual_device}_{model_type}"

        # Check in-memory cache first
        if cache_key in cls._model_cache:
            load_time = cls._load_times.get(cache_key, 0)
            model_name = "ChatterboxTTS" if model_type == "standard" else "ChatterboxMultilingualTTS"
            logger.info(
                f"♻️ Using cached {model_name} model for device: {actual_device} (cache hit, originally loaded in {load_time:.1f}s)"
            )
            return cls._model_cache[cache_key]

        # Load fresh model
        model_name = "ChatterboxTTS" if model_type == "standard" else "ChatterboxMultilingualTTS"
        logger.info(
            f"🔄 Loading {model_name} model for device: {actual_device} (cache miss)"
        )
        
        # Track loading time
        start_time = time.time()
        model = cls._load_fresh_model(actual_device, model_type)
        load_time = time.time() - start_time
        
        # Cache the model and loading time
        cls._model_cache[cache_key] = model
        cls._load_times[cache_key] = load_time
        
        logger.info(f"✅ Model loaded in {load_time:.1f}s and cached for future use in this session")

        return model

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
    def _load_fresh_model(cls, device: str, model_type: str = "standard"):
        """Load a fresh ChatterboxTTS or ChatterboxMultilingualTTS model instance with optimized settings."""
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

                # Import the appropriate model class based on model_type
                if model_type == "multilingual":
                    try:
                        from chatterbox.mtl_tts import ChatterboxMultilingualTTS as ModelClass
                        model_name = "ChatterboxMultilingualTTS"
                    except ImportError:
                        logger.warning("ChatterboxMultilingualTTS not available, falling back to standard ChatterboxTTS")
                        from chatterbox.tts import ChatterboxTTS as ModelClass
                        model_name = "ChatterboxTTS (multilingual fallback)"
                else:
                    from chatterbox.tts import ChatterboxTTS as ModelClass
                    model_name = "ChatterboxTTS"

                # Try to pass attn_implementation – fallback gracefully if the
                # model signature does not yet support the kwarg.
                try:
                    # Use "eager" attention implementation to silence the warning
                    model = ModelClass.from_pretrained(
                        device=device, attn_implementation="eager"
                    )
                except TypeError:
                    # Older library version – ignore kwarg and log info
                    logger.debug(
                        f"{model_name}.from_pretrained() does not accept attn_implementation – falling back without it"
                    )
                    model = ModelClass.from_pretrained(device=device)

            logger.debug(
                f"{model_name} model loaded successfully for device: {device}"
            )
            return model

        except Exception as e:
            model_name = "ChatterboxTTS" if model_type == "standard" else "ChatterboxMultilingualTTS"
            logger.error(
                f"🚨 CRITICAL: Failed to load {model_name} model for device {device}: {e}"
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
        cls._load_times.clear()

    @classmethod
    def get_cache_info(cls) -> Dict[str, Any]:
        """Get information about current cache state."""
        return {
            "cached_devices": list(cls._model_cache.keys()),
            "cache_size": len(cls._model_cache),
            "models_loaded": {
                device: model is not None for device, model in cls._model_cache.items()
            },
            "load_times": cls._load_times.copy(),
            "cache_type": "in-memory (session-only)",
            "cache_behavior": "Cache miss on new program start is normal"
        }

    @classmethod
    def explain_cache_behavior(cls):
        """Explain cache behavior to users."""
        info = cls.get_cache_info()
        
        print("\n" + "=" * 60)
        print("📚 CHATTERBOX MODEL CACHE EXPLANATION")
        print("=" * 60)
        
        print("\n🔍 CURRENT CACHE STATE:")
        print(f"  - Cached devices: {info['cached_devices']}")
        print(f"  - Cache size: {info['cache_size']}")
        print(f"  - Cache type: {info['cache_type']}")
        
        if info['load_times']:
            print(f"  - Load times: {info['load_times']}")
        
        print("\n💡 CACHE BEHAVIOR:")
        print("  ✅ CACHE HIT: When using the same model within one program run")
        print("  ❌ CACHE MISS: When starting a new program run (this is NORMAL)")
        
        print("\n📖 WHY CACHE MISS ON NEW PROGRAM START:")
        print("  • Each Python process has its own memory space")
        print("  • Models are too complex to serialize to disk efficiently")
        print("  • HuggingFace cache still avoids re-downloading the model")
        print("  • Only the initialization takes time, not the download")
        
        print("\n🚀 OPTIMIZATION TIPS:")
        print("  • Process multiple tasks in one run (use job system)")
        print("  • Use --mode all to process all tasks in one session")
        print("  • Consider the cache miss as normal startup time")
        
        print("\n" + "=" * 60)
