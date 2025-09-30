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
            
            # Check if cached model needs optimization for speed
            cached_model = cls._model_cache[cache_key]
            optimized_model = cls._optimize_cached_model(cached_model, actual_device)
            if optimized_model is not cached_model:
                cls._model_cache[cache_key] = optimized_model
                logger.info("🚀 Updated cache with speed-optimized model")
            
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
        
        # OPTIMIZE FRESHLY LOADED MODEL
        optimized_model = cls._optimize_cached_model(model, actual_device)
        if optimized_model is not model:
            logger.info("🚀 Model optimized for speed")
            model = optimized_model
        
        # Cache the optimized model and loading time
        cls._model_cache[cache_key] = model
        cls._load_times[cache_key] = load_time
        
        logger.info(f"✅ Model loaded in {load_time:.1f}s and cached for future use in this session")

        return model

    @classmethod
    def _optimize_cached_model(cls, model, device: str):
        """Optimize a cached model with mixed precision for better performance."""
        import torch
        
        # Only optimize CUDA models
        if device != "cuda":
            return model
            
        try:
            # Try to access model parameters (handle Chatterbox composite models)
            first_param = None
            
            # Method 1: Direct parameters() method
            if hasattr(model, 'parameters'):
                try:
                    first_param = next(model.parameters(), None)
                except:
                    pass
            
            # Method 2: Try sub-models for Chatterbox (s3gen, t3, ve, etc.)
            if first_param is None:
                for attr_name in ['s3gen', 't3', 've', 'model', 'llm', 'transformer', 'encoder', 'decoder']:
                    if hasattr(model, attr_name):
                        sub_model = getattr(model, attr_name)
                        if hasattr(sub_model, 'parameters'):
                            try:
                                first_param = next(sub_model.parameters(), None)
                                logger.debug(f"🔧 Found parameters in model.{attr_name}")
                                break
                            except:
                                continue
            
            # Method 3: Try to convert whole model (if it has .to() method)
            if first_param is None:
                if hasattr(model, 'to'):
                    logger.debug("🔧 No direct parameters found, but model has .to() method - trying conversion")
                    # We'll try the conversion below without checking dtype
                    current_dtype = torch.float32  # Assume fp32
                else:
                    logger.debug("🔧 Model has no parameters or .to() method, skipping optimization")
                    return model
            else:
                current_dtype = first_param.dtype
                logger.debug(f"🔧 Current model dtype: {current_dtype}")
            
            # Note: rsxdalv/chatterbox@faster-multi (v0.2.1+) has built-in CUDA graph optimizations
            # No additional optimization needed - the model handles it internally with:
            # - Automatic CUDA graph capturing for different sequence lengths
            # - StaticCache for KV cache optimization
            # - Optimized sampling loops
            logger.debug("ℹ️ Using chatterbox faster-multi branch with built-in CUDA graph optimizations")
            return model
                
        except Exception as e:
            logger.warning(f"⚠️ Error during model optimization: {e}")
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

                # ==== SPEED OPTIMIZATION WITH BF16/FP16 ====
                # Determine optimal dtype for ~2x speed boost on GPU
                # bf16 (bfloat16) is faster and available on newer GPUs (Ampere+)
                # fp16 (float16) is faster on older GPUs
                torch_dtype = None
                logger.info(f"🔧 DEBUG: Device is '{device}', CUDA available: {torch.cuda.is_available()}")
                
                if device == "cuda":
                    try:
                        capability = torch.cuda.get_device_capability()
                        logger.info(f"🔧 DEBUG: CUDA capability: {capability}")
                        # Check if bfloat16 is supported (CUDA capability >= 8.0, Ampere+)
                        if torch.cuda.is_available() and capability[0] >= 8:
                            torch_dtype = torch.bfloat16
                            logger.info("🚀 Using bfloat16 precision for ~2x speed boost (RTX 30xx/40xx+)")
                        else:
                            torch_dtype = torch.float16
                            logger.info("🚀 Using float16 precision for speed optimization (older GPUs)")
                    except Exception as e:
                        logger.warning(f"🔧 DEBUG: Could not get CUDA capability: {e}, falling back to fp32")
                else:
                    logger.info(f"🔧 DEBUG: Not using CUDA (device={device}), staying with fp32")
                
                # Try to pass attn_implementation and torch_dtype – fallback gracefully if the
                # model signature does not yet support the kwargs.
                try:
                    # Use "eager" attention implementation to silence the warning
                    if torch_dtype:
                        model = ModelClass.from_pretrained(
                            device=device, 
                            attn_implementation="eager",
                            torch_dtype=torch_dtype
                        )
                    else:
                        model = ModelClass.from_pretrained(
                            device=device, 
                            attn_implementation="eager"
                        )
                except TypeError:
                    # Older library version – ignore kwargs and log info
                    logger.info(f"🔧 DEBUG: {model_name} doesn't support torch_dtype in from_pretrained(), trying manual conversion")
                    if torch_dtype:
                        try:
                            model = ModelClass.from_pretrained(device=device, torch_dtype=torch_dtype)
                            logger.info(f"🔧 DEBUG: torch_dtype parameter accepted")
                        except:
                            model = ModelClass.from_pretrained(device=device)
                            logger.info(f"🔧 DEBUG: torch_dtype parameter failed, loading in fp32 then converting")
                    else:
                        model = ModelClass.from_pretrained(device=device)
                
                # Manual dtype conversion if the from_pretrained didn't accept torch_dtype
                if torch_dtype and hasattr(model, 'to'):
                    try:
                        logger.info(f"🔧 DEBUG: Converting model to {torch_dtype} manually...")
                        model = model.to(dtype=torch_dtype)
                        logger.info(f"✅ Successfully converted model to {torch_dtype}")
                    except Exception as e:
                        logger.warning(f"⚠️ Manual dtype conversion failed: {e}, staying with fp32")

                # ==== OLD CODE (BACKUP - UNCOMMENT TO ROLLBACK) ====
                # # Try to pass attn_implementation – fallback gracefully if the
                # # model signature does not yet support the kwarg.
                # try:
                #     # Use "eager" attention implementation to silence the warning
                #     model = ModelClass.from_pretrained(
                #         device=device, attn_implementation="eager"
                #     )
                # except TypeError:
                #     # Older library version – ignore kwarg and log info
                #     logger.debug(
                #         f"{model_name}.from_pretrained() does not accept attn_implementation – falling back without it"
                #     )
                #     model = ModelClass.from_pretrained(device=device)

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
