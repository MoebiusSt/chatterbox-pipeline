"""
Model cache system for ChatterboxTTS to avoid repeated model loading.
Implements singleton pattern with device-specific caching and COMPLETE MODEL SERIALIZATION 
to prevent race conditions in parallel execution.
"""

import hashlib
import logging
import threading
import warnings
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


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
            logger.error(f"🚨 CRITICAL: Failed to load ChatterboxTTS model for device {device}: {e}")
            logger.error("=" * 80)
            logger.error("⚠️  WARNING: TTS MODEL LOADING FAILED!")
            logger.error("⚠️  The system will run in MOCK MODE and generate only NOISE/SILENCE!")
            logger.error("⚠️  Your final audio output will contain NO SPEECH!")
            logger.error("=" * 80)
            logger.error("💡 To fix this issue:")
            logger.error("   1. Check ChatterboxTTS installation: pip install chatterbox-tts")
            logger.error("   2. Check perth dependency: pip install perth")
            logger.error("   3. Update dependencies: pip install --upgrade chatterbox-tts perth")
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


class SerializedModelAccess:
    """
    COMPLETE MODEL SERIALIZATION: Ensures only ONE thread can access the ChatterboxTTS model 
    at any time, preventing all race conditions with model.conds and internal state.
    
    This is the nuclear option for thread-safety - complete serialization of model access.
    """
    
    _global_model_lock = threading.RLock()  # Reentrant lock for nested calls
    _current_conditionals_cache: Dict[str, Any] = {}  # Global conditionals cache
    
    @classmethod
    def with_exclusive_model_access(cls, model, reference_audio_path: str):
        """
        Context manager that provides EXCLUSIVE access to the ChatterboxTTS model.
        Completely serializes model access across all threads.
        
        Args:
            model: ChatterboxTTS model instance
            reference_audio_path: Path to reference audio for conditionals
            
        Returns:
            Context manager that ensures exclusive model access
        """
        return _ExclusiveModelContext(model, reference_audio_path)


class _ExclusiveModelContext:
    """Internal context manager for exclusive model access."""
    
    def __init__(self, model, reference_audio_path: str):
        self.model = model
        self.reference_audio_path = reference_audio_path
        self.thread_id = threading.current_thread().ident
        
    def __enter__(self):
        # Acquire the global model lock - ONLY ONE THREAD can proceed
        logger.debug(f"🔒 Thread {self.thread_id}: Acquiring EXCLUSIVE model access...")
        SerializedModelAccess._global_model_lock.acquire()
        
        try:
            # Check if we need to prepare conditionals
            cache_key = self._get_cache_key(self.reference_audio_path)
            
            if cache_key not in SerializedModelAccess._current_conditionals_cache:
                logger.debug(f"🔄 Thread {self.thread_id}: Preparing conditionals for {Path(self.reference_audio_path).name}")
                start_time = time.time()
                
                # Prepare conditionals - completely serialized
                self.model.prepare_conditionals(wav_fpath=self.reference_audio_path)
                
                # Cache the fact that conditionals are prepared
                SerializedModelAccess._current_conditionals_cache[cache_key] = {
                    'reference_audio': self.reference_audio_path,
                    'prepared_at': time.time(),
                    'thread_id': self.thread_id
                }
                
                elapsed = time.time() - start_time
                logger.debug(f"✅ Thread {self.thread_id}: Conditionals prepared in {elapsed:.2f}s")
            else:
                logger.debug(f"♻️ Thread {self.thread_id}: Using cached conditionals for {Path(self.reference_audio_path).name}")
            
            return self.model
            
        except Exception as e:
            # Release lock on error
            SerializedModelAccess._global_model_lock.release()
            logger.error(f"🚨 Thread {self.thread_id}: Error in exclusive model access: {e}")
            raise
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Release the global model lock
        logger.debug(f"🔓 Thread {self.thread_id}: Releasing EXCLUSIVE model access")
        SerializedModelAccess._global_model_lock.release()
        
        if exc_type is not None:
            logger.error(f"🚨 Thread {self.thread_id}: Exception in exclusive model context: {exc_val}")
    
    def _get_cache_key(self, file_path: str) -> str:
        """Generate cache key for conditionals."""
        try:
            path = Path(file_path)
            if not path.exists():
                return f"missing_{file_path}"
            
            # Use file size and modification time for cache key
            stat = path.stat()
            content = f"{path.name}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(content.encode()).hexdigest()[:12]
        except Exception as e:
            logger.warning(f"Could not generate cache key for {file_path}: {e}")
            return f"error_{file_path}"


class ConditionalCache:
    """
    DEPRECATED: Use SerializedModelAccess instead.
    
    Legacy conditional cache - replaced by complete model serialization.
    Kept for backward compatibility but should be replaced.
    """

    def __init__(self, model):
        """
        Initialize conditional cache (DEPRECATED - use SerializedModelAccess).
        """
        self.model = model
        logger.warning("⚠️ ConditionalCache is DEPRECATED. Use SerializedModelAccess for thread-safety.")

    def ensure_conditionals(self, reference_audio_path: str) -> bool:
        """
        DEPRECATED: Use SerializedModelAccess.with_exclusive_model_access() instead.
        
        This method is kept for backward compatibility but is NOT THREAD-SAFE.
        """
        logger.warning("⚠️ ensure_conditionals() is DEPRECATED and NOT THREAD-SAFE. Use SerializedModelAccess.")
        
        if self.model is None:
            logger.warning("Model not loaded - skipping conditional preparation")
            return False
            
        try:
            # Use the new serialized access for safety
            with SerializedModelAccess.with_exclusive_model_access(self.model, reference_audio_path):
                # Model is now prepared with correct conditionals
                return True
        except Exception as e:
            logger.error(f"Error preparing conditionals: {e}")
            raise

    def get_current_state(self) -> Dict[str, Any]:
        """Get current conditional state information."""
        return {
            "status": "DEPRECATED - Use SerializedModelAccess",
            "thread_id": threading.current_thread().ident,
        }

    def reset(self):
        """Reset conditional cache state."""
        logger.debug("ConditionalCache.reset() called (DEPRECATED)")
