"""
Model cache system for ChatterboxTTS to avoid repeated model loading.
Implements singleton pattern with device-specific caching.

Note: Cache miss on new process start is normal behavior. 
The cache only works within a single program run.
"""

import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from huggingface_hub import hf_hub_download

import torch

logger = logging.getLogger(__name__)

_VIBEVOICE_TYPES = frozenset({"vibevoice", "vibevoice_1_5b", "vibevoice_q4"})


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

    _MODEL_DISPLAY_NAMES: Dict[str, str] = {
        "standard": "ChatterboxTTS",
        "multilingual": "ChatterboxMultilingualTTS",
        "turbo": "ChatterboxTurboTTS",
        "qwen3": "Qwen3TTSModel (1.7B Base)",
        "vibevoice": "VibeVoice-Large-Q8",
        "vibevoice_1_5b": "VibeVoice-1.5B",
        "vibevoice_q4": "VibeVoice-Large-Q4",
    }

    @classmethod
    def get_model(cls, device: str = "auto", model_type: str = "standard", config: Optional[Dict] = None):
        """
        Get a cached TTS model for the specified device and model type.

        Args:
            device: Target device ("auto", "cuda", "mps", "cpu")
            model_type: Model type ("standard", "multilingual", "turbo", "qwen3",
                "vibevoice", "vibevoice_1_5b", or "vibevoice_q4")
            config: Optional full pipeline config dict (used for model-specific load options)

        Returns:
            Model instance (ChatterboxTTS, ChatterboxMultilingualTTS, ChatterboxTurboTTS,
            Qwen3TTSModel, or a VibeVoice model) or None on failure.
        """
        # Resolve auto device
        actual_device = cls._detect_device() if device == "auto" else device
        cache_key = f"{actual_device}_{model_type}"
        if model_type in _VIBEVOICE_TYPES:
            vv_attn_eff = cls._vibevoice_effective_attn(
                model_type, actual_device, config
            )
            if vv_attn_eff:
                cache_key = f"{actual_device}_{model_type}_{vv_attn_eff}"

        # Check in-memory cache first
        if cache_key in cls._model_cache:
            load_time = cls._load_times.get(cache_key, 0)
            model_name = cls._MODEL_DISPLAY_NAMES.get(model_type, model_type)
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

        # Switching between VibeVoice variants: evict the old one before loading the new one.
        # All three variants are large models (1.5B–7B). Keeping the previous variant in VRAM
        # while loading the next causes memory pressure and extremely slow loading.
        if model_type in _VIBEVOICE_TYPES:
            cls._evict_vibevoice(actual_device, except_key=cache_key)

        # Load fresh model
        model_name = cls._MODEL_DISPLAY_NAMES.get(model_type, model_type)
        logger.info(
            f"🔄 Loading {model_name} model for device: {actual_device} (cache miss)"
        )
        
        # Track loading time
        start_time = time.time()
        model = cls._load_fresh_model(actual_device, model_type, config=config)
        load_time = time.time() - start_time
        
        # OPTIMIZE FRESHLY LOADED MODEL
        optimized_model = cls._optimize_cached_model(model, actual_device)
        if optimized_model is not model:
            logger.info("🚀 Model optimized for speed")
            model = optimized_model
        
        # Cache the optimized model and loading time
        cls._model_cache[cache_key] = model
        if model_type in _VIBEVOICE_TYPES and model is not None and hasattr(model, "_vv_processor"):
            cls._model_cache[f"{cache_key}_processor"] = model._vv_processor
        cls._load_times[cache_key] = load_time
        
        logger.info(f"✅ Model loaded in {load_time:.1f}s and cached for future use in this session")

        return model

    # Per-model attention defaults (None = transformers auto-detect, fastest original state).
    # vibevoice (7B Q8): None → auto-detect (was fastest before explicit overrides).
    # vibevoice_1_5b: sdpa → measured 3x speedup vs auto-detected FA2 at 1.5B scale.
    # vibevoice_q4 (7B Q4): sdpa → 4-bit Linear4bit + FA2 is unreliable.
    _VIBEVOICE_ATTN_DEFAULTS: Dict[str, Optional[str]] = {
        "vibevoice": None,
        "vibevoice_1_5b": "sdpa",
        "vibevoice_q4": "sdpa",
    }

    @classmethod
    def _vibevoice_effective_attn(
        cls, model_type: str, device: str, config: Optional[Dict]
    ) -> Optional[str]:
        """Attention backend for VibeVoice – per-model defaults, overridable via config."""
        tester = (config or {}).get("chatterbox_tester") or {}
        raw = tester.get("vibevoice_attn_implementation")
        if raw:
            # Explicit override from tester/env
            vv: Optional[str] = str(raw).strip().lower()
            if vv == "sdpa":
                vv = None
            elif vv not in ("flash_attention_2", "sdpa", "eager"):
                logger.warning(
                    "Invalid vibevoice_attn_implementation %r; falling back to per-model default",
                    raw,
                )
                vv = cls._VIBEVOICE_ATTN_DEFAULTS.get(model_type)
        else:
            # Use per-model default (None = let transformers auto-detect)
            vv = cls._VIBEVOICE_ATTN_DEFAULTS.get(model_type)

        if vv == "flash_attention_2":
            if device != "cuda":
                logger.warning(
                    "VibeVoice: flash_attention_2 requires CUDA; using sdpa"
                )
                vv = "sdpa"
            else:
                try:
                    import flash_attn  # noqa: F401
                except ImportError:
                    logger.warning(
                        "VibeVoice: flash_attn not importable; using sdpa"
                    )
                    vv = "sdpa"
        if model_type == "vibevoice_q4" and vv == "flash_attention_2":
            logger.warning(
                "VibeVoice Q4: flash_attention_2 often incompatible with 4-bit; using sdpa"
            )
            vv = "sdpa"
        return vv

    @classmethod
    def evict_vibevoice(cls, device: str) -> None:
        """Public wrapper: evict all cached VibeVoice models for ``device``.

        Intended for callers that switch from a VibeVoice variant to a non-VV
        model (e.g. Qwen3) and need to free VRAM before the new model loads.
        """
        cls._evict_vibevoice(device, except_key="")

    @classmethod
    def _evict_vibevoice(cls, device: str, except_key: str) -> None:
        """Remove all cached VibeVoice models except the one about to be loaded.

        VibeVoice variants are mutually exclusive in practice (1.5B + 7B Q8 + 7B Q4 together
        would consume 15+ GB VRAM). Evicting before loading the new variant prevents OOM and
        the extremely slow pseudo-loading that happens when CUDA starts swapping to system RAM.
        """
        keys_to_evict = [
            k for k in list(cls._model_cache.keys())
            if k != except_key and any(f"{device}_{vt}" in k for vt in _VIBEVOICE_TYPES)
        ]
        if not keys_to_evict:
            return
        for k in keys_to_evict:
            model_obj = cls._model_cache.pop(k, None)
            cls._load_times.pop(k, None)
            # Move weights off CUDA so the allocator can reclaim the memory immediately
            if model_obj is not None:
                try:
                    model_obj.to("cpu")
                except Exception:
                    pass
                del model_obj
            logger.info(f"🗑️ Evicted VibeVoice cache entry '{k}' to free VRAM before loading new variant")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            free_gb = torch.cuda.mem_get_info()[0] / 1024 ** 3
            logger.info(f"🧹 CUDA cache cleared – {free_gb:.1f} GB free VRAM")

    @classmethod
    def free_vram(cls, device: Optional[str] = None) -> None:
        """Evict all cached TTS models (any type) from VRAM.

        Called at stage boundaries (generation → validation) so the ASR/MOS
        stack can load without competing for VRAM on consumer GPUs. A plain
        ``torch.cuda.empty_cache()`` only releases allocator-cached-but-free
        memory; models still referenced by ``_model_cache`` stay resident and
        starve the next stage. This method removes the references, moves
        weights to CPU so Python's GC can reclaim them, and forces a CUDA
        cache flush.

        If ``device`` is given, only entries for that device are evicted; else
        all entries are evicted regardless of device.
        """
        device_norm = device or ""
        keys_to_evict = [
            k for k in list(cls._model_cache.keys())
            if not device_norm or k.startswith(f"{device_norm}_")
        ]
        if not keys_to_evict:
            return
        for k in keys_to_evict:
            model_obj = cls._model_cache.pop(k, None)
            cls._load_times.pop(k, None)
            # Processor entries are cheap handles; skip the ``.to("cpu")`` call.
            is_processor_entry = k.endswith("_processor")
            if model_obj is not None and not is_processor_entry:
                try:
                    model_obj.to("cpu")
                except Exception:
                    pass
            del model_obj
            logger.info(f"🗑️ Evicted TTS cache entry '{k}' to free VRAM for validation stage")
        # Force a Python GC pass so objects referenced only by cycles (or
        # just-popped locals) are finalized before we ask CUDA for its memory
        # back. Without this the allocator often reports the same "free" GB
        # as before because the tensors still have live refcounts.
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            free_b, total_b = torch.cuda.mem_get_info()
            free_gb = free_b / 1024 ** 3
            total_gb = total_b / 1024 ** 3
            allocated_gb = torch.cuda.memory_allocated() / 1024 ** 3
            reserved_gb = torch.cuda.memory_reserved() / 1024 ** 3
            logger.info(
                f"🧹 CUDA cache cleared – {free_gb:.1f}/{total_gb:.1f} GB free "
                f"(allocated={allocated_gb:.1f} GB, reserved={reserved_gb:.1f} GB)"
            )

    @classmethod
    def _optimize_cached_model(cls, model, device: str):
        """Return model as-is for standard ChatterboxTTS (no optimization needed)."""
        # Standard ChatterboxTTS models are already optimized
        return model

    @classmethod
    def _apply_chatterbox_transformers_compat(cls, model: Any) -> None:
        """Align Chatterbox T3 with transformers>=4.57 (qwen-tts stack).

        chatterbox-tts 0.1.3 loads Llama with ``attn_implementation="sdpa"`` but enables
        ``output_attentions`` for alignment analysis. transformers 4.57+ raises
        ValueError for that combination; eager attention is required (upstream used to
        warn only). Qwen3/VibeVoice sdpa defaults are unchanged.
        """
        try:
            t3 = getattr(model, "t3", None)
            tfmr = getattr(t3, "tfmr", None) if t3 is not None else None
            if tfmr is None:
                return
            cfg = tfmr.config
            current = getattr(cfg, "_attn_implementation", None) or getattr(
                cfg, "attn_implementation", None
            )
            if current and str(current).lower() == "eager":
                return
            if hasattr(cfg, "_attn_implementation"):
                cfg._attn_implementation = "eager"
            if hasattr(cfg, "attn_implementation"):
                cfg.attn_implementation = "eager"
            logger.info(
                "ℹ️ Chatterbox: T3 Llama attention %r → 'eager' "
                "(output_attentions alignment; transformers>=4.57 + sdpa incompatible)",
                current,
            )
        except Exception as e:
            logger.warning("Chatterbox transformers compat patch failed: %s", e)

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
    def _load_fresh_model(cls, device: str, model_type: str = "standard", config: Optional[Dict] = None):
        """Load a fresh TTS model instance for the requested model_type."""
        model_name = cls._MODEL_DISPLAY_NAMES.get(model_type, model_type)
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

                device_obj = torch.device(device)

                if model_type == "multilingual":
                    try:
                        from chatterbox.mtl_tts import ChatterboxMultilingualTTS as ModelClass
                        model_name = "ChatterboxMultilingualTTS"
                    except ImportError:
                        logger.warning(
                            "ChatterboxMultilingualTTS not available, falling back to standard ChatterboxTTS"
                        )
                        from chatterbox.tts import ChatterboxTTS as ModelClass
                        model_name = "ChatterboxTTS (multilingual fallback)"
                    model = ModelClass.from_pretrained(device=device_obj)

                elif model_type == "turbo":
                    try:
                        from chatterbox.tts_turbo import ChatterboxTurboTTS as ModelClass
                    except ImportError as exc:
                        logger.warning(
                            f"ChatterboxTurboTTS not available ({exc}), falling back to standard ChatterboxTTS"
                        )
                        from chatterbox.tts import ChatterboxTTS as ModelClass
                        model_name = "ChatterboxTTS (turbo fallback)"
                    model = ModelClass.from_pretrained(device=device_obj)

                elif model_type == "qwen3":
                    import os
                    from qwen_tts import Qwen3TTSModel
                    # Attention implementation: sdpa by default.
                    # flash_attention_2 is theoretically supported but measured slower than sdpa
                    # for typical TTS workloads (batch=1, short sequences) due to kernel overhead.
                    # Can be overridden via env var or the hidden config key
                    # generation.qwen3_attn_impl (sdpa | flash_attention_2 | eager).
                    cfg_override = str(
                        (config or {}).get("generation", {}).get("qwen3_attn_impl", "")
                    ).strip().lower()
                    env_override = os.environ.get("QWEN3_ATTN_IMPL", "").strip().lower()
                    explicit = env_override or cfg_override
                    if explicit in ("sdpa", "eager", "flash_attention_2"):
                        attn_impl = explicit
                        source = "QWEN3_ATTN_IMPL env" if env_override else "config qwen3_attn_impl"
                        logger.info(f"ℹ️ Qwen3: attention implementation forced to '{attn_impl}' (via {source})")
                    else:
                        attn_impl = "sdpa"
                        logger.info("ℹ️ Qwen3: using sdpa attention")
                    model = Qwen3TTSModel.from_pretrained(
                        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        device_map=device,
                        dtype=torch.bfloat16,
                        attn_implementation=attn_impl,
                    )
                elif model_type in _VIBEVOICE_TYPES:
                    # VibeVoice is vendored locally under src/third_party/vibevoice
                    vibevoice_path = (
                        Path(__file__).resolve().parent.parent / "third_party" / "vibevoice"
                    )
                    if not vibevoice_path.exists():
                        raise RuntimeError(
                            f"Local VibeVoice code not found at: {vibevoice_path}"
                        )
                    vibevoice_path_str = str(vibevoice_path)
                    if vibevoice_path_str not in sys.path:
                        sys.path.insert(0, vibevoice_path_str)

                    from modular.modeling_vibevoice_inference import (
                        VibeVoiceForConditionalGenerationInference,
                    )
                    from processor.vibevoice_processor import VibeVoiceProcessor

                    # HF repo id is always "namespace/name". Folder "4bit" inside the repo
                    # uses subfolder= (see https://huggingface.co/DevParker/VibeVoice7b-low-vram/tree/main/4bit).
                    model_specs: Dict[str, Tuple[str, Optional[str]]] = {
                        "vibevoice": ("FabioSarracino/VibeVoice-Large-Q8", None),
                        "vibevoice_1_5b": ("microsoft/VibeVoice-1.5B", None),
                        "vibevoice_q4": ("DevParker/VibeVoice7b-low-vram", "4bit"),
                    }
                    repo_id, hf_subfolder = model_specs[model_type]
                    load_kwargs: Dict[str, Any] = {
                        "device_map": device,
                        "torch_dtype": torch.bfloat16,
                    }
                    if hf_subfolder:
                        load_kwargs["subfolder"] = hf_subfolder
                    vv_attn = cls._vibevoice_effective_attn(
                        model_type, device, config
                    )
                    if vv_attn:
                        load_kwargs["attn_implementation"] = vv_attn
                    try:
                        model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                            repo_id,
                            **load_kwargs,
                        )
                    except Exception as e:
                        if load_kwargs.get("attn_implementation") == "flash_attention_2":
                            logger.warning(
                                "VibeVoice: flash_attention_2 failed (%s); retrying sdpa",
                                e,
                            )
                            load_kwargs["attn_implementation"] = "sdpa"
                            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                                repo_id,
                                **load_kwargs,
                            )
                        else:
                            raise
                    # Custom VibeVoiceProcessor only does os.path.join(repo_id, ...); it does not
                    # resolve Hub repos or subfolder. Passing subfolder in kwargs breaks the Qwen
                    # tokenizer load (None path). Resolve preprocessor_config.json via hf_hub_download
                    # and load from the on-disk folder (see DevParker/.../4bit).
                    if hf_subfolder:
                        preproc_file = hf_hub_download(
                            repo_id=repo_id,
                            filename="preprocessor_config.json",
                            subfolder=hf_subfolder,
                        )
                        processor = VibeVoiceProcessor.from_pretrained(
                            str(Path(preproc_file).parent)
                        )
                    else:
                        processor = VibeVoiceProcessor.from_pretrained(repo_id)
                    # Attach processor to model for easy downstream access.
                    model._vv_processor = processor

                else:
                    # "standard" and any unknown type
                    from chatterbox.tts import ChatterboxTTS as ModelClass
                    model = ModelClass.from_pretrained(device=device_obj)

            if model_type in ("standard", "multilingual", "turbo"):
                cls._apply_chatterbox_transformers_compat(model)

            logger.debug(f"{model_name} model loaded successfully for device: {device}")
            return model

        except Exception as e:
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
            if model_type == "qwen3":
                logger.error("   1. Install qwen-tts: pip install qwen-tts")
                logger.error(
                    "   2. Download model: huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base"
                )
            elif model_type == "turbo":
                logger.error(
                    "   1. Check ChatterboxTurboTTS availability in chatterbox-tts package"
                )
            else:
                logger.error(
                    "   1. Check ChatterboxTTS installation: pip install chatterbox-tts"
                )
                logger.error(
                    "   2. Check resemble-perth: pip install resemble-perth==1.0.1"
                )
            logger.error("   3. If issue persists, check GPU/CUDA compatibility")
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
        print("  • Use 'resume --all' to process all tasks in one session")
        print("  • Consider the cache miss as normal startup time")
        
        print("\n" + "=" * 60)
