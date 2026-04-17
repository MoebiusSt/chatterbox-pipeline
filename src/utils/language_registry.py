"""
Language and feature registry for all supported TTS model types.

Centralizes supported languages, model capabilities, and parameter sets so that
the pipeline can perform pre-flight checks and graceful parameter filtering without
hardcoding model constraints in multiple places.
"""

from typing import Any, Dict, List, Optional, Set


# VibeVoice Hub variants share languages, features, and TTS param keys (see model_cache).
_VIBEVOICE_LANGS: Set[str] = {"en", "zh"}
_VIBEVOICE_FEATURES: Dict = {
    "speaker_switch": True,
    "ref_text_required": False,
    "paralinguistic_tags": False,
    "builtin_normalization": False,
    "max_ref_audio_seconds": 60.0,
    "language_locked": False,
}
_VIBEVOICE_TTS_PARAMS: Set[str] = {
    "cfg_scale", "temperature", "top_p",
    "diffusion_steps", "voice_speed_factor", "use_sampling",
}

VIBEVOICE_MODEL_TYPES = frozenset({"vibevoice", "vibevoice_1_5b", "vibevoice_q4"})

# Supported language codes per model type (ISO 639-1)
MODEL_LANGUAGE_SUPPORT: Dict[str, Set[str]] = {
    "standard": {"en"},
    "turbo": {"en"},
    "multilingual": {
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi",
        "it", "ja", "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv",
        "sw", "tr", "zh",
    },
    "qwen3": {"zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"},
    "vibevoice": _VIBEVOICE_LANGS,
    "vibevoice_1_5b": _VIBEVOICE_LANGS,
    "vibevoice_q4": _VIBEVOICE_LANGS,
}

# Model feature flags
MODEL_FEATURES: Dict[str, Dict] = {
    "standard": {
        # Speaker switching (different reference audio + tts_params) works for all models.
        # The restriction for standard/turbo is language-only: both are EN-only, so a speaker
        # with language != "en" triggers a language-validation WARNING (not a speaker-switch block).
        "speaker_switch": True,
        "ref_text_required": False,
        "paralinguistic_tags": False,
        "builtin_normalization": False,
        "max_ref_audio_seconds": None,
        "language_locked": True,      # Only "en" supported; language check warns if other langs used
    },
    "multilingual": {
        "speaker_switch": True,
        "ref_text_required": False,
        "paralinguistic_tags": False,
        "builtin_normalization": False,
        "max_ref_audio_seconds": None,
        "language_locked": False,
    },
    "turbo": {
        # Speaker switching fully supported (different voices, different tts_params).
        # Turbo is EN-only: language-validation WARNING fired for non-EN speakers.
        "speaker_switch": True,
        "ref_text_required": False,
        "paralinguistic_tags": True,   # [laugh], [cough], etc.
        "builtin_normalization": True,  # norm_loudness param
        "max_ref_audio_seconds": None,
        "language_locked": True,       # Only "en" supported; language check warns if other langs used
    },
    "qwen3": {
        "speaker_switch": True,
        "ref_text_required": True,   # Needed for ICL mode (full quality)
        "paralinguistic_tags": False,
        "builtin_normalization": False,
        "max_ref_audio_seconds": 3.0,  # Recommended max for voice cloning
        "language_locked": False,
    },
    "vibevoice": _VIBEVOICE_FEATURES,
    "vibevoice_1_5b": _VIBEVOICE_FEATURES,
    "vibevoice_q4": _VIBEVOICE_FEATURES,
}

# TTS parameters supported per model type (keys that may be passed to model.generate / generate_voice_clone)
# Deviation keys and control flags are always consumed internally and never forwarded.
SUPPORTED_TTS_PARAMS: Dict[str, Set[str]] = {
    "standard": {
        "exaggeration", "cfg_weight", "temperature",
        "min_p", "top_p", "repetition_penalty",
    },
    "multilingual": {
        "exaggeration", "cfg_weight", "temperature",
        "min_p", "top_p", "repetition_penalty",
    },
    "turbo": {
        "exaggeration", "cfg_weight", "temperature",
        "min_p", "top_p", "repetition_penalty",
        "top_k", "norm_loudness",
    },
    "qwen3": {
        "temperature", "top_k", "top_p", "repetition_penalty",
        "max_new_tokens", "do_sample",
        "subtalker_dosample", "subtalker_top_k", "subtalker_top_p", "subtalker_temperature",
    },
    "vibevoice": _VIBEVOICE_TTS_PARAMS,
    "vibevoice_1_5b": _VIBEVOICE_TTS_PARAMS,
    "vibevoice_q4": _VIBEVOICE_TTS_PARAMS,
}

# Keys that are always consumed by the pipeline internally and never forwarded to models
INTERNAL_PARAM_KEYS: Set[str] = {
    "exaggeration_max_deviation",
    "cfg_weight_max_deviation",
    "temperature_max_deviation",
    "top_k_max_deviation",
    "subtalker_temperature_max_deviation",
    "subtalker_top_k_max_deviation",
    "cfg_scale_max_deviation",
    "diffusion_steps_max_deviation",
    "enabled",
    "type",
    "seed",
}

# Qwen3 language code to full language name (as expected by the Qwen3 model API)
QWEN3_LANGUAGE_CODE_TO_NAME: Dict[str, str] = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "pt": "Portuguese",
    "es": "Spanish",
    "it": "Italian",
}

# VibeVoice language code to full language name
VIBEVOICE_LANGUAGE_CODE_TO_NAME: Dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
}

# All known model type identifiers
KNOWN_MODEL_TYPES: Set[str] = set(MODEL_LANGUAGE_SUPPORT.keys())


def get_supported_languages(model_type: str) -> Set[str]:
    """Return the set of supported language codes for a model type."""
    return MODEL_LANGUAGE_SUPPORT.get(model_type, set())


def get_model_features(model_type: str) -> Dict:
    """Return the feature flags for a model type."""
    return MODEL_FEATURES.get(model_type, {})


def get_supported_tts_params(model_type: str) -> Set[str]:
    """Return the set of tts_param keys that this model type actually accepts."""
    return SUPPORTED_TTS_PARAMS.get(model_type, set())


def validate_languages_for_model(
    model_type: str,
    used_language_codes: Set[str],
) -> List[str]:
    """
    Check whether all used language codes are supported by the given model type.

    Args:
        model_type: The TTS model type (standard, multilingual, turbo, qwen3, vibevoice, …).
        used_language_codes: Set of ISO 639-1 codes that will actually be used.

    Returns:
        List of human-readable warning strings (empty if all languages are supported).
    """
    supported = get_supported_languages(model_type)
    if not supported:
        # Unknown model type - skip validation
        return []

    warnings: List[str] = []
    for code in sorted(used_language_codes):
        if code and code not in supported:
            warnings.append(
                f"Language '{code}' is not supported by model type '{model_type}'. "
                f"Supported languages: {sorted(supported)}. "
                f"Generation may fail or produce incorrect output."
            )
    return warnings


def filter_params_for_model(
    model_type: str,
    params: Dict,
    logged_keys: Optional[Set[str]] = None,
) -> Dict:
    """
    Filter a tts_params dict to only include keys supported by the given model type.

    Unsupported keys are dropped with a one-time info log. Internal deviation/control
    keys are silently dropped regardless of model type.

    Args:
        model_type: The TTS model type.
        params: Raw parameter dict (may contain unsupported keys).
        logged_keys: Mutable set used to deduplicate info messages across calls.
                     Pass the same set object to suppress repeated messages.

    Returns:
        Filtered parameter dict containing only supported keys.
    """
    import logging
    log = logging.getLogger(__name__)

    supported = get_supported_tts_params(model_type)
    filtered: Dict = {}

    for key, value in params.items():
        if key in INTERNAL_PARAM_KEYS:
            # Always silently consumed - never forwarded
            continue
        if key in supported:
            filtered[key] = value
        else:
            # Log once per (model_type, key) combination
            log_key = f"{model_type}:{key}"
            if logged_keys is None or log_key not in logged_keys:
                log.info(
                    "Skipping unsupported tts_param '%s' for model type '%s'",
                    key, model_type,
                )
                if logged_keys is not None:
                    logged_keys.add(log_key)

    return filtered


def _fmt(value: Any, spec: str = ".2f") -> str:
    """Safe numeric formatter – falls back to str() for non-numerics/None."""
    if value is None:
        return "-"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def format_generation_params_summary(
    params: Optional[Dict[str, Any]],
    model_type: Optional[str] = None,
) -> str:
    """Render a compact, model-aware ``key=value`` summary of generation
    parameters for best-candidate log lines.

    Each TTS backend exposes a different parameter surface (Chatterbox:
    ``exaggeration``/``cfg_weight``/``temperature``; VibeVoice: ``cfg_scale``/
    ``diffusion_steps``; Qwen3: ``subtalker_*``), so a single hard-coded
    format loses information for two of three models. When ``model_type`` is
    omitted the function falls back to key-based heuristics so any call site
    without direct model context still produces a sensible summary.
    """
    p = params or {}

    if not model_type:
        if "cfg_scale" in p or "diffusion_steps" in p:
            model_type = "vibevoice"
        elif (
            "subtalker_temperature" in p or "subtalker_top_k" in p
            or "subtalker_top_p" in p or "subtalker_dosample" in p
        ):
            model_type = "qwen3"
        elif "exaggeration" in p or "cfg_weight" in p:
            model_type = "standard"

    if model_type in VIBEVOICE_MODEL_TYPES:
        return (
            f"cfg_scale: {_fmt(p.get('cfg_scale'), '.3f')}, "
            f"temperature: {_fmt(p.get('temperature'), '.2f')}, "
            f"top_p: {_fmt(p.get('top_p'), '.2f')}, "
            f"steps: {_fmt(p.get('diffusion_steps'), '.0f')}"
        )
    if model_type == "qwen3":
        return (
            f"temperature: {_fmt(p.get('temperature'), '.2f')}, "
            f"top_p: {_fmt(p.get('top_p'), '.2f')}, "
            f"top_k: {_fmt(p.get('top_k'), '.0f')}, "
            f"sub_temp: {_fmt(p.get('subtalker_temperature'), '.2f')}, "
            f"sub_top_k: {_fmt(p.get('subtalker_top_k'), '.0f')}"
        )
    # Default: Chatterbox (standard/multilingual/turbo) or unknown
    return (
        f"exaggeration: {_fmt(p.get('exaggeration'), '.2f')}, "
        f"cfg_weight: {_fmt(p.get('cfg_weight'), '.2f')}, "
        f"temperature: {_fmt(p.get('temperature'), '.2f')}, "
        f"min_p: {_fmt(p.get('min_p'), '.2f')}, "
        f"top_p: {_fmt(p.get('top_p'), '.2f')}"
    )
