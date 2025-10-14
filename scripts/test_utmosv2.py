#!/usr/bin/env python3
"""
Simple UTMOS/UTMOSv2 test script using VERSA (wavlab-speech/versa).

Given an input WAV file, this script computes non-intrusive MOS using
VERSA's pseudo_mos utilities. It prefers UTMOSv2 when available and
falls back to UTMOS otherwise. It also runs multiple repetitions on the
same input to verify deterministic results for UTMOSv2 (fixed chunking).

Usage:
  python scripts/test_utmosv2.py \
    --wav data/output/.../candidates/chunk_004/candidate_01.wav \
    --predictor auto \
    --device auto

Expected MOS scale: 1..5. Good quality typically > 3.5.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, Optional

import numpy as np
import soundfile as sf
import torch


def load_wave_mono(path: str) -> tuple[np.ndarray, int]:
    """Load audio as mono float32 array and return (waveform, sample_rate)."""
    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=-1)
    # Ensure float32
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    return data, int(sr)


def setup_predictor(predictor: str, use_gpu: bool) -> tuple[object, Dict[str, int], str]:
    """
    Setup VERSA pseudo_mos predictors.

    Returns (predictor_dict, predictor_fs, key) where key is 'utmosv2' or 'utmos'.
    """
    from versa.utterance_metrics import pseudo_mos as pm  # Lazy import

    if predictor == "utmosv2":
        predictor_types = ["utmosv2"]
        key = "utmosv2"
        try:
            # Ensure module is imported so VERSA can detect it
            import utmosv2  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                f"UTMOSv2 python package not importable: {e}. Please install sarulab UTMOSv2."
            )
        # Important: reload VERSA pseudo_mos after utmosv2 import so its
        # internal `utmosv2` reference is set (module caches import at load time)
        import importlib
        pm = importlib.reload(pm)
        # Workaround: some VERSA builds keep a module-level sentinel `utmosv2`
        # that remains None unless patched. Provide the imported module.
        try:
            import utmosv2 as _utmosv2  # noqa: F401
            setattr(pm, "utmosv2", _utmosv2)
        except Exception:
            pass
        try:
            predictor_dict, predictor_fs = pm.pseudo_mos_setup(
                predictor_types=predictor_types,
                predictor_args={},
                cache_dir="versa_cache",
                use_gpu=use_gpu,
            )
        except Exception as e:
            raise RuntimeError(
                f"UTMOSv2 setup failed inside VERSA: {e}. If reported as not installed, check VERSA version."
            )
        return predictor_dict, predictor_fs, key

    if predictor == "utmos":
        predictor_types = ["utmos"]
        key = "utmos"
        predictor_dict, predictor_fs = pm.pseudo_mos_setup(
            predictor_types=predictor_types,
            predictor_args={},
            cache_dir="versa_cache",
            use_gpu=use_gpu,
        )
        return predictor_dict, predictor_fs, key

    # auto: try utmosv2 then utmos
    try:
        return setup_predictor("utmosv2", use_gpu)
    except Exception:
        return setup_predictor("utmos", use_gpu)


def compute_mos(
    wav: np.ndarray,
    sr: int,
    predictor_dict: object,
    predictor_fs: Dict[str, int],
    key: str,
    use_gpu: bool,
) -> Optional[float]:
    from versa.utterance_metrics import pseudo_mos as pm  # Lazy import

    scores = pm.pseudo_mos_metric(
        pred=wav,
        fs=int(sr),
        predictor_dict=predictor_dict,
        predictor_fs=predictor_fs,
        use_gpu=use_gpu,
    )
    if not isinstance(scores, dict):
        return None
    if key in scores and scores[key] is not None:
        return float(scores[key])
    # Fallback for some configurations
    if key == "utmosv2" and "utmos" in scores and scores["utmos"] is not None:
        return float(scores["utmos"])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="UTMOS/UTMOSv2 test via VERSA")
    parser.add_argument("--wav", required=True, type=str, help="Path to WAV file")
    parser.add_argument(
        "--predictor",
        choices=["auto", "utmosv2", "utmos"],
        default="auto",
        help="Which predictor to use",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for model execution",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Repeat runs to verify determinism"
    )
    args = parser.parse_args()

    # Device selection
    if args.device == "cuda":
        use_gpu = torch.cuda.is_available()
    elif args.device == "cpu":
        use_gpu = False
    else:  # auto
        use_gpu = torch.cuda.is_available()

    # Load audio
    wav, sr = load_wave_mono(args.wav)

    # Setup predictor(s)
    predictor_dict, predictor_fs, key = setup_predictor(args.predictor, use_gpu)

    # Compute MOS multiple times
    results: list[float] = []
    for _ in range(max(1, int(args.repeats))):
        mos = compute_mos(wav, sr, predictor_dict, predictor_fs, key, use_gpu)
        if mos is None:
            print(json.dumps({
                "predictor": key,
                "mos": None,
                "message": "Scoring failed",
            }, ensure_ascii=False))
            return 1
        results.append(float(mos))

    # Summarize
    summary = {
        "predictor": key,
        "device": ("cuda" if use_gpu else "cpu"),
        "wav": args.wav,
        "mos_values": [round(v, 4) for v in results],
        "mos_mean": round(float(np.mean(results)), 4),
        "mos_min": round(float(np.min(results)), 4),
        "mos_max": round(float(np.max(results)), 4),
        "deterministic": bool(np.max(results) - np.min(results) < 1e-4),
        "scale_hint": "Expected 1..5 (good > 3.5)",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())


