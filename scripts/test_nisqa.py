#!/usr/bin/env python3
"""
Simple NISQA test script via VERSA.

Computes non-intrusive MOS using versa.utterance_metrics.nisqa.
Expected scale: typically 1..5 (depends on checkpoint). Good > 3.5.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict

import numpy as np
import soundfile as sf


def load_wave_mono(path: str) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=-1)
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    return data, int(sr)


def setup_nisqa(use_gpu: bool) -> tuple[object, Dict[str, int]]:
    from versa.utterance_metrics import nisqa as ns  # lazy import
    # Model setup returns a model object; NISQA works at 16 kHz typically
    model = ns.nisqa_model_setup(nisqa_model_path=None, use_gpu=use_gpu)
    predictor_fs = {"nisqa": 16000}
    return model, predictor_fs


def compute_nisqa(wav: np.ndarray, sr: int, model: object, predictor_fs: Dict[str, int]) -> float | None:
    from versa.utterance_metrics import nisqa as ns
    try:
        score = ns.nisqa_metric(model, wav, sr)
        # score is expected to be float in [1..5]
        return float(score) if score is not None else None
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="NISQA test via VERSA")
    parser.add_argument("--wav", required=True, type=str, help="Path to WAV file")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    # Device
    use_gpu = False
    if args.device == "cuda":
        try:
            import torch  # noqa: F401
            import torch.cuda as _cuda
            use_gpu = _cuda.is_available()
        except Exception:
            use_gpu = False
    elif args.device == "auto":
        try:
            import torch.cuda as _cuda  # type: ignore
            use_gpu = _cuda.is_available()
        except Exception:
            use_gpu = False

    wav, sr = load_wave_mono(args.wav)
    model, predictor_fs = setup_nisqa(use_gpu)

    results: list[float] = []
    for _ in range(max(1, int(args.repeats))):
        mos = compute_nisqa(wav, sr, model, predictor_fs)
        if mos is None:
            print(json.dumps({"mos": None, "message": "NISQA scoring failed"}, ensure_ascii=False))
            return 1
        results.append(mos)

    import numpy as np

    summary = {
        "predictor": "nisqa",
        "device": ("cuda" if use_gpu else "cpu"),
        "wav": args.wav,
        "mos_values": [round(v, 4) for v in results],
        "mos_mean": round(float(np.mean(results)), 4),
        "mos_min": round(float(np.min(results)), 4),
        "mos_max": round(float(np.max(results)), 4),
        "deterministic": bool(np.max(results) - np.min(results) < 1e-4),
        "scale_hint": "Expected ~1..5 (good > 3.5)",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
