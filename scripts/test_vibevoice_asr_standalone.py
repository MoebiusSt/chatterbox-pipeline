"""Standalone benchmark for VibeVoice-ASR on a single WAV file.

Loads :class:`VibeVoiceASRBackend` in isolation (no TTS, no pipeline) against
a virgin CUDA context so we can measure pure load/inference latency.

Usage
-----
    source venv/bin/activate
    python scripts/test_vibevoice_asr_standalone.py [path/to/audio.wav]

Without an argument the script defaults to the 4-minute sample at
``data/output/vibevoice-example/input-document_20260416_204120/final/input-document_final.wav``.

The script prints timing for:
  - backend construction
  - model + processor loading (first call to ``_ensure_loaded``)
  - audio decode + resample
  - ``transcribe_with_alignment`` (generation)
and a VRAM snapshot after each step. A transcription preview is dumped at the
end so the caller can sanity-check output quality.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_WAV = (
    REPO_ROOT
    / "data/output/vibevoice-example/input-document_20260416_204120/"
    / "final/input-document_final.wav"
)


def _vram_snapshot(label: str) -> None:
    try:
        import torch

        if not torch.cuda.is_available():
            print(f"[{label}] CUDA not available")
            return
        free_b, total_b = torch.cuda.mem_get_info()
        alloc = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        free_gb = free_b / 1024 ** 3
        total_gb = total_b / 1024 ** 3
        print(
            f"[{label}] VRAM free={free_gb:.2f}/{total_gb:.2f} GB  "
            f"allocated={alloc:.2f} GB  reserved={reserved:.2f} GB"
        )
    except Exception as e:
        print(f"[{label}] VRAM snapshot failed: {e}")


def _load_audio(wav_path: Path):
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(wav_path))
    duration_s = waveform.shape[-1] / float(sample_rate)
    return waveform, sample_rate, duration_s


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "wav",
        nargs="?",
        default=str(DEFAULT_WAV),
        help="Path to audio file to transcribe",
    )
    parser.add_argument(
        "--attn",
        default="auto",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        help="Override attn_implementation (default: auto)",
    )
    parser.add_argument(
        "--language",
        default="de",
        help="Language hint passed to the backend (default: de)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32768,
        help="Cap for max_new_tokens (default: 32768)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable DEBUG logging"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    wav_path = Path(args.wav)
    if not wav_path.exists():
        print(f"ERROR: audio file not found: {wav_path}", file=sys.stderr)
        return 2

    print(f"Audio:  {wav_path}")
    _vram_snapshot("baseline")

    # ------------------------------------------------------------------
    # 1. Backend construction (cheap — just path/config checks)
    # ------------------------------------------------------------------
    from validation.asr.vibevoice_asr_backend import VibeVoiceASRBackend

    cfg = {
        "validation": {
            "vibevoice_asr": {
                "model_id": "microsoft/VibeVoice-ASR",
                "language_model": "Qwen/Qwen2.5-7B",
                "device": "auto",
                "attn_implementation": args.attn,
                "max_new_tokens": args.max_new_tokens,
                "target_sample_rate": 24000,
            }
        }
    }

    t0 = time.perf_counter()
    backend = VibeVoiceASRBackend(cfg)
    t_ctor = time.perf_counter() - t0
    print(f"[ctor]       {t_ctor*1000:.1f} ms")

    # ------------------------------------------------------------------
    # 2. Model + processor load (the big one: weights download/copy/to(device))
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    backend._ensure_loaded()
    t_load = time.perf_counter() - t0
    print(f"[load]       {t_load:.1f} s")
    _vram_snapshot("after-load")

    # ------------------------------------------------------------------
    # 3. Audio decode
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    waveform, sr, duration_s = _load_audio(wav_path)
    t_decode = time.perf_counter() - t0
    print(
        f"[decode]     {t_decode*1000:.1f} ms  (shape={tuple(waveform.shape)} "
        f"sr={sr} duration={duration_s:.1f} s)"
    )

    # ------------------------------------------------------------------
    # 4. Inference
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    result = backend.transcribe_with_alignment(
        audio=waveform, language=args.language, sample_rate=sr
    )
    t_infer = time.perf_counter() - t0
    rtf = t_infer / max(duration_s, 1e-6)
    print(f"[infer]      {t_infer:.1f} s  (RTF={rtf:.3f}x)")
    _vram_snapshot("after-infer")

    print()
    print("=" * 60)
    print("Transcription preview:")
    print("-" * 60)
    preview = (result.transcription or "").strip()
    if len(preview) > 800:
        preview = preview[:800] + "\n… [truncated]"
    print(preview or "[empty]")
    print("=" * 60)
    seg_count = 0
    if result.extra and isinstance(result.extra, dict):
        segs = result.extra.get("segments")
        if isinstance(segs, list):
            seg_count = len(segs)
    print(
        f"Backend={result.backend}  language={result.language}  "
        f"segments={seg_count}  words={len(result.words or [])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
