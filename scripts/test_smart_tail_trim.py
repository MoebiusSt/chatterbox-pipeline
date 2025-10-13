#!/usr/bin/env python3
"""
Quick test utility for Smart Tail-Trim using WhisperX alignment.

Usage:
  python scripts/test_smart_tail_trim.py \
      --audio data/output/.../candidates/chunk_001/candidate_01.wav \
      --text "Einleitung Ende" \
      --language de \
      --out-dir /tmp/trimtest

Note:
  - Saves kept (trimmed) waveform and removed tail for inspection.
  - Does not require the full pipeline; uses TailTrimmer directly.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torchaudio as ta

# Ensure project src/ is on sys.path for direct script execution
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from validation.preprocessors.tail_trimmer import TailTrimmer  # type: ignore


def build_minimal_config(args: argparse.Namespace) -> Dict[str, Any]:
    # Minimal config enabling smart_match; sample_rate taken from the audio file
    return {
        "audio": {
            "sample_rate": int(args.sample_rate) if args.sample_rate else 24000,
        },
        "validation": {
            "preprocessing": {
                "tail_trim": {
                    "enabled": True,
                    "lookback_seconds": 6.0,
                    "post_speech_silence_ms": args.post_silence_ms,
                    "fade_out_ms": args.fade_out_ms,
                    "prefer_whisperx": True,
                    "whisperx_device": args.device,
                    "debug_save_removed_tail": True,
                    "persist_trimmed_candidate": True,
                    "smart_match": {
                        "enabled": True,
                        "last_n_words": args.last_n_words,
                        "fuzzy_ratio": args.fuzzy_ratio,
                        "search_window_words": args.search_window_words,
                        "language_gate": args.language_gate,
                    },
                }
            }
        },
    }


def load_audio(path: Path) -> Tuple[torch.Tensor, int]:
    wave, sr = ta.load(str(path))
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)
    return wave.squeeze(0), int(sr)


def save_audio(path: Path, audio: torch.Tensor, sr: int) -> None:
    audio = audio.detach().cpu()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    ta.save(str(path), audio, sr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Smart Tail-Trim (WhisperX)")
    parser.add_argument("--audio", type=str, required=True, help="Path to WAV file")
    parser.add_argument("--text", type=str, required=True, help="Original chunk text")
    parser.add_argument("--language", type=str, default="en", help="Language code (e.g., en, de)")
    parser.add_argument("--out-dir", type=str, default="./trim_debug", help="Output directory for artifacts")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for WhisperX")
    parser.add_argument("--last-n-words", dest="last_n_words", type=int, default=2)
    parser.add_argument("--fuzzy-ratio", dest="fuzzy_ratio", type=float, default=0.85)
    parser.add_argument("--search-window-words", dest="search_window_words", type=int, default=30)
    parser.add_argument("--language-gate", dest="language_gate", nargs="*", default=["en", "de", "fr", "es", "it"]) 
    parser.add_argument("--post-silence-ms", dest="post_silence_ms", type=int, default=120, help="Post-speech trailing silence to keep (ms)")
    parser.add_argument("--fade-out-ms", dest="fade_out_ms", type=int, default=120)
    parser.add_argument("--sample-rate", dest="sample_rate", type=int, default=None, help="Override sample rate for saving")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = load_audio(audio_path)
    cfg = build_minimal_config(args)
    cfg["audio"]["sample_rate"] = cfg["audio"]["sample_rate"] or sr

    trimmer = TailTrimmer(config=cfg, sample_rate=sr)

    trimmed, removed, meta = trimmer.trim(audio, language=args.language, original_text=args.text)

    # Persist
    if trimmed is not None and trimmed.numel() > 0:
        save_audio(out_dir / "kept_trimmed.wav", trimmed, sr)
    if removed is not None and removed.numel() > 0:
        save_audio(out_dir / "removed_tail.wav", removed, sr)

    with open(out_dir / "tail_trim_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Tail-Trim meta:", json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


