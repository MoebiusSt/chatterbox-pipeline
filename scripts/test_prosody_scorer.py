#!/usr/bin/env python3
import json
import torch

from pathlib import Path

from utils.config_manager import ConfigManager
from validation.preprocessors.tail_trimmer import TailTrimmer
from validation.mos_providers.utmos import UTMOSProvider
from validation.prosody_scorer import ProsodyScorer


def generate_dummy_speech(sr: int = 24000, seconds: float = 2.0) -> torch.Tensor:
    t = torch.linspace(0, seconds, int(sr * seconds))
    # Simple voiced-like tone with fade to silence
    x = 0.05 * torch.sin(2.0 * 3.14159265 * 200.0 * t)
    # Add trailing silence
    pad = torch.zeros(int(sr * 0.8))
    return torch.cat([x, pad], dim=0)


def main():
    project_root = Path(__file__).resolve().parents[1]
    cfg = ConfigManager(project_root).default_config
    sr = cfg.get("audio", {}).get("sample_rate", 24000)

    audio = generate_dummy_speech(sr, 2.0)

    # Tail trim
    trimmer = TailTrimmer(cfg, sample_rate=sr)
    trimmed, meta = trimmer.trim(audio, language="de", original_text="Das ist ein Test.")

    # Prosody scorer
    mos_provider = UTMOSProvider(enabled_languages=["de", "en"])  # may be None if versa missing
    scorer = ProsodyScorer(cfg, mos_provider=mos_provider, sample_rate=sr)
    details = scorer.score(trimmed, language="de", original_text="Das ist ein Test.", asr_transcription="Das ist ein Test.")

    print(json.dumps({"tail_trim": meta, "prosody": details}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
