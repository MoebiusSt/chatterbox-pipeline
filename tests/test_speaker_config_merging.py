#!/usr/bin/env python3
"""
Tests for per-speaker config cascade in ConfigManager (tts_params inheritance).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

from src.utils.config_manager import ConfigManager


def test_qwen3_tts_inherits_subtalker_and_topk_from_default_speaker() -> None:
    """
    When default_speaker S1 has full Qwen3 tts_params and S2 only overrides
    subtalker_temperature, S2 must inherit top_k, subtalker_top_k, subtalker_top_p, etc.
    """
    cm = ConfigManager(Path("/tmp"))
    job_config: Dict[str, Any] = {
        "generation": {
            "default_speaker": "S1",
            "model_type": "qwen3",
            "speakers": [
                {
                    "id": "S1",
                    "reference_audio": "s1.wav",
                    "language": "de",
                    "tts_params": {
                        "temperature": 1.1,
                        "temperature_max_deviation": 0.2,
                        "top_k": 120,
                        "top_k_max_deviation": 50,
                        "subtalker_temperature": 1.45,
                        "subtalker_temperature_max_deviation": -0.3,
                        "subtalker_top_k": 180,
                        "subtalker_top_k_max_deviation": 30,
                        "subtalker_top_p": 0.95,
                    },
                },
                {
                    "id": "S2",
                    "reference_audio": "s2.wav",
                    "language": "de",
                    "tts_params": {
                        "subtalker_temperature": 1.5,
                    },
                },
            ],
        }
    }
    base_config: Dict[str, Any] = {"generation": {"speakers": []}}
    s2 = copy.deepcopy(job_config["generation"]["speakers"][1])
    merged = cm._apply_cascading_inheritance(s2, job_config, base_config)
    tts = merged["tts_params"]

    assert tts["subtalker_temperature"] == 1.5
    assert tts["top_k"] == 120
    assert tts["top_k_max_deviation"] == 50
    assert tts["subtalker_top_k"] == 180
    assert tts["subtalker_top_k_max_deviation"] == 30
    assert tts["subtalker_top_p"] == 0.95
    assert tts["temperature"] == 1.1
    assert tts["temperature_max_deviation"] == 0.2
