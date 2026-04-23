#!/usr/bin/env python3
import os
from pathlib import Path

import yaml

from src.utils.config_manager import ConfigManager


def test_identifier_sanitizer_handles_yaml_reserved_words(tmp_path: Path):
    # Create a minimal default_config.yaml
    project_root = tmp_path
    (project_root / "config").mkdir(parents=True, exist_ok=True)
    (project_root / "data" / "output").mkdir(parents=True, exist_ok=True)

    default_yaml = {
        "job": {"name": "default", "run_label": ""},
        "input": {"text_file": "input-document.txt"},
        "chunking": {},
        "generation": {
            "default_speaker": "base",
            "speakers": [
                {
                    "id": "base",
                    "reference_audio": "voice.wav",
                    "language": "en",
                    "tts_params": {
                        "exaggeration": 0.4,
                        "cfg_weight": 0.2,
                        "temperature": 0.9,
                        "min_p": 0.03,
                        "top_p": 0.99,
                    },
                }
            ],
        },
        "validation": {},
        "audio": {},
    }
    with open(project_root / "config" / "default_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(default_yaml, f, sort_keys=False)

    # Create a job config that uses YAML reserved words for identifiers
    job_yaml = {
        "job": {"name": True, "run_label": None},  # will be coerced to "true" and ""
        "input": {"text_file": "input-document.txt"},
        "generation": {
            "default_speaker": False,  # will be coerced to "false"
            "speakers": [
                {"id": False, "reference_audio": "ra.wav", "language": True, "tts_params": {"exaggeration": 0.4, "cfg_weight": 0.2, "temperature": 0.9, "min_p": 0.03, "top_p": 0.99}},
                {"id": "base"},
            ],
        },
        "chunking": {},
        "validation": {},
        "audio": {},
    }

    job_path = project_root / "config" / "job.yaml"
    with open(job_path, "w", encoding="utf-8") as f:
        yaml.dump(job_yaml, f, sort_keys=False)

    cm = ConfigManager(project_root)
    merged = cm.load_cascading_config(job_path)

    # Expect coerced string identifiers
    assert isinstance(merged["job"]["name"], str) and merged["job"]["name"].lower() == "true"
    assert isinstance(merged["job"]["run_label"], str) and merged["job"]["run_label"] == ""

    spk_ids = [s.get("id") for s in merged["generation"]["speakers"]]
    # One speaker has id "false" (string), and base remains
    assert "false" in spk_ids
    assert "base" in spk_ids

    # default_speaker coerced to string "false"; then fixed by validation to a valid one
    # After merge, default speaker must be a valid existing speaker id
    assert merged["generation"]["default_speaker"] in spk_ids

    # language coerced to string
    first = merged["generation"]["speakers"][0]
    assert isinstance(first.get("language"), str)


