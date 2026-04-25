#!/usr/bin/env python3
"""Test helpers for final-audio regeneration via the reassemble verb."""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pytest
import torch
import torchaudio

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.pipeline.job_manager.cli_mapper import CLIMapper
from src.pipeline.job_manager.execution_types import ExecutionContext
from src.pipeline.job_manager.menu_orchestrator import MenuOrchestrator
from src.pipeline.job_manager.types import Verb

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_mock_final_audio(output_dir: Path) -> Path:
    """Create a mock final audio file."""
    final_dir = output_dir / "final"
    final_dir.mkdir(exist_ok=True)
    sample_rate = 24000
    duration = 5.0
    frequency = 440
    t = torch.linspace(0, duration, int(sample_rate * duration))
    audio = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
    final_path = final_dir / f"{output_dir.name.split('_')[0]}_enhanced.wav"
    torchaudio.save(str(final_path), audio, sample_rate)
    logger.info("Created mock final audio: %s", final_path.name)
    return final_path


def create_mock_scenario(output_dir: Path) -> None:
    """Create a compact mock scenario for testing reassemble."""
    (output_dir / "texts").mkdir(exist_ok=True)
    (output_dir / "candidates").mkdir(exist_ok=True)
    create_mock_final_audio(output_dir)


def test_regenerate_final(tmp_path):
    """Create a scenario and print the new CLI command."""
    test_dir_name = f"test-reassemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    test_dir = tmp_path / test_dir_name
    test_dir.mkdir(exist_ok=True)
    create_mock_scenario(test_dir)
    logger.info("Now test with: python src/cbpipe.py reassemble --job %s", test_dir_name)
    assert test_dir.exists()


def test_reassemble_options():
    """Test forcing final audio regeneration."""
    options = CLIMapper().verb_to_options(Verb.REASSEMBLE)
    assert options.force_final_generation
    assert not options.rerender_all


def test_interactive_reassemble(monkeypatch):
    """Test interactive final audio regeneration via MenuOrchestrator."""
    mock_task = type(
        "Task",
        (),
        {
            "timestamp": "20241201_120000",
            "job_name": "test_job",
            "run_label": "test_label",
            "config_path": Path("/fake/path/config.yaml"),
            "force_final_generation": False,
            "rerender_all": False,
        },
    )()
    context = ExecutionContext(
        existing_tasks=[mock_task],
        job_configs=None,
        execution_path="test",
        job_name="test_job",
    )
    monkeypatch.setattr("builtins.input", lambda _prompt: "f")
    intent = MenuOrchestrator(config_manager=None).resolve_user_intent(context)
    assert intent.verb == Verb.REASSEMBLE
    assert intent.execution_options.force_final_generation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
