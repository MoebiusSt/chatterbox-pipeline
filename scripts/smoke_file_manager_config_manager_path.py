#!/usr/bin/env python3
"""Smoke test: FileManager stores and uses _config_manager for get_default_speaker_id."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config_manager import ConfigManager, TaskConfig
from utils.file_manager.file_manager import FileManager


def _minimal_task_config(cm: ConfigManager) -> TaskConfig:
    cfg = cm.load_default_config()
    return cm.create_task_config(cfg)


def main() -> int:
    cm = ConfigManager(PROJECT_ROOT)
    task_config = _minimal_task_config(cm)
    injected = MagicMock(spec=ConfigManager)
    injected.load_cascading_config = cm.load_cascading_config
    injected.create_task_config = cm.create_task_config
    injected.get_default_speaker_id = MagicMock(return_value="spy_default_speaker")

    fm = FileManager(
        task_config,
        preloaded_config=cm.load_default_config(),
        config_manager=injected,
    )

    errors: list[str] = []

    if not hasattr(fm, "_config_manager"):
        errors.append("FAIL: _config_manager missing after __init__ (old dead-path behavior)")
    elif fm._config_manager is not injected:
        errors.append("FAIL: _config_manager is not the injected ConfigManager instance")
    else:
        print("OK: _config_manager set to injected ConfigManager")

    result = fm.get_default_speaker_id()
    if not injected.get_default_speaker_id.called:
        errors.append(
            "FAIL: ConfigManager.get_default_speaker_id was never called "
            "(inline fallback would have run instead)"
        )
    elif result != "spy_default_speaker":
        errors.append(f"FAIL: expected spy return, got {result!r}")
    else:
        print("OK: get_default_speaker_id delegated to ConfigManager")

    # Pre-fix behavior: hasattr guard was always false because __init__ never set the attr
    class _PreFixSimulator:
        pass

    sim = _PreFixSimulator()
    sim.config = fm.config
    if hasattr(sim, "_config_manager"):
        errors.append("FAIL: simulator unexpectedly has _config_manager")
    else:
        print("OK: pre-fix simulation — without assignment, hasattr(_config_manager) is False")

    if errors:
        for e in errors:
            print(e)
        return 1

    print("\nSmoke test passed: _config_manager path is live, not dead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
