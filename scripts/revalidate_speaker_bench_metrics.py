#!/usr/bin/env python3
"""Backfill validation-derived metrics for existing speaker-bench tasks.

This script does not render candidates and intentionally refuses to run ASR.
It reuses existing whisper results, recalculates prosody/MOS/selection gates,
and rewrites whisper_metrics.json, task_metrics.json, and analysis_metrics.json.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline.task_executor.stage_handlers.validation_handler import ValidationHandler
from utils.config_manager import ConfigManager, TaskConfig
from utils.file_manager.file_manager import FileManager
from utils.file_manager.task_metrics_generator import TaskMetricsGenerator
from validation.quality_scorer import QualityScorer


logger = logging.getLogger("revalidate_speaker_bench_metrics")


class NoASRBackend:
    """ASR guard used to ensure this backfill never transcribes audio."""

    backend_name = "no_asr_backfill"
    supports_alignment = False

    def validate_candidate(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("ASR is disabled for metrics-only backfill")

    def score_transcription(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("ASR is disabled for metrics-only backfill")


class NoGenerationHandler:
    """Generation guard used to ensure this backfill never renders audio."""

    def generate_retry_candidates(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Generation is disabled for metrics-only backfill")


def _task_configs_from_root(root: Path, pattern: str) -> List[Path]:
    config_paths: List[Path] = []
    for task_dir in sorted(p for p in root.glob(pattern) if p.is_dir()):
        config_path = task_dir.with_name(f"{task_dir.name}_config.yaml")
        if config_path.exists():
            config_paths.append(config_path)
        else:
            logger.warning("Skipping %s: missing sibling config %s", task_dir, config_path.name)
    return config_paths


def _prepare_config(config: Dict[str, Any]) -> Dict[str, Any]:
    # This runner must never render retries. It only reuses existing candidates
    # and existing ASR output.
    config.setdefault("generation", {})["max_retries"] = 0
    return config


def _revalidate_task(config_manager: ConfigManager, config_path: Path, dry_run: bool) -> bool:
    task_config: TaskConfig = config_manager.load_task_config(config_path)
    config = _prepare_config(task_config.preloaded_config or config_manager.load_cascading_config(config_path))
    task_dir = task_config.base_output_dir

    if dry_run:
        logger.info("Would backfill %s", task_dir)
        return True

    logger.info("Backfilling %s", task_dir)
    file_manager = FileManager(
        task_config,
        preloaded_config=config,
        config_manager=config_manager,
    )
    validation_handler = ValidationHandler(
        file_manager=file_manager,
        config=config,
        asr_backend=NoASRBackend(),
        quality_scorer=QualityScorer(sample_rate=int(config.get("audio", {}).get("sample_rate", 24000))),
        generation_handler=NoGenerationHandler(),
    )

    if not validation_handler.execute_validation():
        logger.error("Validation backfill failed for %s", task_dir)
        return False

    metrics = TaskMetricsGenerator(task_dir, config)
    ok_task = metrics.generate_task_metrics()
    ok_analysis = metrics.generate_analysis_metrics()
    if not (ok_task and ok_analysis):
        logger.error("Metrics generation failed for %s", task_dir)
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recalculate validation-derived metrics for existing speaker-bench tasks without ASR or rendering."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT / "data" / "output" / "speaker-bench",
        help="Task root containing speaker-bench task directories.",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern for task directories under --root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List tasks without modifying metrics.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        logger.error("Root does not exist: %s", root)
        return 2

    config_paths = _task_configs_from_root(root, args.pattern)
    if not config_paths:
        logger.error("No task configs found under %s", root)
        return 2

    logger.info("Found %d task(s)", len(config_paths))
    config_manager = ConfigManager(PROJECT_ROOT)
    failures = 0
    for config_path in config_paths:
        try:
            if not _revalidate_task(config_manager, config_path, bool(args.dry_run)):
                failures += 1
        except Exception as exc:
            logger.exception("Failed %s: %s", config_path, exc)
            failures += 1

    if failures:
        logger.error("Completed with %d failure(s)", failures)
        return 1
    logger.info("Completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
