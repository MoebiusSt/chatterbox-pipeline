#!/usr/bin/env python3
"""
MetricsIOHandler for quality metrics operations.
Handles saving and loading of quality metrics and validation results.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsIOHandler:
    """Handles quality metrics I/O operations."""

    def __init__(self, task_directory: Path):
        """
        Initialize MetricsIOHandler.

        Args:
            task_directory: Main task directory
        """
        self.task_directory = task_directory

    def save_metrics(self, metrics: dict) -> bool:
        """Save quality metrics and validation results."""
        try:
            path = self.task_directory / "enhanced_metrics.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            return False

    def get_metrics(self) -> dict:
        """Load quality metrics and validation results."""
        path = self.task_directory / "enhanced_metrics.json"
        if not path.exists():
            logger.debug(f"Metrics file not found: {path}")
            return {}

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
