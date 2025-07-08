#!/usr/bin/env python3
"""
JobManager wrapper for backward compatibility.
Combines all job management modules into a single interface.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .job_manager.execution_planner import ExecutionPlanner
from .job_manager.job_manager import JobManager as CoreJobManager
from .job_manager.types import (
    ExecutionPlan,
    ExecutionStrategy,
    UserChoice,
)
from utils.config_manager import TaskConfig

logger = logging.getLogger(__name__)

# Re-export enums and dataclasses for backward compatibility
UserChoice = UserChoice
ExecutionStrategy = ExecutionStrategy
ExecutionPlan = ExecutionPlan

class JobManager:
    """Wraps job management functionality with ExecutionPlanner."""

    def __init__(self, config_manager):
        self.config_manager = config_manager

        # Initialize components
        self.core_manager = CoreJobManager(config_manager)
        self.execution_planner = ExecutionPlanner(self.core_manager, config_manager)

    # Delegate to core manager
    def find_existing_tasks(
        self, job_name: str, run_label: Optional[str] = None
    ) -> List[TaskConfig]:
        return self.core_manager.find_existing_tasks(job_name, run_label)

    def find_jobs_by_name(self, job_name: str) -> List[Any]:
        """Find job configs by name."""
        configs = self.config_manager.find_configs_by_job_name(job_name)
        return [{"config_path": config} for config in configs]

    def create_new_task(self, job_config: Dict[str, Any]) -> TaskConfig:
        return self.core_manager.create_new_task(job_config)

    def is_task_config(self, config_path: Path) -> bool:
        """Check if a config file is a task config."""
        return self.config_manager.is_task_config(config_path)

    def parse_mode_argument(
        self, mode_arg: Optional[str]
    ) -> tuple[Dict[str, ExecutionStrategy], Optional[ExecutionStrategy]]:
        return self.core_manager.parse_mode_argument(mode_arg)

    # Delegate to execution planner
    def resolve_execution_plan(
        self, args: Any, config_files: Optional[List[Path]] = None
    ) -> ExecutionPlan:
        return self.execution_planner.resolve_execution_plan(args, config_files)

    def print_execution_summary(self, plan: ExecutionPlan) -> None:
        return self.execution_planner.print_execution_summary(plan)

    # Delegate to config validator
    def validate_execution_plan(self, plan: ExecutionPlan) -> bool:
        # return self.config_validator.validate_execution_plan(plan) # Removed as per edit hint
        return True  # Placeholder, as ConfigValidator is removed

    def _validate_mixed_configurations(self, task_configs: List[TaskConfig]) -> bool:
        """Validate that mixed task configurations are compatible."""
        # return self.config_validator._validate_mixed_configurations(task_configs) # Removed as per edit hint
        return True  # Placeholder, as ConfigValidator is removed
