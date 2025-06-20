#!/usr/bin/env python3
"""
JobManager wrapper for backward compatibility.
Combines all job management modules into a single interface.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.config_manager import ConfigManager, TaskConfig
from .job_manager.job_manager import JobManager as CoreJobManager
from .job_manager.execution_planner import ExecutionPlanner, ExecutionPlan, ExecutionStrategy
from .job_manager.user_interaction import UserInteraction, UserChoice
from .job_manager.config_validator import ConfigValidator

logger = logging.getLogger(__name__)


# Re-export enums and dataclasses for backward compatibility
UserChoice = UserChoice
ExecutionStrategy = ExecutionStrategy  
ExecutionPlan = ExecutionPlan


class JobManager:
    """
    Wrapper JobManager that combines all job management functionality.
    Maintains backward compatibility with the original monolithic class.
    """

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.project_root = config_manager.project_root
        
        # Initialize components
        self.core_manager = CoreJobManager(config_manager)
        self.execution_planner = ExecutionPlanner(self.core_manager, config_manager)
        self.user_interaction = UserInteraction(config_manager)
        self.config_validator = ConfigValidator(self.core_manager, config_manager)

    # Delegate to core job manager
    def is_task_config(self, config_path: Path) -> bool:
        return self.core_manager.is_task_config(config_path)

    def get_jobs(self, job_name: Optional[str] = None) -> List[TaskConfig]:
        return self.core_manager.get_jobs(job_name)

    def find_jobs_by_name(self, job_name: str) -> List[TaskConfig]:
        return self.core_manager.find_jobs_by_name(job_name)

    def find_all_jobs(self) -> List[TaskConfig]:
        return self.core_manager.find_all_jobs()

    def find_existing_tasks(self, job_name: str) -> List[TaskConfig]:
        return self.core_manager.find_existing_tasks(job_name)

    def create_new_task(self, job_config: Dict[str, Any]) -> TaskConfig:
        return self.core_manager.create_new_task(job_config)

    def parse_mode_argument(self, mode_arg: Optional[str]) -> tuple[Dict[str, ExecutionStrategy], Optional[ExecutionStrategy]]:
        return self.core_manager.parse_mode_argument(mode_arg)

    # Delegate to user interaction
    def prompt_user_selection(self, tasks: List[TaskConfig]) -> UserChoice:
        choice = self.user_interaction.prompt_user_selection(tasks)
        # Copy selected_task_index for backward compatibility
        if hasattr(self.user_interaction, 'selected_task_index'):
            self.selected_task_index = self.user_interaction.selected_task_index
        return choice

    # Delegate to execution planner
    def resolve_execution_plan(
        self, args: Any, config_files: Optional[List[Path]] = None
    ) -> ExecutionPlan:
        return self.execution_planner.resolve_execution_plan(args, config_files)

    def print_execution_summary(self, plan: ExecutionPlan) -> None:
        return self.execution_planner.print_execution_summary(plan)

    # Delegate to config validator
    def validate_execution_plan(self, plan: ExecutionPlan) -> bool:
        return self.config_validator.validate_execution_plan(plan)

    def _validate_mixed_configurations(self, task_configs: List[TaskConfig]) -> bool:
        """Validate that mixed task configurations are compatible."""
        return self.config_validator._validate_mixed_configurations(task_configs)
