#!/usr/bin/env python3
"""
Core JobManager class for job discovery and task creation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.config_manager import ConfigManager, TaskConfig

from .types import ExecutionStrategy

logger = logging.getLogger(__name__)


class JobManager:
    """
    Core job discovery and task creation.

    Handles:
    - Job discovery and task creation
    - Task management operations
    """

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.project_root = config_manager.project_root

    def is_task_config(self, config_path: Path) -> bool:
        return self.config_manager.is_task_config(config_path)

    def get_jobs(self, job_name: Optional[str] = None) -> List[TaskConfig]:
        if job_name:
            return self.find_jobs_by_name(job_name)
        else:
            return self.find_all_jobs()

    def find_jobs_by_name(self, job_name: str) -> List[TaskConfig]:
        task_configs = []

        # Search in config directory for job-yaml files
        config_files = self.config_manager.find_configs_by_job_name(job_name)

        for config_file in config_files:
            try:
                if self.is_task_config(config_file):
                    # Load existing task config
                    task_config = self.config_manager.load_task_config(config_file)
                    task_configs.append(task_config)
                else:
                    # Create new task config from job-yaml
                    config_data = self.config_manager.load_cascading_config(config_file)
                    task_config = self.config_manager.create_task_config(config_data)
                    # Set the original job-yaml path for later use
                    task_config.config_path = config_file
                    task_configs.append(task_config)

            except Exception as e:
                logger.warning(f"Error processing config {config_file}: {e}")

        # Also search for existing tasks in output directory
        existing_tasks = self.config_manager.find_existing_tasks(job_name)
        task_configs.extend(existing_tasks)

        # Remove duplicates and sort by timestamp
        seen_paths = set()
        unique_configs = []
        for config in task_configs:
            if config.config_path not in seen_paths:
                unique_configs.append(config)
                seen_paths.add(config.config_path)

        unique_configs.sort(key=lambda t: t.timestamp, reverse=True)

        logger.info(f"Found {len(unique_configs)} configs for job '{job_name}'")
        return unique_configs

    def find_all_jobs(self) -> List[TaskConfig]:
        task_configs = []

        # Search config directory
        config_dir = self.config_manager.config_dir
        for config_file in config_dir.glob("*.yaml"):
            if config_file.name == "default_config.yaml":
                continue  # Skip default config

            try:
                config_data = self.config_manager.load_cascading_config(config_file)
                task_config = self.config_manager.create_task_config(config_data)
                task_configs.append(task_config)

            except Exception as e:
                logger.warning(f"Error processing config {config_file}: {e}")

        # Search output directory for task configs
        output_dir = self.config_manager.output_dir
        if output_dir.exists():
            for job_dir in output_dir.iterdir():
                if job_dir.is_dir():
                    for config_file in job_dir.glob("*_config.yaml"):
                        try:
                            task_config = self.config_manager.load_task_config(
                                config_file
                            )
                            task_configs.append(task_config)
                        except Exception as e:
                            logger.warning(
                                f"Error processing task config {config_file}: {e}"
                            )

        logger.info(f"Found {len(task_configs)} total job configs")
        return task_configs

    def find_existing_tasks(self, job_name: str, run_label: Optional[str] = None) -> List[TaskConfig]:
        return self.config_manager.find_existing_tasks(job_name, run_label)

    def create_new_task(self, job_config: Dict[str, Any]) -> TaskConfig:
        # Validate and complete config
        if not self.config_manager.validate_config(job_config):
            raise ValueError("Invalid job configuration")

        # Create task config
        task_config = self.config_manager.create_task_config(job_config)

        # Save task config file
        self.config_manager.save_task_config(task_config, job_config)

        logger.info(f"Created new task: {task_config.config_path}")
        return task_config

    def parse_mode_argument(
        self, mode_arg: Optional[str]
    ) -> tuple[Dict[str, Any], Optional[Any]]:
        """
        Parse the unified --mode argument that can be either:
        - Global strategy: "all", "new", "last"
        - Job-specific strategies: "job1:new,job2:all,job3:last"


        Returns:
            Tuple of (job_strategies_dict, global_strategy)
        """
        # ExecutionStrategy is now imported from types module

        if not mode_arg:
            return {}, None

        def normalize_strategy(strategy: str) -> str:
            """Normalize strategy aliases to canonical form."""
            strategy = strategy.strip()
            # Handle aliases - normalize to primary forms
            if strategy == "new-last":
                return "latest-new"
            elif strategy == "last":
                return "latest"
            elif strategy == "last-new":
                return "latest-new"
            elif strategy == "new-all":
                return "all-new"
            return strategy

        # Check if it contains job-specific format (contains colon)
        if ":" in mode_arg:
            # Job-specific strategies: "job1:new,job2:all"
            job_strategies = {}
            try:
                for pair in mode_arg.split(","):
                    job_name, strategy = pair.split(":")
                    normalized_strategy = normalize_strategy(strategy)
                    job_strategies[job_name.strip()] = ExecutionStrategy(
                        normalized_strategy
                    )
                return job_strategies, None
            except ValueError:
                raise ValueError(
                    "Invalid --mode format for job-specific strategies. Use 'job1:strategy,job2:strategy'"
                )
        else:
            # Global strategy: "all", "new", "last"
            try:
                normalized_strategy = normalize_strategy(mode_arg)
                global_strategy = ExecutionStrategy(normalized_strategy)
                return {}, global_strategy
            except ValueError:
                raise ValueError(
                    f"Invalid --mode strategy '{mode_arg}'. Use: latest/last, all, new, latest-new/last-new/new-last, all-new/new-all, or job-specific format 'job1:strategy,job2:strategy'"
                )
