#!/usr/bin/env python3
"""
Core JobManager class for job discovery and task creation.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.config_manager import ConfigManager, TaskConfig

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
                    task_config = self.config_manager.load_task_config_shallow(
                        config_file
                    )
                    task_configs.append(task_config)
                else:
                    # For menu listing, avoid merging; create a shallow TaskConfig signature
                    # Load just minimal job yaml to extract identifiers
                    job_yaml = self.config_manager.load_job_config(config_file)
                    if not isinstance(job_yaml, dict):
                        continue
                    # Create a transient TaskConfig (no saving here)
                    temp = self.config_manager.create_task_config(
                        {
                            "job": {
                                "name": job_yaml.get("job", {}).get("name", "job"),
                                "run_label": job_yaml.get("job", {}).get("run_label", ""),
                            },
                            "input": {"text_file": job_yaml.get("input", {}).get("text_file", "input.txt")},
                        }
                    )
                    temp.config_path = config_file
                    task_configs.append(temp)

            except Exception as e:
                logger.warning(f"Error processing config {config_file}: {e}")

        # Also search for existing tasks in output directory
        existing_tasks = self.config_manager.find_existing_tasks(job_name, shallow=True)
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
                job_yaml = self.config_manager.load_job_config(config_file)
                if not isinstance(job_yaml, dict):
                    continue
                temp = self.config_manager.create_task_config(
                    {
                        "job": {
                            "name": job_yaml.get("job", {}).get("name", "job"),
                            "run_label": job_yaml.get("job", {}).get("run_label", ""),
                        },
                        "input": {"text_file": job_yaml.get("input", {}).get("text_file", "input.txt")},
                    }
                )
                temp.config_path = config_file
                task_configs.append(temp)

            except Exception as e:
                logger.warning(f"Error processing config {config_file}: {e}")

        # Search output directory for task configs
        output_dir = self.config_manager.output_dir
        if output_dir.exists():
            for job_dir in output_dir.iterdir():
                if job_dir.is_dir():
                    for config_file in job_dir.glob("*_config.yaml"):
                        try:
                            task_config = self.config_manager.load_task_config_shallow(
                                config_file
                            )
                            task_configs.append(task_config)
                        except Exception as e:
                            logger.warning(
                                f"Error processing task config {config_file}: {e}"
                            )

        logger.info(f"Found {len(task_configs)} total job configs")
        return task_configs

    def find_existing_tasks(
        self, job_name: str, run_label: Optional[str] = None
    ) -> List[TaskConfig]:
        return self.config_manager.find_existing_tasks(job_name, run_label)

    def create_new_task(self, job_config: Dict[str, Any]) -> TaskConfig:
        # Validate and complete config
        if not self.config_manager.validate_config(job_config):
            raise ValueError("Invalid job configuration")

        # Create task config
        task_config = self.config_manager.create_task_config(job_config)

        # Save task config file
        self.config_manager.save_task_config(task_config, job_config)

        return task_config
