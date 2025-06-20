#!/usr/bin/env python3
"""
Configuration validation functionality for job management.
"""

import logging
from typing import Dict, List

from utils.config_manager import ConfigManager, TaskConfig
from .execution_planner import ExecutionPlan

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Handles validation of task configurations and execution plans."""

    def __init__(self, job_manager, config_manager: ConfigManager):
        self.job_manager = job_manager
        self.config_manager = config_manager

    def validate_execution_plan(self, plan: ExecutionPlan) -> bool:
        # Allow cancelled execution plans
        if plan.execution_mode == "cancelled":
            return True
            
        if not plan.task_configs:
            logger.error("No tasks in execution plan")
            return False

        # Validate mixed configuration scenarios
        if len(plan.task_configs) > 1:
            if not self._validate_mixed_configurations(plan.task_configs):
                return False

        for task_config in plan.task_configs:
            config_data = self.config_manager.load_cascading_config(
                task_config.config_path
            )
            if not self.config_manager.validate_config(config_data):
                logger.error(f"Invalid task config: {task_config.config_path}")
                return False

        return True

    def _validate_mixed_configurations(self, task_configs: List[TaskConfig]) -> bool:
        """Validate that mixed task configurations are compatible."""
        logger.info("🔍 Validating configuration compatibility...")

        # Group by job names
        jobs_by_name: Dict[str, List[TaskConfig]] = {}
        task_configs_by_type: Dict[str, List[TaskConfig]] = {"job_config": [], "task_config": []}

        for task_config in task_configs:
            # Determine if this is from a job config or task config
            if self.job_manager.is_task_config(task_config.config_path):
                task_configs_by_type["task_config"].append(task_config)
            else:
                task_configs_by_type["job_config"].append(task_config)

            # Group by job name
            job_name = task_config.job_name
            if job_name not in jobs_by_name:
                jobs_by_name[job_name] = []
            jobs_by_name[job_name].append(task_config)

        # Check for potential conflicts
        warnings: List[str] = []
        errors: List[str] = []

        # Check 1: Mixed task and job configs
        if task_configs_by_type["job_config"] and task_configs_by_type["task_config"]:
            warnings.append(
                f"Mixed job configs ({len(task_configs_by_type['job_config'])}) "
                f"and task configs ({len(task_configs_by_type['task_config'])})"
            )

        # Check 3: Device compatibility
        device_requirements = set()
        for task_config in task_configs:
            config_data = self.config_manager.load_cascading_config(
                task_config.config_path
            )
            generation_config = config_data.get("generation", {})
            if generation_config.get("device"):
                device_requirements.add(generation_config["device"])

        if len(device_requirements) > 1:
            warnings.append(f"Multiple device requirements: {device_requirements}")

        # Log warnings and errors
        if warnings:
            logger.warning("⚠️ Configuration compatibility warnings:")
            for warning in warnings:
                logger.warning(f"• {warning}")

        if errors:
            logger.error("❌ Configuration compatibility errors:")
            for error in errors:
                logger.error(f"• {error}")
            logger.error(
                "Mixed configurations are incompatible and cannot be processed together"
            )
            return False

        if warnings:
            logger.info("⚠️ Configurations are compatible (with warnings)")
        else:
            logger.info("✅ Configurations are fully compatible")

        return True 