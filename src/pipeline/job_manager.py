#!/usr/bin/env python3
"""
JobManager for job and task management.
Handles job discovery, task selection, and user interactions.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.config_manager import ConfigManager, TaskConfig
import logging

logger = logging.getLogger(__name__)


class UserChoice(Enum):
    """User selection options for task execution."""

    LATEST = "latest"  # Use latest task
    ALL = "all"  # Use all tasks
    NEW = "new"  # Create new task
    LATEST_NEW = "latest-new"  # Use latest task + new final audio
    ALL_NEW = "all-new"  # Use all tasks + new final audio
    SPECIFIC = "specific"  # Select specific task
    SPECIFIC_NEW = "specific-new"  # Select specific task + new final audio
    CANCEL = "cancel"  # Cancel execution


class ExecutionStrategy(Enum):
    """Strategy for task execution."""

    LAST = "last"  # Use latest task
    LATEST = "latest"  # Use latest task (alias for LAST)
    ALL = "all"  # Use all tasks
    NEW = "new"  # Create new task
    LAST_NEW = "last-new"  # Use latest task + new final audio
    ALL_NEW = "all-new"  # Use all tasks + new final audio


@dataclass
class ExecutionPlan:
    """Plan for task execution."""

    task_configs: List[TaskConfig]
    execution_mode: str  # "single", "batch", "interactive"
    requires_user_input: bool = False


class JobManager:
    """
    Central job and task management.

    Handles:
    - Job discovery and task creation
    - User interaction for task selection
    - Execution plan generation
    """

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.project_root = config_manager.project_root

    def is_task_config(self, config_path: Path) -> bool:
        """
        Check if config path points to a task-yaml file.

        Returns:
            True if it's a task-yaml file
        """
        return self.config_manager.is_task_config(config_path)

    def get_jobs(self, job_name: Optional[str] = None) -> List[TaskConfig]:
        """
        Find jobs by name or return all available jobs.

        Args:
            job_name: Specific job name to search for, or None for all

        Returns:
            List of TaskConfig objects
        """
        if job_name:
            return self.find_jobs_by_name(job_name)
        else:
            return self.find_all_jobs()

    def find_jobs_by_name(self, job_name: str) -> List[TaskConfig]:
        """
        Find all jobs (configs) matching a specific job name.

        Returns:
            List of TaskConfig objects
        """
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
        """
        Find all available jobs in the system.

        Returns:
            List of all TaskConfig objects
        """
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

    def find_existing_tasks(self, job_name: str) -> List[TaskConfig]:
        """
        Find existing task configurations for a job.

        Returns:
            List of TaskConfig objects for existing tasks
        """
        return self.config_manager.find_existing_tasks(job_name)

    def create_new_task(self, job_config: Dict[str, Any]) -> TaskConfig:
        """
        Create a new task configuration.

        Args:
            job_config: Job configuration dictionary

        Returns:
            TaskConfig object
        """
        # Validate and complete config
        if not self.config_manager.validate_config(job_config):
            raise ValueError("Invalid job configuration")

        # Create task config
        task_config = self.config_manager.create_task_config(job_config)

        # Save task config file
        self.config_manager.save_task_config(task_config, job_config)

        logger.info(f"Created new task: {task_config.config_path}")
        return task_config

    def prompt_user_selection(self, tasks: List[TaskConfig]) -> UserChoice:
        """
        Prompt user to select task execution strategy.

        Returns:
            UserChoice enum indicating the user's selection.
        """
        if not tasks:
            return UserChoice.NEW

        job_name = tasks[0].job_name if tasks else "Unknown"
        print(f"\nFound existing tasks for job '{job_name}':")
        
        # Store selected task index for SPECIFIC choice
        self.selected_task_index = 0  # Default to latest (first in sorted list)
        
        for i, task in enumerate(tasks, 1):
            # Parse timestamp for better display
            try:
                dt = datetime.strptime(task.timestamp, "%Y%m%d_%H%M%S")
                date_str = dt.strftime("%d.%m.%Y")
                time_str = dt.strftime("%H:%M")
            except ValueError:
                # Fallback parsing for debugging
                date_str = "Parse_Error"
                time_str = task.timestamp
            
            # Get text file name from task config
            text_file = "unknown"
            try:
                if task.config_path.exists():
                    config_data = self.config_manager.load_job_config(task.config_path)
                    text_file = Path(config_data["input"]["text_file"]).stem
            except Exception:
                # Fallback: extract from config filename
                config_name = task.config_path.stem
                if config_name.endswith("_config"):
                    config_name = config_name[:-7]
                file_parts = config_name.split("_")
                if len(file_parts) >= 1:
                    text_file = file_parts[0]
            
            # Format display according to specification:
            # "1. {job-name} - {job-run-label} - {doc-name.txt} - {date as 16.07.2025} - {time as 19:15} (<-- latest)"
            latest_marker = " (<-- latest)" if i == 1 else ""  # First item is newest
            run_label_display = task.run_label if task.run_label else "no-label"
            
            print(f"{i}. {task.job_name} - {run_label_display} - {text_file}.txt - {date_str} - {time_str}{latest_marker}")

        print("\nSelect action:")
        print("[Enter] - Run latest task (Check task)")
        print("n      - Create new task")
        print("a      - Run all tasks (Check tasks)")
        print("ln     - Use latest task + force new final audio")
        print("an     - Run all tasks + force new final audio")
        print("1-{}   - Select specific task".format(len(tasks)))
        print("c      - Cancel")

        choice = input("\n> ").strip().lower()

        if choice == "":
            # Latest task selected - ask for additional options like specific task selection
            latest_task = tasks[0]  # First in sorted list (newest)
            
            # Parse timestamp for display
            try:
                dt = datetime.strptime(latest_task.timestamp, "%Y%m%d_%H%M%S")
                display_time = dt.strftime("%d.%m.%Y %H:%M")
            except ValueError:
                display_time = latest_task.timestamp
                
            print(f"\nSelected latest task: {latest_task.job_name} - {display_time}")
            print("\nWhat to do with this task?")
            print("[Enter] - Run task (Check task)")
            print("n      - Run task + force new final audio")
            print("c      - Cancel")

            sub_choice = input("\n> ").strip().lower()
            if sub_choice == "":
                return UserChoice.LATEST
            elif sub_choice == "n":
                return UserChoice.LATEST_NEW
            elif sub_choice == "c":
                return UserChoice.CANCEL
            else:
                print("Invalid choice, defaulting to check task")
                return UserChoice.LATEST
        elif choice == "n":
            return UserChoice.NEW
        elif choice == "a":
            return UserChoice.ALL
        elif choice == "ln":
            return UserChoice.LATEST_NEW
        elif choice == "an":
            return UserChoice.ALL_NEW
        elif choice == "c":
            return UserChoice.CANCEL
        elif choice.isdigit() and 1 <= int(choice) <= len(tasks):
            # Store the selected task index (convert to 0-based)
            self.selected_task_index = int(choice) - 1
            selected_task = tasks[self.selected_task_index]
            
            # Parse timestamp for display
            try:
                dt = datetime.strptime(selected_task.timestamp, "%Y%m%d_%H%M%S")
                display_time = dt.strftime("%d.%m.%Y %H:%M")
            except ValueError:
                display_time = selected_task.timestamp
                
            print(f"\nSelected task: {selected_task.job_name} - {display_time}")
            print("\nWhat to do with this task?")
            print("[Enter] - Run task (Check task)")
            print("n      - Run task + force new final audio")
            print("c      - Cancel")

            sub_choice = input("\n> ").strip().lower()
            if sub_choice == "":
                return UserChoice.SPECIFIC
            elif sub_choice == "n":
                return UserChoice.SPECIFIC_NEW
            elif sub_choice == "c":
                return UserChoice.CANCEL
            else:
                print("Invalid choice, defaulting to check task")
                return UserChoice.SPECIFIC

        print("Invalid choice, defaulting to latest task")
        return UserChoice.LATEST

    def parse_mode_argument(self, mode_arg: Optional[str]) -> tuple[Dict[str, ExecutionStrategy], Optional[ExecutionStrategy]]:
        """
        Parse the unified --mode argument that can be either:
        - Global strategy: "all", "new", "last"
        - Job-specific strategies: "job1:new,job2:all,job3:last"
    
            
        Returns:
            Tuple of (job_strategies_dict, global_strategy)
        """
        if not mode_arg:
            return {}, None
            
        def normalize_strategy(strategy: str) -> str:
            """Normalize strategy aliases to canonical form."""
            strategy = strategy.strip()
            # Handle aliases
            if strategy == "new-last":
                return "last-new"
            return strategy
            
        # Check if it contains job-specific format (contains colon)
        if ":" in mode_arg:
            # Job-specific strategies: "job1:new,job2:all"
            job_strategies = {}
            try:
                for pair in mode_arg.split(","):
                    job_name, strategy = pair.split(":")
                    normalized_strategy = normalize_strategy(strategy)
                    job_strategies[job_name.strip()] = ExecutionStrategy(normalized_strategy)
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
                    f"Invalid --mode strategy '{mode_arg}'. Use: last/latest, all, new, last-new/new-last, all-new, or job-specific format 'job1:strategy,job2:strategy'"
                )

    def resolve_execution_plan(
        self, args: Any, config_files: Optional[List[Path]] = None
    ) -> ExecutionPlan:
        """
        Resolves the execution plan based on CLI arguments and available job configurations.

        Returns:
            An ExecutionPlan object detailing the tasks to be executed.
        """
        task_configs = []
        execution_mode = "single"
        requires_user_input = False

        # Parse unified --mode argument
        job_strategies, global_strategy = self.parse_mode_argument(args.mode)

        if args.job:
            # --job "jobname" scenario
            job_name = args.job
            existing_tasks = self.find_existing_tasks(job_name)

            if existing_tasks:
                # Apply strategy for this job
                strategy = job_strategies.get(job_name, global_strategy)

                if strategy == ExecutionStrategy.NEW:
                    # Create new task
                    job_configs = self.find_jobs_by_name(job_name)
                    if job_configs:
                        config_data = self.config_manager.load_cascading_config(
                            job_configs[0].config_path
                        )
                        new_task = self.create_new_task(config_data)
                        task_configs = [new_task]
                elif strategy == ExecutionStrategy.ALL:
                    # Use all tasks
                    task_configs = list(existing_tasks)
                    execution_mode = "batch"
                elif strategy == ExecutionStrategy.ALL_NEW:
                    # Use all tasks + force new final audio
                    task_configs = list(existing_tasks)
                    for task in task_configs:
                        task.add_final = True
                    execution_mode = "batch"
                elif strategy == ExecutionStrategy.LAST or strategy == ExecutionStrategy.LATEST:
                    # Use latest task
                    task_configs = [existing_tasks[0]]
                elif strategy == ExecutionStrategy.LAST_NEW:
                    # Use latest task + force new final audio
                    task_config = existing_tasks[0]  # First in sorted list (newest)
                    task_config.add_final = True
                    task_configs = [task_config]
                else:
                    # No strategy specified - interactive selection
                    if global_strategy is None:
                        requires_user_input = True
                    choice = self.prompt_user_selection(existing_tasks)

                    if choice == UserChoice.CANCEL:
                        return ExecutionPlan([], "cancelled")
                    elif choice == UserChoice.LATEST:
                        if existing_tasks:
                            task_configs = [existing_tasks[0]]  # First in sorted list (newest)
                    elif choice == UserChoice.SPECIFIC:
                        if existing_tasks and hasattr(self, 'selected_task_index'):
                            task_configs = [existing_tasks[self.selected_task_index]]
                        elif existing_tasks:
                            task_configs = [existing_tasks[0]]  # Fallback to latest
                    elif choice == UserChoice.LATEST_NEW:
                        if existing_tasks:
                            task_config = existing_tasks[0]  # First in sorted list (newest)
                            task_config.add_final = True
                            task_configs = [task_config]
                    elif choice == UserChoice.SPECIFIC_NEW:
                        if existing_tasks and hasattr(self, 'selected_task_index'):
                            task_config = existing_tasks[self.selected_task_index]
                            task_config.add_final = True
                            task_configs = [task_config]
                        elif existing_tasks:
                            task_config = existing_tasks[0]  # Fallback to latest
                            task_config.add_final = True
                            task_configs = [task_config]
                    elif choice == UserChoice.ALL_NEW:
                        if existing_tasks:
                            for task in existing_tasks:
                                task.add_final = True
                            task_configs = list(existing_tasks)
                            execution_mode = "batch"
                    elif choice == UserChoice.NEW:
                        job_configs = self.find_jobs_by_name(job_name)
                        if job_configs:
                            config_data = self.config_manager.load_cascading_config(
                                job_configs[0].config_path
                            )
                            new_task = self.create_new_task(config_data)
                            task_configs = [new_task]
                    elif choice == UserChoice.ALL:
                        task_configs = list(existing_tasks)
                        execution_mode = "batch"
            else:
                # No existing tasks, create new one
                job_configs = self.find_jobs_by_name(job_name)
                if job_configs:
                    config_data = self.config_manager.load_cascading_config(
                        job_configs[0].config_path
                    )
                    new_task = self.create_new_task(config_data)
                    task_configs = [new_task]
                else:
                    raise ValueError(f"No job configuration found for '{job_name}'")

        elif config_files:
            # Config file(s) provided as arguments
            for config_file in config_files:
                if self.is_task_config(config_file):
                    # Direct task config - execute immediately
                    task_config = self.config_manager.load_task_config(config_file)
                    task_configs.append(task_config)
                else:
                    # Job config - check for existing tasks
                    config_data = self.config_manager.load_cascading_config(config_file)
                    job_name = config_data["job"]["name"]
                    existing_tasks = self.find_existing_tasks(job_name)

                    if existing_tasks:
                        # Apply strategy for this job
                        strategy = job_strategies.get(job_name, global_strategy)

                        if strategy == ExecutionStrategy.NEW:
                            new_task = self.create_new_task(config_data)
                            task_configs.append(new_task)
                        elif strategy == ExecutionStrategy.ALL:
                            task_configs.extend(existing_tasks)
                        elif strategy == ExecutionStrategy.ALL_NEW:
                            # Use all tasks + force new final audio
                            all_tasks = list(existing_tasks)
                            for task in all_tasks:
                                task.add_final = True
                            task_configs.extend(all_tasks)
                        elif strategy == ExecutionStrategy.LAST or strategy == ExecutionStrategy.LATEST:
                            task_configs.append(existing_tasks[0])
                        elif strategy == ExecutionStrategy.LAST_NEW:
                            # Use latest task + force new final audio
                            task_config = existing_tasks[0]  # First in sorted list (newest)
                            task_config.add_final = True
                            task_configs.append(task_config)
                        else:
                            # No strategy specified - interactive selection
                            if global_strategy is None:
                                requires_user_input = True
                            choice = self.prompt_user_selection(existing_tasks)

                            if choice == UserChoice.CANCEL:
                                continue
                            elif choice == UserChoice.LATEST:
                                if existing_tasks:
                                    task_configs.append(existing_tasks[0])  # First in sorted list (newest)
                            elif choice == UserChoice.SPECIFIC:
                                if existing_tasks and hasattr(self, 'selected_task_index'):
                                    task_configs.append(existing_tasks[self.selected_task_index])
                                elif existing_tasks:
                                    task_configs.append(existing_tasks[0])  # Fallback to latest
                            elif choice == UserChoice.LATEST_NEW:
                                if existing_tasks:
                                    task_config = existing_tasks[0]  # First in sorted list (newest)
                                    task_config.add_final = True
                                    task_configs.append(task_config)
                            elif choice == UserChoice.SPECIFIC_NEW:
                                if existing_tasks and hasattr(self, 'selected_task_index'):
                                    task_config = existing_tasks[self.selected_task_index]
                                    task_config.add_final = True
                                    task_configs.append(task_config)
                                elif existing_tasks:
                                    task_config = existing_tasks[0]  # Fallback to latest
                                    task_config.add_final = True
                                    task_configs.append(task_config)
                            elif choice == UserChoice.ALL_NEW:
                                if existing_tasks:
                                    for task in existing_tasks:
                                        task.add_final = True
                                    task_configs.extend(existing_tasks)
                            elif choice == UserChoice.NEW:
                                new_task = self.create_new_task(config_data)
                                task_configs.append(new_task)
                            elif choice == UserChoice.ALL:
                                task_configs.extend(existing_tasks)
                    else:
                        # No existing tasks, create new one
                        new_task = self.create_new_task(config_data)
                        task_configs.append(new_task)

            if len(config_files) > 1 or len(task_configs) > 1:
                execution_mode = "batch"

        else:
            # No arguments - use default job
            default_config = self.config_manager.load_default_config()
            job_name = default_config["job"]["name"]  # "default"
            existing_tasks = self.find_existing_tasks(job_name)

            if existing_tasks:
                # Apply strategy for default job  
                strategy = job_strategies.get(job_name, global_strategy)

                if strategy == ExecutionStrategy.NEW:
                    new_task = self.create_new_task(default_config)
                    task_configs = [new_task]
                elif strategy == ExecutionStrategy.ALL:
                    task_configs = list(existing_tasks)
                    execution_mode = "batch"
                elif strategy == ExecutionStrategy.ALL_NEW:
                    # Use all tasks + force new final audio
                    task_configs = list(existing_tasks)
                    for task in task_configs:
                        task.add_final = True
                    execution_mode = "batch"
                elif strategy == ExecutionStrategy.LAST or strategy == ExecutionStrategy.LATEST:
                    task_configs = [existing_tasks[0]]  # Use first (newest) not last
                elif strategy == ExecutionStrategy.LAST_NEW:
                    # Use latest task + force new final audio
                    task_config = existing_tasks[0]  # First in sorted list (newest)
                    task_config.add_final = True
                    task_configs = [task_config]
                else:
                    # No strategy specified - interactive selection
                    if global_strategy is None:
                        requires_user_input = True
                    choice = self.prompt_user_selection(existing_tasks)

                    if choice == UserChoice.CANCEL:
                        return ExecutionPlan([], "cancelled")
                    elif choice == UserChoice.LATEST:
                        if existing_tasks:
                            task_configs = [existing_tasks[0]]  # First in sorted list (newest)
                    elif choice == UserChoice.SPECIFIC:
                        if existing_tasks and hasattr(self, 'selected_task_index'):
                            task_configs = [existing_tasks[self.selected_task_index]]
                        elif existing_tasks:
                            task_configs = [existing_tasks[0]]  # Fallback to latest
                    elif choice == UserChoice.LATEST_NEW:
                        if existing_tasks:
                            task_config = existing_tasks[0]  # First in sorted list (newest)
                            task_config.add_final = True
                            task_configs = [task_config]
                    elif choice == UserChoice.SPECIFIC_NEW:
                        if existing_tasks and hasattr(self, 'selected_task_index'):
                            task_config = existing_tasks[self.selected_task_index]
                            task_config.add_final = True
                            task_configs = [task_config]
                        elif existing_tasks:
                            task_config = existing_tasks[0]  # Fallback to latest
                            task_config.add_final = True
                            task_configs = [task_config]
                    elif choice == UserChoice.ALL_NEW:
                        if existing_tasks:
                            for task in existing_tasks:
                                task.add_final = True
                            task_configs = list(existing_tasks)
                            execution_mode = "batch"
                    elif choice == UserChoice.NEW:
                        new_task = self.create_new_task(default_config)
                        task_configs = [new_task]
                    elif choice == UserChoice.ALL:
                        task_configs = list(existing_tasks)
                        execution_mode = "batch"
            else:
                # No existing tasks, create new one
                new_task = self.create_new_task(default_config)
                task_configs = [new_task]

        # Set add_final flag for all tasks if requested
        if hasattr(args, 'add_final') and args.add_final:
            for task_config in task_configs:
                task_config.add_final = True

        return ExecutionPlan(
            task_configs=task_configs,
            execution_mode=execution_mode,
            requires_user_input=requires_user_input,
        )

    def validate_execution_plan(self, plan: ExecutionPlan) -> bool:
        """
        Validates the generated execution plan.

        """
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
        """
        Validate that mixed task configurations are compatible.
        Returns:
            True if configurations are compatible
        """
        logger.info("🔍 Validating configuration compatibility...")

        # Group by job names
        jobs_by_name = {}
        task_configs_by_type = {"job_config": [], "task_config": []}

        for task_config in task_configs:
            # Determine if this is from a job config or task config
            if self.is_task_config(task_config.config_path):
                task_configs_by_type["task_config"].append(task_config)
            else:
                task_configs_by_type["job_config"].append(task_config)

            # Group by job name
            job_name = task_config.job_name
            if job_name not in jobs_by_name:
                jobs_by_name[job_name] = []
            jobs_by_name[job_name].append(task_config)

        # Check for potential conflicts
        warnings = []
        errors = []

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

    def print_execution_summary(self, plan: ExecutionPlan) -> None:

        logger.info("")
        logger.info("=" * 50)
        logger.info("📋 EXECUTION PLAN SUMMARY")
        logger.info(f"  Mode: {plan.execution_mode.upper()}")
        logger.info(f"  Tasks: {len(plan.task_configs)}")

        for i, task in enumerate(plan.task_configs, 1):
            run_label = f" ({task.run_label})" if task.run_label else ""
            logger.info(f"  {i}. {task.job_name}: {task.task_name}{run_label}")
            logger.info(f"     └─ {task.base_output_dir}")

        logger.info("=" * 50)
