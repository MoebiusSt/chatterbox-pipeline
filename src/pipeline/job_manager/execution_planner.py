#!/usr/bin/env python3
"""
Execution planning functionality for job management.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional

from utils.config_manager import ConfigManager, TaskConfig

from .types import ExecutionPlan, ExecutionStrategy, UserChoice
from .user_interaction import UserInteraction

logger = logging.getLogger(__name__)


class ExecutionPlanner:
    """Handles execution plan generation and management."""

    def __init__(self, job_manager, config_manager: ConfigManager):
        self.job_manager = job_manager
        self.config_manager = config_manager
        self.user_interaction = UserInteraction(config_manager)

    def _load_existing_tasks_with_config(self, existing_tasks: List[TaskConfig]) -> None:
        """
        Load configs for existing tasks to embed them and avoid redundant loading.
        
        Args:
            existing_tasks: List of TaskConfig objects to load configs for
        """
        for task in existing_tasks:
            if task.preloaded_config is None:  # Only load if not already loaded
                try:
                    config_data = self.config_manager.load_cascading_config(task.config_path)
                    task.preloaded_config = config_data
                except Exception as e:
                    logger.warning(f"Failed to preload config for task {task.task_name}: {e}")

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
        job_strategies, global_strategy = self.job_manager.parse_mode_argument(
            args.mode
        )

        if args.job:
            # --job "jobname" scenario
            job_name = args.job
            existing_tasks = self.job_manager.find_existing_tasks(job_name)

            if existing_tasks:
                # Apply strategy for this job
                strategy = job_strategies.get(job_name, global_strategy)

                if strategy == ExecutionStrategy.NEW:
                    # Create new task
                    job_configs = self.job_manager.find_jobs_by_name(job_name)
                    if job_configs:
                        config_data = self.config_manager.load_cascading_config(
                            job_configs[0].config_path
                        )
                        new_task = self.job_manager.create_new_task(config_data)
                        new_task.preloaded_config = config_data  # Embed config to avoid redundant loading
                        task_configs = [new_task]
                elif strategy == ExecutionStrategy.ALL:
                    # Use all tasks
                    task_configs = list(existing_tasks)
                    self._load_existing_tasks_with_config(task_configs)  # Preload configs
                    execution_mode = "batch"
                elif strategy == ExecutionStrategy.ALL_NEW:
                    # Use all tasks + force new final audio
                    task_configs = list(existing_tasks)
                    for task in task_configs:
                        task.add_final = True
                    execution_mode = "batch"
                elif (
                    strategy == ExecutionStrategy.LATEST
                    or strategy == ExecutionStrategy.LAST
                ):
                    # Use latest task
                    task_configs = [existing_tasks[0]]
                    self._load_existing_tasks_with_config(task_configs)  # Preload configs
                elif (
                    strategy == ExecutionStrategy.LATEST_NEW
                    or strategy == ExecutionStrategy.LAST_NEW
                ):
                    # Use latest task + force new final audio
                    task_config = existing_tasks[0]  # First in sorted list (newest)
                    task_config.add_final = True
                    task_configs = [task_config]
                else:
                    # No strategy specified - interactive selection
                    if global_strategy is None:
                        requires_user_input = True
                    
                    # First prompt - task selection
                    choice = self.user_interaction.prompt_user_selection(existing_tasks)
                    
                    # For LATEST or SPECIFIC choices, show enhanced second prompt
                    if choice in [UserChoice.LATEST, UserChoice.SPECIFIC]:
                        task_config = None
                        if choice == UserChoice.LATEST:
                            task_config = existing_tasks[0]  # Latest task
                        elif choice == UserChoice.SPECIFIC and hasattr(self.user_interaction, "selected_task_index"):
                            task_config = existing_tasks[self.user_interaction.selected_task_index]
                        elif choice == UserChoice.SPECIFIC:
                            task_config = existing_tasks[0]  # Fallback to latest
                        
                        if task_config:
                            # Load config and analyze task state
                            try:
                                config_data = self.config_manager.load_cascading_config(task_config.config_path)
                                from utils.file_manager.file_manager import FileManager
                                temp_file_manager = FileManager(task_config, preloaded_config=config_data, config_manager=self.config_manager)
                                task_state = temp_file_manager.analyze_task_state()
                                
                                # Enhanced second prompt loop that allows cycling between task options and candidate editor
                                while True:
                                    # Show enhanced second prompt
                                    enhanced_choice = self.user_interaction.show_task_options_with_state(task_config, task_state)
                                    
                                    if enhanced_choice == UserChoice.EDIT:
                                        # Enter candidate editor loop
                                        try:
                                            from pipeline.user_candidate_manager import UserCandidateManager
                                            file_manager = FileManager(task_config, preloaded_config=config_data, config_manager=self.config_manager)
                                            candidate_manager = UserCandidateManager(file_manager, task_config)
                                            task_info = self.user_interaction.generate_task_info_dict(task_config)
                                            
                                            # Candidate editor loop
                                            while True:
                                                candidate_manager.show_candidate_overview(task_info)
                                                editor_choice = input("\n> ").strip()
                                                
                                                if editor_choice.lower() == 'c':
                                                    # Return to task options (break candidate editor loop)
                                                    break
                                                elif editor_choice.lower() == 'r':
                                                    # Re-run task - set final choice and exit both loops
                                                    task_config.add_final = True
                                                    choice = UserChoice.LATEST_FILL_GAPS
                                                    break
                                                elif editor_choice.isdigit():
                                                    chunk_idx = int(editor_choice) - 1  # Convert to 0-based
                                                    chunks = file_manager.get_chunks()
                                                    
                                                    if 0 <= chunk_idx < len(chunks):
                                                        # Show candidate selector for this chunk
                                                        result = candidate_manager.show_candidate_selector(chunk_idx, task_info)
                                                        # Continue loop to show overview again
                                                    else:
                                                        print(f"Invalid chunk number. Please enter 1-{len(chunks)} or 'c'")
                                                else:
                                                    print("Invalid choice. Please enter a chunk number, 'r', or 'c'")
                                            
                                            # If user chose 'r' in candidate editor, break main loop too
                                            if editor_choice.lower() == 'r':
                                                break
                                                
                                        except Exception as e:
                                            logger.error(f"Error in candidate editor: {e}")
                                            print(f"Error: {e}")
                                            # Continue to task options on error
                                            continue
                                    else:
                                        # User chose something other than EDIT - set choice and break main loop
                                        choice = enhanced_choice
                                        break
                                
                            except Exception as e:
                                # Fallback to original choice if task state analysis fails
                                logger.warning(f"Task state analysis failed, using original choice: {e}")
                                pass

                    if choice == UserChoice.CANCEL:
                        return ExecutionPlan([], "cancelled")
                    elif choice == UserChoice.LATEST:
                        if existing_tasks:
                            task_configs = [
                                existing_tasks[0]
                            ]  # First in sorted list (newest)
                    elif choice == UserChoice.SPECIFIC:
                        if existing_tasks and hasattr(
                            self.user_interaction, "selected_task_index"
                        ):
                            task_configs = [
                                existing_tasks[
                                    self.user_interaction.selected_task_index
                                ]
                            ]
                        elif existing_tasks:
                            task_configs = [existing_tasks[0]]  # Fallback to latest
                    elif choice == UserChoice.LATEST_NEW:
                        if existing_tasks:
                            task_config = existing_tasks[
                                0
                            ]  # First in sorted list (newest)
                            task_config.add_final = True
                            task_configs = [task_config]
                    elif choice == UserChoice.SPECIFIC_NEW:
                        if existing_tasks and hasattr(
                            self.user_interaction, "selected_task_index"
                        ):
                            task_config = existing_tasks[
                                self.user_interaction.selected_task_index
                            ]
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
                        job_configs = self.job_manager.find_jobs_by_name(job_name)
                        if job_configs:
                            config_data = self.config_manager.load_cascading_config(
                                job_configs[0].config_path
                            )
                            new_task = self.job_manager.create_new_task(config_data)
                            new_task.preloaded_config = config_data  # Embed config to avoid redundant loading
                            task_configs = [new_task]
                    elif choice == UserChoice.ALL:
                        task_configs = list(existing_tasks)
                        execution_mode = "batch"
                    elif choice == UserChoice.LATEST_FILL_GAPS:
                        # Gap-filling with new final audio
                        if existing_tasks:
                            task_config = existing_tasks[0]  # Latest task
                            task_config.add_final = True
                            task_configs = [task_config]
                    elif choice == UserChoice.LATEST_FILL_GAPS_NO_OVERWRITE:
                        # Gap-filling without overwriting final audio
                        if existing_tasks:
                            task_config = existing_tasks[0]  # Latest task
                            task_config.add_final = True  # Enable gap-filling
                            task_config.skip_final_overwrite = True  # Don't overwrite existing final audio
                            task_configs = [task_config]
                    elif choice == UserChoice.LATEST_RERENDER_ALL:
                        # Re-render all candidates from scratch
                        if existing_tasks:
                            task_config = existing_tasks[0]  # Latest task
                            task_config.add_final = True
                            task_config.rerender_all = True  # New flag to indicate full re-rendering
                            task_configs = [task_config]
            else:
                # No existing tasks, create new one
                job_configs = self.job_manager.find_jobs_by_name(job_name)
                if job_configs:
                    config_data = self.config_manager.load_cascading_config(
                        job_configs[0].config_path
                    )
                    new_task = self.job_manager.create_new_task(config_data)
                    new_task.preloaded_config = config_data  # Embed config to avoid redundant loading
                    task_configs = [new_task]
                else:
                    logger.error(f"❌ No job configuration found for '{job_name}'!")
                    logger.error("🔍 Available jobs:")
                    
                    # List available jobs in config directory
                    available_jobs = []
                    for config_file in self.config_manager.config_dir.glob("*.yaml"):
                        if config_file.name == "default_config.yaml":
                            continue
                        try:
                            config_data = self.config_manager.load_job_config(config_file)
                            available_job_name = config_data.get("job", {}).get("name")
                            if available_job_name:
                                available_jobs.append(f"- {available_job_name} (from {config_file.name})")
                        except Exception:
                            pass
                    
                    if available_jobs:
                        for job in available_jobs:
                            logger.error(job)
                    else:
                        logger.error("  No valid job configurations found in the config/ directory.")
                    
                    logger.error(f"⚠️ Use: python {sys.argv[0] if 'sys' in globals() else 'main.py'} --job \"<job_name>\"")
                    
                    # Return cancelled execution plan instead of raising exception
                    return ExecutionPlan([], "cancelled")

        elif config_files:
            # Config file(s) provided as arguments
            for config_file in config_files:
                if self.job_manager.is_task_config(config_file):
                    # Direct task config - execute immediately (config already preloaded in load_task_config)
                    task_config = self.config_manager.load_task_config(config_file)
                    task_configs.append(task_config)
                else:
                    # Job config - check for existing tasks
                    config_data = self.config_manager.load_cascading_config(config_file)
                    job_name = config_data["job"]["name"]
                    existing_tasks = self.job_manager.find_existing_tasks(job_name)

                    if existing_tasks:
                        # Apply strategy for this job
                        strategy = job_strategies.get(job_name, global_strategy)

                        if strategy == ExecutionStrategy.NEW:
                            new_task = self.job_manager.create_new_task(config_data)
                            new_task.preloaded_config = config_data  # Embed config to avoid redundant loading
                            task_configs.append(new_task)
                        elif strategy == ExecutionStrategy.ALL:
                            task_configs.extend(existing_tasks)
                        elif strategy == ExecutionStrategy.ALL_NEW:
                            # Use all tasks + force new final audio
                            all_tasks = list(existing_tasks)
                            for task in all_tasks:
                                task.add_final = True
                            task_configs.extend(all_tasks)
                        elif (
                            strategy == ExecutionStrategy.LATEST
                            or strategy == ExecutionStrategy.LAST
                        ):
                            task_configs.append(existing_tasks[0])
                        elif (
                            strategy == ExecutionStrategy.LATEST_NEW
                            or strategy == ExecutionStrategy.LAST_NEW
                        ):
                            # Use latest task + force new final audio
                            task_config = existing_tasks[
                                0
                            ]  # First in sorted list (newest)
                            task_config.add_final = True
                            task_configs.append(task_config)
                        else:
                            # No strategy specified - interactive selection
                            if global_strategy is None:
                                requires_user_input = True
                            choice = self.user_interaction.prompt_user_selection(
                                existing_tasks
                            )

                            if choice == UserChoice.CANCEL:
                                continue
                            elif choice == UserChoice.LATEST:
                                if existing_tasks:
                                    task_configs.append(
                                        existing_tasks[0]
                                    )  # First in sorted list (newest)
                            elif choice == UserChoice.SPECIFIC:
                                if existing_tasks and hasattr(
                                    self.user_interaction, "selected_task_index"
                                ):
                                    task_configs.append(
                                        existing_tasks[
                                            self.user_interaction.selected_task_index
                                        ]
                                    )
                                elif existing_tasks:
                                    task_configs.append(
                                        existing_tasks[0]
                                    )  # Fallback to latest
                            elif choice == UserChoice.LATEST_NEW:
                                if existing_tasks:
                                    task_config = existing_tasks[
                                        0
                                    ]  # First in sorted list (newest)
                                    task_config.add_final = True
                                    task_configs.append(task_config)
                            elif choice == UserChoice.SPECIFIC_NEW:
                                if existing_tasks and hasattr(
                                    self.user_interaction, "selected_task_index"
                                ):
                                    task_config = existing_tasks[
                                        self.user_interaction.selected_task_index
                                    ]
                                    task_config.add_final = True
                                    task_configs.append(task_config)
                                elif existing_tasks:
                                    task_config = existing_tasks[
                                        0
                                    ]  # Fallback to latest
                                    task_config.add_final = True
                                    task_configs.append(task_config)
                            elif choice == UserChoice.ALL_NEW:
                                if existing_tasks:
                                    for task in existing_tasks:
                                        task.add_final = True
                                    task_configs.extend(existing_tasks)
                            elif choice == UserChoice.NEW:
                                new_task = self.job_manager.create_new_task(config_data)
                                new_task.preloaded_config = config_data  # Embed config to avoid redundant loading
                                task_configs.append(new_task)
                            elif choice == UserChoice.ALL:
                                task_configs.extend(existing_tasks)
                            elif choice == UserChoice.LATEST_FILL_GAPS:
                                # Gap-filling with new final audio
                                if existing_tasks:
                                    task_config = existing_tasks[0]  # Latest task
                                    task_config.add_final = True
                                    task_configs.append(task_config)
                            elif choice == UserChoice.LATEST_FILL_GAPS_NO_OVERWRITE:
                                # Gap-filling without overwriting final audio
                                if existing_tasks:
                                    task_config = existing_tasks[0]  # Latest task
                                    task_config.add_final = True  # Enable gap-filling
                                    task_config.skip_final_overwrite = True  # Don't overwrite existing final audio
                                    task_configs = [task_config]
                            elif choice == UserChoice.LATEST_RERENDER_ALL:
                                # Re-render all candidates from scratch
                                if existing_tasks:
                                    task_config = existing_tasks[0]  # Latest task
                                    task_config.add_final = True
                                    task_config.rerender_all = True  # New flag to indicate full re-rendering
                                    task_configs.append(task_config)
                    else:
                        # No existing tasks, create new one
                        new_task = self.job_manager.create_new_task(config_data)
                        new_task.preloaded_config = config_data  # Embed config to avoid redundant loading
                        task_configs.append(new_task)

            if len(config_files) > 1 or len(task_configs) > 1:
                execution_mode = "batch"

        else:
            # No arguments - use default job
            default_config = self.config_manager.load_default_config()
            job_name = default_config["job"]["name"]  # "default"
            existing_tasks = self.job_manager.find_existing_tasks(job_name)

            if existing_tasks:
                # Apply strategy for default job
                strategy = job_strategies.get(job_name, global_strategy)

                if strategy == ExecutionStrategy.NEW:
                    new_task = self.job_manager.create_new_task(default_config)
                    new_task.preloaded_config = default_config  # Embed config to avoid redundant loading
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
                elif (
                    strategy == ExecutionStrategy.LATEST
                    or strategy == ExecutionStrategy.LAST
                ):
                    task_configs = [existing_tasks[0]]  # Use first (newest) not last
                elif (
                    strategy == ExecutionStrategy.LATEST_NEW
                    or strategy == ExecutionStrategy.LAST_NEW
                ):
                    # Use latest task + force new final audio
                    task_config = existing_tasks[0]  # First in sorted list (newest)
                    task_config.add_final = True
                    task_configs = [task_config]
                else:
                    # No strategy specified - interactive selection
                    if global_strategy is None:
                        requires_user_input = True
                    choice = self.user_interaction.prompt_user_selection(existing_tasks)

                    if choice == UserChoice.CANCEL:
                        return ExecutionPlan([], "cancelled")
                    elif choice == UserChoice.LATEST:
                        if existing_tasks:
                            task_configs = [
                                existing_tasks[0]
                            ]  # First in sorted list (newest)
                    elif choice == UserChoice.SPECIFIC:
                        if existing_tasks and hasattr(
                            self.user_interaction, "selected_task_index"
                        ):
                            task_configs = [
                                existing_tasks[
                                    self.user_interaction.selected_task_index
                                ]
                            ]
                        elif existing_tasks:
                            task_configs = [existing_tasks[0]]  # Fallback to latest
                    elif choice == UserChoice.LATEST_NEW:
                        if existing_tasks:
                            task_config = existing_tasks[
                                0
                            ]  # First in sorted list (newest)
                            task_config.add_final = True
                            task_configs = [task_config]
                    elif choice == UserChoice.SPECIFIC_NEW:
                        if existing_tasks and hasattr(
                            self.user_interaction, "selected_task_index"
                        ):
                            task_config = existing_tasks[
                                self.user_interaction.selected_task_index
                            ]
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
                        new_task = self.job_manager.create_new_task(default_config)
                        new_task.preloaded_config = default_config  # Embed config to avoid redundant loading
                        task_configs = [new_task]
                    elif choice == UserChoice.ALL:
                        task_configs = list(existing_tasks)
                        execution_mode = "batch"
            else:
                # No existing tasks, create new one
                new_task = self.job_manager.create_new_task(default_config)
                new_task.preloaded_config = default_config  # Embed config to avoid redundant loading
                task_configs = [new_task]

        # Set add_final flag for all tasks if requested
        if hasattr(args, "add_final") and args.add_final:
            for task_config in task_configs:
                task_config.add_final = True

        return ExecutionPlan(
            task_configs=task_configs,
            execution_mode=execution_mode,
            requires_user_input=requires_user_input,
        )

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
