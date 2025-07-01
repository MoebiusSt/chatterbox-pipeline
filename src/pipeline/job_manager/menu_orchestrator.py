#!/usr/bin/env python3
"""
Menu Orchestrator - Central menu logic for all execution paths.
Replaces the 3x duplicated menu logic in execution_planner.py.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils.config_manager import ConfigManager, TaskConfig
from utils.file_manager.file_manager import FileManager
from utils.file_manager.state_analyzer import TaskState

from .execution_types import (
    AllTasksChoice,
    ExecutionContext,
    ExecutionIntent,
    ExecutionOptions,
    MenuLevel,
    MenuResult,
    TaskOptionsChoice,
    TaskSelectionChoice,
)

logger = logging.getLogger(__name__)


class MenuOrchestrator:
    """Central menu orchestrator for unified user interaction across all execution paths."""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.selected_task_index = 0

    def resolve_user_intent(self, context: ExecutionContext) -> ExecutionIntent:
        """
        Resolve user intent through hierarchical menu system.
        
        Args:
            context: Execution context with tasks and configuration
            
        Returns:
            ExecutionIntent representing the user's decision
        """
        if not context.has_existing_tasks():
            # No existing tasks - create new one
            return self._create_new_task_intent(context)

        # Hierarchical menu navigation
        while True:
            # Level 1: Task Selection
            selection_result = self._show_task_selection_menu(context)
            
            if selection_result.is_final_choice():
                intent = selection_result.execution_intent
                if intent is not None:
                    return intent
            
            if not selection_result.should_continue_menu():
                # User cancelled or made error
                return self._create_cancelled_intent()
            
            # Level 2: Task Options (if required)
            if selection_result.choice == TaskSelectionChoice.ALL_OPTIONS:
                options_result = self._show_all_tasks_options_menu(context)
            else:
                # Individual task options
                task = self._get_selected_task(context, selection_result)
                if not task:
                    continue  # Back to task selection
                
                options_result = self._show_individual_task_options_menu(task, context, selection_result)
            
            if options_result.is_final_choice():
                intent = options_result.execution_intent
                if intent is not None:
                    return intent
            
            if options_result.choice in [TaskOptionsChoice.RETURN, AllTasksChoice.RETURN]:
                continue  # Back to Level 1
            
            # Level 3: Candidate Editor (if required)
            if hasattr(options_result, 'choice') and options_result.choice == TaskOptionsChoice.EDIT_CANDIDATES:
                task = self._get_selected_task(context, selection_result)
                if task:
                    editor_result = self._show_candidate_editor(task, context)
                    if editor_result.is_final_choice():
                        intent = editor_result.execution_intent
                        if intent is not None:
                            return intent
                # Otherwise continue menu loop

    def _show_task_selection_menu(self, context: ExecutionContext) -> MenuResult:
        """Show Level 1 menu - task selection."""
        job_name = context.job_name
        tasks = context.existing_tasks
        
        print(f"\nFound existing tasks for job '{job_name}':")
        
        # Display tasks with consistent formatting
        for i, task in enumerate(tasks, 1):
            task_display = self._format_task_display(task)
            latest_marker = " (<-- latest)" if i == 1 else ""
            print(f"{i}. {task_display}{latest_marker}")

        print("\nSelect action:")
        print("[Enter] - Options for latest task")
        print("n       - Create and run new task")
        print("a       - Options to run all tasks")
        print(f"1-{len(tasks)}   - Options for specific task")
        print("c       - Cancel")

        choice = input("\n> ").strip().lower()

        if choice == "":
            return MenuResult(
                choice=TaskSelectionChoice.LATEST,
                requires_next_level=True
            )
        elif choice == "n":
            return MenuResult(
                choice=TaskSelectionChoice.NEW,
                execution_intent=self._create_new_task_intent(context)
            )
        elif choice == "a":
            return MenuResult(
                choice=TaskSelectionChoice.ALL_OPTIONS,
                requires_next_level=True
            )
        elif choice == "c":
            return MenuResult(choice=TaskSelectionChoice.CANCEL)
        elif choice.isdigit() and 1 <= int(choice) <= len(tasks):
            task_index = int(choice) - 1
            return MenuResult(
                choice=TaskSelectionChoice.SPECIFIC,
                selected_task_index=task_index,
                requires_next_level=True
            )
        else:
            print("Invalid choice, defaulting to latest task")
            return MenuResult(
                choice=TaskSelectionChoice.LATEST,
                requires_next_level=True
            )

    def _show_all_tasks_options_menu(self, context: ExecutionContext) -> MenuResult:
        """Show Level 2 menu - all tasks options."""
        
        job_name = context.job_name
        task_count = len(context.existing_tasks)
        
        def show_menu():
            print(f"\nOptions for ALL tasks in job '{job_name}' ({task_count} tasks):")
            print()
            print("What to do with ALL tasks?")
            print("[Enter] - Run tasks, fill gaps, CREATE (or overwrite) final audio-files")
            print("s       - Run tasks, fill gaps, KEEP (skip) final audio-files")
            print("r       - Run tasks, RE-RENDER ALL candidates, create new final audio-files")
            print("c       - Return")
        
        show_menu()
        
        while True:
            choice = input("\n> ").strip().lower()
            
            if choice == "":
                return MenuResult(
                    choice=AllTasksChoice.ALL_FILL_GAPS,
                    execution_intent=self._create_all_tasks_intent(context.existing_tasks, ExecutionOptions(force_final_generation=True))
                )
            elif choice == "s":
                return MenuResult(
                    choice=AllTasksChoice.ALL_FILL_GAPS_NO_OVERWRITE,
                    execution_intent=self._create_all_tasks_intent(context.existing_tasks, ExecutionOptions(force_final_generation=True, skip_final_overwrite=True))
                )
            elif choice == "r":
                confirmation = self._confirm_rerender_action("RE-RENDER ALL tasks")
                if confirmation is True:
                    return MenuResult(
                        choice=AllTasksChoice.ALL_RERENDER_ALL,
                        execution_intent=self._create_all_tasks_intent(context.existing_tasks, ExecutionOptions(force_final_generation=True, rerender_all=True))
                    )
                elif confirmation is None:
                    # User cancelled - show menu again and continue loop
                    show_menu()
                    continue
            elif choice == "c":
                return MenuResult(choice=AllTasksChoice.RETURN)
            else:
                print("Invalid choice. Please try again.")

    def _show_individual_task_options_menu(self, task: TaskConfig, context: ExecutionContext, selection_result: MenuResult) -> MenuResult:
        """Show Level 2 menu - individual task options."""
        
        # Load config and analyze task state
        try:
            config_data = self.config_manager.load_cascading_config(task.config_path)
            file_manager = FileManager(task, preloaded_config=config_data, config_manager=self.config_manager)
            task_state = file_manager.analyze_task_state()
        except Exception as e:
            logger.warning(f"Task state analysis failed: {e}")
            # Fallback to simple execution
            return MenuResult(
                choice=TaskOptionsChoice.FILL_GAPS,
                execution_intent=self._create_single_task_intent(task, ExecutionOptions(force_final_generation=True))
            )

        # Display task information
        is_latest = (selection_result.choice == TaskSelectionChoice.LATEST)
        task_type = "latest task" if is_latest else "task"
        
        try:
            dt = datetime.strptime(task.timestamp, "%Y%m%d_%H%M%S")
            display_time = dt.strftime("%d.%m.%Y %H:%M")
        except ValueError:
            display_time = task.timestamp
        
        def show_menu():
            print(f"\nSelected {task_type}: {task.job_name} - {display_time}")
            print(f"\nTask state: {task_state.task_status_message}")
            print()

            print("What to do with this task?")
            print("[Enter] - Run task, fill gaps, CREATE (or overwrite) final audio")
            print("s       - Run task, fill gaps, KEEP (skip) final audio")
            print("r       - Run task, RE-RENDER ALL candidates, create new final audio")
            
            if task_state.candidate_editor_available:
                print("e       - Edit completed task (choose different candidates)")
            else:
                print("N/A     - Edit completed task (not available - task incomplete or no candidate data)")
                
            print("c       - Return")

        show_menu()

        while True:
            choice = input("\n> ").strip().lower()
            
            if choice == "":
                options = ExecutionOptions(force_final_generation=True)
                return MenuResult(
                    choice=TaskOptionsChoice.FILL_GAPS,
                    execution_intent=self._create_single_task_intent(task, options)
                )
            elif choice == "s":
                options = ExecutionOptions(force_final_generation=True, skip_final_overwrite=True)
                return MenuResult(
                    choice=TaskOptionsChoice.FILL_GAPS_NO_OVERWRITE,
                    execution_intent=self._create_single_task_intent(task, options)
                )
            elif choice == "r":
                confirmation = self._confirm_rerender_action("RE-RENDER ALL candidates")
                if confirmation is True:
                    options = ExecutionOptions(force_final_generation=True, rerender_all=True)
                    return MenuResult(
                        choice=TaskOptionsChoice.RERENDER_ALL,
                        execution_intent=self._create_single_task_intent(task, options)
                    )
                elif confirmation is None:
                    # User cancelled - show menu again and continue loop
                    show_menu()
                    continue
            elif choice == "e":
                if task_state.candidate_editor_available:
                    return MenuResult(
                        choice=TaskOptionsChoice.EDIT_CANDIDATES,
                        requires_next_level=True
                    )
                else:
                    print("Edit option not available - task incomplete or no candidate data")
            elif choice == "c":
                return MenuResult(choice=TaskOptionsChoice.RETURN)
            else:
                print("Invalid choice. Please try again.")

    def _show_candidate_editor(self, task: TaskConfig, context: ExecutionContext) -> MenuResult:
        """Show Level 3 menu - candidate editor."""
        
        try:
            from pipeline.user_candidate_manager import UserCandidateManager
            
            config_data = self.config_manager.load_cascading_config(task.config_path)
            file_manager = FileManager(task, preloaded_config=config_data, config_manager=self.config_manager)
            candidate_manager = UserCandidateManager(file_manager, task)
            task_info = self._generate_task_info_dict(task, True)  # Assume latest for editor
            
            # Candidate editor loop
            while True:
                candidate_manager.show_candidate_overview(task_info)
                editor_choice = input("\n> ").strip()
                
                if editor_choice.lower() == 'c':
                    return MenuResult(choice=TaskOptionsChoice.RETURN)
                elif editor_choice.lower() == 'r':
                    # Re-run task - create execution intent
                    options = ExecutionOptions(force_final_generation=True)
                    return MenuResult(
                        choice=TaskOptionsChoice.FILL_GAPS,
                        execution_intent=self._create_single_task_intent(task, options)
                    )
                elif editor_choice.isdigit():
                    chunk_idx = int(editor_choice) - 1
                    chunks = file_manager.get_chunks()
                    
                    if 0 <= chunk_idx < len(chunks):
                        result = candidate_manager.show_candidate_selector(chunk_idx, task_info)
                    else:
                        print(f"Invalid chunk number. Please enter 1-{len(chunks)} or 'c'")
                else:
                    print("Invalid choice. Please enter a chunk number, 'r', or 'c'")
                    
        except Exception as e:
            logger.error(f"Error in candidate editor: {e}")
            print(f"Error: {e}")
            return MenuResult(choice=TaskOptionsChoice.RETURN)

    def _format_task_display(self, task: TaskConfig) -> str:
        """Format task for display in selection menu."""
        
        # Parse timestamp for better display
        try:
            dt = datetime.strptime(task.timestamp, "%Y%m%d_%H%M%S")
            date_str = dt.strftime("%d.%m.%Y")
            time_str = dt.strftime("%H:%M")
        except ValueError:
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

        # Format: "job-name - run-label - doc-name.txt - date - time"
        run_label_display = task.run_label if task.run_label else "no-label"
        return f"{task.job_name} - {run_label_display} - {text_file}.txt - {date_str} - {time_str}"

    def _confirm_rerender_action(self, action_description: str) -> Optional[bool]:
        """Show safety prompt for re-render actions."""
        
        print(f"\n⚠️  WARNING: {action_description}")
        print("This will DELETE (!) ALL audio chunks and final audio files from pre-existing runs!")
        print("Are you sure? (y = YES, RE-RENDER | c = CANCEL)")
        
        while True:
            choice = input("\n> ").strip().lower()
            
            if choice in ["y", "yes"]:
                return True
            elif choice in ["c", "cancel"]:
                return None  # Cancel - return to previous menu
            else:
                print("Please enter 'y' for yes or 'c' to cancel")

    def _generate_task_info_dict(self, task: TaskConfig, is_latest: bool) -> Dict:
        """Generate task info dictionary for candidate editor."""
        
        try:
            dt = datetime.strptime(task.timestamp, "%Y%m%d_%H%M%S")
            display_time = dt.strftime("%d.%m.%Y %H:%M")
        except ValueError:
            display_time = task.timestamp
        
        return {
            'job_name': task.job_name,
            'display_time': display_time,
            'task_type': 'latest task' if is_latest else 'task'
        }

    def _get_selected_task(self, context: ExecutionContext, selection_result: MenuResult) -> Optional[TaskConfig]:
        """Get the selected task based on menu selection."""
        
        if selection_result.choice == TaskSelectionChoice.LATEST:
            return context.get_latest_task()
        elif selection_result.choice == TaskSelectionChoice.SPECIFIC:
            if selection_result.selected_task_index is not None:
                index = selection_result.selected_task_index
                if 0 <= index < len(context.existing_tasks):
                    return context.existing_tasks[index]
        
        return None

    def _create_new_task_intent(self, context: ExecutionContext) -> ExecutionIntent:
        """Create execution intent for new task creation."""
        
        return ExecutionIntent(
            tasks=[],  # Will be populated by execution planner
            execution_mode="single",
            execution_options=ExecutionOptions(force_final_generation=True),
            source="menu"
        )

    def _create_single_task_intent(self, task: TaskConfig, options: ExecutionOptions) -> ExecutionIntent:
        """Create execution intent for single task operation."""
        
        # Apply options to task (legacy field mapping for compatibility)
        task.add_final = options.force_final_generation
        task.skip_final_overwrite = options.skip_final_overwrite
        task.rerender_all = options.rerender_all
        
        return ExecutionIntent(
            tasks=[task],
            execution_mode="single",
            execution_options=options,
            source="menu"
        )

    def _create_all_tasks_intent(self, tasks: List[TaskConfig], options: ExecutionOptions) -> ExecutionIntent:
        """Create execution intent for all tasks operation."""
        
        # Apply options to all tasks (legacy field mapping for compatibility)
        for task in tasks:
            task.add_final = options.force_final_generation
            task.skip_final_overwrite = options.skip_final_overwrite
            task.rerender_all = options.rerender_all
        
        return ExecutionIntent(
            tasks=tasks,
            execution_mode="batch",
            execution_options=options,
            source="menu"
        )

    def _create_cancelled_intent(self) -> ExecutionIntent:
        """Create execution intent for cancelled operation."""
        
        return ExecutionIntent(
            tasks=[],
            execution_mode="cancelled",
            execution_options=ExecutionOptions(),
            source="menu"
        ) 