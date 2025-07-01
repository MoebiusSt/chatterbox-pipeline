#!/usr/bin/env python3
"""
User interaction functionality for job management.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils.config_manager import ConfigManager, TaskConfig
from utils.file_manager.state_analyzer import TaskState

from .types import UserChoice

logger = logging.getLogger(__name__)


class UserInteraction:
    """Handles user interactions for task selection."""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.selected_task_index = 0

    def prompt_user_selection(self, tasks: List[TaskConfig]) -> UserChoice:
        """Prompt user to select task execution strategy."""
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

            print(
                f"{i}. {task.job_name} - {run_label_display} - {text_file}.txt - {date_str} - {time_str}{latest_marker}"
            )

        print("\nSelect action:")
        print("[Enter] - Options for latest task")
        print("n       - Create and run new task")
        print("a       - Options to run all tasks")
        print("1-{}   - Options for specific task".format(len(tasks)))
        print("c       - Cancel")

        choice = input("\n> ").strip().lower()

        if choice == "":
            return UserChoice.LATEST
        elif choice == "n":
            return UserChoice.NEW
        elif choice == "a":
            return UserChoice.ALL_OPTIONS
        elif choice == "c":
            return UserChoice.CANCEL
        elif choice.isdigit() and 1 <= int(choice) <= len(tasks):
            # Store the selected task index (convert to 0-based)
            self.selected_task_index = int(choice) - 1
            return UserChoice.SPECIFIC
        
        print("Invalid choice, defaulting to latest task")
        return UserChoice.LATEST
    
    def confirm_rerender_action(self, action_description: str) -> Optional[bool]:
        """
        Show safety prompt for re-render actions that will delete existing audio chunks.
        
        Args:
            action_description: Description of the action being confirmed
            
        Returns:
            True if user confirms, None if user cancels
        """
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

    def show_task_options_with_state(self, task: TaskConfig, task_state: TaskState, is_latest: bool = True) -> UserChoice:
        """
        Show task options with state information - the enhanced second prompt.
        
        Args:
            task: Selected TaskConfig
            task_state: TaskState analysis
            is_latest: Whether this is the latest task or a specifically selected one
            
        Returns:
            UserChoice for the action to take
        """
        # Parse timestamp for display
        try:
            dt = datetime.strptime(task.timestamp, "%Y%m%d_%H%M%S")
            display_time = dt.strftime("%d.%m.%Y %H:%M")
        except ValueError:
            display_time = task.timestamp
            
        task_type = "latest task" if is_latest else "task"
        
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
                return UserChoice.LATEST_FILL_GAPS
            elif choice == "s":
                return UserChoice.LATEST_FILL_GAPS_NO_OVERWRITE
            elif choice == "r":
                confirmation = self.confirm_rerender_action("RE-RENDER ALL candidates")
                if confirmation is True:
                    return UserChoice.LATEST_RERENDER_ALL
                elif confirmation is None:
                    # User cancelled - show menu again and continue loop
                    show_menu()
                    continue
            elif choice == "e":
                if task_state.candidate_editor_available:
                    return UserChoice.EDIT
                else:
                    print("Edit option not available - task incomplete or no candidate data")
            elif choice == "c":
                return UserChoice.RETURN
            else:
                print("Invalid choice. Please try again.")
                
    def show_all_tasks_options(self, tasks: List[TaskConfig]) -> UserChoice:
        """
        Show options for all tasks - the new third menu.
        
        Args:
            tasks: List of all available TaskConfigs
            
        Returns:
            UserChoice for the action to take on all tasks
        """
        job_name = tasks[0].job_name if tasks else "Unknown"
        
        def show_menu():
            print(f"\nOptions for ALL tasks in job '{job_name}' ({len(tasks)} tasks):")
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
                return UserChoice.ALL_FILL_GAPS
            elif choice == "s":
                return UserChoice.ALL_FILL_GAPS_NO_OVERWRITE
            elif choice == "r":
                confirmation = self.confirm_rerender_action("RE-RENDER ALL tasks")
                if confirmation is True:
                    return UserChoice.ALL_RERENDER_ALL
                elif confirmation is None:
                    # User cancelled - show menu again and continue loop
                    show_menu()
                    continue
            elif choice == "c":
                return UserChoice.RETURN
            else:
                print("Invalid choice. Please try again.")

    def generate_task_info_dict(self, task: TaskConfig, is_latest: bool = True) -> Dict:
        """Generate task info dictionary for display purposes."""
        try:
            dt = datetime.strptime(task.timestamp, "%Y%m%d_%H%M%S")
            display_time = dt.strftime("%d.%m.%Y %H:%M")
        except ValueError:
            display_time = task.timestamp
            
        task_type = "latest task" if is_latest else "task"
        
        return {
            "job_name": task.job_name,
            "run_label": task.run_label,
            "display_time": display_time,
            "timestamp": task.timestamp,
            "task_type": task_type
        }
