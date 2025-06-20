#!/usr/bin/env python3
"""
User interaction functionality for job management.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List

from utils.config_manager import ConfigManager, TaskConfig

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
