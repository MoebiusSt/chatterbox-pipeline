#!/usr/bin/env python3
"""
Menu Orchestrator - Central menu logic for all execution paths.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from utils.config_manager import ConfigManager, TaskConfig
from utils.file_manager.file_manager import FileManager

from .execution_types import ExecutionContext, ExecutionIntent, ExecutionOptions, MenuResult
from .types import Verb

logger = logging.getLogger(__name__)


class MenuOrchestrator:
    """Central menu orchestrator for unified user interaction across execution paths."""

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager

    def resolve_user_intent(self, context: ExecutionContext) -> ExecutionIntent:
        """Resolve user intent through the interactive menu."""
        if not context.has_existing_tasks():
            job_label = f"'{context.job_name}'" if context.job_name else "this job"
            print(f"\nNo existing tasks found for {job_label}.")
            print("Create a new task? [Y/n]")
            choice = input("> ").strip().lower()
            if choice in ("", "y", "yes"):
                return self._create_new_task_intent()
            return self._create_cancelled_intent()

        while True:
            selection_result = self._show_task_selection_menu(context)
            if selection_result.is_final_choice():
                intent = selection_result.execution_intent
                if intent is not None:
                    return intent
            if selection_result.should_return:
                return self._create_cancelled_intent()
            if not selection_result.should_continue_menu():
                continue

            if selection_result.all_tasks:
                options_result = self._show_task_operation_menu(
                    context.existing_tasks,
                    title=f"ALL tasks in job '{context.job_name}'",
                    allow_edit=False,
                )
            else:
                task = self._get_selected_task(context, selection_result)
                if task is None:
                    continue
                options_result = self._show_individual_task_menu(task, selection_result)

            if options_result.is_final_choice():
                intent = options_result.execution_intent
                if intent is not None:
                    return intent
            if options_result.should_return:
                continue

    def _show_task_selection_menu(self, context: ExecutionContext) -> MenuResult:
        """Show the top-level task and verb selection menu."""
        tasks = context.existing_tasks
        print(f"\nFound existing tasks for job '{context.job_name}':")
        for i, task in enumerate(tasks, 1):
            task_display = self._format_task_display(task)
            latest_marker = " (<-- latest)" if i == 1 else ""
            print(f"{i}. {task_display}{latest_marker}")

        print("\nSelect action:")
        print("[Enter] - Resume latest task (fill gaps; final only if missing)")
        print("f       - Reassemble final audio for latest task")
        print("r       - Rebuild latest task from scratch")
        print("e       - Edit candidates for latest task")
        print("a       - Options for all tasks")
        print("n       - Create and run new task")
        spacing = " " * (6 - len(str(len(tasks))))
        print(f"1-{len(tasks)}{spacing}- Options for specific task")
        print("c       - Cancel")

        choice = input("\n> ").strip().lower()
        latest = context.get_latest_task()

        if choice == "" and latest is not None:
            return MenuResult(
                choice=Verb.RESUME,
                execution_intent=self._create_single_task_intent(latest, Verb.RESUME),
            )
        if choice == "f" and latest is not None:
            return MenuResult(
                choice=Verb.REASSEMBLE,
                execution_intent=self._create_single_task_intent(latest, Verb.REASSEMBLE),
            )
        if choice == "r" and latest is not None:
            if self._confirm_rebuild_action("latest task") is True:
                return MenuResult(
                    choice=Verb.REBUILD,
                    execution_intent=self._create_single_task_intent(latest, Verb.REBUILD),
                )
            return MenuResult(requires_next_level=False)
        if choice == "e" and latest is not None:
            result = self._maybe_open_candidate_editor(latest, is_latest=True)
            # "c" inside the editor means "back to this menu", not "exit everything"
            if result.should_return:
                return MenuResult(requires_next_level=False)
            return result
        if choice == "a":
            return MenuResult(all_tasks=True, requires_next_level=True)
        if choice == "n":
            return MenuResult(
                choice=Verb.CREATE,
                execution_intent=self._create_new_task_intent(),
            )
        if choice == "c":
            return MenuResult(should_return=True)
        if choice.isdigit() and 1 <= int(choice) <= len(tasks):
            return MenuResult(
                selected_task_index=int(choice) - 1,
                requires_next_level=True,
            )

        print("Invalid choice. Please try again.")
        return MenuResult(requires_next_level=False)

    def _show_individual_task_menu(
        self, task: TaskConfig, selection_result: MenuResult
    ) -> MenuResult:
        """Show verb options for one selected task."""
        is_latest = selection_result.selected_task_index in (None, 0)
        try:
            config_data = self.config_manager.load_cascading_config(task.config_path)
            file_manager = FileManager(
                task, preloaded_config=config_data, config_manager=self.config_manager
            )
            task_state = file_manager.analyze_task_state()
            allow_edit = bool(task_state.candidate_editor_available)
            state_message = task_state.task_status_message
        except Exception as e:
            logger.warning("Task state analysis failed: %s", e)
            allow_edit = False
            state_message = "unknown"

        try:
            dt = datetime.strptime(task.timestamp, "%Y%m%d_%H%M%S")
            display_time = dt.strftime("%d.%m.%Y %H:%M")
        except ValueError:
            display_time = task.timestamp

        task_type = "latest task" if is_latest else "task"
        title = f"{task_type}: {task.job_name} - {display_time}\nTask state: {state_message}"
        return self._show_task_operation_menu([task], title=title, allow_edit=allow_edit)

    def _show_task_operation_menu(
        self, tasks: List[TaskConfig], title: str, allow_edit: bool
    ) -> MenuResult:
        """Show verb choices for an already selected task scope."""

        def show_menu() -> None:
            print(f"\nSelected {title}")
            print()
            print("What to do?")
            print("[Enter] - Resume (fill gaps; final only if missing)")
            print("f       - Reassemble final audio")
            print("r       - Rebuild from scratch")
            if allow_edit:
                print("e       - Edit candidates")
            elif len(tasks) == 1:
                print("N/A     - Edit candidates (task incomplete or no candidate data)")
            print("c       - Return")

        show_menu()
        while True:
            choice = input("\n> ").strip().lower()
            if choice == "":
                return self._create_scoped_intent(tasks, Verb.RESUME)
            if choice == "f":
                return self._create_scoped_intent(tasks, Verb.REASSEMBLE)
            if choice == "r":
                if self._confirm_rebuild_action(title) is True:
                    return self._create_scoped_intent(tasks, Verb.REBUILD)
                show_menu()
                continue
            if choice == "e" and allow_edit and len(tasks) == 1:
                return self._show_candidate_editor(tasks[0])
            if choice == "c":
                return MenuResult(should_return=True)
            print("Invalid choice. Please try again.")

    def _show_candidate_editor(
        self,
        task: TaskConfig,
        file_manager: Optional[FileManager] = None,
        config_data: Optional[dict] = None,
    ) -> MenuResult:
        """Show candidate editor and return the chosen follow-up intent."""
        try:
            from pipeline.user_candidate_manager import UserCandidateManager

            if config_data is None:
                config_data = self.config_manager.load_cascading_config(task.config_path)
            if file_manager is None:
                file_manager = FileManager(
                    task, preloaded_config=config_data, config_manager=self.config_manager
                )
            candidate_manager = UserCandidateManager(file_manager, task)
            task_info = self._generate_task_info_dict(task, True)

            while True:
                candidate_manager.show_candidate_overview(task_info)
                editor_choice = input("\n> ").strip()

                if editor_choice.lower() == "c":
                    return MenuResult(should_return=True)
                if editor_choice.lower() == "r":
                    return self._create_scoped_intent([task], Verb.REASSEMBLE)
                if editor_choice.isdigit():
                    chunk_idx = int(editor_choice) - 1
                    chunks = file_manager.get_chunks()
                    if 0 <= chunk_idx < len(chunks):
                        while 0 <= chunk_idx < len(chunks):
                            result = candidate_manager.show_candidate_selector(
                                chunk_idx, task_info
                            )
                            if result == -2:
                                chunk_idx += 1
                            elif result == -3:
                                chunk_idx -= 1
                            else:
                                break
                    else:
                        print(
                            f"Invalid chunk number. Please enter 1-{len(chunks)} or 'c'"
                        )
                    continue
                print("Invalid choice. Please enter a chunk number, 'r', or 'c'")

        except Exception as e:
            logger.error("Error in candidate editor: %s", e)
            print(f"Error: {e}")
            return MenuResult(should_return=True)

    def _maybe_open_candidate_editor(
        self, task: TaskConfig, is_latest: bool
    ) -> MenuResult:
        """Open the editor if candidate data is available for the task."""
        try:
            config_data = self.config_manager.load_cascading_config(task.config_path)
            file_manager = FileManager(
                task, preloaded_config=config_data, config_manager=self.config_manager
            )
            task_state = file_manager.analyze_task_state()
            if task_state.candidate_editor_available:
                return self._show_candidate_editor(task, file_manager=file_manager, config_data=config_data)
        except Exception as e:
            logger.warning("Candidate editor availability check failed: %s", e)

        task_type = "latest task" if is_latest else "task"
        print(f"Edit candidates is not available for this {task_type}.")
        return MenuResult(requires_next_level=False)

    def _format_task_display(self, task: TaskConfig) -> str:
        """Format task for display in selection menu."""
        try:
            dt = datetime.strptime(task.timestamp, "%Y%m%d_%H%M%S")
            date_str = dt.strftime("%d.%m.%Y")
            time_str = dt.strftime("%H:%M")
        except ValueError:
            date_str = "Parse_Error"
            time_str = task.timestamp

        text_file = "unknown"
        try:
            if task.config_path.exists():
                config_data = self.config_manager.load_job_config(task.config_path)
                text_file = Path(config_data["input"]["text_file"]).stem
        except Exception:
            config_name = task.config_path.stem
            if config_name.endswith("_config"):
                config_name = config_name[:-7]
            file_parts = config_name.split("_")
            if len(file_parts) >= 4:
                text_file = "_".join(file_parts[1:-2])
            elif file_parts:
                text_file = file_parts[0]

        run_label_display = task.run_label if task.run_label else "no-label"
        return f"{task.job_name} - {run_label_display} - {text_file}.txt - {date_str} - {time_str}"

    def _confirm_rebuild_action(self, action_description: str) -> Optional[bool]:
        """Show safety prompt for rebuild actions."""
        print(f"\nWARNING: Rebuild {action_description}")
        print("This will delete all audio chunks and final audio files for this scope.")
        print("Are you sure? (y = YES, REBUILD | c = CANCEL)")

        while True:
            choice = input("\n> ").strip().lower()
            if choice in ["y", "yes"]:
                return True
            if choice in ["c", "cancel"]:
                return None
            print("Please enter 'y' for yes or 'c' to cancel")

    def _generate_task_info_dict(self, task: TaskConfig, is_latest: bool) -> Dict:
        """Generate task info dictionary for candidate editor."""
        try:
            dt = datetime.strptime(task.timestamp, "%Y%m%d_%H%M%S")
            display_time = dt.strftime("%d.%m.%Y %H:%M")
        except ValueError:
            display_time = task.timestamp

        return {
            "job_name": task.job_name,
            "display_time": display_time,
            "task_type": "latest task" if is_latest else "task",
        }

    def _get_selected_task(
        self, context: ExecutionContext, selection_result: MenuResult
    ) -> Optional[TaskConfig]:
        """Get the selected task based on menu selection."""
        if selection_result.selected_task_index is None:
            return context.get_latest_task()
        index = selection_result.selected_task_index
        if 0 <= index < len(context.existing_tasks):
            return context.existing_tasks[index]
        return None

    def _options_for_verb(self, verb: Verb) -> ExecutionOptions:
        """Map a verb to TaskConfig-compatible execution options."""
        if verb == Verb.REASSEMBLE:
            return ExecutionOptions(force_final_generation=True)
        if verb == Verb.REBUILD:
            return ExecutionOptions(force_final_generation=True, rerender_all=True)
        if verb == Verb.EDIT:
            return ExecutionOptions(edit_mode=True)
        return ExecutionOptions()

    def _create_new_task_intent(self) -> ExecutionIntent:
        """Create execution intent for new task creation."""
        return ExecutionIntent(
            verb=Verb.CREATE,
            tasks=[],
            execution_mode="single",
            execution_options=self._options_for_verb(Verb.CREATE),
            source="menu",
        )

    def _create_single_task_intent(self, task: TaskConfig, verb: Verb) -> ExecutionIntent:
        """Create execution intent for single task operation."""
        options = self._options_for_verb(verb)
        task.force_final_generation = options.force_final_generation
        task.rerender_all = options.rerender_all
        return ExecutionIntent(
            verb=verb,
            tasks=[task],
            execution_mode="single",
            execution_options=options,
            source="menu",
        )

    def _create_scoped_intent(self, tasks: List[TaskConfig], verb: Verb) -> MenuResult:
        """Create a menu result containing an execution intent for a task scope."""
        options = self._options_for_verb(verb)
        for task in tasks:
            task.force_final_generation = options.force_final_generation
            task.rerender_all = options.rerender_all
        intent = ExecutionIntent(
            verb=verb,
            tasks=tasks,
            execution_mode="batch" if len(tasks) > 1 else "single",
            execution_options=options,
            source="menu",
        )
        return MenuResult(choice=verb, execution_intent=intent)

    def resolve_edit_intent(self, context: ExecutionContext) -> ExecutionIntent:
        """Entry point for the CLI edit verb.

        If the job has more than one task, shows a task-selection menu first.
        Pressing 'c' inside the editor returns to that selection; 'c' at the
        selection level cancels the whole operation.
        """
        tasks = context.existing_tasks
        if not tasks:
            print(f"\nNo existing tasks found for job '{context.job_name}'.")
            return self._create_cancelled_intent()

        while True:
            if len(tasks) == 1:
                task = tasks[0]
            else:
                print(f"\nSelect task to edit for job '{context.job_name}':")
                for i, t in enumerate(tasks, 1):
                    latest_marker = " (<-- latest)" if i == 1 else ""
                    print(f"{i}. {self._format_task_display(t)}{latest_marker}")
                spacing = " " * (6 - len(str(len(tasks))))
                print(f"\n[Enter]{' ' * 1}- Edit latest task")
                print(f"1-{len(tasks)}{spacing}- Edit specific task")
                print(f"c      - Cancel")
                choice = input("\n> ").strip().lower()
                if choice == "c":
                    return self._create_cancelled_intent()
                if choice == "" or (choice.isdigit() and int(choice) == 1):
                    task = tasks[0]
                elif choice.isdigit() and 1 <= int(choice) <= len(tasks):
                    task = tasks[int(choice) - 1]
                else:
                    print("Invalid choice. Please try again.")
                    continue

            result = self._maybe_open_candidate_editor(
                task, is_latest=(task is tasks[0])
            )
            if result.is_final_choice() and result.execution_intent is not None:
                return result.execution_intent
            # "c" in editor or unavailable editor
            if len(tasks) == 1:
                return self._create_cancelled_intent()
            # Multi-task: go back to task selection

    def _create_cancelled_intent(self) -> ExecutionIntent:
        """Create execution intent for cancelled operation."""
        return ExecutionIntent(
            verb=Verb.RESUME,
            tasks=[],
            execution_mode="cancelled",
            execution_options=ExecutionOptions(),
            source="menu",
        )
