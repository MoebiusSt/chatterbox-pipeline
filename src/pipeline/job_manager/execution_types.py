#!/usr/bin/env python3
"""
Central execution types for the menu orchestrator system.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from utils.config_manager import TaskConfig

from .types import Verb


@dataclass
class ExecutionOptions:
    """Unified execution options for all task operations."""

    force_final_generation: bool = False
    rerender_all: bool = False
    edit_mode: bool = False


@dataclass
class ExecutionContext:
    """Context for menu decision-making across all execution paths."""

    existing_tasks: List[TaskConfig]
    job_configs: Optional[List[Path]]
    execution_path: str  # "job-name", "config-files", "default"
    job_name: str

    def has_existing_tasks(self) -> bool:
        """Check if there are existing tasks to work with."""
        return bool(self.existing_tasks)

    def get_latest_task(self) -> Optional[TaskConfig]:
        """Get the latest (newest) task if available."""
        return self.existing_tasks[0] if self.existing_tasks else None


@dataclass
class ExecutionIntent:
    """Structured execution intent."""

    verb: Verb
    tasks: List[TaskConfig]
    execution_mode: str  # "single", "batch", "cancelled"
    execution_options: ExecutionOptions
    source: str  # "menu", "cli", "config"

    def is_batch_mode(self) -> bool:
        """Check if this is a batch execution."""
        return self.execution_mode == "batch" or len(self.tasks) > 1

    def requires_final_generation(self) -> bool:
        """Check if final audio generation is required."""
        return self.execution_options.force_final_generation

    def is_gap_filling_operation(self) -> bool:
        """Check if this intent processes existing tasks."""
        return self.verb in {Verb.RESUME, Verb.REASSEMBLE, Verb.REBUILD}


@dataclass
class MenuResult:
    """Result from menu interaction with navigation context."""

    choice: Optional[Verb] = None
    selected_task_index: Optional[int] = None
    requires_next_level: bool = False
    execution_intent: Optional[ExecutionIntent] = None
    all_tasks: bool = False
    should_return: bool = False

    def is_final_choice(self) -> bool:
        """Check if this result represents a final user decision."""
        return self.execution_intent is not None

    def should_continue_menu(self) -> bool:
        """Check if menu navigation should continue."""
        return self.requires_next_level and not self.is_final_choice()
