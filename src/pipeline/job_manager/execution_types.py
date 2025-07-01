#!/usr/bin/env python3
"""
Central execution types for the menu orchestrator system.
Replaces scattered UserChoice enums with structured data models.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from utils.config_manager import TaskConfig
from .types import ExecutionStrategy


@dataclass
class ExecutionOptions:
    """Unified execution options for all task operations."""
    
    force_final_generation: bool = False  # Forces final audio regeneration
    rerender_all: bool = False           # Delete all candidates and re-render everything from scratch
    gap_filling_mode: bool = False       # Indicates this is a gap-filling operation


@dataclass
class ExecutionContext:
    """Context for menu decision-making across all execution paths."""
    
    existing_tasks: List[TaskConfig]
    job_configs: Optional[List[Path]]
    execution_path: str  # "job-name", "config-files", "default"
    job_name: str
    available_strategies: Dict[str, ExecutionStrategy]
    
    def has_existing_tasks(self) -> bool:
        """Check if there are existing tasks to work with."""
        return bool(self.existing_tasks)
    
    def get_latest_task(self) -> Optional[TaskConfig]:
        """Get the latest (newest) task if available."""
        return self.existing_tasks[0] if self.existing_tasks else None


@dataclass 
class ExecutionIntent:
    """Structured execution intent - replaces scattered UserChoice handling."""
    
    tasks: List[TaskConfig]
    execution_mode: str  # "single", "batch"
    execution_options: ExecutionOptions
    source: str  # "menu", "cli", "config"
    
    def is_batch_mode(self) -> bool:
        """Check if this is a batch execution."""
        return self.execution_mode == "batch" or len(self.tasks) > 1
    
    def requires_final_generation(self) -> bool:
        """Check if final audio generation is required."""
        return self.execution_options.force_final_generation
    
    def is_gap_filling_operation(self) -> bool:
        """Check if this is a gap-filling operation."""
        return self.execution_options.gap_filling_mode


class MenuLevel(Enum):
    """Menu hierarchy levels for consistent navigation."""
    
    TASK_SELECTION = "task_selection"     # Level 1: Latest/Specific/New/All-Options
    TASK_OPTIONS = "task_options"         # Level 2: Fill-Gaps/Skip-Overwrite/Rerender/Edit
    CANDIDATE_EDITOR = "candidate_editor" # Level 3: Candidate selection and editing


class TaskSelectionChoice(Enum):
    """Level 1 menu choices - task selection."""
    
    LATEST = "latest"           # Use latest task
    SPECIFIC = "specific"       # Select specific task by number
    NEW = "new"                # Create new task
    ALL_OPTIONS = "all_options" # Show options for all tasks
    CANCEL = "cancel"          # Cancel execution


class TaskOptionsChoice(Enum):
    """Level 2 menu choices - task-specific options."""
    
    FILL_GAPS = "fill_gaps"                         # Run task, fill gaps, CREATE/overwrite final audio
    FILL_GAPS_NO_OVERWRITE = "fill_gaps_no_overwrite" # Run task, fill gaps, KEEP existing final audio
    RERENDER_ALL = "rerender_all"                   # Re-render ALL candidates, create new final audio
    EDIT_CANDIDATES = "edit_candidates"             # Edit completed task (choose different candidates)
    RETURN = "return"                              # Return to previous menu


class AllTasksChoice(Enum):
    """Options for all-tasks operations."""
    
    ALL_FILL_GAPS = "all_fill_gaps"                         # Run all tasks, fill gaps, CREATE/overwrite final audio
    ALL_FILL_GAPS_NO_OVERWRITE = "all_fill_gaps_no_overwrite" # Run all tasks, fill gaps, KEEP existing final audio
    ALL_RERENDER_ALL = "all_rerender_all"                   # Re-render ALL candidates for all tasks
    RETURN = "return"                                      # Return to main menu


@dataclass
class MenuResult:
    """Result from menu interaction with navigation context."""
    
    choice: Enum
    selected_task_index: Optional[int] = None
    requires_next_level: bool = False
    execution_intent: Optional[ExecutionIntent] = None
    
    def is_final_choice(self) -> bool:
        """Check if this result represents a final user decision."""
        return self.execution_intent is not None
    
    def should_continue_menu(self) -> bool:
        """Check if menu navigation should continue."""
        return self.requires_next_level and not self.is_final_choice() 