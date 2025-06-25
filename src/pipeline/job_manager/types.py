#!/usr/bin/env python3
"""
Common types for job management.
Contains enums and dataclasses shared across job management modules.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List

from utils.config_manager import TaskConfig


class UserChoice(Enum):
    """User selection options for task execution."""

    LATEST = "latest"  # Use latest task
    ALL = "all"  # Use all tasks
    NEW = "new"  # Create new task
    LATEST_NEW = "latest-new"  # Use latest task + new final audio
    ALL_NEW = "all-new"  # Use all tasks + new final audio
    SPECIFIC = "specific"  # Select specific task
    SPECIFIC_NEW = "specific-new"  # Select specific task + new final audio
    EDIT = "edit"  # Edit completed task candidates
    CANCEL = "cancel"  # Cancel execution
    LATEST_FILL_GAPS = "latest-fill-gaps"  # Use latest task + fill gaps + new final audio
    LATEST_FILL_GAPS_NO_OVERWRITE = "latest-fill-gaps-no-overwrite"  # Use latest task + fill gaps, don't overwrite final audio
    LATEST_RERENDER_ALL = "latest-rerender-all"  # Use latest task + delete all candidates + rerender everything


class ExecutionStrategy(Enum):
    """Strategy for task execution."""

    LATEST = "latest"  # Use latest task (primary)
    LAST = "latest"  # Use latest task (alias for LATEST)
    ALL = "all"  # Use all tasks
    NEW = "new"  # Create new task
    LATEST_NEW = "latest-new"  # Use latest task + new final audio (primary)
    LAST_NEW = "latest-new"  # Use latest task + new final audio (alias for LATEST_NEW)
    ALL_NEW = "all-new"  # Use all tasks + new final audio


@dataclass
class ExecutionPlan:
    """Plan for task execution."""

    task_configs: List[TaskConfig]
    execution_mode: str  # "single", "batch", "interactive"
    requires_user_input: bool = False
