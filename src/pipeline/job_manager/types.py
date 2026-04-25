#!/usr/bin/env python3
"""
Common types for job management.
Contains enums and dataclasses shared across job management modules.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List

from utils.config_manager import TaskConfig


class Verb(Enum):
    """Top-level task verbs exposed by the CLI and menu."""

    CREATE = "create"
    RESUME = "resume"
    REASSEMBLE = "reassemble"
    REBUILD = "rebuild"
    EDIT = "edit"


@dataclass
class ExecutionPlan:
    """Plan for task execution."""

    task_configs: List[TaskConfig]
    execution_mode: str  # "single", "batch", "interactive"
    requires_user_input: bool = False
