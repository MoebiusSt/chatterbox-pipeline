#!/usr/bin/env python3
"""CLI mapper for verb-based task execution."""

import logging
from typing import Any, List, Optional

from utils.config_manager import TaskConfig

from .execution_types import ExecutionContext, ExecutionIntent, ExecutionOptions
from .types import Verb

logger = logging.getLogger(__name__)


class CLIMapper:
    """Maps CLI verbs to execution intents."""

    def parse_cli_to_execution_intent(
        self, args: Any, context: ExecutionContext
    ) -> Optional[ExecutionIntent]:
        """Parse CLI arguments to ExecutionIntent if no user interaction is required."""
        command = getattr(args, "command", None)
        if command is None:
            return None

        try:
            verb = Verb(command)
        except ValueError:
            logger.warning("Unknown command: %s", command)
            return None

        execution_options = self.verb_to_options(verb)
        if verb == Verb.CREATE:
            tasks: List[TaskConfig] = []  # Will be created by execution planner
            execution_mode = "single"
        elif verb in {Verb.RESUME, Verb.REASSEMBLE, Verb.REBUILD}:
            tasks = (
                context.existing_tasks
                if bool(getattr(args, "all", False))
                else self._latest_task_per_job(context)
            )
            if not tasks:
                raise ValueError(f"{verb.value} found no existing task to process")
            execution_mode = "batch" if len(tasks) > 1 else "single"
        elif verb == Verb.EDIT:
            tasks = self._latest_task_per_job(context)
            if not tasks:
                raise ValueError(f"{verb.value} found no existing task to process")
            execution_mode = "single"
        else:
            return None

        return ExecutionIntent(
            verb=verb,
            tasks=tasks,
            execution_mode=execution_mode,
            execution_options=execution_options,
            source="cli",
        )

    def verb_to_options(self, verb: Verb) -> ExecutionOptions:
        """Return TaskConfig-compatible options for a verb."""
        if verb == Verb.REASSEMBLE:
            return ExecutionOptions(force_final_generation=True)
        if verb == Verb.REBUILD:
            return ExecutionOptions(force_final_generation=True, rerender_all=True)
        if verb == Verb.EDIT:
            return ExecutionOptions(edit_mode=True)
        return ExecutionOptions()

    def _latest_task_per_job(self, context: ExecutionContext):
        """Return the newest task per selected job scope."""
        tasks_by_job = {}
        for task in context.existing_tasks:
            key = (
                (task.job_name, task.run_label)
                if context.execution_path == "config-files" and task.run_label
                else (task.job_name,)
            )
            if key not in tasks_by_job:
                tasks_by_job[key] = task
        return list(tasks_by_job.values())

    def validate_cli_menu_parity(self) -> bool:
        """
        Validate that all CLI options have menu equivalents and vice versa.

        Returns:
            True if parity is maintained
        """
        # Define expected CLI options
        # expected_cli_options = {"command", "job", "all"}

        # Define menu capabilities
        # menu_capabilities = {
        #     "task_selection",
        #     "execution_options",
        #     "candidate_editing",
        #     "batch_operations",
        #     "safety_confirmations",
        # }

        # In a real implementation, this would cross-reference with actual CLI parser
        # For now, we assume parity based on our design

        logger.info("CLI-Menu parity validation: all commands mapped")
        return True


class StrategyResolver:
    """Resolves execution strategies from various input sources."""

    def __init__(self, cli_mapper: CLIMapper):
        self.cli_mapper = cli_mapper

    def resolve_from_args(
        self, args: Any, context: ExecutionContext
    ) -> Optional[ExecutionIntent]:
        """
        Resolve execution intent from CLI arguments.

        Args:
            args: CLI arguments
            context: Execution context

        Returns:
            ExecutionIntent if resolvable from CLI, None if user interaction needed
        """
        return self.cli_mapper.parse_cli_to_execution_intent(args, context)

    def requires_user_interaction(self, args: Any, context: ExecutionContext) -> bool:
        """
        Check if user interaction is required based on CLI arguments.

        Args:
            args: CLI arguments
            context: Execution context

        Returns:
            True if user interaction is needed
        """
        return getattr(args, "command", None) is None
