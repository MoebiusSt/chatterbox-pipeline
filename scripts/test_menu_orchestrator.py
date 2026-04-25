#!/usr/bin/env python3
"""Unit tests for verb-based CLI/menu execution mapping."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from src.pipeline.job_manager.cli_mapper import CLIMapper, StrategyResolver
from src.pipeline.job_manager.execution_types import (
    ExecutionContext,
    ExecutionIntent,
    ExecutionOptions,
    MenuResult,
)
from src.pipeline.job_manager.menu_orchestrator import MenuOrchestrator
from src.pipeline.job_manager.types import Verb
from src.utils.config_manager import ConfigManager, TaskConfig


def _task(job_name: str = "test_job", run_label: str = "test_label") -> Mock:
    task = Mock(spec=TaskConfig)
    task.timestamp = "20240320_120000"
    task.job_name = job_name
    task.run_label = run_label
    task.config_path = Mock(spec=Path)
    task.config_path.exists.return_value = True
    task.config_path.stem = "config"
    task.force_final_generation = False
    task.rerender_all = False
    return task


def _context(tasks=None) -> ExecutionContext:
    return ExecutionContext(
        existing_tasks=list(tasks or []),
        job_configs=None,
        execution_path="test",
        job_name="test_job",
    )


class TestExecutionTypes:
    """Test execution data structures."""

    def test_execution_options_defaults(self):
        options = ExecutionOptions()
        assert not options.force_final_generation
        assert not options.rerender_all
        assert not options.edit_mode

    def test_execution_intent_methods(self):
        intent = ExecutionIntent(
            verb=Verb.REASSEMBLE,
            tasks=[_task(), _task()],
            execution_mode="batch",
            execution_options=ExecutionOptions(force_final_generation=True),
            source="menu",
        )
        assert intent.is_batch_mode()
        assert intent.requires_final_generation()
        assert intent.is_gap_filling_operation()

    def test_menu_result_navigation(self):
        result = MenuResult(should_return=True)
        assert not result.is_final_choice()
        assert not result.should_continue_menu()

        result_next = MenuResult(requires_next_level=True)
        assert result_next.should_continue_menu()

        intent = ExecutionIntent(
            verb=Verb.CREATE,
            tasks=[],
            execution_mode="single",
            execution_options=ExecutionOptions(),
            source="menu",
        )
        result_final = MenuResult(choice=Verb.CREATE, execution_intent=intent)
        assert result_final.is_final_choice()


class TestCLIMapper:
    """Test verb-to-intent mapping."""

    def setup_method(self):
        self.cli_mapper = CLIMapper()

    def test_verb_to_options_mapping(self):
        assert not self.cli_mapper.verb_to_options(Verb.CREATE).force_final_generation
        assert not self.cli_mapper.verb_to_options(Verb.RESUME).force_final_generation
        assert self.cli_mapper.verb_to_options(Verb.REASSEMBLE).force_final_generation

        rebuild_options = self.cli_mapper.verb_to_options(Verb.REBUILD)
        assert rebuild_options.force_final_generation
        assert rebuild_options.rerender_all

    def test_parse_resume_latest(self):
        args = Mock(command="resume", all=False)
        context = _context([_task("job_a"), _task("job_b")])
        intent = self.cli_mapper.parse_cli_to_execution_intent(args, context)
        assert intent is not None
        assert intent.verb == Verb.RESUME
        assert intent.execution_mode == "batch"
        assert len(intent.tasks) == 2

    def test_parse_resume_latest_keeps_config_file_run_labels(self):
        args = Mock(command="resume", all=False)
        context = ExecutionContext(
            existing_tasks=[
                _task("speaker-bench", "run-a"),
                _task("speaker-bench", "run-b"),
            ],
            job_configs=None,
            execution_path="config-files",
            job_name="",
        )
        intent = self.cli_mapper.parse_cli_to_execution_intent(args, context)
        assert intent is not None
        assert len(intent.tasks) == 2

    def test_parse_reassemble_all(self):
        args = Mock(command="reassemble", all=True)
        context = _context([_task("job_a"), _task("job_a", "older")])
        intent = self.cli_mapper.parse_cli_to_execution_intent(args, context)
        assert intent is not None
        assert intent.verb == Verb.REASSEMBLE
        assert intent.execution_options.force_final_generation
        assert len(intent.tasks) == 2

    def test_requires_user_interaction_without_command(self):
        resolver = StrategyResolver(self.cli_mapper)
        args = Mock(command=None)
        assert resolver.requires_user_interaction(args, _context([_task()]))


class TestMenuOrchestrator:
    """Test menu helper behavior."""

    def setup_method(self):
        self.config_manager = Mock(spec=ConfigManager)
        self.orchestrator = MenuOrchestrator(self.config_manager)

    def test_format_task_display(self):
        mock_task = _task()
        self.config_manager.load_job_config.return_value = {
            "input": {"text_file": "test_document.txt"}
        }

        display = self.orchestrator._format_task_display(mock_task)
        assert "test_job" in display
        assert "test_label" in display
        assert "test_document.txt" in display
        assert "20.03.2024" in display
        assert "12:00" in display

    def test_create_execution_intents(self):
        create_intent = self.orchestrator._create_new_task_intent()
        assert create_intent.verb == Verb.CREATE
        assert create_intent.execution_mode == "single"
        assert create_intent.tasks == []

        task = _task()
        result = self.orchestrator._create_scoped_intent([task], Verb.REBUILD)
        assert result.execution_intent is not None
        assert result.execution_intent.verb == Verb.REBUILD
        assert task.force_final_generation
        assert task.rerender_all


if __name__ == "__main__":
    pytest.main([__file__])
