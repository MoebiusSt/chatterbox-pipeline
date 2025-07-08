#!/usr/bin/env python3
"""
Unit tests for the MenuOrchestrator refactoring components.
Tests the new unified menu system and CLI-Menu parity.
"""

from pathlib import Path
from typing import List
from unittest.mock import Mock, patch

import pytest

from src.pipeline.job_manager.cli_mapper import CLIMapper, StrategyResolver
from src.pipeline.job_manager.execution_types import (
    ExecutionContext,
    ExecutionIntent,
    ExecutionOptions,
    MenuResult,
    TaskOptionsChoice,
    TaskSelectionChoice,
)
from src.pipeline.job_manager.menu_orchestrator import MenuOrchestrator
from src.pipeline.job_manager.types import ExecutionStrategy
from src.utils.config_manager import ConfigManager, TaskConfig


class TestExecutionTypes:
    """Test the new execution data structures."""

    def test_execution_options_defaults(self):
        """Test ExecutionOptions default values."""
        options = ExecutionOptions()
        assert options.force_final_generation == False

        assert options.rerender_all == False
        assert options.gap_filling_mode == False

    def test_execution_options_renamed_field(self):
        """Test that force_final_generation replaces add_final semantically."""
        options = ExecutionOptions(force_final_generation=True)
        assert options.force_final_generation == True
        # This should be more semantic than the old 'add_final' name

    def test_execution_context_methods(self):
        """Test ExecutionContext helper methods."""
        # Empty context
        context = ExecutionContext(
            existing_tasks=[],
            job_configs=None,
            execution_path="test",
            job_name="test_job",
            available_strategies={},
        )
        assert not context.has_existing_tasks()
        assert context.get_latest_task() is None

        # Context with tasks
        mock_task = Mock(spec=TaskConfig)
        context_with_tasks = ExecutionContext(
            existing_tasks=[mock_task],
            job_configs=None,
            execution_path="test",
            job_name="test_job",
            available_strategies={},
        )
        assert context_with_tasks.has_existing_tasks()
        assert context_with_tasks.get_latest_task() == mock_task

    def test_execution_intent_methods(self):
        """Test ExecutionIntent helper methods."""
        options = ExecutionOptions(force_final_generation=True, gap_filling_mode=True)
        intent = ExecutionIntent(
            tasks=[Mock(), Mock()],
            execution_mode="batch",
            execution_options=options,
            source="menu",
        )

        assert intent.is_batch_mode()  # 2 tasks = batch
        assert intent.requires_final_generation()
        assert intent.is_gap_filling_operation()

    def test_menu_result_navigation(self):
        """Test MenuResult navigation logic."""
        # Simple choice without continuation
        result = MenuResult(choice=TaskSelectionChoice.CANCEL)
        assert not result.is_final_choice()
        assert not result.should_continue_menu()

        # Choice requiring next level
        result_next = MenuResult(
            choice=TaskSelectionChoice.LATEST, requires_next_level=True
        )
        assert not result_next.is_final_choice()
        assert result_next.should_continue_menu()

        # Final choice with execution intent
        intent = ExecutionIntent([], "single", ExecutionOptions(), "menu")
        result_final = MenuResult(
            choice=TaskSelectionChoice.NEW, execution_intent=intent
        )
        assert result_final.is_final_choice()
        assert not result_final.should_continue_menu()


class TestCLIMapper:
    """Test CLI-Menu parity and strategy resolution."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cli_mapper = CLIMapper()

    def test_strategy_to_options_mapping(self):
        """Test that all strategies map to appropriate options."""
        # NEW strategy should not force final generation
        new_options = self.cli_mapper.strategy_to_options[ExecutionStrategy.NEW]
        assert not new_options.force_final_generation

        # ALL_NEW should force final generation
        all_new_options = self.cli_mapper.strategy_to_options[ExecutionStrategy.ALL_NEW]
        assert all_new_options.force_final_generation

        # LATEST_NEW should force final generation
        latest_new_options = self.cli_mapper.strategy_to_options[
            ExecutionStrategy.LATEST_NEW
        ]
        assert latest_new_options.force_final_generation

    def test_parse_mode_argument(self):
        """Test mode argument parsing."""
        # Simple global strategy
        job_strat, global_strat = self.cli_mapper._parse_mode_argument("latest-new")
        assert global_strat == ExecutionStrategy.LATEST_NEW
        assert job_strat == {}

        # Job-specific strategy
        job_strat, global_strat = self.cli_mapper._parse_mode_argument("job1:all-new")
        assert global_strat is None
        assert job_strat["job1"] == ExecutionStrategy.ALL_NEW

        # Mixed strategies
        job_strat, global_strat = self.cli_mapper._parse_mode_argument(
            "job1:all,job2:latest-new,new"
        )
        assert global_strat == ExecutionStrategy.NEW
        assert job_strat["job1"] == ExecutionStrategy.ALL
        assert job_strat["job2"] == ExecutionStrategy.LATEST_NEW

        # Test alias functionality
        job_strat, global_strat = self.cli_mapper._parse_mode_argument("last")
        assert global_strat == ExecutionStrategy.LATEST
        assert job_strat == {}

        job_strat, global_strat = self.cli_mapper._parse_mode_argument("last-new")
        assert global_strat == ExecutionStrategy.LATEST_NEW
        assert job_strat == {}

        job_strat, global_strat = self.cli_mapper._parse_mode_argument("new-last")
        assert global_strat == ExecutionStrategy.LATEST_NEW
        assert job_strat == {}

        job_strat, global_strat = self.cli_mapper._parse_mode_argument("new-all")
        assert global_strat == ExecutionStrategy.ALL_NEW
        assert job_strat == {}

        # Test job-specific aliases
        job_strat, global_strat = self.cli_mapper._parse_mode_argument(
            "job1:last,job2:last-new"
        )
        assert global_strat is None
        assert job_strat["job1"] == ExecutionStrategy.LATEST
        assert job_strat["job2"] == ExecutionStrategy.LATEST_NEW

    def test_menu_choice_to_cli_args(self):
        """Test conversion from menu choices to CLI arguments."""
        # Single task with force final
        options = ExecutionOptions(force_final_generation=True, rerender_all=True)
        intent = ExecutionIntent(
            tasks=[Mock()],
            execution_mode="single",
            execution_options=options,
            source="menu",
        )

        cli_args = self.cli_mapper.menu_choice_to_cli_args(intent)
        assert cli_args["mode"] == "latest-new"
        assert cli_args["force_final_generation"] == True
        assert cli_args["rerender_all"] == True

        # Multiple tasks
        multi_intent = ExecutionIntent(
            tasks=[Mock(), Mock()],
            execution_mode="batch",
            execution_options=ExecutionOptions(force_final_generation=True),
            source="menu",
        )

        cli_args = self.cli_mapper.menu_choice_to_cli_args(multi_intent)
        assert cli_args["mode"] == "all-new"

    def test_cli_menu_parity_validation(self):
        """Test that CLI-Menu parity validation works."""
        # This should always pass with our design
        assert self.cli_mapper.validate_cli_menu_parity()

    def test_cli_help_text_generation(self):
        """Test CLI help text generation."""
        help_text = self.cli_mapper.get_cli_help_text()
        assert "CLI-Menu Equivalents" in help_text
        assert "--mode latest" in help_text
        assert "Menu:" in help_text


class TestStrategyResolver:
    """Test strategy resolution from CLI arguments."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cli_mapper = CLIMapper()
        self.resolver = StrategyResolver(self.cli_mapper)

    def test_requires_user_interaction(self):
        """Test detection of when user interaction is needed."""
        # Mock args without mode
        args = Mock()
        args.mode = None

        context = ExecutionContext(
            existing_tasks=[Mock()],
            job_configs=None,
            execution_path="test",
            job_name="test_job",
            available_strategies={},
        )

        # Should require interaction when no strategy specified
        assert self.resolver.requires_user_interaction(args, context)

        # Should not require interaction when strategy is clear
        args.mode = "latest"
        assert not self.resolver.requires_user_interaction(args, context)


class TestMenuOrchestrator:
    """Test the central MenuOrchestrator functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.config_manager = Mock(spec=ConfigManager)
        self.orchestrator = MenuOrchestrator(self.config_manager)

    def test_format_task_display(self):
        """Test task display formatting."""
        # Create mock task
        mock_task = Mock(spec=TaskConfig)
        mock_task.timestamp = "20240320_120000"
        mock_task.job_name = "test_job"
        mock_task.run_label = "test_label"
        mock_task.config_path = Path("/fake/path/config.yaml")

        # Mock config loading
        self.config_manager.load_job_config.return_value = {
            "input": {"text_file": "test_document.txt"}
        }
        mock_task.config_path.exists.return_value = True

        display = self.orchestrator._format_task_display(mock_task)

        # Should format as: "job-name - run-label - doc-name.txt - date - time"
        assert "test_job" in display
        assert "test_label" in display
        assert "test_document.txt" in display
        assert "20.03.2024" in display
        assert "12:00" in display

    def test_create_execution_intents(self):
        """Test creation of different execution intents."""
        context = ExecutionContext(
            existing_tasks=[],
            job_configs=None,
            execution_path="test",
            job_name="test_job",
            available_strategies={},
        )

        # Test new task intent
        new_intent = self.orchestrator._create_new_task_intent(context)
        assert new_intent.execution_mode == "single"
        assert new_intent.source == "menu"
        assert not new_intent.execution_options.force_final_generation

        # Test cancelled intent
        cancelled_intent = self.orchestrator._create_cancelled_intent()
        assert cancelled_intent.execution_mode == "cancelled"
        assert cancelled_intent.tasks == []

        # Test single task intent with options
        mock_task = Mock(spec=TaskConfig)
        options = ExecutionOptions(force_final_generation=True, rerender_all=True)
        single_intent = self.orchestrator._create_single_task_intent(mock_task, options)

        assert single_intent.execution_mode == "single"
        assert len(single_intent.tasks) == 1
        assert single_intent.execution_options.force_final_generation
        assert single_intent.execution_options.rerender_all

        # Verify legacy field mapping
        assert mock_task.force_final_generation == True
        assert mock_task.rerender_all == True


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for common user scenarios."""

    def test_cli_to_menu_equivalence(self):
        """Test that CLI args produce same result as menu choices."""
        cli_mapper = CLIMapper()

        # Scenario: User wants to run latest task with final generation

        # CLI way
        mock_args = Mock()
        mock_args.mode = "latest-new"
        mock_args.force_final_generation = True

        context = ExecutionContext(
            existing_tasks=[Mock()],
            job_configs=None,
            execution_path="cli",
            job_name="test_job",
            available_strategies={},
        )

        cli_intent = cli_mapper.parse_cli_to_execution_intent(mock_args, context)

        # Menu way (simulated)
        menu_options = ExecutionOptions(force_final_generation=True)
        menu_intent = ExecutionIntent(
            tasks=context.existing_tasks[:1],
            execution_mode="single",
            execution_options=menu_options,
            source="menu",
        )

        # Should produce equivalent results
        assert cli_intent.execution_mode == menu_intent.execution_mode
        assert (
            cli_intent.execution_options.force_final_generation
            == menu_intent.execution_options.force_final_generation
        )
        assert len(cli_intent.tasks) == len(menu_intent.tasks)

    def test_semantic_improvement_add_final(self):
        """Test that force_final_generation is more semantic than add_final."""

        # Old way (semantic confusion)
        # task.add_final = True  # Does this ADD a final file or FORCE generation?

        # New way (clear semantics)
        options = ExecutionOptions(
            force_final_generation=True
        )  # Clearly forces generation

        # Test the improvement
        assert hasattr(options, "force_final_generation")
        assert options.force_final_generation == True

        # The new name clearly indicates the action: forcing generation of final audio
        # rather than the ambiguous "add_final" which could mean adding an additional file


if __name__ == "__main__":
    pytest.main([__file__])
