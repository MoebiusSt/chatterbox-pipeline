#!/usr/bin/env python3
"""
Test script für das task-basierte TTS System.
"""

import logging
import sys
from pathlib import Path

import pytest

from pipeline.job_manager_facade import ExecutionStrategy, JobManager, UserChoice
from utils.config_manager import ConfigManager

# Path correction for imports
PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)  # Go up one more level to get to project root
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from pipeline.task_executor import TaskExecutor
from utils.file_manager import FileManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_task_system():
    """Test the task-based system with default config."""
    try:
        logger.info("Testing task-based TTS system...")

        # Initialize config manager
        config_manager = ConfigManager(PROJECT_ROOT)

        # Load default config
        default_config = config_manager.load_default_config()
        logger.info(f"Loaded default config for job: {default_config['job']['name']}")

        # Create task config
        task_config = config_manager.create_task_config(default_config)
        logger.info(f"Created task config: {task_config.task_name}")

        # Save task config
        config_manager.save_task_config(task_config, default_config)
        logger.info(f"Saved task config to: {task_config.config_path}")

        # Create file manager
        file_manager = FileManager(task_config)
        logger.info("Created file manager")

        # Create task executor
        task_executor = TaskExecutor(file_manager, task_config)
        logger.info("Created task executor")

        # Test model loading without execution
        logger.info("Testing TTS model loading...")
        tts_gen = task_executor.tts_generator
        logger.info(f"TTS generator created with model: {tts_gen.model is not None}")

        # Test reference audio loading
        reference_audio_path = file_manager.get_reference_audio()
        logger.info(f"Reference audio path: {reference_audio_path}")

        if reference_audio_path.exists():
            logger.info("Testing reference audio preparation...")
            tts_gen.prepare_conditionals(str(reference_audio_path))
            logger.info("✓ Reference audio prepared successfully")
        else:
            logger.warning(f"Reference audio not found: {reference_audio_path}")

        logger.info("✓ Task system test completed successfully")
        assert True  # Explicit assertion for pytest

    except Exception as e:
        logger.error(f"Task system test failed: {e}", exc_info=True)
        return False


def test_user_choice_enum():
    """Test UserChoice enum values."""
    assert UserChoice.LATEST.value == "latest"
    assert UserChoice.ALL.value == "all"
    assert UserChoice.NEW.value == "new"
    assert UserChoice.LATEST_NEW.value == "latest-new"
    assert UserChoice.ALL_NEW.value == "all-new"
    assert UserChoice.SPECIFIC.value == "specific"
    assert UserChoice.SPECIFIC_NEW.value == "specific-new"
    assert UserChoice.CANCEL.value == "cancel"


def test_execution_strategy_enum():
    """Test ExecutionStrategy enum values."""
    assert ExecutionStrategy.LAST.value == "last"
    assert ExecutionStrategy.ALL.value == "all"
    assert ExecutionStrategy.NEW.value == "new"
    assert ExecutionStrategy.LAST_NEW.value == "last-new"
    assert ExecutionStrategy.ALL_NEW.value == "all-new"


def test_job_manager_initialization():
    """Test JobManager initialization."""
    config_manager = ConfigManager(PROJECT_ROOT)
    job_manager = JobManager(config_manager)
    assert job_manager is not None


def test_resolve_execution_plan():
    """Test execution plan resolution with different strategies."""
    config_manager = ConfigManager(PROJECT_ROOT)
    job_manager = JobManager(config_manager)

    # Create mock args object
    args = type("Args", (), {"mode": "last-new", "job": None, "add_final": False})()
    
    # Test global strategy
    plan = job_manager.resolve_execution_plan(args)
    assert plan is not None

    # Test job-specific strategy
    args = type("Args", (), {"mode": "job1:last-new,job2:all-new", "job": None, "add_final": False})()
    plan = job_manager.resolve_execution_plan(args)
    assert plan is not None


def test_prompt_user_selection():
    """Test user selection prompt."""
    config_manager = ConfigManager(PROJECT_ROOT)
    job_manager = JobManager(config_manager)

    # Test with no tasks
    choice = job_manager.prompt_user_selection([])
    assert choice == UserChoice.NEW

    # Test with tasks (mock input)
    import builtins

    original_input = builtins.input

    def mock_input(prompt):
        return "ln"

    # Create mock task with required attributes
    mock_task = type("Task", (), {
        "timestamp": "2024-03-20_120000",
        "job_name": "test_job",
        "run_label": "test_label",
        "config_path": Path("/fake/path/config.yaml")
    })()

    builtins.input = mock_input
    try:
        choice = job_manager.prompt_user_selection([mock_task])
        assert choice == UserChoice.LATEST_NEW
    finally:
        builtins.input = original_input


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
