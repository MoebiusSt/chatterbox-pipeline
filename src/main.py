#!/usr/bin/env python3
"""
Refactored main script for the task-based TTS pipeline.
Implements unified job and task management with automatic state detection and recovery.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import torch

# Path correction for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from pipeline.batch_executor import BatchExecutor
from pipeline.job_manager import ExecutionPlan, JobManager
from pipeline.task_executor import TaskExecutor
from utils.config_manager import ConfigManager, TaskConfig
from utils.file_manager import FileManager
from utils.logging_config import LoggingConfigurator

# Ensure logs directory exists
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)

# Early enhanced logging setup with icons
LoggingConfigurator.configure(
    log_file=PROJECT_ROOT / "logs" / "main.log",
    console_level=logging.INFO,
    file_level=logging.DEBUG,
    append=True,
    verbose_mode=True,  # Will be updated later based on args
    use_icons=True,
)

logger = logging.getLogger(__name__)


def detect_device() -> str:
    """Automatically detect the best available device."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.primary(f"Using device: {device}")
    return device


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for the new task-based system.

    Supported usage patterns:
    - python main.py                           -> default job
    - python main.py config1.yaml config2.yaml -> specific configs
    - python main.py --job "jobname"           -> find job by name
    - python main.py --parallel               -> enable parallel execution
    """
    parser = argparse.ArgumentParser(
        description="TTS Pipeline - Task-Based Execution System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  %(prog)s                           # Run default job
  %(prog)s job1.yaml job2.yaml       # Run specific config files
  %(prog)s --job "my_job"            # Run job by name
  %(prog)s --job "my_job" --parallel # Run job in parallel mode
  %(prog)s --mode new                # Create new tasks for all jobs
  %(prog)s --mode all                # Run all tasks of all jobs
  %(prog)s --mode last               # Run latest task of all jobs
  %(prog)s --mode "job1:new,job2:all"  # Different strategies per job
  %(prog)s --add-final               # Force regeneration of final audio from existing candidates
        """,
    )

    # Job selection arguments
    parser.add_argument("--job", "-j", type=str, help="Job name to execute")
    parser.add_argument(
        "config_files", nargs="*", type=Path, help="Configuration file(s) to process"
    )

    # Execution strategy arguments
    parser.add_argument(
        "--mode",
        type=str,
        help="Execution strategy: global (last/all/new) or job-specific (job1:new,job2:all). Examples: --mode all, --mode 'job1:new,job2:last'",
    )
    parser.add_argument(
        "--add-final",
        action="store_true",
        help="Force regeneration of final audio from existing candidates, even if final audio exists",
    )

    # Execution options
    parser.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        help="Enable parallel execution for multiple tasks",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum number of parallel workers (default: 2)",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for processing (default: auto)",
    )

    return parser.parse_args()


def resolve_config_files(args: argparse.Namespace, project_root: Path) -> List[Path]:
    """
    Resolve configuration files from command line arguments.

    Args:
        args: Parsed command line arguments
        project_root: Project root directory

    Returns:
        List of resolved configuration file paths
    """
    config_files = []

    if args.config_files:
        for config_file in args.config_files:
            # Handle relative paths
            if not config_file.is_absolute():
                # Try relative to current directory first
                if config_file.exists():
                    config_files.append(config_file.resolve())
                # Then try relative to config directory
                elif (project_root / "config" / config_file).exists():
                    config_files.append(
                        (project_root / "config" / config_file).resolve()
                    )
                else:
                    raise FileNotFoundError(
                        f"Configuration file not found: {config_file}"
                    )
            else:
                if config_file.exists():
                    config_files.append(config_file)
                else:
                    raise FileNotFoundError(
                        f"Configuration file not found: {config_file}"
                    )

    return config_files


def main() -> int:
    """Main entry point for the refactored TTS pipeline."""

    try:
        # Parse arguments
        args = parse_arguments()

        # DEVELOPMENT: Force verbose mode during development
        verbose_mode = True  # Set to args.verbose for production

        # Early logging configuration with verbose mode
        if verbose_mode:
            logging.getLogger().setLevel(logging.DEBUG)

        # Detect device
        device = detect_device() if args.device == "auto" else args.device

        logger.primary("=" * 60)
        logger.primary("TTS PIPELINE - TASK-BASED EXECUTION SYSTEM")
        logger.primary("=" * 60)

        # Initialize core components
        project_root = PROJECT_ROOT
        config_manager = ConfigManager(project_root)
        job_manager = JobManager(config_manager)

        # Resolve execution plan
        config_files = resolve_config_files(args, project_root)
        execution_plan = job_manager.resolve_execution_plan(args, config_files)

        # Handle cancelled execution
        if execution_plan.execution_mode == "cancelled":
            logger.primary("Execution cancelled by user")
            return 0

        # Validate execution plan
        if not job_manager.validate_execution_plan(execution_plan):
            logger.error("Invalid execution plan")
            return 1

        # Print execution summary
        job_manager.print_execution_summary(execution_plan)

        # Execute tasks
        if (
            execution_plan.execution_mode == "batch"
            or len(execution_plan.task_configs) > 1
        ):
            # Batch execution
            logger.verbose("Starting batch execution mode")

            batch_executor = BatchExecutor(config_manager)

            # Handle task dependencies
            ordered_tasks = batch_executor.handle_task_dependencies(
                execution_plan.task_configs
            )

            # Execute with parallel option
            batch_result = batch_executor.execute_multiple_tasks(
                ordered_tasks, parallel=args.parallel, max_workers=args.max_workers
            )

            # Generate detailed report
            if len(execution_plan.task_configs) > 1:
                batch_executor.print_detailed_results(batch_result)

                # Generate batch report file
                report_path = batch_executor.generate_batch_report(batch_result)
            logger.primary(f"Detailed report saved: {report_path}")

            # Return appropriate exit code
            return 0 if batch_result.failed_tasks == 0 else 1

        else:
            # Single task execution
            logger.verbose("Starting single task execution mode")

            task_config = execution_plan.task_configs[0]

            # Load config once via ConfigManager
            task_config_path = task_config.config_path
            project_root = task_config_path.parent.parent.parent.parent
            cm = ConfigManager(project_root)
            loaded_config = cm.load_cascading_config(task_config_path)

            # Create file manager with preloaded config
            file_manager = FileManager(task_config, preloaded_config=loaded_config)

            # Create and execute task - TaskExecutor will not reload config
            task_executor = TaskExecutor(file_manager, task_config)
            # Set the loaded config directly to avoid re-loading in TaskExecutor
            task_executor.config = loaded_config
            
            result = task_executor.execute_task()

            # Report results
            if result.success:
                logger.primary("=" * 60)
                logger.primary("TASK COMPLETED SUCCESSFULLY")
                logger.primary("=" * 60)
                logger.primary(f"Job: {result.task_config.job_name}")
                logger.primary(f"Task: {result.task_config.task_name}")
                if result.task_config.run_label:
                    logger.primary(f"Label: {result.task_config.run_label}")
                logger.primary(f"Execution time: {result.execution_time:.2f} seconds")
                logger.primary(f"Final stage: {result.completion_stage.value}")

                if result.final_audio_path:
                    logger.primary(f"Final audio: {result.final_audio_path}")

                return 0
            else:
                logger.error("=" * 60)
                logger.error("✗ TASK EXECUTION FAILED")
                logger.error("=" * 60)
                logger.error(f"Job: {result.task_config.job_name}")
                logger.error(f"Error: {result.error_message}")
                logger.error(f"Stage reached: {result.completion_stage.value}")

                return 1

    except KeyboardInterrupt:
        logger.primary("\nExecution interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
