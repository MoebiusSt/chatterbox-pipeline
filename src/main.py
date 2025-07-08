#!/usr/bin/env python3
"""
Refactored main script for the task-based TTS pipeline.
Implements unified job and task management with automatic state detection and recovery.
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import torch

# Project root detection
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from pipeline.batch_executor import BatchExecutor
from pipeline.job_manager_wrapper import JobManager
from pipeline.task_executor import TaskExecutor
from utils.config_manager import ConfigManager, TaskConfig
from utils.file_manager.file_manager import FileManager
from utils.logging_config import LoggingConfigurator

# Ensure logs directory exists
(PROJECT_ROOT / "logs").mkdir(exist_ok=True)

# Early enhanced logging setup with icons (verbose mode will be configured later)
LoggingConfigurator.configure(
    log_file=PROJECT_ROOT / "logs" / "main.log",
    console_level=logging.INFO,
    file_level=logging.DEBUG,
    append=True,
    verbose_mode=False,  # Will be updated later based on args
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

    logger.info(f"Using device: {device}")
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
  %(prog)s job1.yaml                 # Run specific config files
  %(prog)s --job "my_job"            # Run job by name
  %(prog)s --job "testjob*"          # Run all jobs starting with "testjob"
  %(prog)s --job "test?job"          # Run jobs like "test1job", "test2job" etc.

  %(prog)s --mode new                # Global:Create new tasks for all given jobs
  %(prog)s --mode all                # Global: Run all exisiting tasks of all given jobs
  %(prog)s --mode latest             # Global: Run latest task of all given jobs
  %(prog)s --mode "job1:new,job2:all"  # Specify different strategies per job
        """,
    )

    # Job selection arguments
    parser.add_argument(
        "--job",
        "-j",
        type=str,
        help="Job name or pattern to execute (supports wildcards: testjob*, test?job, testjob[12])",
    )
    parser.add_argument(
        "config_files", nargs="*", type=Path, help="Configuration file(s) to process"
    )

    # Execution strategy arguments
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        help="Execution strategy: global (last/all/new) or job-specific (job1:new,job2:all). Examples: --mode all, --mode 'job1:new,job2:last'",
    )
    parser.add_argument(
        "--force-final-generation",
        "-f",
        action="store_true",
        help="Force regeneration of final audio from existing candidates, even if final audio exists",
    )
    parser.add_argument(
        "--rerender-all",
        "-r",
        action="store_true",
        help="Delete all existing candidates and re-render everything from scratch",
    )

    # Execution options

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to use for processing (default: auto)",
    )

    parser.add_argument(
        "--cli-menu-help",
        action="store_true",
        help="Show CLI-Menu equivalents and advanced usage information",
    )

    return parser.parse_args()


def resolve_config_files(args: argparse.Namespace, project_root: Path) -> List[Path]:
    """
    Resolve configuration files from command line arguments.

    Args:
        args: Parsed command line arguments
        project_root: Project root directory

    Returns:
        List of resolved configuration file paths (deduplicated)
    """
    config_files = []
    seen_paths = set()  # Track resolved paths to avoid duplicates

    if args.config_files:
        original_count = len(args.config_files)

        for config_file in args.config_files:
            resolved_path = None

            # Handle relative paths
            if not config_file.is_absolute():
                # Try relative to current directory first
                if config_file.exists():
                    resolved_path = config_file.resolve()
                # Then try relative to config directory
                elif (project_root / "config" / config_file).exists():
                    resolved_path = (project_root / "config" / config_file).resolve()
                else:
                    raise FileNotFoundError(
                        f"Configuration file not found: {config_file}"
                    )
            else:
                if config_file.exists():
                    resolved_path = config_file
                else:
                    raise FileNotFoundError(
                        f"Configuration file not found: {config_file}"
                    )

            # Add only if not already seen (silent deduplication)
            if resolved_path not in seen_paths:
                config_files.append(resolved_path)
                seen_paths.add(resolved_path)

        # Silent log of deduplication if duplicates were found
        if len(config_files) < original_count:
            duplicates_removed = original_count - len(config_files)
            logger.debug(f"🔄 Removed {duplicates_removed} duplicate config file(s)")

    return config_files


def confirm_cli_rerender_action(task_configs: List[TaskConfig]) -> bool:
    """
    CLI safety confirmation for --rerender-all flag.
    Shows affected task directories and asks for user confirmation.

    Args:
        task_configs: List of TaskConfig objects that will be affected

    Returns:
        True if user confirms, False if cancelled
    """

    print("\n⚠️  WARNING: RE-RENDER ALL CANDIDATES (CLI Mode)")
    print(
        "This will DELETE (!) ALL audio chunks and final audio files from the following task directories:"
    )
    print()

    # List all affected task directories
    for i, task in enumerate(task_configs, 1):
        # Format task display similar to MenuOrchestrator
        try:
            dt = datetime.strptime(task.timestamp, "%Y%m%d_%H%M%S")
            display_time = dt.strftime("%d.%m.%Y %H:%M")
        except ValueError:
            display_time = task.timestamp

        run_label_display = task.run_label if task.run_label else "no-label"
        print(f"  {i}. Job: {task.job_name} ({run_label_display}) - {display_time}")
        print(f"     Directory: {task.base_output_dir}")

    print()
    print("Are you sure you want to proceed? This action cannot be undone!")
    print("(y = YES, PROCEED | c = CANCEL)")

    while True:
        choice = input("\n> ").strip().lower()

        if choice in ["y", "yes"]:
            return True
        elif choice in ["c", "cancel", ""]:  # Include empty input as cancel
            return False
        else:
            print("Please enter 'y' for yes or 'c' to cancel")


def main() -> int:
    """Main entry point for the refactored TTS pipeline."""

    try:
        # Parse arguments
        args = parse_arguments()

        # Handle CLI-Menu help request
        if hasattr(args, "cli_menu_help") and args.cli_menu_help:
            from pipeline.job_manager.cli_mapper import CLIMapper

            cli_mapper = CLIMapper()
            print(cli_mapper.get_cli_help_text())
            return 0

        # Validate CLI arguments for problematic combinations
        if args.job and args.config_files:
            logger.error(
                "❌ Invalid combination of arguments! You cannot specify --job {string} and config-files at the same time."
            )
            logger.info(" Use EITHER:")
            logger.info(
                f"   python {sys.argv[0]} --job  \"{args.job}\" --mode {args.mode or 'new'}"
            )
            logger.info(" OR:")
            logger.info(
                f"   python {sys.argv[0]} {' '.join(str(f) for f in args.config_files)} --mode {args.mode or 'new'}"
            )
            return 1

        # Update verbose mode based on arguments
        verbose_mode = args.verbose  # Set to args.verbose for production

        # Reconfigure logging with correct verbose mode
        LoggingConfigurator.configure(
            log_file=PROJECT_ROOT / "logs" / "main.log",
            console_level=logging.INFO,
            file_level=logging.DEBUG,
            append=True,
            verbose_mode=verbose_mode,
            use_icons=True,
        )

        # Also set root logger level for debug if verbose
        if verbose_mode:
            logging.getLogger().setLevel(logging.DEBUG)

        logger.info("\n" + "=" * 50)
        logger.info("TTS PIPELINE - TASK-BASED EXECUTION SYSTEM")
        logger.info("=" * 50)

        # Detect device
        detect_device() if args.device == "auto" else args.device

        # Initialize core components
        project_root = PROJECT_ROOT
        config_manager = ConfigManager(project_root)
        job_manager = JobManager(config_manager)

        # Resolve execution plan
        config_files = resolve_config_files(args, project_root)
        execution_plan = job_manager.resolve_execution_plan(args, config_files)

        # Handle cancelled execution
        if execution_plan.execution_mode == "cancelled":
            return 0

        # Validate execution plan
        if not job_manager.validate_execution_plan(execution_plan):
            logger.error("Invalid execution plan")
            return 1

        # Print execution summary
        job_manager.print_execution_summary(execution_plan)

        # CLI Safety check for --rerender-all flag
        if args.rerender_all and execution_plan.task_configs:
            if not confirm_cli_rerender_action(execution_plan.task_configs):
                logger.info("❌ Operation cancelled by user")
                return 0

        # Execute tasks
        if (
            execution_plan.execution_mode == "batch"
            or len(execution_plan.task_configs) > 1
        ):
            # Batch execution
            logger.debug("Starting batch execution mode")

            batch_executor = BatchExecutor(config_manager)

            # Execute tasks
            task_results = batch_executor.execute_batch(execution_plan.task_configs)
            batch_result = batch_executor.get_batch_summary(task_results)

            # Return appropriate exit code
            return 0 if batch_result.failed_tasks == 0 else 1

        else:
            # Single task execution
            logger.debug("▶️  Starting single task execution mode")

            task_config = execution_plan.task_configs[0]

            # Use preloaded config if available, otherwise load from ConfigManager
            if task_config.preloaded_config:
                logger.debug("⚙️ Using preloaded config (avoiding redundant loading)")
                loaded_config = task_config.preloaded_config
            else:
                logger.debug(f"⚙️ Loading config: {task_config.config_path}")
                loaded_config = config_manager.load_cascading_config(
                    task_config.config_path
                )

            # Create file manager with preloaded config and shared ConfigManager
            file_manager = FileManager(
                task_config,
                preloaded_config=loaded_config,
                config_manager=config_manager,
            )

            # Create and execute task with preloaded config (avoiding redundant loading)
            task_executor = TaskExecutor(
                file_manager, task_config, config=loaded_config
            )

            result = task_executor.execute_task()

            # Report results
            if result.success:
                logger.info("=" * 50)
                logger.info("TASK COMPLETED SUCCESSFULLY")
                logger.info("=" * 50)
                logger.info(f"Job: {result.task_config.job_name}")
                logger.info(f"📊 Task: {result.task_config.task_name}")
                if result.task_config.run_label:
                    logger.info(f"Label: {result.task_config.run_label}")
                total_seconds = result.execution_time
                # Format duration as HH:MM:SS or MM:SS
                hours, remainder = divmod(int(total_seconds), 3600)
                minutes, seconds = divmod(remainder, 60)

                formatted_time = (
                    f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                    if hours > 0
                    else f"{minutes:02d}:{seconds:02d}"
                )
                logger.info(f"⏳ Execution time: {formatted_time}")
                logger.info(f"Final stage: {result.completion_stage.value}")

                if result.final_audio_path:
                    logger.info(f"📁 Final audio: {result.final_audio_path}")

                return 0
            else:
                logger.error("=" * 50)
                logger.error("❌ TASK EXECUTION FAILED")
                logger.error("=" * 50)
                logger.error(f"Job: {result.task_config.job_name}")
                logger.error(f"Error: {result.error_message}")
                logger.error(f"Stage reached: {result.completion_stage.value}")

                return 1

    except KeyboardInterrupt:
        logger.info("\nExecution interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
