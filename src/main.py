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

# Project root detection
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from pipeline.batch_executor import BatchExecutor
from pipeline.job_manager.types import ExecutionPlan
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
  %(prog)s job1.yaml job2.yaml       # Run specific config files
  %(prog)s --job "my_job"            # Run job by name
  %(prog)s --job "testjob*"          # Run all jobs starting with "testjob"
  %(prog)s --job "test?job"          # Run jobs like "test1job", "test2job" etc.
  %(prog)s --job "my_job" --parallel # Run job in parallel mode
  %(prog)s --mode new                # Create new tasks for all jobs
  %(prog)s --mode all                # Run all exisiting tasks of all jobs
  %(prog)s --mode last               # Run latest task of all jobs
  %(prog)s --mode "job1:new,job2:all"  # Different strategies per job
  %(prog)s --force-final-generation  # Force regeneration of final audio from existing candidates
  %(prog)s --rerender-all            # Delete all existing candidates and re-render everything from scratch
  %(prog)s --verbose                 # Enable verbose logging
        """,
    )

    # Job selection arguments
    parser.add_argument("--job", "-j", type=str, help="Job name or pattern to execute (supports wildcards: testjob*, test?job, testjob[12])")
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
        "--force-final-generation",
        action="store_true",
        help="Force regeneration of final audio from existing candidates, even if final audio exists",
    )
    parser.add_argument(
        "--rerender-all",
        action="store_true",
        help="Delete all existing candidates and re-render everything from scratch",
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


def main() -> int:
    """Main entry point for the refactored TTS pipeline."""

    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate CLI arguments for problematic combinations
        if args.job and args.config_files:
            logger.error("❌ Ungültige Argumentkombination!")
            logger.error("Sie können nicht gleichzeitig --job und config-Dateien angeben.")
            logger.error("Verwenden Sie ENTWEDER:")
            logger.error(f"  python {sys.argv[0]} --job \"{args.job}\" --mode {args.mode or 'new'}")
            logger.error(f"ODER:")
            logger.error(f"  python {sys.argv[0]} {' '.join(str(f) for f in args.config_files)}")
            logger.error("Wenn Sie mehrere Jobs gleichzeitig ausführen möchten, verwenden Sie:")
            logger.error(f"  python {sys.argv[0]} {' '.join(str(f) for f in args.config_files)} --mode {args.mode or 'new'}")
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
        device = detect_device() if args.device == "auto" else args.device

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

        # Execute tasks
        if (
            execution_plan.execution_mode == "batch"
            or len(execution_plan.task_configs) > 1
        ):
            # Batch execution
            logger.debug("Starting batch execution mode")

            batch_executor = BatchExecutor(
                config_manager, 
                max_workers=args.max_workers if args.parallel else None,
                parallel_enabled=args.parallel
            )

            # Execute with parallel option
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
                loaded_config = config_manager.load_cascading_config(task_config.config_path)

            # Create file manager with preloaded config and shared ConfigManager
            file_manager = FileManager(task_config, preloaded_config=loaded_config, config_manager=config_manager)

            # Create and execute task with preloaded config (avoiding redundant loading)
            task_executor = TaskExecutor(file_manager, task_config, config=loaded_config)

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
                logger.info(f"⏳ Execution time: {result.execution_time:.2f} seconds")
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
