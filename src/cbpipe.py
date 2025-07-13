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

from generation.model_cache import ChatterboxModelCache
from pipeline.job_manager_wrapper import JobManager
from pipeline.task_orchestrator import TaskOrchestrator
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
    - python cbpipe.py                           -> default job
    - python cbpipe.py config1.yaml config2.yaml -> specific configs
    - python cbpipe.py --job "jobname"           -> find job by name
    - python cbpipe.py --parallel               -> enable parallel execution
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
        "--explain-cache",
        action="store_true",
        help="Explain model cache behavior and exit",
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
                    # Provide more helpful error message for relative paths
                    attempted_paths = [
                        str(config_file.resolve()),
                        str(project_root / "config" / config_file)
                    ]
                    raise FileNotFoundError(
                        f"Configuration file not found: {config_file}\n"
                        f"Attempted paths: {', '.join(attempted_paths)}"
                    )
            else:
                if config_file.exists():
                    resolved_path = config_file
                else:
                    # Provide more helpful error message for absolute paths
                    raise FileNotFoundError(
                        f"Configuration file not found: {config_file}\n"
                        f"Absolute path does not exist: {config_file}"
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

        # Handle cache explanation request
        if hasattr(args, "explain_cache") and args.explain_cache:
            ChatterboxModelCache.explain_cache_behavior()
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

        # Execute tasks using task orchestrator
        orchestrator = TaskOrchestrator(config_manager)
        results = orchestrator.execute_tasks(execution_plan.task_configs)
        
        # Return appropriate exit code
        return orchestrator.get_exit_code(results)

    except KeyboardInterrupt:
        logger.info("\nExecution interrupted by user")
        return 1

    except FileNotFoundError as e:
        # Handle configuration file not found errors gracefully
        error_msg = str(e)
        if "Configuration file not found" in error_msg:
            logger.error("❌ Configuration file not found!")
            
            # Parse the error message to extract the file path and attempted paths
            lines = error_msg.split('\n')
            file_path = lines[0].split(': ')[1] if ': ' in lines[0] else lines[0]
            logger.error(f"   Path: {file_path}")
            
            # Show attempted paths if available
            if len(lines) > 1 and "Attempted paths:" in lines[1]:
                attempted_paths = lines[1].split("Attempted paths: ")[1].split(", ")
                logger.info("")
                logger.info("🔍 Attempted paths:")
                for i, path in enumerate(attempted_paths, 1):
                    logger.info(f"   {i}. {path}")
            
            logger.info("")
            logger.info("💡 Possible solutions:")
            logger.info("   1. Check the path to the configuration file")
            logger.info("   2. Use forward slashes (/) instead of backslashes (\\)")
            logger.info("   3. Make sure the file exists")
            logger.info("   4. Use relative paths from the project directory")
            logger.info("")
            logger.info("📁 Available configuration files in config/ directory:")
            config_dir = PROJECT_ROOT / "config"
            if config_dir.exists():
                yaml_files = list(config_dir.glob("*.yaml"))
                if yaml_files:
                    for config_file in yaml_files:
                        logger.info(f"   - {config_file.name}")
                else:
                    logger.info("   (no .yaml files found)")
            else:
                logger.info("   (config/ directory not found)")
            logger.info("")
            logger.info("🔧 Example of correct usage:")
            logger.info(f"   python {sys.argv[0]} config/your-file.yaml")
            logger.info(f"   python {sys.argv[0]} /absolute/path/to/file.yaml")
        else:
            logger.error(f"❌ File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main()) 