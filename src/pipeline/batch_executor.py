#!/usr/bin/env python3
"""
BatchExecutor for managing multiple TTS tasks in parallel.
Handles batch processing with progress tracking and error management.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pipeline.task_executor import CompletionStage, TaskExecutor, TaskResult
from utils.config_manager import ConfigManager, TaskConfig
from utils.file_manager import FileManager

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of batch execution."""

    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    execution_time: float
    task_results: List[TaskResult]

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        return (
            (self.successful_tasks / self.total_tasks) * 100
            if self.total_tasks > 0
            else 0.0
        )


class BatchExecutor:
    """Executes multiple TTS tasks in parallel with progress tracking."""

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize BatchExecutor.

        Args:
            max_workers: Maximum number of parallel workers. If None, uses system default.
        """
        self.max_workers = max_workers
        logger.info(f"BatchExecutor initialized with max_workers={max_workers}")

    def execute_batch(self, task_configs: List[TaskConfig]) -> List[TaskResult]:
        """
        Execute a batch of TTS tasks.

        Args:
            task_configs: List of task configurations to execute

        Returns:
            List of TaskResult objects
        """
        start_time = time.time()
        total_tasks = len(task_configs)

        logger.info(f"🚀 Starting batch execution of {total_tasks} tasks")

        results = []

        if total_tasks == 1:
            # Single task - execute directly for better logging
            logger.info("📋 EXECUTION PLAN SUMMARY")
            logger.info("  Mode: SINGLE")
            logger.info(f"  Tasks: {total_tasks}")

            task_config = task_configs[0]
            logger.info(f"  1. {task_config.job_name}: {task_config.task_name}")
            logger.info(f"     └─ {task_config.base_output_dir}")
            logger.info("=" * 50)

            result = self._execute_single_task(task_config)
            results.append(result)

        else:
            # Multiple tasks - use parallel execution
            logger.info("📋 EXECUTION PLAN SUMMARY")
            logger.info("  Mode: PARALLEL")
            logger.info(f"  Tasks: {total_tasks}")
            logger.info(f"  Max Workers: {self.max_workers or 'auto'}")

            for i, task_config in enumerate(task_configs, 1):
                logger.info(f"  {i}. {task_config.job_name}: {task_config.task_name}")
                logger.info(f"     └─ {task_config.base_output_dir}")
            logger.info("=" * 50)

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_config = {
                    executor.submit(self._execute_single_task, config): config
                    for config in task_configs
                }

                # Collect results as they complete
                for future in as_completed(future_to_config):
                    config = future_to_config[future]
                    try:
                        result = future.result()
                        results.append(result)

                        # Log completion
                        status = "✅ SUCCESS" if result.success else "❌ FAILED"
                        logger.info(f"{status}: {config.job_name}:{config.task_name}")

                    except Exception as e:
                        logger.error(
                            f"❌ FAILED: {config.job_name}:{config.task_name} - {e}"
                        )
                        # Create a failed result
                        failed_result = TaskResult(
                            task_config=config,
                            success=False,
                            completion_stage=CompletionStage.NOT_STARTED,
                            execution_time=0.0,
                            error_message=str(e),
                            final_audio_path=None,
                        )
                        results.append(failed_result)

        # Calculate batch statistics
        execution_time = time.time() - start_time
        successful_tasks = sum(1 for r in results if r.success)
        failed_tasks = total_tasks - successful_tasks

        # Log batch summary
        logger.info("=" * 50)
        logger.info("📊 BATCH EXECUTION SUMMARY")
        logger.info(f"  Total tasks: {total_tasks}")
        logger.info(f"  Successful: {successful_tasks}")
        logger.info(f"  Failed: {failed_tasks}")
        logger.info(f"  Success rate: {(successful_tasks/total_tasks)*100:.1f}%")
        logger.info(f"  Total time: {execution_time:.2f} seconds")
        logger.info("=" * 50)

        return results

    def _execute_single_task(self, task_config: TaskConfig) -> TaskResult:
        """
        Execute a single task.

        Args:
            task_config: Task configuration

        Returns:
            TaskResult object
        """
        # Load config once to avoid duplicate loading
        project_root = Path.cwd()  # Assume we're running from project root
        config_manager = ConfigManager(project_root)

        # Debug output
        logger.debug(f"task_config.config_path: {task_config.config_path}")
        logger.debug(f"task_config.config_path type: {type(task_config.config_path)}")

        loaded_config = config_manager.load_cascading_config(task_config.config_path)

        # Create file manager with preloaded config
        file_manager = FileManager(task_config, preloaded_config=loaded_config)

        # Create task executor with required parameters
        task_executor = TaskExecutor(file_manager, task_config)
        task_executor.config = loaded_config

        # Execute task
        return task_executor.execute_task()

    def get_batch_summary(self, results: List[TaskResult]) -> BatchResult:
        """
        Generate a summary of batch execution results.

        Args:
            results: List of task results

        Returns:
            BatchResult summary
        """
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.success)
        failed_tasks = total_tasks - successful_tasks

        # Calculate total execution time
        total_execution_time = sum(r.execution_time for r in results)

        return BatchResult(
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            execution_time=total_execution_time,
            task_results=results,
        )
