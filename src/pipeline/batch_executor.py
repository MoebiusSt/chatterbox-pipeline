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
from utils.file_manager.file_manager import FileManager

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

    def __init__(self, config_manager: ConfigManager, max_workers: Optional[int] = None, parallel_enabled: bool = True):
        """
        Initialize BatchExecutor.

        Args:
            config_manager: Shared ConfigManager instance (avoids redundant loading)
            max_workers: Maximum number of parallel workers. If None, uses system default.
            parallel_enabled: Whether parallel execution is enabled. If False, executes sequentially.
        """
        self.config_manager = config_manager
        self.max_workers = max_workers
        self.parallel_enabled = parallel_enabled
        logger.info(f"BatchExecutor initialized with max_workers={max_workers}, parallel_enabled={parallel_enabled}")

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

        # Determine execution mode based on task count and parallel flag
        use_parallel = self.parallel_enabled and total_tasks > 1

        if not use_parallel:
            # Sequential execution - single task or parallel disabled
            execution_mode = "SINGLE" if total_tasks == 1 else "SEQUENTIAL"
            logger.debug(f"Execution mode: {execution_mode}")

            # Execute tasks sequentially
            for i, task_config in enumerate(task_configs, 1):
                if total_tasks > 1:
                    logger.info(f"▶️  Executing task {i}/{total_tasks}: {task_config.job_name}:{task_config.task_name}")
                
                result = self._execute_single_task(task_config)
                results.append(result)

                # Log completion for multi-task sequential execution
                if total_tasks > 1:
                    status = "✅ SUCCESS" if result.success else "❌ FAILED"
                    logger.info(f"{status}: {task_config.job_name}:{task_config.task_name}")

        else:
            # Parallel execution - multiple tasks with parallel enabled
            logger.debug(f"Execution mode: PARALLEL, Max Workers: {self.max_workers or 'auto'}")

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
        # Use preloaded config if available, otherwise load from ConfigManager
        if task_config.preloaded_config:
            logger.debug("⚙️ Using preloaded config (avoiding redundant loading)")
            loaded_config = task_config.preloaded_config
        else:
            logger.debug(f"⚙️ Loading config: {task_config.config_path}")
            loaded_config = self.config_manager.load_cascading_config(task_config.config_path)

        # Create file manager with preloaded config and shared ConfigManager
        file_manager = FileManager(task_config, preloaded_config=loaded_config, config_manager=self.config_manager)

        # Create task executor with preloaded config (avoiding redundant loading)
        task_executor = TaskExecutor(file_manager, task_config, config=loaded_config)

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
