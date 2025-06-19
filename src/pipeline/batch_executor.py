#!/usr/bin/env python3
"""
BatchExecutor for processing multiple tasks.
Handles batch execution of multiple jobs/tasks with progress tracking and error handling.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.pipeline.task_executor import CompletionStage, TaskExecutor, TaskResult
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
    error_summary: List[str]


class BatchExecutor:
    """
    Batch executor for processing multiple tasks.

    Features:
    - Sequential and parallel execution modes
    - Progress tracking and reporting
    - Error handling and recovery
    - Job dependency management
    """

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager

    def execute_multiple_tasks(
        self,
        task_configs: List[TaskConfig],
        parallel: bool = False,
        max_workers: int = 2,
    ) -> BatchResult:
        """
        Execute multiple tasks in batch mode.

        Args:
            task_configs: List of TaskConfig objects to execute
            parallel: Whether to execute tasks in parallel
            max_workers: Maximum number of parallel workers

        Returns:
            BatchResult object with execution summary
        """
        start_time = time.time()
        logger.info("=" * 50)
        logger.info(f"Starting batch execution of {len(task_configs)} tasks")
        logger.info(f"Execution mode: {'parallel' if parallel else 'sequential'}")

        if parallel:
            results = self._execute_parallel(task_configs, max_workers)
        else:
            results = self._execute_sequential(task_configs)

        execution_time = time.time() - start_time

        # Analyze results
        successful_tasks = sum(1 for result in results if result.success)
        failed_tasks = len(results) - successful_tasks
        error_summary = [
            f"{result.task_config.job_name}: {result.error_message}"
            for result in results
            if not result.success and result.error_message
        ]

        batch_result = BatchResult(
            total_tasks=len(task_configs),
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            execution_time=execution_time,
            task_results=results,
            error_summary=error_summary,
        )

        self._log_batch_summary(batch_result)
        return batch_result

    def _execute_sequential(self, task_configs: List[TaskConfig]) -> List[TaskResult]:
        """
        Execute tasks sequentially.

        Args:
            task_configs: List of TaskConfig objects

        Returns:
            List of TaskResult objects
        """
        results = []

        for i, task_config in enumerate(task_configs, 1):
            logger.info(
                f"🔧 Processing task {i}/{len(task_configs)}: {task_config.job_name}"
            )

            try:
                result = self._execute_single_task(task_config)
                results.append(result)

                if result.success:
                    logger.info(f"Task {i} completed successfully\n")
                else:
                    logger.error(f"✗ Task {i} failed: {result.error_message}\n")

            except Exception as e:
                logger.error(f"✗ Task {i} crashed: {e}", exc_info=True)

                error_result = TaskResult(
                    task_config=task_config,
                    success=False,
                    completion_stage=CompletionStage.NOT_STARTED,
                    error_message=str(e),
                )
                results.append(error_result)

        return results

    def _execute_parallel(
        self, task_configs: List[TaskConfig], max_workers: int
    ) -> List[TaskResult]:
        """
        Execute tasks in parallel.

        Args:
            task_configs: List of TaskConfig objects
            max_workers: Maximum number of parallel workers

        Returns:
            List of TaskResult objects
        """
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._execute_single_task, task_config): task_config
                for task_config in task_configs
            }

            # Collect results as they complete
            for future in as_completed(future_to_task):
                task_config = future_to_task[future]

                try:
                    result = future.result()
                    results.append(result)

                    if result.success:
                        logger.info(f"✓ Task completed: {task_config.job_name}")
                    else:
                        logger.error(
                            f"✗ Task failed: {task_config.job_name} - {result.error_message}"
                        )

                except Exception as e:
                    logger.error(
                        f"✗ Task crashed: {task_config.job_name} - {e}", exc_info=True
                    )

                    error_result = TaskResult(
                        task_config=task_config,
                        success=False,
                        completion_stage=CompletionStage.NOT_STARTED,
                        error_message=str(e),
                    )
                    results.append(error_result)

        # Sort results to match original order
        task_order = {id(task): i for i, task in enumerate(task_configs)}
        results.sort(key=lambda result: task_order.get(id(result.task_config), 999))

        return results

    def _execute_single_task(self, task_config: TaskConfig) -> TaskResult:
        """
        Execute a single task.

        Args:
            task_config: TaskConfig object

        Returns:
            TaskResult object
        """
        # Load config once to avoid duplicate loading
        loaded_config = self.config_manager.load_cascading_config(task_config.config_path)
        
        # Create file manager with preloaded config
        file_manager = FileManager(task_config, preloaded_config=loaded_config)

        # Create task executor and set config to avoid re-loading
        task_executor = TaskExecutor(file_manager, task_config)
        task_executor.config = loaded_config

        # Execute task
        return task_executor.execute_task()

    def handle_task_dependencies(self, tasks: List[TaskConfig]) -> List[TaskConfig]:
        """
        Analyze and handle task dependencies.

        Args:
            tasks: List of TaskConfig objects

        Returns:
            List of TaskConfig objects in dependency order
        """
        # For now, simple implementation - just return tasks as-is
        # Future enhancement: analyze dependencies and reorder tasks

        # Group tasks by job name
        job_groups = {}
        for task in tasks:
            job_name = task.job_name
            if job_name not in job_groups:
                job_groups[job_name] = []
            job_groups[job_name].append(task)

        # Sort each group by timestamp (oldest first for sequential processing)
        ordered_tasks = []
        for job_name in sorted(job_groups.keys()):
            job_tasks = job_groups[job_name]
            job_tasks.sort(key=lambda t: t.timestamp)
            ordered_tasks.extend(job_tasks)

        logger.info(f"Organized {len(tasks)} tasks from {len(job_groups)} jobs")
        return ordered_tasks

    def _log_batch_summary(self, batch_result: BatchResult):
        """
        Log a summary of batch execution results.

        Args:
            batch_result: BatchResult object
        """
        logger.info("\n" + "=" * 50)
        logger.info("📋 BATCH EXECUTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"- Total tasks: {batch_result.total_tasks}")
        logger.info(f"- Successful: {batch_result.successful_tasks}")
        logger.info(f"- Failed: {batch_result.failed_tasks}")
        logger.info(f"- Total execution time: {batch_result.execution_time:.2f} seconds")

        if batch_result.successful_tasks > 0:
            avg_time = (
                sum(r.execution_time for r in batch_result.task_results if r.success)
                / batch_result.successful_tasks
            )
            logger.info(f"- Average task time: {avg_time:.2f} seconds")

        if batch_result.error_summary:
            logger.info("\nErrors:")
            for error in batch_result.error_summary:
                logger.info(f"  • {error}")

        logger.info("=" * 50)

    def print_detailed_results(self, batch_result: BatchResult):
        """
        Print detailed results of batch execution.

        Args:
            batch_result: BatchResult object
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 DETAILED BATCH RESULTS")
        logger.info("=" * 80)

        for i, result in enumerate(batch_result.task_results, 1):
            status = "✅" if result.success else "❌"
            logger.info(f"{status} {i:2d}. {result.task_config.job_name} - {result.success}")
            logger.info(f"- Task: {result.task_config.task_name}")
            if result.task_config.run_label:
                logger.info(f"- Label: {result.task_config.run_label}")
            logger.info(f"- Time: {result.execution_time:.2f}s")
            logger.info(f"- Stage: {result.completion_stage.value}")

            if result.final_audio_path:
                logger.info(f"- Output: {result.final_audio_path}")

            if not result.success and result.error_message:
                logger.info(f"❌ Error: {result.error_message}")

            logger.info("")

        logger.info("=" * 80)

    def generate_batch_report(
        self, batch_result: BatchResult, output_path: Optional[Path] = None
    ) -> Path:
        """
        Generate a detailed batch execution report.

        Args:
            batch_result: BatchResult object
            output_path: Optional path for report file

        Returns:
            Path to generated report file
        """
        if output_path is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"batch_report_{timestamp}.txt")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("TTS PIPELINE BATCH EXECUTION REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Execution Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Tasks: {batch_result.total_tasks}\n")
            f.write(f"Successful: {batch_result.successful_tasks}\n")
            f.write(f"Failed: {batch_result.failed_tasks}\n")
            f.write(
                f"Success Rate: {batch_result.successful_tasks/batch_result.total_tasks*100:.1f}%\n"
            )
            f.write(f"Total Time: {batch_result.execution_time:.2f} seconds\n\n")

            f.write("TASK DETAILS:\n")
            f.write("-" * 50 + "\n\n")

            for i, result in enumerate(batch_result.task_results, 1):
                f.write(f"{i:2d}. {result.task_config.job_name}\n")
                f.write(f"    Status: {'SUCCESS' if result.success else 'FAILED'}\n")
                f.write(f"    Task: {result.task_config.task_name}\n")
                f.write(f"    Run Label: {result.task_config.run_label or 'None'}\n")
                f.write(f"    Execution Time: {result.execution_time:.2f}s\n")
                f.write(f"    Completion Stage: {result.completion_stage.value}\n")

                if result.final_audio_path:
                    f.write(f"    Final Audio: {result.final_audio_path}\n")

                if not result.success and result.error_message:
                    f.write(f"    Error: {result.error_message}\n")

                f.write("\n")

            if batch_result.error_summary:
                f.write("ERROR SUMMARY:\n")
                f.write("-" * 50 + "\n")
                for error in batch_result.error_summary:
                    f.write(f"• {error}\n")

        logger.info(f"Batch report file generated: {output_path}")
        return output_path
