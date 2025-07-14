#!/usr/bin/env python3
"""
Task Orchestrator - Orchestrates execution of one or multiple tasks.
Handles progress tracking and batch summaries, but delegates actual task execution.
"""

import logging
import time
from typing import List

from utils.config_manager import ConfigManager, TaskConfig
from utils.file_manager.file_manager import FileManager
from .task_executor.task_executor import TaskExecutor, TaskResult

logger = logging.getLogger(__name__)


class TaskOrchestrator:
    """Orchestrates execution of single or multiple tasks."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
    
    def execute_tasks(self, task_configs: List[TaskConfig]) -> List[TaskResult]:
        """Execute one or more tasks with unified orchestration."""
        if not task_configs:
            return []
        
        results = []
        total_tasks = len(task_configs)
        
        # Batch start message for multiple tasks
        if total_tasks > 1:
            logger.info(f"🚀 Starting batch execution of {total_tasks} tasks")
        
        for i, task_config in enumerate(task_configs, 1):
            # Progress log for multiple tasks
            if total_tasks > 1:
                logger.info(f"▶️  Executing task {i}/{total_tasks}: {task_config.job_name}:{task_config.task_name}")
            
            # Execute task (always same logic)
            result = self._execute_single_task(task_config)
            results.append(result)
            
            # Detailed completion log for multiple tasks
            if total_tasks > 1:
                # Show detailed task execution summary for each task in batch mode
                self._log_single_task_summary(result)
        
        # Show appropriate summary based on number of tasks
        if total_tasks == 1:
            self._log_single_task_summary(results[0])
        else:
            self._log_batch_summary(results)
        
        return results
    
    def _execute_single_task(self, task_config: TaskConfig) -> TaskResult:
        """Execute a single task by delegating to TaskExecutor."""
        # Use preloaded config if available, otherwise load from ConfigManager
        if task_config.preloaded_config:
            logger.debug("⚙️ Using preloaded config")
            loaded_config = task_config.preloaded_config
        else:
            logger.debug(f"⚙️ Loading config: {task_config.config_path}")
            loaded_config = self.config_manager.load_cascading_config(
                task_config.config_path
            )
        
        # Create file manager and task executor
        file_manager = FileManager(
            task_config,
            preloaded_config=loaded_config,
            config_manager=self.config_manager,
        )
        task_executor = TaskExecutor(file_manager, task_config, config=loaded_config)
        
        # Execute task
        return task_executor.execute_task()
    
    def _format_execution_time(self, seconds: float) -> str:
        """Format execution time as HH:MM:SS or MM:SS."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        return (
            f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            if hours > 0
            else f"{minutes:02d}:{seconds:02d}"
        )
    
    def _log_summary_header(self, title: str) -> None:
        logger.info("=" * 50)
        logger.info(f"📊 {title}")
    
    def _log_single_task_summary(self, result: TaskResult):
        """Log single task summary with execution time."""
        formatted_time = self._format_execution_time(result.execution_time)
        status = "✅ SUCCESS" if result.success else "❌ FAILED"
        
        self._log_summary_header("TASK EXECUTION SUMMARY")
        logger.info(f"  Status: {status}")
        logger.info(f"  Task: {result.task_config.job_name}:{result.task_config.task_name}")
        logger.info(f"  Execution time: {formatted_time}")
        if result.final_audio_path:
            logger.info(f"  Final audio: {result.final_audio_path}")
        if result.error_message:
            logger.info(f"  Error: {result.error_message}")
        logger.info("=" * 50)
    
    def _log_batch_summary(self, results: List[TaskResult]):
        """Log detailed batch summary with success rate and formatted time."""
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.success)
        failed_tasks = total_tasks - successful_tasks
        total_time = sum(r.execution_time for r in results)
        
        formatted_time = self._format_execution_time(total_time)
        
        self._log_summary_header("BATCH EXECUTION SUMMARY")
        logger.info(f"  Total tasks: {total_tasks}")
        logger.info(f"  Successful: {successful_tasks}")
        logger.info(f"  Failed: {failed_tasks}")
        logger.info(f"  Success rate: {(successful_tasks/total_tasks)*100:.1f}%")
        logger.info(f"  Total time: {formatted_time}")
        logger.info("=" * 50)
    
    def get_exit_code(self, results: List[TaskResult]) -> int:
        """Get exit code: 0 if all successful, 1 if any failed."""
        return 0 if all(r.success for r in results) else 1 