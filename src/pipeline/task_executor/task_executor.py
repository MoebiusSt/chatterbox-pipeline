#!/usr/bin/env python3
"""TaskExecutor for unified task execution with separated stage handlers."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from chunking.spacy_chunker import SpaCyChunker
from generation.candidate_manager import CandidateManager
from generation.tts_generator import TTSGenerator
from utils.config_manager import ConfigManager, TaskConfig
from utils.file_manager import CompletionStage, FileManager, TaskState
from utils.progress_tracker import ProgressTracker
from validation.quality_scorer import QualityScorer
from validation.whisper_validator import WhisperValidator

from .stage_handlers import (
    AssemblyHandler,
    GenerationHandler,
    PreprocessingHandler,
    ValidationHandler,
)

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of task execution."""

    task_config: TaskConfig
    success: bool
    completion_stage: CompletionStage
    error_message: Optional[str] = None
    execution_time: float = 0.0
    final_audio_path: Optional[Path] = None


class TaskExecutor:
    """Unified task executor with separated stage handlers."""

    def __init__(self, file_manager: FileManager, task_config: TaskConfig):
        self.file_manager = file_manager
        self.task_config = task_config

        # Load config data only if not already set
        if not hasattr(self, "config") or self.config is None:
            cm = ConfigManager(
                task_config.config_path.parent.parent.parent.parent
            )
            self.config = cm.load_cascading_config(task_config.config_path)
            file_manager.config = self.config

        # Initialize progress tracking
        self.progress_tracker = None

        # Initialize components (lazy loading)
        self._chunker = None
        self._tts_generator = None
        self._whisper_validator = None
        self._quality_scorer = None
        self._candidate_manager = None

        # Initialize handlers (lazy loading)
        self._preprocessing_handler = None
        self._generation_handler = None
        self._validation_handler = None
        self._assembly_handler = None

    @property
    def chunker(self) -> SpaCyChunker:
        """Lazy-loaded chunker."""
        if self._chunker is None:
            chunking_config = self.config["chunking"]
            self._chunker = SpaCyChunker(
                model_name=chunking_config.get("spacy_model", "en_core_web_sm"),
                target_limit=chunking_config.get("target_chunk_limit", 480),
                max_limit=chunking_config.get("max_chunk_limit", 600),
                min_length=chunking_config.get("min_chunk_length", 50),
            )
        return self._chunker

    @property
    def tts_generator(self) -> TTSGenerator:
        """Lazy-loaded TTS generator."""
        if self._tts_generator is None:
            device = self._detect_device()
            self._tts_generator = TTSGenerator(
                config=self.config["generation"], device=device, seed=12345
            )
            logger.debug("TTSGenerator initialized with automatic model loading")
        return self._tts_generator

    @property
    def whisper_validator(self) -> WhisperValidator:
        """Lazy-loaded Whisper validator."""
        if self._whisper_validator is None:
            validation_config = self.config["validation"]
            self._whisper_validator = WhisperValidator(
                model_size=validation_config.get("whisper_model", "base"),
                device="auto",
                similarity_threshold=validation_config.get("similarity_threshold", 0.7),
                min_quality_score=validation_config.get("min_quality_score", 0.75),
            )
        return self._whisper_validator

    @property
    def quality_scorer(self) -> QualityScorer:
        """Lazy-loaded quality scorer."""
        if self._quality_scorer is None:
            self._quality_scorer = QualityScorer(sample_rate=24000)
        return self._quality_scorer

    @property
    def candidate_manager(self):
        """Lazy-loaded candidate manager."""
        if self._candidate_manager is None:
            self._candidate_manager = CandidateManager(
                tts_generator=self.tts_generator,
                config=self.config,
                output_dir=self.file_manager.task_directory,
            )
            self._candidate_manager.file_manager = self.file_manager
        return self._candidate_manager

    @property
    def preprocessing_handler(self) -> PreprocessingHandler:
        """Lazy-loaded preprocessing handler."""
        if self._preprocessing_handler is None:
            self._preprocessing_handler = PreprocessingHandler(
                self.file_manager, self.config, self.chunker
            )
        return self._preprocessing_handler

    @property
    def generation_handler(self) -> GenerationHandler:
        """Lazy-loaded generation handler."""
        if self._generation_handler is None:
            self._generation_handler = GenerationHandler(
                self.file_manager,
                self.config,
                self.tts_generator,
                self.candidate_manager,
            )
        return self._generation_handler

    @property
    def validation_handler(self) -> ValidationHandler:
        """Lazy-loaded validation handler."""
        if self._validation_handler is None:
            self._validation_handler = ValidationHandler(
                self.file_manager,
                self.config,
                self.whisper_validator,
                self.quality_scorer,
                self.generation_handler,  # Pass for retry logic
            )
        return self._validation_handler

    @property
    def assembly_handler(self) -> AssemblyHandler:
        """Lazy-loaded assembly handler."""
        if self._assembly_handler is None:
            self._assembly_handler = AssemblyHandler(
                self.file_manager, self.config, self.task_config
            )
        return self._assembly_handler

    def create_progress_tracker(
        self, total_items: int, description: str
    ) -> ProgressTracker:
        """Create a progress tracker for a specific stage."""
        return ProgressTracker(total_items, description)

    def execute_task(self) -> TaskResult:
        """Execute a complete task with automatic state detection and resumption."""
        start_time = time.time()

        try:
            logger.debug(f"Task directory: {self.task_config.base_output_dir}")

            # Analyze current state
            task_state = self.file_manager.analyze_task_state()
            logger.info(
                f"Current completion stage: {task_state.completion_stage.value}"
            )

            # Migrate existing whisper files to enhanced metrics format
            self.file_manager.migrate_whisper_to_enhanced_metrics()

            if task_state.missing_components:
                logger.debug(
                    f"Missing components: {', '.join(task_state.missing_components)}"
                )

            # Check if we should force final audio regeneration
            force_final = self.task_config.add_final
            if force_final and task_state.completion_stage == CompletionStage.COMPLETE:
                logger.info("🔄 Forcing final audio regeneration")

                has_missing_candidates = any(
                    "candidates_chunk" in comp for comp in task_state.missing_components
                )
                has_missing_whisper = any(
                    "whisper_chunk" in comp for comp in task_state.missing_components
                )

                if has_missing_candidates or has_missing_whisper:
                    logger.info(
                        "Missing data detected, will regenerate before final assembly"
                    )
                    # Continue with normal pipeline execution
                else:
                    # Jump directly to assembly
                    if not self.assembly_handler.execute_assembly():
                        return TaskResult(
                            task_config=self.task_config,
                            success=False,
                            completion_stage=task_state.completion_stage,
                            error_message="Final assembly failed",
                            execution_time=time.time() - start_time,
                        )

                    return TaskResult(
                        task_config=self.task_config,
                        success=True,
                        completion_stage=CompletionStage.COMPLETE,
                        execution_time=time.time() - start_time,
                        final_audio_path=self.file_manager.get_final_audio(),
                    )

            # Execute stages based on current state
            if not self._execute_stages_from_state(task_state):
                return TaskResult(
                    task_config=self.task_config,
                    success=False,
                    completion_stage=task_state.completion_stage,
                    error_message="Pipeline execution failed",
                    execution_time=time.time() - start_time,
                )

            # Success - get final stage and paths
            final_audio_path = self.file_manager.get_final_audio()

            return TaskResult(
                task_config=self.task_config,
                success=True,
                completion_stage=CompletionStage.COMPLETE,
                execution_time=time.time() - start_time,
                final_audio_path=final_audio_path,
            )

        except Exception as e:
            logger.error(f"Task execution failed: {e}", exc_info=True)
            return TaskResult(
                task_config=self.task_config,
                success=False,
                completion_stage=CompletionStage.NOT_STARTED,
                error_message=str(e),
                execution_time=time.time() - start_time,
            )

    def _execute_stages_from_state(self, task_state: TaskState) -> bool:
        """Execute the pipeline stages based on the current task state."""
        if task_state.completion_stage == CompletionStage.COMPLETE:
            logger.info("Task already complete")
            return True

        # Execute stages in order based on what's missing
        if task_state.completion_stage in [
            CompletionStage.NOT_STARTED,
            CompletionStage.PREPROCESSING,
        ]:
            if not self.preprocessing_handler.execute_preprocessing():
                return False

        if task_state.completion_stage in [
            CompletionStage.NOT_STARTED,
            CompletionStage.PREPROCESSING,
            CompletionStage.GENERATION,
        ]:
            if not self.generation_handler.execute_generation():
                return False

        if task_state.completion_stage in [
            CompletionStage.NOT_STARTED,
            CompletionStage.PREPROCESSING,
            CompletionStage.GENERATION,
            CompletionStage.VALIDATION,
        ]:
            if not self.validation_handler.execute_validation():
                return False

        if task_state.completion_stage in [
            CompletionStage.NOT_STARTED,
            CompletionStage.PREPROCESSING,
            CompletionStage.GENERATION,
            CompletionStage.VALIDATION,
            CompletionStage.ASSEMBLY,
        ]:
            if not self.assembly_handler.execute_assembly():
                return False

        return True

    def _detect_device(self) -> str:
        """Detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu" 