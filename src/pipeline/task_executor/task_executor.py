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
from utils.file_manager.file_manager import FileManager
from utils.file_manager.state_analyzer import CompletionStage, TaskState
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

    def __init__(self, file_manager: FileManager, task_config: TaskConfig, config: Optional[Dict[str, Any]] = None):
        self.file_manager = file_manager
        self.task_config = task_config

        # Use provided config or load from file
        if config is not None:
            self.config = config
        else:
            cm = ConfigManager(task_config.config_path.parent.parent.parent.parent)
            self.config = cm.load_cascading_config(task_config.config_path)
            
        # Ensure file_manager has the config
        if not hasattr(file_manager, 'config') or file_manager.config is None:
            file_manager.config = self.config
            
        # Also ensure the state analyzer has the correct config
        if hasattr(file_manager, '_state_analyzer') and file_manager._state_analyzer:
            file_manager._state_analyzer.config = self.config

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

            # Check if we need to delete all candidates and start fresh
            if self.task_config.rerender_all:
                logger.info("🔄 Re-rendering all candidates from scratch - deleting existing candidates and validation data")
                self._delete_all_candidates_and_validation()

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
                    # Continue with normal pipeline execution to fill gaps
                    # Do NOT return here - let it flow through to _execute_stages_from_state
                else:
                    # Jump directly to assembly if no missing data
                    if not self.assembly_handler.execute_assembly():
                        return TaskResult(
                            task_config=self.task_config,
                            success=False,
                            completion_stage=task_state.completion_stage,
                            error_message="Final assembly failed",
                            execution_time=time.time() - start_time,
                        )

                    # Find the actual final audio file path
                    final_audio_path = None
                    final_files = list(self.file_manager.final_dir.glob("*_final.wav"))
                    if final_files:
                        final_audio_path = max(
                            final_files, key=lambda f: f.stat().st_mtime
                        )

                    return TaskResult(
                        task_config=self.task_config,
                        success=True,
                        completion_stage=CompletionStage.COMPLETE,
                        execution_time=time.time() - start_time,
                        final_audio_path=final_audio_path,
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

            # Success - get final audio file path
            final_audio_path = None
            final_files = list(self.file_manager.final_dir.glob("*_final.wav"))
            if final_files:
                final_audio_path = max(final_files, key=lambda f: f.stat().st_mtime)

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
        # Check if this is a gap-filling scenario (unified detection)
        has_missing_candidates = any("candidates_chunk" in comp for comp in task_state.missing_components)
        has_missing_whisper = any("whisper_chunk" in comp for comp in task_state.missing_components)
        has_existing_metrics = bool(self.file_manager.get_metrics())
        
        is_gap_filling = (
            has_existing_metrics and 
            (has_missing_candidates or has_missing_whisper) and
            self.task_config.add_final
        )
        
        if task_state.completion_stage == CompletionStage.COMPLETE and not is_gap_filling:
            logger.info("Task already complete")
            return True
        
        if is_gap_filling:
            logger.info("🔄 Gap-filling mode detected - regenerating missing components")
            # Determine which chunks need validation in gap-filling mode
            missing_chunk_indices = []
            for comp in task_state.missing_components:
                if "candidates_chunk" in comp:
                    # Extract chunk index from component name like "candidates_chunk_33"
                    try:
                        chunk_idx = int(comp.split("_")[-1])
                        missing_chunk_indices.append(chunk_idx)
                    except (ValueError, IndexError):
                        logger.warning(f"Could not extract chunk index from component: {comp}")
        
        # Execute stages in order based on what's missing
        if task_state.completion_stage in [
            CompletionStage.NOT_STARTED,
            CompletionStage.PREPROCESSING,
        ]:
            if not self.preprocessing_handler.execute_preprocessing():
                return False

        if (task_state.completion_stage in [
            CompletionStage.NOT_STARTED,
            CompletionStage.PREPROCESSING,
            CompletionStage.GENERATION,
        ]) or is_gap_filling:  # Also run generation for gap-filling
            if not self.generation_handler.execute_generation():
                return False

        if (task_state.completion_stage in [
            CompletionStage.NOT_STARTED,
            CompletionStage.PREPROCESSING,
            CompletionStage.GENERATION,
            CompletionStage.VALIDATION,
        ]) or is_gap_filling:  # Also run validation for gap-filling
            
            if is_gap_filling:
                # Use selective validation to preserve existing selected_candidates
                logger.info("🔧 Gap-filling detected: Using selective validation to preserve user candidate selections")
                if not self.validation_handler.execute_selective_validation(chunks_to_validate=missing_chunk_indices):
                    return False
            else:
                # Use full validation for initial runs or when no existing metrics
                if not self.validation_handler.execute_validation():
                    return False

        if task_state.completion_stage in [
            CompletionStage.NOT_STARTED,
            CompletionStage.PREPROCESSING,
            CompletionStage.GENERATION,
            CompletionStage.VALIDATION,
            CompletionStage.ASSEMBLY,
        ] or (task_state.completion_stage == CompletionStage.COMPLETE and self.task_config.add_final):
            # Execute assembly if needed or if forcing final audio regeneration
            if not self.assembly_handler.execute_assembly():
                return False

        return True

    def _delete_all_candidates_and_validation(self) -> None:
        """Delete all candidate audio files and validation data to start fresh."""
        import shutil
        
        try:
            # Delete candidates directory
            candidates_dir = self.file_manager.candidates_dir
            if candidates_dir.exists():
                logger.info(f"Deleting candidates directory: {candidates_dir}")
                shutil.rmtree(candidates_dir)
                candidates_dir.mkdir(parents=True, exist_ok=True)
            
            # Delete whisper directory (validation outputs)
            whisper_dir = self.file_manager.whisper_dir
            if whisper_dir.exists():
                logger.info(f"Deleting whisper directory: {whisper_dir}")
                shutil.rmtree(whisper_dir)
                whisper_dir.mkdir(parents=True, exist_ok=True)
            
            # Delete enhanced_metrics.json to start fresh with validation
            metrics_file = self.file_manager.task_directory / "enhanced_metrics.json"
            if metrics_file.exists():
                logger.info(f"Deleting metrics file: {metrics_file}")
                metrics_file.unlink()
            
            # Delete final audio to ensure it gets regenerated
            final_dir = self.file_manager.final_dir
            if final_dir.exists():
                final_files = list(final_dir.glob("*_final.wav"))
                for final_file in final_files:
                    logger.info(f"Deleting final audio: {final_file}")
                    final_file.unlink()
                
                metadata_files = list(final_dir.glob("*_final_metadata.json"))
                for metadata_file in metadata_files:
                    logger.info(f"Deleting final metadata: {metadata_file}")
                    metadata_file.unlink()
            
            logger.info("✅ All candidate and validation data deleted - ready for fresh re-rendering")
            
        except Exception as e:
            logger.error(f"Error deleting candidate and validation data: {e}")
            raise

    def _detect_device(self) -> str:
        """Detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
