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
from utils.file_manager.task_metrics_generator import TaskMetricsGenerator
from utils.progress_tracker import ProgressTracker
from utils.task_run_logger import TaskRunLogger
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
    total_execution_time: float = 0.0
    final_audio_path: Optional[Path] = None


class TaskExecutor:
    """Unified task executor with separated stage handlers."""

    def __init__(
        self,
        file_manager: FileManager,
        task_config: TaskConfig,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.file_manager = file_manager
        self.task_config = task_config

        # Use provided config or load from file
        if config is not None:
            self.config = config
        else:
            cm = ConfigManager(task_config.config_path.parent.parent.parent.parent)
            self.config = cm.load_cascading_config(task_config.config_path)

        # Ensure file_manager has the config
        if not hasattr(file_manager, "config") or file_manager.config is None:
            file_manager.config = self.config

        # Also ensure the state analyzer has the correct config
        if hasattr(file_manager, "_state_analyzer") and file_manager._state_analyzer:
            file_manager._state_analyzer.config = self.config

        # Initialize progress tracking
        self.progress_tracker = None
        # Persistent runtime logger
        try:
            self.run_logger = TaskRunLogger(self.file_manager.task_directory)
        except Exception:
            # Fail-safe: do not break pipeline
            self.run_logger = None
        # Expose to file_manager for downstream optional usage
        try:
            setattr(self.file_manager, "run_logger", self.run_logger)
        except Exception:
            pass

    # Safe helper to tick persistent runtime
    def _tick_runtime(self) -> None:
        try:
            if self.run_logger:
                self.run_logger.tick()
        except Exception:
            pass

        # Initialize components (lazy loading)
        self._chunker = None
        self._tts_generator = None
        self._whisper_validator = None
        self._quality_scorer = None
        self._candidate_manager = None

        # Initialize handlers (lazy loading)
        self._preprocessing_handler: Optional[PreprocessingHandler] = None
        self._generation_handler: Optional[GenerationHandler] = None
        self._validation_handler: Optional[ValidationHandler] = None
        self._assembly_handler: Optional[AssemblyHandler] = None

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
                force_paragraph_chunks=chunking_config.get("force_paragraph_chunks", False),
                refinement_enabled=chunking_config.get("refinement_enabled", True),
                micro_chunk_max_chars=chunking_config.get("micro_chunk_max_chars", None),
                short_par_chars=chunking_config.get("short_par_chars", None),
                respect_headings_in_speaker_mode=chunking_config.get(
                    "respect_headings_in_speaker_mode", True
                ),
            )
        return self._chunker

    @property
    def tts_generator(self) -> TTSGenerator:
        """Lazy-loaded TTS generator."""
        if self._tts_generator is None:
            device = self._detect_device()
            self._tts_generator = TTSGenerator(
                config=self.config, device=device, seed=None
            )
            logger.debug("TTSGenerator initialized with automatic model loading")
        return self._tts_generator

    @property
    def whisper_validator(self) -> WhisperValidator:
        """Lazy-loaded Whisper validator."""
        if self._whisper_validator is None:
            validation_config = self.config["validation"]
            validator = WhisperValidator(
                model_size=validation_config.get("whisper_model", "base"),
                device="auto",
                similarity_threshold=validation_config.get("similarity_threshold", 0.7),
                min_quality_score=validation_config.get("min_quality_score", 0.75),
            )
            # Inject config for optional prompt and normalization access
            try:
                setattr(validator, "_config", self.config)
            except Exception:
                pass
            self._whisper_validator = validator
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

            # Start persistent runtime session
            run_info = {
                "rerender_all": bool(self.task_config.rerender_all),
                "force_final_generation": bool(self.task_config.force_final_generation),
            }
            if self.run_logger:
                self.run_logger.start_session(run_info)

            # Check if we need to delete all candidates and start fresh
            if self.task_config.rerender_all:
                logger.info(
                    "🔄 Re-rendering all candidates from scratch - deleting existing candidates and validation data"
                )
                if self.run_logger:
                    self.run_logger.add_event("rerender_all", {"enabled": True})
                self._delete_all_candidates_and_validation()
                self._tick_runtime()

            # Analyze current state
            task_state = self.file_manager.analyze_task_state()
            logger.info(
                f"Current completion stage: {task_state.completion_stage.value}"
            )
            if self.run_logger:
                self.run_logger.add_event(
                    "analyzed_state",
                    {
                        "stage": task_state.completion_stage.value,
                        "missing_components": task_state.missing_components,
                    },
                )
            self._tick_runtime()

            # Migrate existing whisper files into consolidated whisper metrics
            self.file_manager.migrate_whisper_to_enhanced_metrics()
            if self.run_logger:
                self.run_logger.add_event("migrated_whisper_metrics", {})
            self._tick_runtime()

            # Migrate legacy selected_candidates from whisper metrics to task_metrics.json
            try:
                task_metrics_generator = TaskMetricsGenerator(self.file_manager.task_directory, self.config)
                task_metrics_generator.migrate_selected_candidates_from_whisper()
            except Exception as e:
                logger.warning(f"Failed legacy selection migration: {e}")

            if task_state.missing_components:
                logger.debug(
                    f"Missing components: {', '.join(task_state.missing_components)}"
                )

            # Check if we should force final audio regeneration
            force_final = self.task_config.force_final_generation
            if force_final and task_state.completion_stage == CompletionStage.COMPLETE:
                logger.info("🔄 Forcing final audio regeneration")
                if self.run_logger:
                    self.run_logger.add_event("force_final_regeneration", {})

                has_missing_candidates = any(
                    "candidates_chunk" in comp for comp in task_state.missing_components
                )
                has_missing_whisper = any(
                    "whisper_chunk" in comp for comp in task_state.missing_components
                )

                # Detect incomplete selected_candidates even when candidates/whisper look complete
                try:
                    # Ensure task_metrics.json exists before checking selections
                    tmg = TaskMetricsGenerator(self.file_manager.task_directory, self.config)
                    tmg.generate_task_metrics()
                except Exception:
                    pass

                try:
                    chunks_for_selection = self.file_manager.get_chunks()
                    selected_for_selection = self.file_manager.get_selected_candidates()
                    missing_selection_indices_force = [
                        idx for idx in range(len(chunks_for_selection)) if idx not in selected_for_selection
                    ]
                except Exception:
                    missing_selection_indices_force = []

                if has_missing_candidates or has_missing_whisper:
                    logger.info(
                        "Missing data detected, will regenerate before final assembly"
                    )
                    # Continue with normal pipeline execution to fill gaps
                    # Do NOT return here - let it flow through to _execute_stages_from_state
                elif missing_selection_indices_force:
                    # We have all files/whispers, but selections are incomplete.
                    # Fill selections via selective validation without overwriting existing ones.
                    logger.info(
                        f"🧩 Incomplete selected_candidates detected for chunks: {[i+1 for i in missing_selection_indices_force]} - performing selective validation to complete selections"
                    )
                    if self.run_logger:
                        self.run_logger.add_event("stage_start", {"stage": "validation", "selective": True, "reason": "complete_selections"})
                    self._tick_runtime()
                    if not self.validation_handler.execute_selective_validation(
                        chunks_to_validate=missing_selection_indices_force
                    ):
                        if self.run_logger:
                            self.run_logger.add_event("stage_end", {"stage": "validation", "success": False})
                        return TaskResult(
                            task_config=self.task_config,
                            success=False,
                            completion_stage=task_state.completion_stage,
                            error_message="Selective validation to complete selections failed",
                            execution_time=time.time() - start_time,
                            total_execution_time=(self.run_logger.total_execution_seconds if self.run_logger else time.time() - start_time),
                        )
                    if self.run_logger:
                        self.run_logger.add_event("stage_end", {"stage": "validation", "success": True})
                    self._tick_runtime()

                    # Proceed directly to assembly in forced flow to honor overwrite intent
                    if self.run_logger:
                        self.run_logger.add_event("stage_start", {"stage": "assembly", "forced": True, "reason": "complete_selections"})
                    self._tick_runtime()
                    if not self.assembly_handler.execute_assembly():
                        if self.run_logger:
                            self.run_logger.add_event("stage_end", {"stage": "assembly", "success": False})
                            try:
                                self.run_logger.end_session(False, error_message="Final assembly failed")
                            except Exception:
                                pass
                        return TaskResult(
                            task_config=self.task_config,
                            success=False,
                            completion_stage=task_state.completion_stage,
                            error_message="Final assembly failed",
                            execution_time=time.time() - start_time,
                            total_execution_time=(self.run_logger.total_execution_seconds if self.run_logger else time.time() - start_time),
                        )
                    if self.run_logger:
                        self.run_logger.add_event("stage_end", {"stage": "assembly", "success": True})

                    # Find the actual final audio file path
                    final_audio_path = None
                    final_files = list(self.file_manager.final_dir.glob("*_final.wav"))
                    if final_files:
                        final_audio_path = max(final_files, key=lambda f: f.stat().st_mtime)

                    # Finalize session BEFORE creating result to include final tick in totals
                    if self.run_logger:
                        self.run_logger.end_session(True, final_audio_path=final_audio_path)

                    # Generate task metrics overview after final audio creation
                    try:
                        task_metrics_generator = TaskMetricsGenerator(self.file_manager.task_directory, self.config)
                        task_metrics_generator.generate_task_metrics()
                    except Exception as e:
                        logger.warning(f"Failed to generate task metrics: {e}")
                    result = TaskResult(
                        task_config=self.task_config,
                        success=True,
                        completion_stage=CompletionStage.COMPLETE,
                        execution_time=time.time() - start_time,
                        total_execution_time=(self.run_logger.total_execution_seconds if self.run_logger else time.time() - start_time),
                        final_audio_path=final_audio_path,
                    )
                    return result
                else:
                    # Jump directly to assembly if no missing data
                    if self.run_logger:
                        self.run_logger.add_event("stage_start", {"stage": "assembly", "forced": True})
                    self._tick_runtime()
                    if not self.assembly_handler.execute_assembly():
                        if self.run_logger:
                            self.run_logger.add_event("stage_end", {"stage": "assembly", "success": False})
                            try:
                                self.run_logger.end_session(False, error_message="Final assembly failed")
                            except Exception:
                                pass
                        return TaskResult(
                            task_config=self.task_config,
                            success=False,
                            completion_stage=task_state.completion_stage,
                            error_message="Final assembly failed",
                            execution_time=time.time() - start_time,
                            total_execution_time=(self.run_logger.total_execution_seconds if self.run_logger else time.time() - start_time),
                        )
                    if self.run_logger:
                        self.run_logger.add_event("stage_end", {"stage": "assembly", "success": True})

                    # Find the actual final audio file path
                    final_audio_path = None
                    final_files = list(self.file_manager.final_dir.glob("*_final.wav"))
                    if final_files:
                        final_audio_path = max(
                            final_files, key=lambda f: f.stat().st_mtime
                        )

                    # Finalize session BEFORE creating result to include final tick in totals
                    if self.run_logger:
                        self.run_logger.end_session(True, final_audio_path=final_audio_path)

                    # Generate task metrics overview after final audio creation
                    try:
                        task_metrics_generator = TaskMetricsGenerator(self.file_manager.task_directory, self.config)
                        task_metrics_generator.generate_task_metrics()
                    except Exception as e:
                        logger.warning(f"Failed to generate task metrics: {e}")
                    result = TaskResult(
                        task_config=self.task_config,
                        success=True,
                        completion_stage=CompletionStage.COMPLETE,
                        execution_time=time.time() - start_time,
                        total_execution_time=(self.run_logger.total_execution_seconds if self.run_logger else time.time() - start_time),
                        final_audio_path=final_audio_path,
                    )
                    return result

            # Execute stages based on current state
            if not self._execute_stages_from_state(task_state):
                if self.run_logger:
                    try:
                        self.run_logger.end_session(False, error_message="Pipeline execution failed")
                    except Exception:
                        pass
                return TaskResult(
                    task_config=self.task_config,
                    success=False,
                    completion_stage=task_state.completion_stage,
                    error_message="Pipeline execution failed",
                    execution_time=time.time() - start_time,
                    total_execution_time=(self.run_logger.total_execution_seconds if self.run_logger else time.time() - start_time),
                )

            # Success - get final audio file path
            final_audio_path = None
            final_files = list(self.file_manager.final_dir.glob("*_final.wav"))
            if final_files:
                final_audio_path = max(final_files, key=lambda f: f.stat().st_mtime)

            # Finalize session BEFORE creating result to include final tick in totals
            if self.run_logger:
                self.run_logger.end_session(True, final_audio_path=final_audio_path)
            
            # Generate task metrics overview
            try:
                task_metrics_generator = TaskMetricsGenerator(self.file_manager.task_directory, self.config)
                task_metrics_generator.generate_task_metrics()
            except Exception as e:
                logger.warning(f"Failed to generate task metrics: {e}")
            
            result = TaskResult(
                task_config=self.task_config,
                success=True,
                completion_stage=CompletionStage.COMPLETE,
                execution_time=time.time() - start_time,
                total_execution_time=(self.run_logger.total_execution_seconds if self.run_logger else time.time() - start_time),
                final_audio_path=final_audio_path,
            )
            return result

        except Exception as e:
            logger.error(f"Task execution failed: {e}", exc_info=True)
            if self.run_logger:
                try:
                    self.run_logger.end_session(False, error_message=str(e))
                except Exception:
                    pass
            return TaskResult(
                task_config=self.task_config,
                success=False,
                completion_stage=CompletionStage.NOT_STARTED,
                error_message=str(e),
                execution_time=time.time() - start_time,
                total_execution_time=(self.run_logger.total_execution_seconds if self.run_logger else time.time() - start_time),
            )

    def _execute_stages_from_state(self, task_state: TaskState) -> bool:
        """Execute the pipeline stages based on the current task state."""
        # Check if this is a gap-filling scenario (unified detection)
        has_missing_candidates = any(
            "candidates_chunk" in comp for comp in task_state.missing_components
        )
        has_missing_whisper = any(
            "whisper_chunk" in comp for comp in task_state.missing_components
        )
        # Ensure whisper metrics are available for gap-fill analysis
        try:
            _ = self.file_manager.get_metrics()
            has_existing_metrics = True
        except Exception:
            has_existing_metrics = False
        # Detect incomplete selected_candidates to trigger gap-filling path as well
        missing_selection_indices = []
        try:
            chunks_sel = self.file_manager.get_chunks()
            selected_sel = self.file_manager.get_selected_candidates()
            missing_selection_indices = [
                idx for idx in range(len(chunks_sel)) if idx not in selected_sel
            ]
        except Exception:
            missing_selection_indices = []
        has_missing_selections = len(missing_selection_indices) > 0

        is_gap_filling = (
            has_existing_metrics
            and (has_missing_candidates or has_missing_whisper or has_missing_selections)
            and self.task_config.force_final_generation
        )

        if (
            task_state.completion_stage == CompletionStage.COMPLETE
            and not is_gap_filling
            and not self.task_config.force_final_generation
        ):
            logger.info("Task already complete")
            return True

        if is_gap_filling:
            logger.info(
                "🔄 Gap-filling mode detected - regenerating missing components"
            )
            # Determine which chunks need validation in gap-filling mode
            missing_chunk_indices = []
            for comp in task_state.missing_components:
                if "candidates_chunk" in comp:
                    # Extract chunk index from component name like "candidates_chunk_33"
                    try:
                        chunk_idx = int(comp.split("_")[-1])
                        missing_chunk_indices.append(chunk_idx)
                    except (ValueError, IndexError):
                        logger.warning(
                            f"Could not extract chunk index from component: {comp}"
                        )
                elif "whisper_chunk" in comp:
                    # Also validate chunks with missing whisper results
                    try:
                        chunk_idx = int(comp.split("_")[-1])
                        missing_chunk_indices.append(chunk_idx)
                    except (ValueError, IndexError):
                        logger.warning(
                            f"Could not extract chunk index from component: {comp}"
                        )

            # Also include chunks that have no selected_candidates yet
            if missing_selection_indices:
                for idx in missing_selection_indices:
                    if idx not in missing_chunk_indices:
                        missing_chunk_indices.append(idx)

        # Execute stages in order based on what's missing
        if task_state.completion_stage in [
            CompletionStage.NOT_STARTED,
            CompletionStage.PREPROCESSING,
        ]:
            if not self.preprocessing_handler.execute_preprocessing():
                return False

        if (
            task_state.completion_stage
            in [
                CompletionStage.NOT_STARTED,
                CompletionStage.PREPROCESSING,
                CompletionStage.GENERATION,
            ]
        ) or is_gap_filling:  # Also run generation for gap-filling
            if not self.generation_handler.execute_generation():
                return False

        if (
            task_state.completion_stage
            in [
                CompletionStage.NOT_STARTED,
                CompletionStage.PREPROCESSING,
                CompletionStage.GENERATION,
                CompletionStage.VALIDATION,
            ]
        ) or is_gap_filling:  # Also run validation for gap-filling

            if is_gap_filling:
                # Use selective validation to preserve existing selected_candidates
                logger.info(
                    "🔧 Gap-filling detected: Using selective validation to preserve user candidate selections"
                )
                if not self.validation_handler.execute_selective_validation(
                    chunks_to_validate=missing_chunk_indices
                ):
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
        ] or (
            task_state.completion_stage == CompletionStage.COMPLETE
            and self.task_config.force_final_generation
        ):
            # Ensure task_metrics.json exists before assembly in gap-fill/forced flows
            try:
                tmg = TaskMetricsGenerator(self.file_manager.task_directory, self.config)
                tmg.generate_task_metrics()
            except Exception:
                pass

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

            # Delete whisper_metrics.json to start fresh with validation
            metrics_file = self.file_manager.task_directory / "whisper" / "whisper_metrics.json"
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

            # Delete task metrics to ensure it gets regenerated
            task_metrics_file = self.file_manager.task_directory / "task_metrics.json"
            if task_metrics_file.exists():
                logger.info(f"Deleting task metrics: {task_metrics_file}")
                task_metrics_file.unlink()

            logger.info(
                "✅ All candidate and validation data deleted - ready for fresh re-rendering"
            )

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
