#!/usr/bin/env python3
"""
TaskExecutor for unified task execution.
Combines the old main pipeline and recovery logic into a single execution path.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from chunking.chunk_validator import ChunkValidator

# Import pipeline components
from chunking.spacy_chunker import SpaCyChunker
from generation.audio_processor import AudioProcessor
from generation.candidate_manager import CandidateManager
from generation.tts_generator import TTSGenerator
from utils.config_manager import ConfigManager, TaskConfig
from utils.file_manager import (
    AudioCandidate,
    CompletionStage,
    FileManager,
    TaskState,
    TextChunk,
)
import logging
from utils.progress_tracker import ProgressTracker
from validation.fuzzy_matcher import FuzzyMatcher
from validation.quality_scorer import QualityScorer
from validation.whisper_validator import WhisperValidator

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
    """
    Unified task executor that combines main pipeline and recovery logic.

    Features:
    - Automatic state detection and gap analysis
    - Smart resumption from any stage
    - Unified execution path for all scenarios
    """

    def __init__(self, file_manager: FileManager, task_config: TaskConfig):
        """
        Initializes the TaskExecutor.
        """
        self.file_manager = file_manager
        self.task_config = task_config

        # Load config data only if not already set (avoid duplicate loading)
        if not hasattr(self, "config") or self.config is None:
            cm = ConfigManager(
                task_config.config_path.parent.parent.parent.parent
            )  # Go up to project root
            self.config = cm.load_cascading_config(task_config.config_path)

            # Set the loaded config in FileManager to avoid duplicate loading
            file_manager.config = self.config

        # Initialize progress tracking (will be created when needed)
        self.progress_tracker = None

        # Initialize components (lazy loading)
        self._chunker = None
        self._tts_generator = None
        self._whisper_validator = None
        self._quality_scorer = None
        self._candidate_manager = None

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
            # Detect device for TTS generation
            device = self._detect_device()

            # Create TTSGenerator which will load the ChatterboxTTS model automatically
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
            from validation.quality_scorer import ScoringStrategy

            validation_config = self.config["validation"]
            self._quality_scorer = QualityScorer(
                sample_rate=24000,
            )
        return self._quality_scorer

    @property
    def candidate_manager(self):
        """Lazy-loaded candidate manager."""
        if self._candidate_manager is None:
            from generation.candidate_manager import CandidateManager

            self._candidate_manager = CandidateManager(
                tts_generator=self.tts_generator,
                config=self.config,
                output_dir=self.file_manager.task_directory,
            )
            # Pass FileManager reference for correct saving structure
            self._candidate_manager.file_manager = self.file_manager
        return self._candidate_manager

    def create_progress_tracker(
        self, total_items: int, description: str
    ) -> ProgressTracker:
        """Create a progress tracker for a specific stage."""
        return ProgressTracker(total_items, description)

    def execute_task(self) -> TaskResult:
        """
        Execute a complete task with automatic state detection and resumption.

        Returns:
            TaskResult object with execution details
        """
        start_time = time.time()

        try:
            logger.debug(f"Task directory: {self.task_config.base_output_dir}")

            # Analyze current state
            task_state = self.file_manager.analyze_task_state()
            logger.info(
                f"Current completion stage: {task_state.completion_stage.value}"
            )

            # Migrate existing whisper files to enhanced metrics format (hybrid system)
            self.file_manager.migrate_whisper_to_enhanced_metrics()

            if task_state.missing_components:
                logger.debug(
                    f"Missing components: {', '.join(task_state.missing_components)}"
                )

            # Check if we should force final audio regeneration
            force_final = self.task_config.add_final
            if force_final and task_state.completion_stage == CompletionStage.COMPLETE:
                logger.info("🔄 Forcing final audio regeneration")

                # Check if we have missing candidates or whisper validations
                has_missing_candidates = any(
                    "candidates_chunk" in comp for comp in task_state.missing_components
                )
                has_missing_whisper = any(
                    "whisper_chunk" in comp for comp in task_state.missing_components
                )

                if has_missing_candidates:
                    logger.info(
                        "⚠️ Missing candidates detected - must generate first"
                    )
                    task_state.completion_stage = CompletionStage.GENERATION
                elif has_missing_whisper:
                    logger.info(
                        "⚠️ Missing whisper validations detected - must validate first"
                    )
                    task_state.completion_stage = CompletionStage.VALIDATION
                else:
                    logger.info(
                        "✅ All candidates available - proceeding to assembly only"
                    )
                    task_state.completion_stage = CompletionStage.ASSEMBLY

                # Remove existing final audio files
                final_files = list(self.file_manager.final_dir.glob("*_final.wav"))
                for final_file in final_files:
                    final_file.unlink()
                    logger.debug(f"Removed existing final audio file: {final_file}")
            elif task_state.missing_components and any(
                "whisper_chunk" in comp for comp in task_state.missing_components
            ):
                # If only whisper validations are missing, go directly to validation
                logger.info(
                    "🔍 Missing whisper validations detected - running validation phase"
                )
                task_state.completion_stage = CompletionStage.VALIDATION

            # Execute stages based on gap analysis
            success = self._execute_stages_from_state(task_state)

            if success:
                # Final state check
                final_state = self.file_manager.analyze_task_state()
                final_audio_path = None

                # Find the actual file path
                final_files = list(self.file_manager.final_dir.glob("*_final.wav"))
                if final_files:
                    final_audio_path = max(final_files, key=lambda f: f.stat().st_mtime)

                execution_time = time.time() - start_time

                logger.info(
                    f"⏳ Task completed successfully in {execution_time:.2f} seconds"
                )

                return TaskResult(
                    task_config=self.task_config,
                    success=True,
                    completion_stage=final_state.completion_stage,
                    execution_time=execution_time,
                    final_audio_path=final_audio_path,
                )
            else:
                execution_time = time.time() - start_time
                return TaskResult(
                    task_config=self.task_config,
                    success=False,
                    completion_stage=task_state.completion_stage,
                    error_message="Execution failed",
                    execution_time=execution_time,
                )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task execution failed: {e}", exc_info=True)

            return TaskResult(
                task_config=self.task_config,
                success=False,
                completion_stage=CompletionStage.NOT_STARTED,
                error_message=str(e),
                execution_time=execution_time,
            )

    def _execute_stages_from_state(self, task_state: TaskState) -> bool:
        """
        Executes the pipeline stages based on the current task state.

        Returns:
            True if all required stages completed successfully, False otherwise.
        """
        success = True
        if task_state.completion_stage == CompletionStage.COMPLETE:
            logger.info("Task already complete")
            return True

        # Execute stages in order based on what's missing
        if task_state.completion_stage in [
            CompletionStage.NOT_STARTED,
            CompletionStage.PREPROCESSING,
        ]:
            if not self.execute_preprocessing():
                return False

        if task_state.completion_stage in [
            CompletionStage.NOT_STARTED,
            CompletionStage.PREPROCESSING,
            CompletionStage.GENERATION,
        ]:
            if not self.execute_generation():
                return False

        if task_state.completion_stage in [
            CompletionStage.NOT_STARTED,
            CompletionStage.PREPROCESSING,
            CompletionStage.GENERATION,
            CompletionStage.VALIDATION,
        ]:
            if not self.execute_validation():
                return False

        if task_state.completion_stage in [
            CompletionStage.NOT_STARTED,
            CompletionStage.PREPROCESSING,
            CompletionStage.GENERATION,
            CompletionStage.VALIDATION,
            CompletionStage.ASSEMBLY,
        ]:
            if not self.execute_assembly():
                return False

        return True

    def execute_preprocessing(self) -> bool:
        """
        Execute the text preprocessing stage.

        Returns:
            True if preprocessing is successful, False otherwise.
        """
        logger.info("🚀 Starting Preprocessing Stage")
        try:
            logger.info("Starting preprocessing stage")

            # Load input text
            input_text = self.file_manager.get_input_text()
            logger.debug(f"Loaded input text: {len(input_text)} characters")

            # Chunk text
            text_chunks = self.chunker.chunk_text(input_text)
            logger.info(f"Generated {len(text_chunks)} text chunks")

            # Update indices for TextChunk objects
            for i, chunk in enumerate(text_chunks):
                chunk.idx = i

            # Validate chunks
            chunking_config = self.config["chunking"]
            chunk_validator = ChunkValidator(
                max_limit=chunking_config.get("max_chunk_limit", 600),
                min_length=chunking_config.get("min_chunk_length", 50),
            )
            if not chunk_validator.run_all_validations(text_chunks):
                # Sentence-boundary Warnungen oder geringfügige Längen-Abweichungen
                # sollen den weiteren Pipeline-Ablauf nicht mehr blockieren.
                logger.warning("Chunk validation reported issues – proceeding anyway")

            # Save chunks
            if not self.file_manager.save_chunks(text_chunks):
                logger.error("Failed to save chunks")
                return False

            logger.info("Preprocessing stage completed successfully")
            return True

        except Exception as e:
            logger.error(f"Preprocessing stage failed: {e}", exc_info=True)
            return False

    def execute_generation(self) -> bool:
        """
        Execute the candidate generation stage.

        Returns:
            True if generation is successful, False otherwise.
        """
        logger.info("🎙️ Starting Generation Stage")
        try:
            logger.info("")
            logger.info("Starting generation stage")

            # Load chunks
            chunks = self.file_manager.get_chunks()
            if not chunks:
                logger.error("No chunks found for generation")
                return False

            # Load reference audio
            reference_audio_path = self.file_manager.get_reference_audio()

            # Initialize TTS generator with reference audio
            self.tts_generator.load_reference_audio(str(reference_audio_path))

            # Generate candidates for each chunk
            total_chunks = len(chunks)
            logger.info(f"⚡ GENERATION PHASE: Processing {total_chunks} chunks")
            logger.info("=" * 50)

            for chunk in chunks:
                # Enhanced chunk header with clear visual separation
                chunk_num = chunk.idx + 1
                logger.info("")
                logger.info(f"🎯 CHUNK {chunk_num}/{total_chunks}")
                logger.debug(f"Text length: {len(chunk.text)} characters")
                if len(chunk.text) > 80:
                    preview = chunk.text[:80] + "..."
                else:
                    preview = chunk.text
                logger.debug(f'Preview: "{preview}"')
                logger.info("-" * 50)

                # Check if we need to generate missing candidates
                existing_candidates = self.file_manager.get_candidates(chunk.idx)
                chunk_candidates = existing_candidates.get(chunk.idx, [])

                generation_config = self.config["generation"]
                num_candidates = generation_config["num_candidates"]

                # Check actual files in chunk directory for accurate count
                chunk_dir = (
                    self.file_manager.candidates_dir / f"chunk_{chunk.idx+1:03d}"
                )
                existing_file_count = 0
                if chunk_dir.exists():
                    candidate_files = list(chunk_dir.glob("candidate_*.wav"))
                    existing_file_count = len(candidate_files)

                if existing_file_count >= num_candidates:
                    logger.debug(
                        f"✓ Candidates already exist for chunk {chunk_num} ({existing_file_count}/{num_candidates}), skipping"
                    )
                    continue
                elif existing_file_count > 0:
                    logger.info(
                        f"⚡ Found {existing_file_count}/{num_candidates} candidates - generating {num_candidates - existing_file_count} missing candidates"
                    )

                # Generate missing candidates
                missing_count = num_candidates - existing_file_count

                if missing_count > 0:

                    # Find which specific candidate indices are missing
                    existing_indices = set()
                    if chunk_dir.exists():
                        candidate_files = list(chunk_dir.glob("candidate_*.wav"))
                        for candidate_file in candidate_files:
                            try:
                                candidate_num = int(candidate_file.stem.split("_")[1])
                                candidate_idx = candidate_num - 1  # Convert to 0-based
                                existing_indices.add(candidate_idx)
                            except (IndexError, ValueError):
                                continue

                    # Find missing indices in the range [0, num_candidates)
                    missing_indices = []
                    for i in range(num_candidates):
                        if i not in existing_indices:
                            missing_indices.append(i)

                    # Generate missing candidates for specific indices
                    new_candidates = self._generate_missing_candidates(
                        chunk, missing_indices
                    )

                    if not new_candidates:
                        logger.error(
                            f"❌ Failed to generate missing candidates for chunk {chunk_num+1}"
                        )
                        return False

                    # CandidateManager already saved the new candidates, no need to save again
                    logger.debug(
                        f"✅ Successfully generated {len(new_candidates)} missing candidates"
                    )
                else:
                    # Generate all candidates (original logic for empty chunks)
                    logger.info(f"⚡ Generating candidates...")
                    candidates = self._generate_candidates_for_chunk(chunk)

                    if not candidates:
                        logger.error(
                            f"❌ Failed to generate candidates for chunk {chunk_num+1}"
                        )
                        return False

                    # Save candidates (new chunk, safe to overwrite)
                    if not self.file_manager.save_candidates(
                        chunk.idx, candidates, overwrite_existing=True
                    ):
                        logger.error(
                            f"❌ Failed to save candidates for chunk {chunk_num+1}"
                        )
                        return False

                    logger.info(
                        f"✅ Successfully generated {len(candidates)} candidates"
                    )
            logger.info("✅ Generation stage completed successfully")
            return True

        except Exception as e:
            logger.error(f"Generation stage failed: {e}", exc_info=True)
            return False

    def _generate_candidates_for_chunk(self, chunk: TextChunk) -> List[AudioCandidate]:
        """
        Generates candidates for a single text chunk.

        Returns:
            List of generated AudioCandidate objects.
        """
        logger.debug(f"Generating {self.candidate_manager.max_candidates} candidates for chunk '{chunk.text[:50]}...'")
        try:
            # Use TTSGenerator's built-in candidate generation with parameter variation
            generation_config = self.config["generation"]
            num_candidates = generation_config["num_candidates"]

            # Generate candidates with parameter variation
            # Get TTS parameters from the tts_params section
            tts_params = generation_config.get("tts_params", {})
            candidates = self.tts_generator.generate_candidates(
                text=chunk.text,
                num_candidates=num_candidates,
                exaggeration=tts_params.get("exaggeration"),
                cfg_weight=tts_params.get("cfg_weight"),
                temperature=tts_params.get("temperature"),
                conservative_config=generation_config.get(
                    "conservative_candidate", None
                ),
                tts_params=tts_params,
            )

            # Update chunk_idx and chunk_text for all candidates
            for candidate in candidates:
                candidate.chunk_idx = chunk.idx
                candidate.chunk_text = chunk.text

            return candidates

        except Exception as e:
            logger.error(f"Error generating candidates for chunk {chunk.idx+1}: {e}")
            return []

    def _generate_missing_candidates(
        self, chunk: TextChunk, missing_indices: List[int]
    ) -> List[AudioCandidate]:
        """
        Generates specific missing candidates for a chunk.

        Returns:
            List of newly generated AudioCandidate objects.
        """
        logger.info(f"Generating missing candidates {missing_indices} for chunk {chunk.idx+1}")
        try:
            logger.debug(
                f"starting _generate_missing_candidates(): Generating {len(missing_indices)} candidates for indices: {missing_indices}"
            )

            # Convert 0-based indices to 1-based for CandidateManager
            one_based_indices = [idx + 1 for idx in missing_indices]

            # Use CandidateManager for consistent candidate generation and whisper file deletion
            missing_candidates = self.candidate_manager.generate_specific_candidates(
                text_chunk=chunk,
                chunk_index=chunk.idx,
                candidate_indices=one_based_indices,
                output_dir=self.file_manager.task_directory,
            )

            logger.debug(
                f"Returning from candidate manager: generated {len(missing_candidates)}/{len(missing_indices)} missing candidates"
            )
            return missing_candidates

        except Exception as e:
            logger.error(f"Error in missing candidate generation: {e}")
            return []

    def _delete_whisper_file(self, chunk_index: int, candidate_idx: int):
        """
        Deletes a specific whisper validation file (e.g., when regenerating a candidate).
        """
        self.candidate_manager._delete_whisper_file(self.file_manager.task_directory, chunk_index, candidate_idx)

    def _generate_retry_candidates(
        self, chunk: TextChunk, max_retries: int, start_candidate_idx: int
    ) -> List[AudioCandidate]:
        """
        Generates additional conservative candidates if initial generation fails quality.

        Returns:
            List of additional AudioCandidate objects.
        """
        retry_candidates = []
        try:
            generation_config = self.config["generation"]
            conservative_config = generation_config.get("conservative_candidate", {})

            if not conservative_config.get("enabled", False):
                logger.warning(
                    "Conservative candidate not enabled, using default values for retries"
                )
                # Use default conservative values
                base_exaggeration = 0.45
                base_cfg_weight = 0.4
                base_temperature = 0.8
            else:
                # Use configured conservative values
                base_exaggeration = conservative_config.get("exaggeration", 0.45)
                base_cfg_weight = conservative_config.get("cfg_weight", 0.4)
                base_temperature = conservative_config.get("temperature", 0.8)

            logger.debug(
                f"Generating {max_retries} retry candidates with conservative base values: "
                f"exag={base_exaggeration:.2f}, cfg={base_cfg_weight:.2f}, temp={base_temperature:.2f}"
            )

            for i in range(max_retries):
                try:
                    # Calculate variation offset (-0.05 to +0.05)
                    # First retry uses exact conservative values, subsequent ones add variations
                    if i == 0:
                        variation_factor = 0.0  # First retry: exact conservative values
                    else:
                        # Spread variations evenly across ±0.05 range
                        variation_factor = (
                            (i - 1) / max(1, max_retries - 2)
                        ) * 2.0 - 1.0  # -1.0 to +1.0
                        variation_factor *= 0.05  # Scale to ±0.05

                    # Apply variations to conservative parameters
                    retry_exaggeration = max(
                        0.1, min(1.0, base_exaggeration + variation_factor)
                    )
                    retry_cfg_weight = max(
                        0.1, min(1.0, base_cfg_weight + variation_factor)
                    )
                    retry_temperature = max(
                        0.1, min(2.0, base_temperature + variation_factor)
                    )

                    # Set unique seed for this retry
                    retry_seed = (
                        self.tts_generator.seed
                        + (chunk.idx * 1000)
                        + (start_candidate_idx + i) * 100
                    )

                    logger.debug(
                        f"Retry {i+1}/{max_retries}: exag={retry_exaggeration:.3f}, "
                        f"cfg={retry_cfg_weight:.3f}, temp={retry_temperature:.3f}, seed={retry_seed}"
                    )

                    # Generate single candidate with these parameters
                    import torch

                    torch.manual_seed(retry_seed)

                    retry_audio = self.tts_generator.generate_single(
                        text=chunk.text,
                        exaggeration=retry_exaggeration,
                        cfg_weight=retry_cfg_weight,
                        temperature=retry_temperature,
                    )

                    # Create AudioCandidate with correct index
                    candidate_idx = start_candidate_idx + i
                    generation_params = {
                        "exaggeration": retry_exaggeration,
                        "cfg_weight": retry_cfg_weight,
                        "temperature": retry_temperature,
                        "seed": retry_seed,
                        "type": "RETRY_CONSERVATIVE",
                        "variation_factor": variation_factor,
                        "retry_attempt": i + 1,
                    }

                    from pathlib import Path

                    from utils.file_manager import AudioCandidate

                    retry_candidate = AudioCandidate(
                        chunk_idx=chunk.idx,
                        candidate_idx=candidate_idx,
                        audio_path=Path(),  # Will be set when saving
                        audio_tensor=retry_audio,
                        generation_params=generation_params,
                        chunk_text=chunk.text,
                    )

                    retry_candidates.append(retry_candidate)

                    logger.debug(
                        f"✅ Generated retry candidate {i+1} (idx={candidate_idx}) with duration={retry_audio.shape[-1]/24000:.2f}s\n"
                    )

                except Exception as e:
                    logger.error(f"Failed to generate retry candidate {i+1}: {e}")
                    continue

            logger.debug(
                f"Successfully generated {len(retry_candidates)}/{max_retries} retry candidates"
            )
            return retry_candidates

        except Exception as e:
            logger.error(f"Error in retry candidate generation: {e}")
            return []

    def execute_validation(self) -> bool:
        """
        Execute the validation stage.

        Returns:
            True if validation is successful, False otherwise.
        """
        logger.info("🧐 Starting Validation Stage")
        try:
            logger.info("=" * 50)
            logger.info("Starting validation stage")

            # Load chunks and candidates
            chunks = self.file_manager.get_chunks()
            all_candidates = self.file_manager.get_candidates()

            if not chunks or not all_candidates:
                logger.error("No chunks or candidates found for validation")
                return False

            # Validate each candidate
            validation_results = {}

            logger.info(f"🚦 VALIDATION PHASE: Processing {len(chunks)} chunks")
            logger.info("=" * 50)

            for chunk in chunks:
                chunk_candidates = all_candidates.get(chunk.idx, [])
                if not chunk_candidates:
                    logger.warning(f"No candidates found for chunk {chunk.idx}")
                    continue

                chunk_num = chunk.idx + 1
                logger.info("")  # Empty line for spacing
                logger.info(f"🎯 CHUNK {chunk_num}/{len(chunks)}")
                logger.debug(f"Candidates to validate: {len(chunk_candidates)}")
                logger.info("-" * 40)

                chunk_results = {}

                for candidate in chunk_candidates:
                    candidate_num = (
                        candidate.candidate_idx + 1
                    )  # Start numbering from 1 for user display
                    logger.debug(f"🔍 Validating candidate {candidate_num}...")

                    # Check if whisper result already exists
                    existing_whisper = self.file_manager.get_whisper(
                        chunk.idx, candidate.candidate_idx
                    )
                    if candidate.candidate_idx in existing_whisper:
                        logger.debug(
                            f"✓ Whisper result already exists for candidate {candidate_num}"
                        )
                        chunk_results[candidate.candidate_idx] = existing_whisper[
                            candidate.candidate_idx
                        ]
                        continue

                    # Set chunk text for validation compatibility
                    candidate.chunk_text = chunk.text

                    # Perform Whisper validation
                    whisper_result = self.whisper_validator.validate_candidate(
                        candidate, chunk.text
                    )

                    if whisper_result:
                        # Perform quality scoring
                        quality_result = self.quality_scorer.score_candidate(
                            candidate, whisper_result
                        )

                        # Combine results - convert ValidationResult to dict
                        combined_result = {
                            "is_valid": whisper_result.is_valid,
                            "transcription": whisper_result.transcription,
                            "similarity_score": whisper_result.similarity_score,
                            "quality_score": whisper_result.quality_score,
                            "validation_time": whisper_result.validation_time,
                            "error_message": whisper_result.error_message,
                            "overall_quality_score": quality_result.overall_score,
                            "quality_details": quality_result.details,
                        }

                        # Save whisper result
                        self.file_manager.save_whisper(
                            chunk.idx, candidate.candidate_idx, combined_result
                        )
                        chunk_results[candidate.candidate_idx] = combined_result

                        # Log validation result with overall quality score for consistency
                        status = "✅ Valid" if whisper_result.is_valid else "❌ Invalid"
                        logger.debug(
                            f"{status} - candidate {candidate_num} (similarity: {whisper_result.similarity_score:.3f}, quality: {whisper_result.quality_score:.3f}, overall: {quality_result.overall_score:.3f})"
                        )
                    else:
                        logger.warning(
                            f"❌ Whisper validation failed for candidate {candidate_num}"
                        )

                validation_results[chunk.idx] = chunk_results
                valid_count = sum(
                    1
                    for result in chunk_results.values()
                    if result.get("is_valid", False)
                )

                # Log summary with overall quality scores for consistency with final metrics
                if chunk_results:
                    overall_scores = [
                        result.get("overall_quality_score", 0.0)
                        for result in chunk_results.values()
                    ]
                    min_score = min(overall_scores)
                    max_score = max(overall_scores)
                    logger.info(
                        f"✅ Validation complete: {valid_count}/{len(chunk_candidates)} candidates valid (overall scores: {min_score:.3f} to {max_score:.3f})"
                    )
                else:
                    logger.info(
                        f"✅ Validation complete: {valid_count}/{len(chunk_candidates)} candidates valid"
                    )

                # Only generate retry candidates during initial validation, not when re-running validation
                generation_config = self.config.get("generation", {})
                max_retries = generation_config.get("max_retries", 0)
                num_candidates = generation_config.get("num_candidates", 5)
                max_total_candidates = num_candidates + max_retries

                # Check if we should retry based on actual file count, not loaded candidates
                chunk_dir = (
                    self.file_manager.candidates_dir / f"chunk_{chunk.idx+1:03d}"
                )
                highest_candidate_idx = -1
                if chunk_dir.exists():
                    # Find all candidate files and get the highest index
                    candidate_files = list(chunk_dir.glob("candidate_*.wav"))
                    for candidate_file in candidate_files:
                        try:
                            # Extract candidate number from filename (candidate_05.wav -> 4)
                            candidate_num = int(candidate_file.stem.split("_")[1])
                            candidate_idx = candidate_num - 1  # Convert to 0-based
                            highest_candidate_idx = max(
                                highest_candidate_idx, candidate_idx
                            )
                        except (IndexError, ValueError):
                            continue

                # Check if we've already reached the maximum number of candidates
                max_candidate_idx = max_total_candidates - 1  # Convert to 0-based
                already_at_max = highest_candidate_idx >= max_candidate_idx

                should_retry = (
                    valid_count == 0 and max_retries > 0 and not already_at_max
                )

                if already_at_max and valid_count == 0:
                    logger.warning(
                        f"⚠️ All candidates invalid but maximum retry limit reached (max: {max_total_candidates} candidates)"
                    )
                elif should_retry:
                    # Calculate how many retries we can still generate
                    next_candidate_idx = highest_candidate_idx + 1
                    remaining_slots = max_total_candidates - (highest_candidate_idx + 1)
                    actual_retries = min(max_retries, remaining_slots)

                    logger.info(
                        f"⚠️ All candidates invalid - generating {actual_retries} retry candidates"
                    )
                    logger.debug(
                        f"Highest existing candidate: {highest_candidate_idx}, next: {next_candidate_idx}, max allowed: {max_candidate_idx}"
                    )

                    # Generate retry candidates with conservative parameters + variations
                    retry_candidates = self._generate_retry_candidates(
                        chunk, actual_retries, next_candidate_idx
                    )

                    # Delete whisper files for retry candidates to ensure re-validation
                    for retry_candidate in retry_candidates:
                        self._delete_whisper_file(
                            chunk.idx, retry_candidate.candidate_idx + 1
                        )

                    if retry_candidates:
                        logger.info(
                            f"🔁 Validating {len(retry_candidates)} retry candidates..."
                        )

                        # Validate retry candidates immediately
                        for retry_candidate in retry_candidates:
                            candidate_num = retry_candidate.candidate_idx + 1
                            logger.debug(
                                f"🔍 Validating retry candidate {candidate_num}..."
                            )

                            # Set chunk text for validation compatibility
                            retry_candidate.chunk_text = chunk.text

                            # Perform Whisper validation
                            whisper_result = self.whisper_validator.validate_candidate(
                                retry_candidate, chunk.text
                            )

                            if whisper_result:
                                # Perform quality scoring
                                quality_result = self.quality_scorer.score_candidate(
                                    retry_candidate, whisper_result
                                )

                                # Combine results
                                combined_result = {
                                    "is_valid": whisper_result.is_valid,
                                    "transcription": whisper_result.transcription,
                                    "similarity_score": whisper_result.similarity_score,
                                    "quality_score": whisper_result.quality_score,
                                    "validation_time": whisper_result.validation_time,
                                    "error_message": whisper_result.error_message,
                                    "overall_quality_score": quality_result.overall_score,
                                    "quality_details": quality_result.details,
                                }

                                # Save whisper result
                                self.file_manager.save_whisper(
                                    chunk.idx,
                                    retry_candidate.candidate_idx,
                                    combined_result,
                                )
                                chunk_results[retry_candidate.candidate_idx] = (
                                    combined_result
                                )

                                # Log validation result with overall quality score for consistency
                                status = (
                                    "✅ Valid"
                                    if whisper_result.is_valid
                                    else "❌ Invalid"
                                )
                                logger.debug(
                                    f"{status} - retry candidate {candidate_num} (similarity: {whisper_result.similarity_score:.3f}, quality: {whisper_result.quality_score:.3f}, overall: {quality_result.overall_score:.3f})"
                                )

                        # Add retry candidates to the chunk candidates list and update all_candidates
                        all_candidates[chunk.idx].extend(retry_candidates)

                        # Save the updated candidates list (original + retry) to disk
                        if not self.file_manager.save_candidates(
                            chunk.idx,
                            all_candidates[chunk.idx],
                            overwrite_existing=False,
                        ):
                            logger.warning(
                                f"Failed to save retry candidates for chunk {chunk_num+1}"
                            )
                        else:
                            logger.debug(
                                f"✓ Saved {len(retry_candidates)} retry candidates to disk"
                            )

                        # Update validation results
                        validation_results[chunk.idx] = chunk_results

                        # Recalculate valid count and show updated overall scores
                        new_valid_count = sum(
                            1
                            for result in chunk_results.values()
                            if result.get("is_valid", False)
                        )
                        if chunk_results:
                            overall_scores = [
                                result.get("overall_quality_score", 0.0)
                                for result in chunk_results.values()
                            ]
                            min_score = min(overall_scores)
                            max_score = max(overall_scores)
                            score_summary = (
                                f" (overall scores: {min_score:.3f} to {max_score:.3f})"
                            )
                        else:
                            score_summary = ""

                        if new_valid_count > valid_count:
                            logger.info(
                                f"🎉 Retry success: {new_valid_count-valid_count} additional valid candidates found!{score_summary}"
                            )
                        else:
                            logger.info(
                                f"😞 Retry complete: Still no valid candidates{score_summary}"
                            )
                    else:
                        logger.warning(
                            f"Failed to generate retry candidates for chunk {chunk_num+1}"
                        )

            # Create and save enhanced metrics
            metrics = self._create_enhanced_metrics(
                chunks, all_candidates, validation_results
            )

            if not self.file_manager.save_metrics(metrics):
                logger.error("Failed to save validation metrics")
                return False

            logger.info("")  # Empty line for spacing
            logger.info("✅ Validation stage completed successfully")
            return True

        except Exception as e:
            logger.error(f"Validation stage failed: {e}", exc_info=True)
            return False

    def _create_enhanced_metrics(
        self,
        chunks: List[TextChunk],
        candidates: Dict[int, List[AudioCandidate]],
        validation_results: Dict[int, Dict[int, dict]],
    ) -> Dict[str, Any]:
        """
        Creates a dictionary of enhanced metrics for all chunks and candidates.

        Returns:
            A dictionary where keys are chunk indices and values are lists of enhanced metrics for candidates.
        """
        metrics = {
            "timestamp": time.time(),
            "total_chunks": len(chunks),
            "chunks": {},
            "selected_candidates": {},
        }

        # Process each chunk
        for chunk in chunks:
            chunk_candidates = candidates.get(chunk.idx, [])
            chunk_validation = validation_results.get(chunk.idx, {})

            if not chunk_validation or not chunk_candidates:
                continue

            # Convert cached JSON results back to ValidationResult objects
            validation_results_list = []
            candidates_list = []

            for candidate in chunk_candidates:
                if candidate.candidate_idx in chunk_validation:
                    result_dict = chunk_validation[candidate.candidate_idx]

                    # Reconstruct ValidationResult from cached JSON
                    from validation.whisper_validator import ValidationResult

                    validation_result = ValidationResult(
                        is_valid=result_dict.get("is_valid", False),
                        transcription=result_dict.get("transcription", ""),
                        similarity_score=result_dict.get("similarity_score", 0.0),
                        quality_score=result_dict.get("quality_score", 0.0),
                        validation_time=result_dict.get("validation_time", 0.0),
                        error_message=result_dict.get("error_message"),
                    )

                    validation_results_list.append(validation_result)
                    candidates_list.append(candidate)

            if not candidates_list:
                continue

            # Find best candidate using already calculated scores from validation
            try:
                # Use already calculated overall_quality_score from validation results
                best_candidate_idx = None
                best_score_value = -1.0

                # Find candidate with highest overall_quality_score
                for candidate in candidates_list:
                    result_dict = chunk_validation[candidate.candidate_idx]
                    candidate_score = result_dict.get("overall_quality_score", 0.0)

                    if candidate_score > best_score_value:
                        best_score_value = candidate_score
                        best_candidate_idx = candidate.candidate_idx

                # Create chunk metrics for reporting
                chunk_metrics = {
                    "chunk_text": (
                        chunk.text[:100] + "..."
                        if len(chunk.text) > 100
                        else chunk.text
                    ),
                    "candidates": {},
                    "best_candidate": best_candidate_idx,
                    "best_score": best_score_value,
                }

                # Add candidate details for reporting and log final results
                candidate_scores = []
                for candidate in candidates_list:
                    result_dict = chunk_validation[candidate.candidate_idx]
                    candidate_score = result_dict.get("overall_quality_score", 0.0)
                    candidate_scores.append(candidate_score)

                    chunk_metrics["candidates"][candidate.candidate_idx] = {
                        "transcription": result_dict.get("transcription", ""),
                        "similarity_score": result_dict.get("similarity_score", 0.0),
                        "validation_score": result_dict.get("quality_score", 0.0),
                        "overall_quality_score": candidate_score,
                        "quality_details": result_dict.get("quality_details", {}),
                        "final_score": candidate_score,
                    }

                # Log corrected chunk results with actual quality scores
                if candidate_scores:
                    min_score = min(candidate_scores)
                    max_score = max(candidate_scores)
                    best_candidate_display = (
                        best_candidate_idx + 1 if best_candidate_idx is not None else 0
                    )

                    # Get TTS parameters from best candidate for logging
                    if best_candidate_idx is not None:
                        best_candidate_obj = next(
                            (
                                c
                                for c in candidates_list
                                if c.candidate_idx == best_candidate_idx
                            ),
                            None,
                        )
                        if best_candidate_obj and best_candidate_obj.generation_params:
                            best_params = best_candidate_obj.generation_params
                            exaggeration = best_params.get("exaggeration", 0.0)
                            cfg_weight = best_params.get("cfg_weight", 0.0)
                            temperature = best_params.get("temperature", 0.0)
                        else:
                            exaggeration = cfg_weight = temperature = 0.0
                    else:
                        exaggeration = cfg_weight = temperature = 0.0

                    logger.info(
                        f"Chunk_{chunk.idx:02d}: score {min_score:.3f} to {max_score:.3f}. "
                        f"Best candidate: {best_candidate_display} of {len(candidates_list)} (score: {best_score_value:.3f}) "
                        f"– exaggeration: {exaggeration:.2f}, cfg_weight: {cfg_weight:.2f}, temperature: {temperature:.2f}"
                    )

                metrics["chunks"][chunk.idx] = chunk_metrics
                metrics["selected_candidates"][chunk.idx] = best_candidate_idx

            except Exception as e:
                logger.warning(
                    f"Failed to select best candidate for chunk {chunk.idx}: {e}"
                )
                # Fallback to first valid candidate
                if candidates_list:
                    metrics["selected_candidates"][chunk.idx] = candidates_list[
                        0
                    ].candidate_idx

        return metrics

    def execute_assembly(self) -> bool:
        """
        Execute assembly stage (audio concatenation and post-processing).

        Returns:
            True if successful
        """
        logger.info("🎵 Starting Assembly Stage")
        try:
            logger.info("Starting assembly stage")

            # Load metrics to get selected candidates
            metrics = self.file_manager.get_metrics()
            if not metrics or "selected_candidates" not in metrics:
                logger.error("No metrics or selected candidates found for assembly")
                return False

            selected_candidates = metrics["selected_candidates"]
            logger.info(
                f"Assembling audio from {len(selected_candidates)} selected candidates"
            )

            # Load audio segments
            audio_segments = self.file_manager.get_audio_segments(selected_candidates)

            if not audio_segments:
                logger.error("No audio segments loaded for assembly")
                return False

            # Load chunks for paragraph break information
            chunks = self.file_manager.get_chunks()
            has_paragraph_breaks = [chunk.has_paragraph_break for chunk in chunks]

            # Assemble audio with appropriate silences
            final_audio = self._assemble_audio_with_silences(
                audio_segments, has_paragraph_breaks
            )

            # Apply post-processing if any component is enabled
            postprocessing_config = self.config.get("postprocessing", {})
            audio_cleaning_enabled = postprocessing_config.get(
                "audio_cleaning", {}
            ).get("enabled", False)
            auto_editor_enabled = postprocessing_config.get("auto_editor", {}).get(
                "enabled", False
            )

            if audio_cleaning_enabled or auto_editor_enabled:
                final_audio = self._apply_post_processing(final_audio)

            # Create metadata
            sample_rate = self.config.get("audio", {}).get("sample_rate", 24000)
            audio_duration_seconds = len(final_audio) / sample_rate

            metadata = {
                "job_name": self.task_config.job_name,
                "task_name": self.task_config.task_name,
                "run_label": self.task_config.run_label,
                "timestamp": self.task_config.timestamp,
                "total_chunks": len(chunks),
                "selected_candidates": selected_candidates,
                "audio_duration_seconds": audio_duration_seconds,
                "sample_rate": sample_rate,
            }

            # Save final audio
            if not self.file_manager.save_final_audio(final_audio, metadata):
                logger.error("Failed to save final audio")
                return False

            logger.info("✅ Assembly stage completed successfully")
            return True

        except Exception as e:
            logger.error(f"Assembly stage failed: {e}", exc_info=True)
            return False

    def _assemble_audio_with_silences(
        self, audio_segments: List[torch.Tensor], has_paragraph_breaks: List[bool]
    ) -> torch.Tensor:
        """
        Assemble audio segments with appropriate silences.

        Args:
            audio_segments: List of audio tensors
            has_paragraph_breaks: List indicating paragraph breaks

        Returns:
            The concatenated final audio tensor.
        """
        if not audio_segments:
            return torch.tensor([])

        sample_rate = self.config.get("audio", {}).get("sample_rate", 24000)
        silence_config = self.config.get("audio", {}).get("silence_duration", {})
        normal_silence = int(sample_rate * silence_config.get("normal", 0.2))
        paragraph_silence = int(sample_rate * silence_config.get("paragraph", 0.8))

        assembled_segments = []

        for i, segment in enumerate(audio_segments):
            assembled_segments.append(segment)

            # Add silence between segments (except after the last one)
            if i < len(audio_segments) - 1:
                if i < len(has_paragraph_breaks) and has_paragraph_breaks[i]:
                    silence = torch.zeros(paragraph_silence)
                else:
                    silence = torch.zeros(normal_silence)
                assembled_segments.append(silence)

        return torch.cat(assembled_segments)

    def _apply_post_processing(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            Post-processed audio tensor
        """
        try:
            postprocessing_config = self.config.get("postprocessing", {})

            # Initialize post-processing components based on config
            processed_audio = audio.clone()

            # Apply audio cleaning if enabled
            audio_cleaning_enabled = postprocessing_config.get(
                "audio_cleaning", {}
            ).get("enabled", False)
            if audio_cleaning_enabled:
                from postprocessing.audio_cleaner import AudioCleaner, CleaningSettings

                # Create cleaning settings from config
                cleaning_settings = CleaningSettings(
                    spectral_gating=True,
                    normalize_audio=True,
                    target_rms=0.2,
                    remove_dc_offset=True,
                )

                audio_cleaner = AudioCleaner(
                    sample_rate=self.config.get("audio", {}).get("sample_rate", 24000),
                    settings=cleaning_settings,
                )

                processed_audio = audio_cleaner.clean_audio(processed_audio)
                logger.debug("Audio cleaning applied")

            # Apply Auto-Editor if enabled and available
            auto_editor_config = postprocessing_config.get("auto_editor", {})
            auto_editor_enabled = auto_editor_config.get("enabled", False)

            if auto_editor_enabled:
                try:
                    from postprocessing.auto_editor_wrapper import AutoEditorWrapper

                    auto_editor = AutoEditorWrapper(
                        margin_before=auto_editor_config.get("margin_before", 0.1),
                        margin_after=auto_editor_config.get("margin_after", 0.1),
                        preserve_natural_sounds=auto_editor_config.get(
                            "preserve_natural_sounds", True
                        ),
                    )

                    # Get reference audio path for threshold calculation
                    reference_audio_path = str(self.file_manager.get_reference_audio())

                    # Calculate custom threshold using noise_threshold_factor
                    custom_threshold = None
                    noise_threshold_factor = postprocessing_config.get(
                        "noise_threshold_factor"
                    )
                    if noise_threshold_factor is not None:
                        # Get the recommended threshold from noise analysis
                        from postprocessing.noise_analyzer import NoiseAnalyzer

                        noise_analyzer = NoiseAnalyzer(
                            sample_rate=self.config.get("audio", {}).get(
                                "sample_rate", 24000
                            )
                        )

                        if reference_audio_path and Path(reference_audio_path).exists():
                            profile = noise_analyzer.analyze_reference_audio(
                                reference_audio_path
                            )
                        else:
                            profile = noise_analyzer.analyze_noise_floor(
                                processed_audio
                            )

                        # Apply noise_threshold_factor as multiplier
                        custom_threshold = (
                            profile.recommended_threshold * noise_threshold_factor
                        )
                        logger.debug(
                            f"Custom threshold: {profile.recommended_threshold:.6f} * {noise_threshold_factor} = {custom_threshold:.6f}"
                        )

                    processed_audio = auto_editor.clean_audio(
                        processed_audio,
                        sample_rate=self.config.get("audio", {}).get(
                            "sample_rate", 24000
                        ),
                        reference_audio_path=reference_audio_path,
                        custom_threshold=custom_threshold,
                    )
                    logger.debug("Auto-Editor processing applied")

                except ImportError:
                    logger.warning("Auto-Editor not available, skipping")
                except Exception as e:
                    logger.warning(f"Auto-Editor processing failed: {e}")

            logger.info("Post-processing applied successfully")
            return processed_audio

        except Exception as e:
            logger.warning(f"Post-processing failed, using original audio: {e}")
            return audio

    def _detect_device(self) -> str:
        """Detect the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
