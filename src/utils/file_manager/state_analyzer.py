#!/usr/bin/env python3
"""
StateAnalyzer for task state analysis.
Analyzes current task completion state.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


class CompletionStage(Enum):
    """Pipeline completion stages for state analysis."""

    NOT_STARTED = "not_started"
    PREPROCESSING = "preprocessing"
    GENERATION = "generation"
    VALIDATION = "validation"
    ASSEMBLY = "assembly"
    COMPLETE = "complete"


@dataclass
class TaskState:
    """Complete task state analysis."""

    task_path: Path
    has_input: bool
    has_chunks: bool
    chunk_count: int
    has_candidates: Dict[int, int]  # chunk_idx: candidate_count
    has_whisper: Dict[int, Set[int]]  # chunk_idx: candidate_indices
    has_metrics: bool
    has_final_audio: bool
    completion_stage: CompletionStage
    missing_components: List[str]
    # New fields for candidate editor functionality
    has_candidate_selection: bool = False
    candidate_editor_available: bool = False
    missing_candidates: List[int] = None
    task_status_message: str = ""
    
    def __post_init__(self):
        """Initialize mutable default values."""
        if self.missing_candidates is None:
            self.missing_candidates = []


class StateAnalyzer:
    """Analyzes task completion state."""

    def __init__(
        self,
        task_directory: Path,
        candidates_dir: Path,
        config: dict,
        chunk_handler,
        candidate_handler,
        whisper_handler,
        metrics_handler,
        final_audio_handler,
        get_input_text_func,
    ):
        """
        Initialize StateAnalyzer.

        Args:
            task_directory: Main task directory
            candidates_dir: Directory for candidate files
            config: Configuration dictionary
            chunk_handler: ChunkIOHandler instance
            candidate_handler: CandidateIOHandler instance
            whisper_handler: WhisperIOHandler instance
            metrics_handler: MetricsIOHandler instance
            final_audio_handler: FinalAudioIOHandler instance
            get_input_text_func: Function to get input text
        """
        self.task_directory = task_directory
        self.candidates_dir = candidates_dir
        self.config = config
        self.chunk_handler = chunk_handler
        self.candidate_handler = candidate_handler
        self.whisper_handler = whisper_handler
        self.metrics_handler = metrics_handler
        self.final_audio_handler = final_audio_handler
        self.get_input_text_func = get_input_text_func

    def analyze_task_state(self) -> TaskState:
        """
        Analyze current task completion state.

        Returns:
            TaskState object with complete analysis
        """
        missing_components = []

        # Check input text
        has_input = False
        try:
            self.get_input_text_func()
            has_input = True
        except FileNotFoundError:
            missing_components.append("input_text")

        # Check chunks
        chunks = self.chunk_handler.get_chunks()
        has_chunks = len(chunks) > 0
        if not has_chunks:
            missing_components.append("chunks")

        # Check candidates - improved to check file system completeness
        candidates = self.candidate_handler.get_candidates()
        has_candidates = {}
        expected_candidates_per_chunk = self.config.get("generation", {}).get(
            "num_candidates", 5
        )

        for chunk_idx in range(len(chunks)):
            chunk_candidates = candidates.get(chunk_idx, [])
            has_candidates[chunk_idx] = len(chunk_candidates)

            # Also check file system for expected candidates
            chunk_dir = self.candidates_dir / f"chunk_{chunk_idx+1:03d}"
            if chunk_dir.exists():
                # Count actual .wav files in chunk directory
                actual_wav_files = list(chunk_dir.glob("candidate_*.wav"))
                file_count = len(actual_wav_files)

                # Check if we have the expected number of candidates
                if file_count < expected_candidates_per_chunk:
                    missing_components.append(f"candidates_chunk_{chunk_idx}")
                    logger.debug(
                        f"Gap detected - Chunk {chunk_idx}: expected {expected_candidates_per_chunk}, found {file_count} files"
                    )
                else:
                    logger.debug(
                        f"Chunk {chunk_idx}: expected {expected_candidates_per_chunk}, found {file_count} files - OK"
                    )
            else:
                missing_components.append(f"candidates_chunk_{chunk_idx}")

        # Check whisper results - improved to check per candidate file
        has_whisper = {}
        for chunk_idx in range(len(chunks)):
            whisper_results = self.whisper_handler.get_whisper(chunk_idx)
            has_whisper[chunk_idx] = set(whisper_results.keys())

            # Check against actual candidate files, not just loaded candidates
            chunk_dir = self.candidates_dir / f"chunk_{chunk_idx+1:03d}"
            if chunk_dir.exists():
                actual_wav_files = list(chunk_dir.glob("candidate_*.wav"))
                expected_whisper_count = len(actual_wav_files)

                if len(whisper_results) < expected_whisper_count:
                    missing_components.append(f"whisper_chunk_{chunk_idx}")
                    logger.debug(
                        f"Chunk {chunk_idx}: expected {expected_whisper_count} whisper validations, found {len(whisper_results)}"
                    )

                    # Log which specific candidates are missing whisper validation
                    for wav_file in actual_wav_files:
                        # Extract candidate index from filename (candidate_01.wav -> 0)
                        try:
                            candidate_num = int(wav_file.stem.split("_")[1]) - 1
                            if candidate_num not in whisper_results:
                                logger.debug(
                                    f"Missing whisper validation for chunk {chunk_idx}, candidate {candidate_num}"
                                )
                        except (IndexError, ValueError):
                            logger.warning(
                                f"Could not parse candidate index from {wav_file}"
                            )
            else:
                # No chunk directory means no candidates at all
                if len(whisper_results) > 0:
                    logger.warning(
                        f"Found whisper results for chunk {chunk_idx} but no candidate directory"
                    )

        # Check metrics
        metrics = self.metrics_handler.get_metrics()
        has_metrics = len(metrics) > 0
        if not has_metrics:
            missing_components.append("metrics")

                # Check final audio
        final_audio = self.final_audio_handler.get_final_audio()
        has_final_audio = final_audio is not None
        if not has_final_audio:
            missing_components.append("final_audio")
            
        # Determine completion stage - moved before candidate selection logic
        if not has_input:
            completion_stage = CompletionStage.NOT_STARTED
        elif not has_chunks:
            completion_stage = CompletionStage.PREPROCESSING
        elif not all(count > 0 for count in has_candidates.values()):
            completion_stage = CompletionStage.GENERATION
        elif not has_metrics:
            completion_stage = CompletionStage.VALIDATION
        elif not has_final_audio:
            completion_stage = CompletionStage.ASSEMBLY
        else:
            completion_stage = CompletionStage.COMPLETE
            
        # Check candidate selection data (new)
        has_candidate_selection = False
        candidate_editor_available = False
        missing_candidates_list = []
        
        if has_metrics:
            metrics = self.metrics_handler.get_metrics()
            selected_candidates = metrics.get("selected_candidates", {})
            has_candidate_selection = len(selected_candidates) > 0
            
            # Check if all chunks have candidate selections
            for chunk_idx in range(len(chunks)):
                chunk_key = str(chunk_idx)
                if chunk_key not in selected_candidates:
                    missing_candidates_list.append(chunk_idx)
            
            # Editor is available if we have metrics and complete candidate data
            candidate_editor_available = (
                has_metrics and
                has_candidate_selection and
                len(missing_candidates_list) == 0 and
                completion_stage in [CompletionStage.ASSEMBLY, CompletionStage.COMPLETE]
            )
        
        # Generate task status message
        task_status_message = self._generate_task_status_message(
            completion_stage, has_final_audio, has_candidate_selection,
            len(missing_candidates_list) > 0
        )

        return TaskState(
            task_path=self.task_directory,
            has_input=has_input,
            has_chunks=has_chunks,
            chunk_count=len(chunks),
            has_candidates=has_candidates,
            has_whisper=has_whisper,
            has_metrics=has_metrics,
            has_final_audio=has_final_audio,
            completion_stage=completion_stage,
            missing_components=missing_components,
            has_candidate_selection=has_candidate_selection,
            candidate_editor_available=candidate_editor_available,
            missing_candidates=missing_candidates_list,
            task_status_message=task_status_message,
        )

    def _generate_task_status_message(
        self, 
        completion_stage: CompletionStage, 
        has_final_audio: bool, 
        has_candidate_selection: bool,
        has_missing_candidates: bool
    ) -> str:
        """Generate human-readable task status message."""
        if completion_stage == CompletionStage.COMPLETE and has_final_audio:
            if has_candidate_selection and not has_missing_candidates:
                return "Task is complete with final audio available."
            elif has_missing_candidates:
                return "Task is complete with final audio but is missing some candidates and should be re-assembled."
            else:
                return "Task is complete with final audio available."
        elif completion_stage == CompletionStage.ASSEMBLY:
            return "Task is ready for final assembly."
        elif completion_stage in [CompletionStage.VALIDATION, CompletionStage.GENERATION, CompletionStage.PREPROCESSING]:
            return "Task is incomplete and needs to finish."
        else:
            return "Task has not been started."
