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
                        f"Chunk {chunk_idx}: expected {expected_candidates_per_chunk}, found {file_count} files"
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

        # Determine completion stage
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
        )
