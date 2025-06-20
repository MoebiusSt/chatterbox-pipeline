#!/usr/bin/env python3
"""
FileManager for centralized file operations.
Handles all file I/O operations for the TTS pipeline with consistent schemas.
Refactored version with delegation to specialized handlers.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from utils.config_manager import ConfigManager, TaskConfig
from .io_handlers import (
    ChunkIOHandler,
    CandidateIOHandler, 
    WhisperIOHandler,
    MetricsIOHandler,
    FinalAudioIOHandler
)
from .state_analyzer import StateAnalyzer, TaskState, CompletionStage
from .validation_helpers import ValidationHelpers

# Re-export classes for backward compatibility
from .io_handlers.candidate_io import AudioCandidate

logger = logging.getLogger(__name__)


class FileManager:
    """
    Central file manager for all pipeline I/O operations.
    Maintains consistent file schemas and directory structures.
    Delegates operations to specialized handlers.
    """

    def __init__(self, task_config: Union[TaskConfig, Dict[str, Any]], preloaded_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize FileManager with task configuration.

        Args:
            task_config: TaskConfig object or config dictionary
            preloaded_config: Optional pre-loaded config to avoid redundant loading
        """
        # Use duck typing instead of strict isinstance checks
        if hasattr(task_config, 'base_output_dir') and hasattr(task_config, 'job_name') and hasattr(task_config, 'config_path'):
            # TaskConfig-like object (duck typing)
            self.task_config = task_config
            self.task_directory = task_config.base_output_dir
            self.job_name = task_config.job_name

            # Use preloaded config if provided, otherwise load from file
            if preloaded_config is not None:
                self.config = preloaded_config
            else:
                # Load the config data from file
                cm = ConfigManager(Path.cwd())
                self.config = cm.load_cascading_config(task_config.config_path)

        elif isinstance(task_config, dict):
            # Config dictionary (fallback for backward compatibility)
            self.config = task_config

            # Create task config from dictionary
            cm = ConfigManager(Path.cwd())
            tc = cm.create_task_config(task_config)
            self.task_config = tc
            self.task_directory = tc.base_output_dir
            self.job_name = tc.job_name

        else:
            raise TypeError(f"Expected TaskConfig-like object or dict, got {type(task_config)}")

        # Set up directory structure
        self.task_directory.mkdir(parents=True, exist_ok=True)
        self.candidates_dir = self.task_directory / "candidates"
        self.texts_dir = self.task_directory / "texts"
        self.final_dir = self.task_directory / "final"
        self.whisper_dir = self.task_directory / "whisper"

        # Ensure directories exist
        for dir_path in [
            self.candidates_dir,
            self.texts_dir,
            self.final_dir,
            self.whisper_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Project paths
        self.project_root = self._find_project_root()
        self.input_texts_dir = self.project_root / "data" / "input" / "texts"
        self.reference_audio_dir = (
            self.project_root / "data" / "input" / "reference_audio"
        )

        # Initialize specialized handlers
        self._chunk_handler = ChunkIOHandler(self.texts_dir)
        self._candidate_handler = CandidateIOHandler(self.candidates_dir, self.config)
        self._whisper_handler = WhisperIOHandler(self.whisper_dir, self.task_directory, self.candidates_dir)
        self._metrics_handler = MetricsIOHandler(self.task_directory)
        self._final_audio_handler = FinalAudioIOHandler(self.final_dir, self.candidates_dir, self.config)
        
        # Initialize helpers
        self._validation_helpers = ValidationHelpers(self.candidates_dir, self.whisper_dir, self._whisper_handler)
        self._state_analyzer = StateAnalyzer(
            self.task_directory, 
            self.candidates_dir, 
            self.config,
            self._chunk_handler,
            self._candidate_handler,
            self._whisper_handler,
            self._metrics_handler,
            self._final_audio_handler,
            self.get_input_text
        )

    def _find_project_root(self) -> Path:
        """Find project root by looking for config directory."""
        current = Path.cwd()
        while current.parent != current:
            if (current / "config").exists():
                return current
            current = current.parent
        return Path.cwd()

    # Input Operations
    def get_input_text(self) -> str:
        """Load input text file."""
        text_file = self.config["input"]["text_file"]
        text_path = self.input_texts_dir / text_file

        if not text_path.exists():
            raise FileNotFoundError(f"Input text file not found: {text_path}")

        with open(text_path, "r", encoding="utf-8") as f:
            content = f.read()

        logger.debug(f"Loaded input text: {text_path} ({len(content)} characters)")
        return content

    def get_reference_audio(self) -> Path:
        """Get reference audio file path."""
        reference_audio = self.config["input"]["reference_audio"]
        audio_path = self.reference_audio_dir / reference_audio

        if not audio_path.exists():
            raise FileNotFoundError(f"Reference audio file not found: {audio_path}")

        return audio_path

    # Delegated Operations - Chunk Handler
    def save_chunks(self, chunks: List) -> bool:
        """Save text chunks to files."""
        return self._chunk_handler.save_chunks(chunks)

    def get_chunks(self) -> List:
        """Load text chunks from files."""
        return self._chunk_handler.get_chunks()

    # Delegated Operations - Candidate Handler
    def save_candidates(self, chunk_idx: int, candidates: List[AudioCandidate], overwrite_existing: bool = False) -> bool:
        """Save audio candidates for a chunk."""
        return self._candidate_handler.save_candidates(chunk_idx, candidates, overwrite_existing)

    def get_candidates(self, chunk_idx: Optional[int] = None) -> Dict[int, List[AudioCandidate]]:
        """Load audio candidates."""
        return self._candidate_handler.get_candidates(chunk_idx)

    # Delegated Operations - Whisper Handler
    def save_whisper(self, chunk_idx: int, candidate_idx: int, result: dict) -> bool:
        """Save Whisper validation result."""
        return self._whisper_handler.save_whisper(chunk_idx, candidate_idx, result)

    def get_whisper(self, chunk_idx: int, candidate_idx: Optional[int] = None) -> Dict[int, dict]:
        """Load Whisper validation results."""
        return self._whisper_handler.get_whisper(chunk_idx, candidate_idx)

    def migrate_whisper_to_enhanced_metrics(self) -> bool:
        """Migrate existing individual whisper files to enhanced_metrics.json format."""
        return self._whisper_handler.migrate_whisper_to_enhanced_metrics()

    def cleanup_duplicate_whisper_files(self, keep_individual_files: bool = True) -> bool:
        """Clean up duplicate validation data after successful migration."""
        return self._whisper_handler.cleanup_duplicate_whisper_files(keep_individual_files)

    # Delegated Operations - Metrics Handler
    def save_metrics(self, metrics: dict) -> bool:
        """Save quality metrics and validation results."""
        return self._metrics_handler.save_metrics(metrics)

    def get_metrics(self) -> dict:
        """Load quality metrics and validation results."""
        return self._metrics_handler.get_metrics()

    # Delegated Operations - Final Audio Handler
    def save_final_audio(self, audio, metadata: dict) -> bool:
        """Save final assembled audio with metadata."""
        return self._final_audio_handler.save_final_audio(audio, metadata)

    def get_final_audio(self):
        """Load final assembled audio."""
        return self._final_audio_handler.get_final_audio()

    def get_audio_segments(self, selected_candidates: Dict[int, int]) -> List:
        """Load selected audio segments for assembly."""
        return self._final_audio_handler.get_audio_segments(selected_candidates)

    # Delegated Operations - State Analyzer
    def analyze_task_state(self) -> TaskState:
        """Analyze current task completion state."""
        return self._state_analyzer.analyze_task_state()

    # Delegated Operations - Validation Helpers
    def _remove_corrupt_candidate(self, chunk_idx: int, candidate_idx: int) -> bool:
        """Remove corrupt candidate file and its validation data."""
        return self._validation_helpers.remove_corrupt_candidate(chunk_idx, candidate_idx) 