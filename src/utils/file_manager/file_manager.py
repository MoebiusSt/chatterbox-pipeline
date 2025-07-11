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
    CandidateIOHandler,
    ChunkIOHandler,
    FinalAudioIOHandler,
    MetricsIOHandler,
    WhisperIOHandler,
)

# Re-export classes for backward compatibility
from .io_handlers.candidate_io import AudioCandidate
from .state_analyzer import StateAnalyzer, TaskState
from .validation_helpers import ValidationHelpers

logger = logging.getLogger(__name__)

class FileManager:
    """
    Central file manager for all pipeline I/O operations.
    Maintains consistent file schemas and directory structures.
    Delegates operations to specialized handlers.
    """

    def __init__(
        self,
        task_config: Union[TaskConfig, Dict[str, Any]],
        preloaded_config: Optional[Dict[str, Any]] = None,
        config_manager: Optional[ConfigManager] = None,
    ) -> None:
        """
        Initialize FileManager with task configuration.

        Args:
            task_config: TaskConfig object or config dictionary
            preloaded_config: Optional pre-loaded config to avoid redundant loading
            config_manager: Optional ConfigManager instance to reuse existing cache
        """
        # Use duck typing instead of strict isinstance checks
        if (
            hasattr(task_config, "base_output_dir")
            and hasattr(task_config, "job_name")
            and hasattr(task_config, "config_path")
        ):
            # TaskConfig-like object (duck typing)
            self.task_config = task_config
            self.task_directory = task_config.base_output_dir
            self.job_name = task_config.job_name

            # Use preloaded config if provided, otherwise load from file
            if preloaded_config is not None:
                self.config = preloaded_config
            else:
                # Use provided ConfigManager or create new one (with shared cache)
                if config_manager is not None:
                    cm = config_manager
                else:
                    cm = ConfigManager(Path.cwd())
                self.config = cm.load_cascading_config(task_config.config_path)

        elif isinstance(task_config, dict):
            # Config dictionary (fallback for backward compatibility)
            self.config = task_config

            # Create task config from dictionary
            if config_manager is not None:
                cm = config_manager
            else:
                cm = ConfigManager(Path.cwd())
            tc = cm.create_task_config(task_config)
            self.task_config = tc
            self.task_directory = tc.base_output_dir
            self.job_name = tc.job_name

        else:
            raise TypeError(
                f"Expected TaskConfig-like object or dict, got {type(task_config)}"
            )

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
        self._whisper_handler = WhisperIOHandler(
            self.whisper_dir, self.task_directory, self.candidates_dir
        )
        self._metrics_handler = MetricsIOHandler(self.task_directory)
        self._final_audio_handler = FinalAudioIOHandler(
            self.final_dir, self.candidates_dir, self.config
        )

        # Initialize helpers
        self._validation_helpers = ValidationHelpers(
            self.candidates_dir, self.whisper_dir, self._whisper_handler
        )
        
        # Now inject ValidationHelpers into IO handlers that need it
        self._candidate_handler.validation_helpers = self._validation_helpers
        self._final_audio_handler.validation_helpers = self._validation_helpers
        self._state_analyzer = StateAnalyzer(
            self.task_directory,
            self.candidates_dir,
            self.config,
            self._chunk_handler,
            self._candidate_handler,
            self._whisper_handler,
            self._metrics_handler,
            self._final_audio_handler,
            self.get_input_text,
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
            # Try to provide helpful information about available files
            available_files = []
            if self.input_texts_dir.exists():
                available_files = [f.name for f in self.input_texts_dir.glob("*.txt")]
                # Also include other common text file extensions
                for ext in ["*.md", "*.rtf"]:
                    available_files.extend(
                        [f.name for f in self.input_texts_dir.glob(ext)]
                    )

            error_msg = f"Input text file not found: {text_path}"
            if available_files:
                error_msg += f"\n📂 Available text files: {', '.join(available_files)}"
            else:
                error_msg += f"\n📂 Input texts dir: {self.input_texts_dir} (empty or doesn't exist)"

            raise FileNotFoundError(error_msg)

        with open(text_path, "r", encoding="utf-8") as f:
            content = f.read()

        logger.debug(f"Loaded input text: {text_path} ({len(content)} characters)")
        return content

    def check_input_text_exists(self) -> bool:
        """Check if input text file exists without raising an exception."""
        try:
            text_file = self.config["input"]["text_file"]
            text_path = self.input_texts_dir / text_file
            return text_path.exists()
        except Exception:
            return False
    
    # Delegated Operations - Chunk Handler
    def save_chunks(self, chunks: List) -> bool:
        """Save text chunks to files."""
        return self._chunk_handler.save_chunks(chunks)

    def get_chunks(self) -> List:
        """Load text chunks from files."""
        return self._chunk_handler.get_chunks()

    # Delegated Operations - Candidate Handler
    def save_candidates(
        self,
        chunk_idx: int,
        candidates: List[AudioCandidate],
        overwrite_existing: bool = False,
    ) -> bool:
        """Save audio candidates for a chunk."""
        return self._candidate_handler.save_candidates(
            chunk_idx, candidates, overwrite_existing
        )

    def get_candidates(
        self, chunk_idx: Optional[int] = None
    ) -> Dict[int, List[AudioCandidate]]:
        """Load audio candidates."""
        return self._candidate_handler.get_candidates(chunk_idx)

    # Delegated Operations - Whisper Handler
    def save_whisper(self, chunk_idx: int, candidate_idx: int, result: dict) -> bool:
        """Save Whisper validation result."""
        return self._whisper_handler.save_whisper(chunk_idx, candidate_idx, result)

    def get_whisper(
        self, chunk_idx: int, candidate_idx: Optional[int] = None
    ) -> Dict[int, dict]:
        """Load Whisper validation results."""
        return self._whisper_handler.get_whisper(chunk_idx, candidate_idx)

    def delete_whisper(self, chunk_idx: int, candidate_idx: int) -> bool:
        """Delete Whisper validation result."""
        return self._whisper_handler.delete_whisper(chunk_idx, candidate_idx)

    def migrate_whisper_to_enhanced_metrics(self) -> bool:
        """Migrate existing individual whisper files to enhanced_metrics.json format."""
        return self._whisper_handler.migrate_whisper_to_enhanced_metrics()

    def cleanup_duplicate_whisper_files(
        self, keep_individual_files: bool = True
    ) -> bool:
        """Clean up duplicate validation data after successful migration."""
        return self._whisper_handler.cleanup_duplicate_whisper_files(
            keep_individual_files
        )

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
        return self._validation_helpers.remove_corrupt_candidate(
            chunk_idx, candidate_idx
        )

    # Speaker system methods
    def get_reference_audio_for_speaker(self, speaker_id: str) -> Path:
        """
        Get reference_audio for specific speaker.

        Args:
            speaker_id: Speaker ID

        Returns:
            Path to reference_audio file

        Raises:
            ValueError: When no reference_audio is defined for speaker
            FileNotFoundError: When the audio file does not exist
        """
        # Get speaker configuration via ConfigManager if available
        # Fallback: Search directly in config
        speakers = self.config.get("generation", {}).get("speakers", [])

        if not speakers:
            raise RuntimeError("No speakers configured")

        # Normalize speaker_id (default speaker aliases)
        if speaker_id in ["0", "default", "reset"]:
            # Use explicit default_speaker from config
            default_speaker = self.config.get("generation", {}).get("default_speaker")
            if default_speaker:
                speaker_id = default_speaker
            else:
                # Fallback to first speaker if default_speaker not configured
                speaker_id = speakers[0].get("id", "default")

        # Search for speaker by ID
        speaker_config = None
        for speaker in speakers:
            if speaker.get("id") == speaker_id:
                speaker_config = speaker
                break

        if not speaker_config:
            # Use explicit default_speaker from config
            default_speaker = self.config.get("generation", {}).get("default_speaker")
            if default_speaker:
                logger.warning(
                    f"Speaker '{speaker_id}' not found, using default speaker '{default_speaker}'"
                )
                # Find default speaker config
                for speaker in speakers:
                    if speaker.get("id") == default_speaker:
                        speaker_config = speaker
                        speaker_id = default_speaker
                        break

            # Final fallback to first speaker
            if not speaker_config:
                logger.warning("Default speaker not found, using first speaker")
                speaker_config = speakers[0]
                speaker_id = speaker_config.get("id", "default")

        reference_audio = speaker_config.get("reference_audio")
        if not reference_audio:
            raise ValueError(f"No reference_audio defined for speaker '{speaker_id}'")

        audio_path = self.reference_audio_dir / reference_audio

        if not audio_path.exists():
            available_files = [f.name for f in self.reference_audio_dir.glob("*.wav")]
            raise FileNotFoundError(
                f"Reference audio not found: {audio_path}\n"
                f"Available files: {available_files}"
            )

        return audio_path

    def get_all_speaker_ids(self) -> List[str]:
        """
        Get all available speaker IDs.

        Returns:
            List of all speaker IDs
        """
        speakers = self.config.get("generation", {}).get("speakers", [])
        return [speaker.get("id", "default") for speaker in speakers]

    def validate_speakers_reference_audio(self) -> Dict[str, Any]:
        """
        Validate reference_audio for all speakers.

        Returns:
            Dictionary with detailed validation results:
            {
                "valid": bool,
                "failed_speakers": List[str],
                "missing_files": Dict[str, str],  # speaker_id -> missing_file_path
                "available_files": List[str],
                "configured_speakers": List[str],
                "details": Dict[str, Dict[str, Any]]  # speaker_id -> {success: bool, error: str|None}
            }
        """
        validation_results = {}
        failed_speakers = []
        missing_files = {}
        error_details = {}
        
        speakers = self.config.get("generation", {}).get("speakers", [])
        configured_speakers = [speaker.get("id", "unknown") for speaker in speakers]
        
        # Get available files in reference_audio directory
        available_files = []
        if self.reference_audio_dir.exists():
            available_files = [f.name for f in self.reference_audio_dir.glob("*.wav")]

        for speaker in speakers:
            speaker_id = speaker.get("id", "unknown")
            try:
                self.get_reference_audio_for_speaker(speaker_id)
                validation_results[speaker_id] = True
                error_details[speaker_id] = {"success": True, "error": None}
            except (FileNotFoundError, ValueError) as e:
                validation_results[speaker_id] = False
                failed_speakers.append(speaker_id)
                error_details[speaker_id] = {"success": False, "error": str(e)}
                
                # Extract the missing file path from the error message
                if isinstance(e, FileNotFoundError):
                    # Extract the filename from the error message
                    reference_audio = speaker.get("reference_audio", "unknown")
                    missing_files[speaker_id] = reference_audio

        return {
            "valid": len(failed_speakers) == 0,
            "failed_speakers": failed_speakers,
            "missing_files": missing_files,
            "available_files": available_files,
            "configured_speakers": configured_speakers,
            "details": error_details
        }

    def get_default_speaker_id(self) -> str:
        """
        Get the ID of the default speaker using explicit default_speaker key.

        Returns:
            Default speaker ID
        """
        generation_config = self.config.get("generation", {})
        default_speaker = generation_config.get("default_speaker")

        if default_speaker:
            # Verify the default_speaker exists in speakers list
            speakers = generation_config.get("speakers", [])
            speaker_ids = [speaker.get("id", "") for speaker in speakers]

            if default_speaker in speaker_ids:
                return default_speaker
            else:
                logger.warning(
                    f"default_speaker '{default_speaker}' not found in speakers list, falling back to first speaker"
                )

        # Fallback to first speaker if default_speaker key is missing or invalid
        speakers = generation_config.get("speakers", [])
        if not speakers:
            raise RuntimeError("No speakers configured")

        fallback_id = speakers[0].get("id", "default")
        logger.debug(f"Using fallback default speaker: '{fallback_id}'")
        return fallback_id
