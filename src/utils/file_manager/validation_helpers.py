#!/usr/bin/env python3
"""
ValidationHelpers for validation-related helper functions.
Handles corrupt candidate removal and validation data cleanup.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationHelpers:
    """Helper functions for validation operations."""

    def __init__(self, candidates_dir: Path, whisper_dir: Path, whisper_handler):
        """
        Initialize ValidationHelpers.

        Args:
            candidates_dir: Directory for candidate files
            whisper_dir: Directory for whisper validation files
            whisper_handler: WhisperIOHandler instance for stale data removal
        """
        self.candidates_dir = candidates_dir
        self.whisper_dir = whisper_dir
        self.whisper_handler = whisper_handler

    def remove_corrupt_candidate(self, chunk_idx: int, candidate_idx: int) -> bool:
        """
        Remove corrupt candidate file and its validation data.

        Args:
            chunk_idx: Chunk index
            candidate_idx: Candidate index

        Returns:
            True if removal successful
        """
        try:
            removed_files = []

            # Remove audio file
            chunk_dir = self.candidates_dir / f"chunk_{chunk_idx+1:03d}"
            audio_file = chunk_dir / f"candidate_{candidate_idx+1:02d}.wav"
            if audio_file.exists():
                audio_file.unlink()
                removed_files.append(str(audio_file))
                logger.info(f"🗑️ Removed corrupt audio file: {audio_file}")

            # Remove whisper validation file
            whisper_file = (
                self.whisper_dir
                / f"chunk_{chunk_idx+1:03d}_candidate_{candidate_idx+1:02d}_whisper.json"
            )
            if whisper_file.exists():
                whisper_file.unlink()
                removed_files.append(str(whisper_file))
                logger.info(f"🗑️ Removed stale whisper validation: {whisper_file}")

            # Remove from enhanced metrics
            self.whisper_handler._remove_stale_validation_data(chunk_idx, candidate_idx)

            if removed_files:
                logger.warning(
                    f"⚠️ Cleaned up {len(removed_files)} files for corrupt candidate {candidate_idx+1} in chunk {chunk_idx+1}"
                )

            return True

        except Exception as e:
            logger.error(
                f"Failed to remove corrupt candidate {candidate_idx+1} for chunk {chunk_idx+1}: {e}"
            )
            return False
