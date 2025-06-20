#!/usr/bin/env python3
"""
CandidateIOHandler for audio candidate operations.
Handles saving and loading of audio candidates.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torchaudio

logger = logging.getLogger(__name__)


class AudioCandidate:
    """Audio candidate data structure."""

    def __init__(
        self,
        chunk_idx: int,
        candidate_idx: int,
        audio_path: Path,
        audio_tensor: Optional[torch.Tensor] = None,
        generation_params: Optional[Dict[str, Any]] = None,
        chunk_text: Optional[str] = None,
    ):
        self.chunk_idx = chunk_idx
        self.candidate_idx = candidate_idx
        self.audio_path = audio_path
        self.audio_tensor = audio_tensor
        self.generation_params = generation_params
        self.chunk_text = chunk_text


class CandidateIOHandler:
    """Handles audio candidate I/O operations."""

    def __init__(self, candidates_dir: Path, config: dict):
        """
        Initialize CandidateIOHandler.

        Args:
            candidates_dir: Directory for candidate files
            config: Configuration dictionary
        """
        self.candidates_dir = candidates_dir
        self.config = config
        self.candidates_dir.mkdir(parents=True, exist_ok=True)

    def save_candidates(
        self,
        chunk_idx: int,
        candidates: List[AudioCandidate],
        overwrite_existing: bool = False,
    ) -> bool:
        """
        Save audio candidates for a chunk.

        Args:
            chunk_idx: Chunk index
            candidates: List of AudioCandidate objects
            overwrite_existing: If True, overwrites existing files. If False, only saves new candidates.

        Returns:
            True if successful
        """
        try:
            chunk_dir = self.candidates_dir / f"chunk_{chunk_idx+1:03d}"
            chunk_dir.mkdir(exist_ok=True)

            saved_count = 0
            skipped_count = 0

            for candidate in candidates:
                # Save audio file
                audio_filename = f"candidate_{candidate.candidate_idx+1:02d}.wav"
                audio_path = chunk_dir / audio_filename

                # Check if file already exists and we shouldn't overwrite
                if not overwrite_existing and audio_path.exists():
                    skipped_count += 1
                    logger.debug(f"Skipping existing candidate file: {audio_filename}")
                    # Update candidate path even if not saving
                    candidate.audio_path = audio_path
                    continue

                if candidate.audio_tensor is not None:
                    # Save tensor as audio file
                    sample_rate = self.config.get("audio", {}).get("sample_rate", 24000)
                    # Move tensor to CPU for saving and ensure correct dimensions
                    audio_cpu = candidate.audio_tensor.cpu()
                    if audio_cpu.ndim == 1:
                        audio_cpu = audio_cpu.unsqueeze(0)  # Add channel dimension
                    torchaudio.save(str(audio_path), audio_cpu, sample_rate)
                    saved_count += 1
                    logger.debug(f"Saved new candidate file: {audio_filename}")
                elif candidate.audio_path and candidate.audio_path.exists():
                    # VALIDATE before copying to prevent corrupt files from propagating
                    try:
                        # Test if the file can be loaded properly
                        test_waveform, test_sample_rate = torchaudio.load(
                            str(candidate.audio_path)
                        )
                        if test_waveform.numel() == 0:
                            raise ValueError("Empty audio file")
                        if (
                            torch.isnan(test_waveform).any()
                            or torch.isinf(test_waveform).any()
                        ):
                            raise ValueError("Audio contains NaN or Inf values")

                        # File is valid, safe to copy
                        shutil.copy2(candidate.audio_path, audio_path)
                        saved_count += 1
                        logger.debug(
                            f"Copied validated candidate file: {audio_filename}"
                        )

                    except Exception as e:
                        # CRITICAL: Corrupt file detected - do NOT copy!
                        logger.error(
                            f"🚨 CORRUPT AUDIO FILE DETECTED: {candidate.audio_path}"
                        )
                        logger.error(f"   Error: {e}")
                        logger.error(
                            f"   Skipping candidate {candidate.candidate_idx+1} for chunk {chunk_idx+1}"
                        )
                        logger.error(
                            f"   This candidate will be excluded from final audio assembly!"
                        )

                        # Remove the corrupt file and its validation data
                        self._remove_corrupt_candidate(
                            chunk_idx, candidate.candidate_idx
                        )
                        continue  # Skip this candidate entirely
                else:
                    # No audio tensor AND no valid audio file - this candidate is unusable
                    logger.warning(
                        f"⚠️ Unusable candidate {candidate.candidate_idx+1} for chunk {chunk_idx+1}: no audio tensor or valid file"
                    )
                    continue  # Skip this candidate

                # Update candidate path
                candidate.audio_path = audio_path

            # Save candidate metadata
            candidate_metadata = {
                "chunk_idx": chunk_idx,
                "total_candidates": len(candidates),
                "candidates": [
                    {
                        "candidate_idx": c.candidate_idx,
                        "audio_filename": f"candidate_{c.candidate_idx+1:02d}.wav",
                        "generation_params": c.generation_params,
                    }
                    for c in candidates
                ],
            }

            metadata_path = chunk_dir / "candidates_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(candidate_metadata, f, indent=2)

            if overwrite_existing:
                logger.debug(
                    f"Saved {saved_count} candidates for chunk {chunk_idx + 1} (overwrite mode)"
                )
            else:
                logger.debug(
                    f"Saved {saved_count} new candidates for chunk {chunk_idx + 1} (skipped {skipped_count} existing)"
                )
            return True

        except Exception as e:
            logger.error(f"Error saving candidates for chunk {chunk_idx+1}: {e}")
            return False

    def get_candidates(
        self, chunk_idx: Optional[int] = None
    ) -> Dict[int, List[AudioCandidate]]:
        """
        Load audio candidates.

        Args:
            chunk_idx: Specific chunk index, or None for all chunks

        Returns:
            Dictionary mapping chunk_idx to list of AudioCandidate objects
        """
        candidates = {}

        if chunk_idx is not None:
            # Load specific chunk
            chunk_indices = [chunk_idx]
        else:
            # Load all chunks
            chunk_dirs = [
                d
                for d in self.candidates_dir.iterdir()
                if d.is_dir() and d.name.startswith("chunk_")
            ]
            chunk_indices = [
                int(d.name.split("_")[1]) - 1 for d in chunk_dirs
            ]  # Convert back to 0-based

        for idx in chunk_indices:
            chunk_dir = self.candidates_dir / f"chunk_{idx+1:03d}"
            if not chunk_dir.exists():
                continue

            chunk_candidates = []

            # Load metadata
            metadata_path = chunk_dir / "candidates_metadata.json"
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

            # Load audio files
            audio_files = sorted(chunk_dir.glob("candidate_*.wav"))
            for audio_file in audio_files:
                # Extract candidate index
                candidate_idx = (
                    int(audio_file.stem.split("_")[1]) - 1
                )  # Convert back to 0-based

                # Get metadata for this candidate
                candidate_meta = None
                if metadata and "candidates" in metadata:
                    candidate_meta = next(
                        (
                            c
                            for c in metadata["candidates"]
                            if c["candidate_idx"] == candidate_idx
                        ),
                        None,
                    )

                generation_params = (
                    candidate_meta.get("generation_params") if candidate_meta else None
                )

                # Load audio tensor if file exists
                audio_tensor = None
                if audio_file.exists():
                    try:
                        waveform, sample_rate = torchaudio.load(str(audio_file))
                        # Convert to mono if needed and remove channel dimension
                        if waveform.shape[0] > 1:
                            waveform = waveform.mean(dim=0, keepdim=True)
                        audio_tensor = waveform.squeeze(
                            0
                        )  # Remove channel dimension for consistency
                    except Exception as e:
                        logger.warning(f"Failed to load audio file {audio_file}: {e}")

                candidate = AudioCandidate(
                    chunk_idx=idx,
                    candidate_idx=candidate_idx,
                    audio_path=audio_file,
                    audio_tensor=audio_tensor,
                    generation_params=generation_params,
                )
                chunk_candidates.append(candidate)

            # Sort by candidate index
            chunk_candidates.sort(key=lambda c: c.candidate_idx)
            candidates[idx] = chunk_candidates

        total_candidates = sum(len(cands) for cands in candidates.values())
        logger.debug(
            f"Loaded {total_candidates} candidates for {len(candidates)} chunks"
        )
        return candidates

    def save_candidates_to_disk(
        self,
        candidates: List[AudioCandidate],
        chunk_index: int,
        sample_rate: int = 24000,
        output_dir: Optional[Path] = None,
    ) -> List[str]:
        """
        Saves generated audio candidates to disk for inspection/debugging.

        Args:
            candidates: List of AudioCandidate objects
            chunk_index: Chunk index (0-based)
            sample_rate: Audio sample rate
            output_dir: Output directory for whisper file deletion

        Returns:
            List of file paths where candidates were saved.
        """
        if not candidates:
            return []

        chunk_dir = self.candidates_dir / f"chunk_{chunk_index+1:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for candidate in candidates:
            try:
                filename = f"candidate_{candidate.candidate_idx+1:02d}.wav"
                filepath = chunk_dir / filename

                # Delete corresponding whisper file if it exists (ensures re-validation)
                if output_dir:
                    self._delete_whisper_file(
                        output_dir, chunk_index, candidate.candidate_idx + 1
                    )

                # Ensure audio tensor is 2D for torchaudio.save (channels, samples)
                audio_tensor = candidate.audio_tensor.cpu()
                if audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension

                torchaudio.save(str(filepath), audio_tensor, sample_rate)

                # Update candidate metadata with correct path
                candidate.audio_path = filepath

                saved_paths.append(str(filepath))
                logger.debug(f"Saved candidate to: {filepath}")

            except Exception as e:
                logger.error(
                    f"Failed to save candidate {candidate.candidate_idx+1} for chunk {chunk_index}: {e}"
                )
                continue

        # Save candidate metadata (consistent with FileManager)
        if saved_paths:
            self._save_candidate_metadata(candidates, chunk_index, chunk_dir)

        return saved_paths

    def _save_candidates_in_correct_structure(
        self, candidates: List[AudioCandidate], chunk_index: int
    ):
        """Helper to save candidates when FileManager is not directly available."""
        chunk_dir = self.candidates_dir / f"chunk_{chunk_index+1:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        sample_rate = self.config.get("audio", {}).get("sample_rate", 24000)

        for candidate in candidates:
            try:
                filename = f"candidate_{candidate.candidate_idx+1:02d}.wav"
                filepath = chunk_dir / filename

                audio_tensor = candidate.audio_tensor.cpu()
                if audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)

                torchaudio.save(str(filepath), audio_tensor, sample_rate)
                candidate.audio_path = filepath

                logger.debug(f"Saved candidate to correct structure: {filepath}")

            except Exception as e:
                logger.error(
                    f"Failed to save candidate {candidate.candidate_idx+1}: {e}"
                )

    def _save_candidate_metadata(
        self, candidates: List[AudioCandidate], chunk_index: int, chunk_dir: Path
    ):
        """Saves metadata for generated candidates in a JSON file within the chunk directory."""
        try:
            candidate_metadata = {
                "chunk_idx": chunk_index,
                "total_candidates": len(candidates),
                "candidates": [
                    {
                        "candidate_idx": c.candidate_idx,
                        "audio_filename": f"candidate_{c.candidate_idx+1:02d}.wav",
                        "generation_params": c.generation_params,
                    }
                    for c in candidates
                ],
            }

            metadata_path = chunk_dir / "candidates_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(candidate_metadata, f, indent=2)

            logger.debug(f"Saved candidate metadata: {metadata_path}")

        except Exception as e:
            logger.error(f"Failed to save candidate metadata: {e}")

    def _delete_whisper_file(
        self, output_dir: Path, chunk_index: int, candidate_idx: int
    ):
        """Delete corresponding whisper validation file for a candidate (ensures re-validation)."""
        whisper_dir = output_dir / "whisper"
        whisper_file = (
            whisper_dir
            / f"chunk_{chunk_index+1:03d}_candidate_{candidate_idx:02d}_whisper.json"
        )

        if whisper_file.exists():
            whisper_file.unlink()
            logger.debug(f"🗑️ Deleted old whisper file: {whisper_file.name}")

        alt_whisper_file = (
            whisper_dir
            / f"chunk_{chunk_index+1:03d}_candidate_{candidate_idx:02d}_whisper.txt"
        )
        if alt_whisper_file.exists():
            alt_whisper_file.unlink()
            logger.debug(f"🗑️ Deleted old whisper TXT file: {alt_whisper_file.name}")

    def _remove_corrupt_candidate(self, chunk_idx: int, candidate_idx: int) -> bool:
        """Remove corrupt candidate file - simplified version without validation data."""
        try:
            chunk_dir = self.candidates_dir / f"chunk_{chunk_idx+1:03d}"
            audio_file = chunk_dir / f"candidate_{candidate_idx+1:02d}.wav"
            if audio_file.exists():
                audio_file.unlink()
                logger.info(f"🗑️ Removed corrupt audio file: {audio_file}")
                return True
        except Exception as e:
            logger.error(
                f"Failed to remove corrupt candidate {candidate_idx+1} for chunk {chunk_idx+1}: {e}"
            )
        return False
