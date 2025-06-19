#!/usr/bin/env python3
"""
FileManager for centralized file operations.
Handles all file I/O operations for the TTS pipeline with consistent schemas.
"""

import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import torch
import torchaudio
import yaml

from chunking.base_chunker import TextChunk
from utils.config_manager import ConfigManager, TaskConfig

import logging

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


@dataclass
class AudioCandidate:
    """Audio candidate data structure."""

    chunk_idx: int
    candidate_idx: int
    audio_path: Path
    audio_tensor: Optional[torch.Tensor] = None
    generation_params: Optional[Dict[str, Any]] = None
    chunk_text: Optional[str] = None  # For validation compatibility


class FileManager:
    """
    Central file manager for all pipeline I/O operations.
    Maintains consistent file schemas and directory structures.
    """

    def __init__(self, task_config: Union[TaskConfig, Dict[str, Any]], preloaded_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize FileManager with task configuration.

        Args:
            task_config: TaskConfig object or config dictionary
            preloaded_config: Optional pre-loaded config to avoid redundant loading
        """
        from utils.config_manager import ConfigManager

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

    # Chunk Operations
    def save_chunks(self, chunks: List[TextChunk]) -> bool:
        """
        Save text chunks to files.

        Args:
            chunks: List of TextChunk objects

        Returns:
            True if successful
        """
        try:
            # Save individual chunk files
            for chunk in chunks:
                chunk_filename = f"chunk_{chunk.idx+1:03d}.txt"
                chunk_path = self.texts_dir / chunk_filename

                with open(chunk_path, "w", encoding="utf-8") as f:
                    f.write(chunk.text)

            # Save chunk metadata
            chunk_metadata = {
                "total_chunks": len(chunks),
                "chunks": [
                    {
                        "idx": chunk.idx,
                        "text_length": len(chunk.text),
                        "is_paragraph_break": chunk.has_paragraph_break,
                        "filename": f"chunk_{chunk.idx+1:03d}.txt",
                    }
                    for chunk in chunks
                ],
            }

            metadata_path = self.texts_dir / "chunks_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(chunk_metadata, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved {len(chunks)} chunks to {self.texts_dir}")
            return True

        except Exception as e:
            logger.error(f"Error saving chunks: {e}")
            return False

    def get_chunks(self) -> List[TextChunk]:
        """
        Load text chunks from files.

        Returns:
            List of TextChunk objects
        """
        chunks = []

        # Load metadata if available
        metadata_path = self.texts_dir / "chunks_metadata.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

        # Load chunk files
        chunk_files = sorted(self.texts_dir.glob("chunk_*.txt"))
        for chunk_file in chunk_files:
            # Extract chunk index from filename (convert from 1-based filename to 0-based idx)
            chunk_idx = int(chunk_file.stem.split("_")[1]) - 1

            # Load text content
            with open(chunk_file, "r", encoding="utf-8") as f:
                text = f.read()

            # Get metadata for this chunk (metadata uses 0-based idx)
            chunk_meta = None
            if metadata and "chunks" in metadata:
                chunk_meta = next(
                    (c for c in metadata["chunks"] if c["idx"] == chunk_idx), None
                )

            is_paragraph_break = (
                chunk_meta.get("is_paragraph_break", False) if chunk_meta else False
            )

            chunk = TextChunk(
                idx=chunk_idx,
                text=text,
                start_pos=0,  # Not tracked in saved chunks
                end_pos=len(text),  # Not tracked in saved chunks
                has_paragraph_break=is_paragraph_break,
                estimated_tokens=len(text.split()),
                is_fallback_split=False,
            )
            chunks.append(chunk)

        # Sort by index
        chunks.sort(key=lambda c: c.idx)

        logger.debug(f"Loaded {len(chunks)} chunks from {self.texts_dir}")
        return chunks

    # Candidate Operations
    def save_candidates(self, chunk_idx: int, candidates: List[AudioCandidate], overwrite_existing: bool = False) -> bool:
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
                        test_waveform, test_sample_rate = torchaudio.load(str(candidate.audio_path))
                        if test_waveform.numel() == 0:
                            raise ValueError("Empty audio file")
                        if torch.isnan(test_waveform).any() or torch.isinf(test_waveform).any():
                            raise ValueError("Audio contains NaN or Inf values")
                        
                        # File is valid, safe to copy
                        import shutil
                        shutil.copy2(candidate.audio_path, audio_path)
                        saved_count += 1
                        logger.debug(f"Copied validated candidate file: {audio_filename}")
                        
                    except Exception as e:
                        # CRITICAL: Corrupt file detected - do NOT copy!
                        logger.error(f"🚨 CORRUPT AUDIO FILE DETECTED: {candidate.audio_path}")
                        logger.error(f"   Error: {e}")
                        logger.error(f"   Skipping candidate {candidate.candidate_idx+1} for chunk {chunk_idx+1}")
                        logger.error(f"   This candidate will be excluded from final audio assembly!")
                        
                        # Remove the corrupt file and its validation data
                        self._remove_corrupt_candidate(chunk_idx, candidate.candidate_idx)
                        continue  # Skip this candidate entirely
                else:
                    # No audio tensor AND no valid audio file - this candidate is unusable
                    logger.warning(f"⚠️ Unusable candidate {candidate.candidate_idx+1} for chunk {chunk_idx+1}: no audio tensor or valid file")
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
                logger.debug(f"Saved {saved_count} candidates for chunk {chunk_idx + 1} (overwrite mode)")
            else:
                logger.debug(f"Saved {saved_count} new candidates for chunk {chunk_idx + 1} (skipped {skipped_count} existing)")
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
            chunk_indices = [int(d.name.split("_")[1]) - 1 for d in chunk_dirs]  # Convert back to 0-based

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
                candidate_idx = int(audio_file.stem.split("_")[1]) - 1  # Convert back to 0-based

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

    # Whisper Operations
    def save_whisper(self, chunk_idx: int, candidate_idx: int, result: dict) -> bool:
        """Save Whisper validation result to both individual file and enhanced metrics."""
        try:
            # Save individual file (for Recovery System compatibility)
            filename = f"chunk_{chunk_idx+1:03d}_candidate_{candidate_idx+1:02d}_whisper.json"
            path = self.whisper_dir / filename
            
            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Also sync to enhanced metrics (maintaining dual system consistency)
            sync_success = self._sync_whisper_to_enhanced_metrics(chunk_idx, candidate_idx, result)
            if sync_success:
                logger.debug(f"✓ Synced whisper result to enhanced metrics: chunk {chunk_idx}, candidate {candidate_idx}")
            
            return True
        except Exception as e:
            logger.error(f"Error saving Whisper result for chunk {chunk_idx}, candidate {candidate_idx}: {e}")
            return False

    def _sync_whisper_to_enhanced_metrics(self, chunk_idx: int, candidate_idx: int, result: dict) -> bool:
        """Synchronize whisper result to enhanced_metrics.json if it exists."""
        try:
            metrics_path = self.task_directory / "enhanced_metrics.json"
            
            # Load existing metrics or create new structure
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            else:
                # Create minimal structure for sync
                metrics = {
                    "timestamp": time.time(),
                    "total_chunks": 0,
                    "chunks": {},
                    "selected_candidates": {}
                }
            
            # Ensure chunk structure exists
            chunk_key = str(chunk_idx)
            if chunk_key not in metrics["chunks"]:
                metrics["chunks"][chunk_key] = {
                    "chunk_text": "",  # Will be filled by validation stage
                    "candidates": {},
                    "best_candidate": None,
                    "best_score": 0.0
                }
            
            # Update candidate data in enhanced metrics
            candidate_key = str(candidate_idx)
            candidate_data = {
                "transcription": result.get("transcription", ""),
                "similarity_score": result.get("similarity_score", 0.0),
                "validation_score": result.get("quality_score", 0.0),
                "overall_quality_score": result.get("overall_quality_score", 0.0),
                "quality_details": result.get("quality_details", {}),
                "is_valid": result.get("is_valid", False)  # Store original validation result
            }
            
            metrics["chunks"][chunk_key]["candidates"][candidate_key] = candidate_data
            
            # Save updated metrics back
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
                
            return True
            
        except Exception as e:
            # Don't fail the whole operation if sync fails
            logger.warning(f"Failed to sync whisper result to enhanced metrics: {e}")
            return False

    def get_whisper(self, chunk_idx: int, candidate_idx: Optional[int] = None) -> Dict[int, dict]:
        """
        Load Whisper validation results, preferring enhanced_metrics.json over individual files.
        Validates that corresponding audio files still exist to prevent stale validation data.
        """
        results = {}
        
        # Try to load from enhanced metrics first (primary source)
        enhanced_results = self._get_whisper_from_enhanced_metrics(chunk_idx, candidate_idx)
        if enhanced_results:
            # Validate that audio files still exist for each result
            validated_results = {}
            for cand_idx, validation_data in enhanced_results.items():
                if self._audio_file_exists(chunk_idx, cand_idx):
                    validated_results[cand_idx] = validation_data
                else:
                    logger.debug(f"Skipping stale validation data for chunk {chunk_idx}, candidate {cand_idx} - audio file no longer exists")
                    # Clean up stale data from enhanced metrics
                    self._remove_stale_validation_data(chunk_idx, cand_idx)
            
            results.update(validated_results)
        
        # Fallback to individual files for missing data (compatibility)
        if candidate_idx is not None:
            if candidate_idx not in results and self._audio_file_exists(chunk_idx, candidate_idx):
                filename = f"chunk_{chunk_idx+1:03d}_candidate_{candidate_idx+1:02d}_whisper.json"
                path = self.whisper_dir / filename
                if path.exists():
                    with open(path, "r", encoding="utf-8") as f:
                        results[candidate_idx] = json.load(f)
        else:
            # Load all candidates for chunk - check for missing ones from individual files
            pattern = f"chunk_{chunk_idx+1:03d}_candidate_*_whisper.json"
            for file in self.whisper_dir.glob(pattern):
                cand_idx = int(file.stem.split("_")[3]) - 1
                if cand_idx not in results and self._audio_file_exists(chunk_idx, cand_idx):
                    with open(file, "r", encoding="utf-8") as f:
                        results[cand_idx] = json.load(f)
        
        return results

    def _get_whisper_from_enhanced_metrics(self, chunk_idx: int, candidate_idx: Optional[int] = None) -> Dict[int, dict]:
        """Extract whisper validation results from enhanced_metrics.json."""
        try:
            metrics_path = self.task_directory / "enhanced_metrics.json"
            if not metrics_path.exists():
                return {}
            
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            
            chunk_key = str(chunk_idx)
            if chunk_key not in metrics.get("chunks", {}):
                return {}
            
            chunk_data = metrics["chunks"][chunk_key]
            candidates = chunk_data.get("candidates", {})
            
            results = {}
            if candidate_idx is not None:
                # Get specific candidate
                candidate_key = str(candidate_idx)
                if candidate_key in candidates:
                    candidate_data = candidates[candidate_key]
                    # Convert back to whisper format - use original is_valid from validation
                    # Don't recalculate is_valid here, use the stored value from original validation
                    original_is_valid = candidate_data.get("is_valid", False)
                    if "is_valid" not in candidate_data:
                        # Fallback: recalculate with proper thresholds only if not stored
                        similarity_score = candidate_data.get("similarity_score", 0.0)
                        quality_score = candidate_data.get("validation_score", 0.0)
                        # Use config thresholds instead of hardcoded values
                        original_is_valid = similarity_score >= 0.85 and quality_score >= 0.6
                    
                    results[candidate_idx] = {
                        "is_valid": original_is_valid,
                        "transcription": candidate_data.get("transcription", ""),
                        "similarity_score": candidate_data.get("similarity_score", 0.0),
                        "quality_score": candidate_data.get("validation_score", 0.0),
                        "validation_time": 0.0,  # Not stored in enhanced metrics
                        "error_message": None,
                        "overall_quality_score": candidate_data.get("overall_quality_score", 0.0),
                        "quality_details": candidate_data.get("quality_details", {})
                    }
            else:
                # Get all candidates for chunk
                for candidate_key, candidate_data in candidates.items():
                    cand_idx = int(candidate_key)
                    
                    # Use original is_valid from validation, don't recalculate
                    original_is_valid = candidate_data.get("is_valid", False)
                    if "is_valid" not in candidate_data:
                        # Fallback: recalculate with proper thresholds only if not stored
                        similarity_score = candidate_data.get("similarity_score", 0.0)
                        quality_score = candidate_data.get("validation_score", 0.0)
                        # Use config thresholds instead of hardcoded values
                        original_is_valid = similarity_score >= 0.85 and quality_score >= 0.6
                    
                    results[cand_idx] = {
                        "is_valid": original_is_valid,
                        "transcription": candidate_data.get("transcription", ""),
                        "similarity_score": candidate_data.get("similarity_score", 0.0),
                        "quality_score": candidate_data.get("validation_score", 0.0),
                        "validation_time": 0.0,
                        "error_message": None,
                        "overall_quality_score": candidate_data.get("overall_quality_score", 0.0),
                        "quality_details": candidate_data.get("quality_details", {})
                    }
            
            return results
            
        except Exception as e:
            logger.warning(f"Failed to extract whisper results from enhanced metrics: {e}")
            return {}

    def _audio_file_exists(self, chunk_idx: int, candidate_idx: int) -> bool:
        """Check if the corresponding audio file exists for a validation result."""
        try:
            chunk_dir = self.candidates_dir / f"chunk_{chunk_idx+1:03d}"
            audio_file = chunk_dir / f"candidate_{candidate_idx+1:02d}.wav"
            return audio_file.exists()
        except Exception as e:
            logger.warning(f"Error checking audio file existence for chunk {chunk_idx}, candidate {candidate_idx}: {e}")
            return False

    def _remove_stale_validation_data(self, chunk_idx: int, candidate_idx: int) -> bool:
        """Remove stale validation data from enhanced_metrics.json when audio file no longer exists."""
        try:
            metrics_path = self.task_directory / "enhanced_metrics.json"
            if not metrics_path.exists():
                return True  # Nothing to clean up
            
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            
            chunk_key = str(chunk_idx)
            candidate_key = str(candidate_idx)
            
            # Remove stale candidate data
            if (chunk_key in metrics.get("chunks", {}) and 
                candidate_key in metrics["chunks"][chunk_key].get("candidates", {})):
                
                del metrics["chunks"][chunk_key]["candidates"][candidate_key]
                logger.debug(f"Removed stale validation data for chunk {chunk_idx}, candidate {candidate_idx}")
                
                # If no candidates left in chunk, clean up chunk-level data
                if not metrics["chunks"][chunk_key]["candidates"]:
                    metrics["chunks"][chunk_key]["best_candidate"] = None
                    metrics["chunks"][chunk_key]["best_score"] = 0.0
                    logger.debug(f"Reset chunk {chunk_idx} best candidate info due to no valid candidates")
                
                # Remove from selected candidates if it was selected
                if chunk_key in metrics.get("selected_candidates", {}) and metrics["selected_candidates"][chunk_key] == candidate_idx:
                    del metrics["selected_candidates"][chunk_key]
                    logger.debug(f"Removed stale selected candidate for chunk {chunk_idx}")
                
                # Save updated metrics
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                
                return True
            
        except Exception as e:
            logger.warning(f"Failed to remove stale validation data for chunk {chunk_idx}, candidate {candidate_idx}: {e}")
            
        return False

    def migrate_whisper_to_enhanced_metrics(self) -> bool:
        """
        Migrate existing individual whisper files to enhanced_metrics.json format.
        This ensures backward compatibility and unified data access.
        
        Returns:
            True if migration successful
        """
        try:
            logger.debug("🔄 Migrating existing whisper files to enhanced metrics format...")
            
            # Find all existing whisper files
            whisper_files = list(self.whisper_dir.glob("chunk_*_candidate_*_whisper.json"))
            if not whisper_files:
                logger.debug("No existing whisper files found - no migration needed")
                return True
            
            migration_count = 0
            
            for whisper_file in whisper_files:
                try:
                    # Parse chunk and candidate indices from filename
                    parts = whisper_file.stem.split("_")
                    chunk_idx = int(parts[1]) - 1  # Convert from 1-based to 0-based
                    candidate_idx = int(parts[3]) - 1  # Convert from 1-based to 0-based
                    
                    # Only migrate if corresponding audio file still exists
                    if not self._audio_file_exists(chunk_idx, candidate_idx):
                        logger.debug(f"Skipping migration for chunk {chunk_idx}, candidate {candidate_idx} - audio file no longer exists")
                        continue
                    
                    # Load whisper result
                    with open(whisper_file, "r", encoding="utf-8") as f:
                        result = json.load(f)
                    
                    # Sync to enhanced metrics (without saving individual file again)
                    if self._sync_whisper_to_enhanced_metrics(chunk_idx, candidate_idx, result):
                        migration_count += 1
                        # logger.debug(f"✓ Migrated whisper result for chunk {chunk_idx}, candidate {candidate_idx}")
                    
                except Exception as e:
                    logger.warning(f"Failed to migrate {whisper_file}: {e}")
                    continue
            
            logger.debug(f"✅ Migration completed: {migration_count}/{len(whisper_files)} whisper files migrated to enhanced metrics")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False

    def cleanup_duplicate_whisper_files(self, keep_individual_files: bool = True) -> bool:
        """
        Clean up duplicate validation data after successful migration.
        
        Args:
            keep_individual_files: If True, keeps individual whisper files for Recovery System compatibility
                                 If False, removes them after successful migration to enhanced metrics
        
        Returns:
            True if cleanup successful
        """
        if keep_individual_files:
            logger.info("Keeping individual whisper files for Recovery System compatibility")
            return True
            
        try:
            logger.debug("🧹 Cleaning up individual whisper files after migration...")
            
            # Verify enhanced metrics exists and has data
            metrics = self.get_metrics()
            if not metrics or "chunks" not in metrics:
                logger.warning("Enhanced metrics not found or empty - skipping cleanup for safety")
                return False
            
            # Remove individual whisper files
            whisper_files = list(self.whisper_dir.glob("chunk_*_candidate_*_whisper.json"))
            removed_count = 0
            
            for whisper_file in whisper_files:
                try:
                    whisper_file.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove {whisper_file}: {e}")
            
            logger.debug(f"✅ Cleanup completed: {removed_count} individual whisper files removed")
            return True
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False

    # Metrics Operations
    def save_metrics(self, metrics: dict) -> bool:
        """Save quality metrics and validation results."""
        try:
            path = self.task_directory / "enhanced_metrics.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            return False

    def get_metrics(self) -> dict:
        """Load quality metrics and validation results."""
        path = self.task_directory / "enhanced_metrics.json"
        if not path.exists():
            logger.debug(f"Metrics file not found: {path}")
            return {}
        
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Final Audio Operations  
    def save_final_audio(self, audio: torch.Tensor, metadata: dict) -> bool:
        """Save final assembled audio with metadata."""
        try:
            text_base = Path(self.config["input"]["text_file"]).stem
            run_label = self.config["job"].get("run-label", "")
            filename = f"{text_base}_{run_label}_final.wav" if run_label else f"{text_base}_final.wav"
            
            audio_path = self.final_dir / filename
            sample_rate = self.config.get("audio", {}).get("sample_rate", 24000)
            torchaudio.save(str(audio_path), audio.unsqueeze(0), sample_rate)
            
            metadata_path = self.final_dir / filename.replace('.wav', '_metadata.json')
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving final audio: {e}")
            return False

    def get_final_audio(self) -> Optional[torch.Tensor]:
        """Load final assembled audio."""
        final_files = list(self.final_dir.glob("*_final.wav"))
        if not final_files:
            return None
        
        final_file = max(final_files, key=lambda f: f.stat().st_mtime)
        try:
            audio, _ = torchaudio.load(str(final_file))
            return audio.squeeze(0)
        except Exception as e:
            logger.error(f"Error loading final audio {final_file}: {e}")
            return None

    def get_audio_segments(
        self, selected_candidates: Dict[int, int]
    ) -> List[torch.Tensor]:
        """
        Load selected audio segments for assembly.

        Args:
            selected_candidates: Dictionary mapping chunk_idx to candidate_idx

        Returns:
            List of audio tensors in chunk order
        """
        audio_segments = []

        for chunk_idx in sorted(selected_candidates.keys()):
            candidate_idx = selected_candidates[chunk_idx]

            # Ensure indices are integers (JSON loads as strings)
            chunk_idx = int(chunk_idx)
            candidate_idx = int(candidate_idx)

            # Find audio file
            chunk_dir = self.candidates_dir / f"chunk_{chunk_idx+1:03d}"
            audio_file = chunk_dir / f"candidate_{candidate_idx+1:02d}.wav"

            if audio_file.exists():
                try:
                    audio, _ = torchaudio.load(str(audio_file))
                    
                    # VALIDATE loaded audio
                    if audio.numel() == 0:
                        raise ValueError("Empty audio file")
                    if torch.isnan(audio).any() or torch.isinf(audio).any():
                        raise ValueError("Audio contains NaN or Inf values")
                        
                    audio_segments.append(audio.squeeze(0))  # Remove batch dimension
                    
                except Exception as e:
                    logger.error(f"🚨 CORRUPT AUDIO DETECTED in final assembly: {audio_file}")
                    logger.error(f"   Error: {e}")
                    logger.error(f"   Falling back to silence for chunk {chunk_idx}")
                    
                    # Remove corrupt file and its validation data
                    self._remove_corrupt_candidate(chunk_idx, candidate_idx)
                    
                    # Add silence as fallback instead of breaking the entire final audio
                    sample_rate = self.config.get("audio", {}).get("sample_rate", 24000)
                    silence_duration = 2.0  # 2 seconds of silence as fallback
                    silence = torch.zeros(int(sample_rate * silence_duration))
                    audio_segments.append(silence)
            else:
                logger.warning(f"Audio file not found: {audio_file}")
                # Add silence as fallback
                sample_rate = self.config.get("audio", {}).get("sample_rate", 24000)
                silence = torch.zeros(int(sample_rate * 0.5))
                audio_segments.append(silence)

        logger.debug(f"Loaded {len(audio_segments)} audio segments")
        return audio_segments

    def _remove_corrupt_candidate(self, chunk_idx: int, candidate_idx: int) -> bool:
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
            whisper_file = self.whisper_dir / f"chunk_{chunk_idx+1:03d}_candidate_{candidate_idx+1:02d}_whisper.json"
            if whisper_file.exists():
                whisper_file.unlink()
                removed_files.append(str(whisper_file))
                logger.info(f"🗑️ Removed stale whisper validation: {whisper_file}")
            
            # Remove from enhanced metrics
            self._remove_stale_validation_data(chunk_idx, candidate_idx)
            
            if removed_files:
                logger.warning(f"⚠️ Cleaned up {len(removed_files)} files for corrupt candidate {candidate_idx+1} in chunk {chunk_idx+1}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove corrupt candidate {candidate_idx+1} for chunk {chunk_idx+1}: {e}")
            return False

    # State Analysis
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
            self.get_input_text()
            has_input = True
        except FileNotFoundError:
            missing_components.append("input_text")

        # Check chunks
        chunks = self.get_chunks()
        has_chunks = len(chunks) > 0
        if not has_chunks:
            missing_components.append("chunks")

        # Check candidates - improved to check file system completeness
        candidates = self.get_candidates()
        has_candidates = {}
        expected_candidates_per_chunk = self.config.get("generation", {}).get("num_candidates", 5)
        
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
                    logger.debug(f"Chunk {chunk_idx}: expected {expected_candidates_per_chunk}, found {file_count} files")
            else:
                missing_components.append(f"candidates_chunk_{chunk_idx}")

        # Check whisper results - improved to check per candidate file
        has_whisper = {}
        for chunk_idx in range(len(chunks)):
            whisper_results = self.get_whisper(chunk_idx)
            has_whisper[chunk_idx] = set(whisper_results.keys())
            
            # Check against actual candidate files, not just loaded candidates
            chunk_dir = self.candidates_dir / f"chunk_{chunk_idx+1:03d}"
            if chunk_dir.exists():
                actual_wav_files = list(chunk_dir.glob("candidate_*.wav"))
                expected_whisper_count = len(actual_wav_files)
                
                if len(whisper_results) < expected_whisper_count:
                    missing_components.append(f"whisper_chunk_{chunk_idx}")
                    logger.debug(f"Chunk {chunk_idx}: expected {expected_whisper_count} whisper validations, found {len(whisper_results)}")
                    
                    # Log which specific candidates are missing whisper validation
                    for wav_file in actual_wav_files:
                        # Extract candidate index from filename (candidate_01.wav -> 0)
                        try:
                            candidate_num = int(wav_file.stem.split('_')[1]) - 1
                            if candidate_num not in whisper_results:
                                logger.debug(f"Missing whisper validation for chunk {chunk_idx}, candidate {candidate_num}")
                        except (IndexError, ValueError):
                            logger.warning(f"Could not parse candidate index from {wav_file}")
            else:
                # No chunk directory means no candidates at all
                if len(whisper_results) > 0:
                    logger.warning(f"Found whisper results for chunk {chunk_idx} but no candidate directory")

        # Check metrics
        metrics = self.get_metrics()
        has_metrics = len(metrics) > 0
        if not has_metrics:
            missing_components.append("metrics")

        # Check final audio
        final_audio = self.get_final_audio()
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
