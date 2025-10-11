#!/usr/bin/env python3
"""
WhisperIOHandler for Whisper validation operations.
Handles saving and loading of Whisper validation results.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

class WhisperIOHandler:
    """Handles Whisper validation I/O operations."""

    def __init__(self, whisper_dir: Path, task_directory: Path, candidates_dir: Path):
        """
        Initialize WhisperIOHandler.

        Args:
            whisper_dir: Directory for whisper validation files
            task_directory: Main task directory (for enhanced_metrics.json)
            candidates_dir: Directory for candidate files (for validation)
        """
        self.whisper_dir = whisper_dir
        self.task_directory = task_directory
        self.candidates_dir = candidates_dir
        self.whisper_dir.mkdir(parents=True, exist_ok=True)

    def save_whisper(self, chunk_idx: int, candidate_idx: int, result: dict) -> bool:
        """Save Whisper validation result to individual file and sync to whisper metrics (without selections)."""
        try:
            # Save individual file (for Recovery System compatibility)
            filename = (
                f"chunk_{chunk_idx+1:03d}_candidate_{candidate_idx+1:02d}_whisper.json"
            )
            path = self.whisper_dir / filename

            with open(path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            # Also sync to whisper metrics (without any selected_candidates)
            sync_success = self._sync_whisper_to_enhanced_metrics(
                chunk_idx, candidate_idx, result
            )
            if sync_success:
                logger.debug(
                    f"✓ Synced whisper result to enhanced metrics: chunk {chunk_idx}, candidate {candidate_idx}"
                )

            return True
        except Exception as e:
            logger.error(
                f"Error saving Whisper result for chunk {chunk_idx}, candidate {candidate_idx}: {e}"
            )
            return False

    def get_whisper(
        self, chunk_idx: int, candidate_idx: Optional[int] = None
    ) -> Dict[int, dict]:
        """
        Load Whisper validation results, preferring enhanced_metrics.json over individual files.
        Validates that corresponding audio files still exist to prevent stale validation data.
        """
        results = {}

        # Try to load from enhanced metrics first (primary source)
        enhanced_results = self._get_whisper_from_enhanced_metrics(
            chunk_idx, candidate_idx
        )
        if enhanced_results:
            # Validate that audio files still exist for each result
            validated_results = {}
            for cand_idx, validation_data in enhanced_results.items():
                if self._audio_file_exists(chunk_idx, cand_idx):
                    validated_results[cand_idx] = validation_data
                else:
                    logger.debug(
                        f"Skipping stale validation data for chunk {chunk_idx}, candidate {cand_idx} - audio file no longer exists"
                    )
                    # Clean up stale data from enhanced metrics
                    self._remove_stale_validation_data(chunk_idx, cand_idx)

            results.update(validated_results)

        # Fallback to individual files for missing data (compatibility)
        if candidate_idx is not None:
            if candidate_idx not in results and self._audio_file_exists(
                chunk_idx, candidate_idx
            ):
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
                if cand_idx not in results and self._audio_file_exists(
                    chunk_idx, cand_idx
                ):
                    with open(file, "r", encoding="utf-8") as f:
                        results[cand_idx] = json.load(f)

        return results

    def delete_whisper(self, chunk_idx: int, candidate_idx: int) -> bool:
        """Delete Whisper validation result for a specific candidate."""
        try:
            # Delete individual file
            filename = (
                f"chunk_{chunk_idx+1:03d}_candidate_{candidate_idx+1:02d}_whisper.json"
            )
            path = self.whisper_dir / filename
            if path.exists():
                path.unlink()
                logger.debug(f"Deleted individual whisper file: {path}")

            # Remove from enhanced metrics
            self._remove_stale_validation_data(chunk_idx, candidate_idx)
            logger.debug(
                f"Removed whisper data from enhanced metrics for chunk {chunk_idx}, candidate {candidate_idx}"
            )

            return True
        except Exception as e:
            logger.error(
                f"Error deleting Whisper result for chunk {chunk_idx}, candidate {candidate_idx}: {e}"
            )
            return False

    def migrate_whisper_to_enhanced_metrics(self) -> bool:
        """
        Migrate existing individual whisper files into consolidated whisper metrics.
        Note: No selected_candidates are written into whisper metrics.

        Returns:
            True if migration successful
        """
        try:
            logger.debug(
                "🔄 Migrating existing whisper files to enhanced metrics format..."
            )

            # Find all existing whisper files
            whisper_files = list(
                self.whisper_dir.glob("chunk_*_candidate_*_whisper.json")
            )
            total_files = len(whisper_files)
            if total_files == 0:
                logger.info("Whisper migration: no files found - nothing to migrate")
                return True

            migrated_count = 0
            skipped_missing_audio = 0
            invalid_filename_or_json = 0
            sync_failures = 0

            for whisper_file in whisper_files:
                try:
                    # Parse chunk and candidate indices from filename
                    parts = whisper_file.stem.split("_")
                    chunk_idx = int(parts[1]) - 1  # Convert from 1-based to 0-based
                    candidate_idx = int(parts[3]) - 1  # Convert from 1-based to 0-based

                    # Only migrate if corresponding audio file still exists
                    if not self._audio_file_exists(chunk_idx, candidate_idx):
                        skipped_missing_audio += 1
                        continue

                    # Load whisper result
                    with open(whisper_file, "r", encoding="utf-8") as f:
                        result = json.load(f)

                    # Sync to enhanced metrics (without saving individual file again)
                    if self._sync_whisper_to_enhanced_metrics(
                        chunk_idx, candidate_idx, result
                    ):
                        migrated_count += 1
                    else:
                        sync_failures += 1

                except Exception:
                    # Filename parse errors, IO errors, or JSON decode errors → count and continue
                    invalid_filename_or_json += 1
                    continue

            logger.info(
                "Whisper migration: processed %d, migrated %d, skipped_missing_audio %d, invalid %d, sync_failures %d",
                total_files,
                migrated_count,
                skipped_missing_audio,
                invalid_filename_or_json,
                sync_failures,
            )
            return True

        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False

    def cleanup_duplicate_whisper_files(
        self, keep_individual_files: bool = True
    ) -> bool:
        """
        Clean up duplicate validation data after successful migration.

        Args:
            keep_individual_files: If True, keeps individual whisper files for Recovery System compatibility
                                 If False, removes them after successful migration to enhanced metrics

        Returns:
            True if cleanup successful
        """
        if keep_individual_files:
            logger.info(
                "Keeping individual whisper files for Recovery System compatibility"
            )
            return True

        try:
            logger.debug("🧹 Cleaning up individual whisper files after migration...")

            # Verify enhanced metrics exists and has data
            metrics_path = self.task_directory / "whisper" / "whisper_metrics.json"
            if not metrics_path.exists():
                logger.warning(
                    "Whisper metrics not found or empty - skipping cleanup for safety"
                )
                return False

            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            if not metrics or "chunks" not in metrics:
                logger.warning(
                    "Whisper metrics not found or empty - skipping cleanup for safety"
                )
                return False

            # Remove individual whisper files
            whisper_files = list(
                self.whisper_dir.glob("chunk_*_candidate_*_whisper.json")
            )
            removed_count = 0

            for whisper_file in whisper_files:
                try:
                    whisper_file.unlink()
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove {whisper_file}: {e}")

            logger.debug(
                f"✅ Cleanup completed: {removed_count} individual whisper files removed"
            )
            return True

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return False

    def _sync_whisper_to_enhanced_metrics(
        self, chunk_idx: int, candidate_idx: int, result: dict
    ) -> bool:
        """Synchronize whisper result to whisper_metrics.json (no selections stored)."""
        try:
            metrics_path = self.task_directory / "whisper" / "whisper_metrics.json"

            # Load existing metrics or create new structure
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            else:
                # Create minimal structure for sync
                metrics = {
                    "timestamp": 0.0,  # Changed from time.time() to 0.0 as time is removed
                    "total_chunks": 0,
                    "chunks": {},
                }

            # Ensure chunk structure exists
            chunk_key = str(chunk_idx)
            if chunk_key not in metrics["chunks"]:
                metrics["chunks"][chunk_key] = {
                    "chunk_text": "",  # Will be filled by validation stage
                    "candidates": {},
                    "best_candidate": None,
                    "best_score": 0.0,
                }

            # Update candidate data in enhanced metrics - minimal structure
            candidate_key = str(candidate_idx)
            quality_details = result.get("quality_details", {})
            individual_scores = quality_details.get("individual_scores", {})
            overall_score = individual_scores.get("overall_score", 0.0)

            new_candidate_data = {
                "transcription": result.get("transcription", ""),
                "quality_details": quality_details,
                "is_valid": result.get("is_valid", False),
                # Add top-level fields for median calculation compatibility
                "overall_quality_score": overall_score,
                "similarity_score": individual_scores.get("similarity_score", 0.0),
            }

            # Merge into existing candidate entry if present to preserve fields like final_score
            existing_candidate = metrics["chunks"][chunk_key]["candidates"].get(candidate_key)
            if isinstance(existing_candidate, dict):
                merged = existing_candidate.copy()
                merged.update(new_candidate_data)
                # Ensure final_score is present and consistent
                if "final_score" not in merged:
                    merged["final_score"] = overall_score
                metrics["chunks"][chunk_key]["candidates"][candidate_key] = merged
            else:
                # Ensure final_score on fresh insert
                new_candidate_data["final_score"] = overall_score
                metrics["chunks"][chunk_key]["candidates"][candidate_key] = new_candidate_data

            # Ensure whisper metrics never contain selected_candidates
            if isinstance(metrics, dict) and "selected_candidates" in metrics:
                try:
                    metrics.pop("selected_candidates", None)
                except Exception:
                    pass

            # Sort chunks and candidates for stable, readable output
            try:
                chunks = metrics.get("chunks", {})
                if isinstance(chunks, dict):
                    # Sort candidates per chunk
                    sorted_chunks = {}
                    for k in sorted(chunks.keys(), key=lambda x: int(x)):
                        chunk_data = chunks.get(k, {})
                        if isinstance(chunk_data, dict):
                            cands = chunk_data.get("candidates", {})
                            if isinstance(cands, dict):
                                sorted_cands = {ck: cands[ck] for ck in sorted(cands.keys(), key=lambda x: int(x))}
                                chunk_copy = chunk_data.copy()
                                chunk_copy["candidates"] = sorted_cands
                                sorted_chunks[k] = chunk_copy
                            else:
                                sorted_chunks[k] = chunk_data
                        else:
                            sorted_chunks[k] = chunk_data
                    metrics["chunks"] = sorted_chunks
            except Exception:
                # Non-critical: keep original ordering if sorting fails
                pass

            # Save updated metrics back
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            # Don't fail the whole operation if sync fails; keep noise low in normal runs
            logger.debug(f"Failed to sync whisper result to enhanced metrics: {e}")
            return False

    def _get_whisper_from_enhanced_metrics(
        self, chunk_idx: int, candidate_idx: Optional[int] = None
    ) -> Dict[int, dict]:
        """Extract whisper validation results from whisper_metrics.json."""
        try:
            metrics_path = self.task_directory / "whisper" / "whisper_metrics.json"
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
                    # Extract from quality_details to match new structure
                    quality_details = candidate_data.get("quality_details", {})
                    individual_scores = quality_details.get("individual_scores", {}) if isinstance(quality_details, dict) else {}
                    validation_metrics = quality_details.get("validation_metrics", {}) if isinstance(quality_details, dict) else {}

                    similarity_score = individual_scores.get("similarity_score", 0.0)
                    quality_score = validation_metrics.get("whisper_quality", 0.0)
                    overall_quality_score = individual_scores.get("overall_score", 0.0)

                    original_is_valid = candidate_data.get("is_valid", False)
                    if "is_valid" not in candidate_data:
                        # Fallback: recalculate with proper thresholds only if not stored
                        # Use config thresholds instead of hardcoded values
                        original_is_valid = (
                            similarity_score >= 0.85 and quality_score >= 0.6
                        )

                    results[candidate_idx] = {
                        "is_valid": original_is_valid,
                        "transcription": candidate_data.get("transcription", ""),
                        "similarity_score": similarity_score,
                        "quality_score": quality_score,
                        "validation_time": 0.0,  # Not stored in enhanced metrics
                        "error_message": None,
                        "overall_quality_score": overall_quality_score,
                        "quality_details": quality_details,
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
                        original_is_valid = (
                            similarity_score >= 0.85 and quality_score >= 0.6
                        )

                    results[cand_idx] = {
                        "is_valid": original_is_valid,
                        "transcription": candidate_data.get("transcription", ""),
                        "similarity_score": candidate_data.get("similarity_score", 0.0),
                        "quality_score": candidate_data.get("validation_score", 0.0),
                        "validation_time": 0.0,
                        "error_message": None,
                        "overall_quality_score": candidate_data.get(
                            "overall_quality_score", 0.0
                        ),
                        "quality_details": candidate_data.get("quality_details", {}),
                    }

            return results

        except Exception as e:
            logger.warning(
                f"Failed to extract whisper results from enhanced metrics: {e}"
            )
            return {}

    def _audio_file_exists(self, chunk_idx: int, candidate_idx: int) -> bool:
        """Check if the corresponding audio file exists for a validation result."""
        try:
            chunk_dir = self.candidates_dir / f"chunk_{chunk_idx+1:03d}"
            audio_file = chunk_dir / f"candidate_{candidate_idx+1:02d}.wav"
            return audio_file.exists()
        except Exception as e:
            logger.warning(
                f"Error checking audio file existence for chunk {chunk_idx}, candidate {candidate_idx}: {e}"
            )
            return False

    def _remove_stale_validation_data(self, chunk_idx: int, candidate_idx: int) -> bool:
        """Remove stale validation data from whisper_metrics.json when audio file no longer exists."""
        try:
            metrics_path = self.task_directory / "whisper" / "whisper_metrics.json"
            if not metrics_path.exists():
                return True  # Nothing to clean up

            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)

            chunk_key = str(chunk_idx)
            candidate_key = str(candidate_idx)

            # Remove stale candidate data
            if chunk_key in metrics.get("chunks", {}) and candidate_key in metrics[
                "chunks"
            ][chunk_key].get("candidates", {}):

                del metrics["chunks"][chunk_key]["candidates"][candidate_key]
                logger.debug(
                    f"Removed stale validation data for chunk {chunk_idx}, candidate {candidate_idx}"
                )

                # If no candidates left in chunk, clean up chunk-level data
                if not metrics["chunks"][chunk_key]["candidates"]:
                    metrics["chunks"][chunk_key]["best_candidate"] = None
                    metrics["chunks"][chunk_key]["best_score"] = 0.0
                    logger.debug(
                        f"Reset chunk {chunk_idx} best candidate info due to no valid candidates"
                    )

                # Save updated metrics
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)

                return True

        except Exception as e:
            logger.warning(
                f"Failed to remove stale validation data for chunk {chunk_idx}, candidate {candidate_idx}: {e}"
            )

        return False
