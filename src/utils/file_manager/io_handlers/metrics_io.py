#!/usr/bin/env python3
"""
MetricsIOHandler for quality metrics operations.
Handles saving and loading of quality metrics and validation results.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Set

logger = logging.getLogger(__name__)


class MetricsIOHandler:
    """Handles quality metrics I/O operations."""

    def __init__(self, task_directory: Path):
        """
        Initialize MetricsIOHandler.

        Args:
            task_directory: Main task directory
        """
        self.task_directory = task_directory

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

    def load_enhanced_metrics(self) -> Dict:
        """Load enhanced metrics with error handling."""
        return self.get_metrics()

    def update_selected_candidates(self, selections: Dict[int, int]) -> bool:
        """
        Update selected candidates in enhanced metrics.

        Args:
            selections: Dictionary mapping chunk_idx to candidate_idx

        Returns:
            True if update successful, False otherwise
        """
        try:
            metrics = self.get_metrics()
            if not metrics:
                logger.error("No metrics found to update")
                return False

            # Convert int keys to string keys for JSON compatibility
            string_selections = {str(k): v for k, v in selections.items()}

            # Sort by chunk index to maintain proper order
            sorted_selections = dict(
                sorted(string_selections.items(), key=lambda x: int(x[0]))
            )
            metrics["selected_candidates"] = sorted_selections

            return self.save_metrics(metrics)
        except Exception as e:
            logger.error(f"Error updating selected candidates: {e}")
            return False

    def backup_original_selections(self) -> Dict:
        """
        Create backup of original selected candidates.

        Returns:
            Dictionary of original selections
        """
        try:
            metrics = self.get_metrics()
            return metrics.get("selected_candidates", {}).copy()
        except Exception as e:
            logger.error(f"Error backing up original selections: {e}")
            return {}

    def get_changed_candidates(self, original: Dict, current: Dict) -> Set[int]:
        """
        Get set of chunk indices where candidate selection changed.

        Args:
            original: Original selected candidates
            current: Current selected candidates

        Returns:
            Set of changed chunk indices
        """
        changed = set()

        # Check all chunk indices from both dictionaries
        all_chunks = set(original.keys()) | set(current.keys())

        for chunk_key in all_chunks:
            original_candidate = original.get(chunk_key)
            current_candidate = current.get(chunk_key)

            if original_candidate != current_candidate:
                try:
                    chunk_idx = int(chunk_key)
                    changed.add(chunk_idx)
                except ValueError:
                    logger.warning(f"Invalid chunk key: {chunk_key}")

        return changed

    def update_metrics_selectively(
        self, new_chunk_data: Dict[int, Dict], preserve_selected_candidates: bool = True
    ) -> bool:
        """
        Update enhanced metrics with new chunk data while preserving existing selected candidates.

        Args:
            new_chunk_data: Dictionary mapping chunk_idx to chunk validation data
            preserve_selected_candidates: Whether to preserve existing selected_candidates

        Returns:
            True if update successful, False otherwise
        """
        try:
            metrics = self.get_metrics()
            if not metrics:
                logger.error("No existing metrics found to update")
                return False

            # Preserve selected candidates if requested
            original_selected_candidates = {}
            if preserve_selected_candidates:
                original_selected_candidates = metrics.get(
                    "selected_candidates", {}
                ).copy()

            # Update chunk data
            if "chunks" not in metrics:
                metrics["chunks"] = {}

            for chunk_idx, chunk_data in new_chunk_data.items():
                chunk_key = str(chunk_idx)

                if chunk_key in metrics["chunks"]:
                    # Update existing chunk data - merge candidate data
                    existing_candidates = metrics["chunks"][chunk_key].get(
                        "candidates", {}
                    )
                    new_candidates = chunk_data.get("candidates", {})

                    # Normalize keys to strings to avoid duplicate numeric vs string keys
                    existing_candidates_str = {
                        str(k): v for k, v in existing_candidates.items()
                    }
                    new_candidates_str = {str(k): v for k, v in new_candidates.items()}

                    # Merge candidates (new override existing)
                    merged_candidates = existing_candidates_str.copy()
                    for cand_key, cand_val in new_candidates_str.items():
                        # If we already have data for this candidate, merge shallowly to preserve existing fields like final_score
                        if cand_key in merged_candidates and isinstance(merged_candidates[cand_key], dict) and isinstance(cand_val, dict):
                            merged = merged_candidates[cand_key].copy()
                            merged.update(cand_val)
                            # Ensure final_score is present if overall_quality_score exists
                            if "final_score" not in merged and "overall_quality_score" in merged:
                                merged["final_score"] = merged["overall_quality_score"]
                            merged_candidates[cand_key] = merged
                        else:
                            # Ensure final_score consistency on fresh inserts as well
                            if isinstance(cand_val, dict) and "final_score" not in cand_val and "overall_quality_score" in cand_val:
                                cand_val = cand_val.copy()
                                cand_val["final_score"] = cand_val["overall_quality_score"]
                            merged_candidates[cand_key] = cand_val

                    # Update chunk data but avoid re-setting candidates with possibly non-normalized keys
                    chunk_data_copy = chunk_data.copy()
                    if "candidates" in chunk_data_copy:
                        del chunk_data_copy["candidates"]
                    metrics["chunks"][chunk_key].update(chunk_data_copy)
                    metrics["chunks"][chunk_key]["candidates"] = merged_candidates
                else:
                    # Add new chunk data (normalize candidate keys)
                    normalized_chunk = chunk_data.copy()
                    candidates = normalized_chunk.get("candidates", {})
                    normalized_chunk["candidates"] = {str(k): v for k, v in candidates.items()}
                    # Ensure final_score consistency for all new candidates
                    for k, v in normalized_chunk["candidates"].items():
                        if isinstance(v, dict) and "final_score" not in v and "overall_quality_score" in v:
                            v["final_score"] = v["overall_quality_score"]
                    metrics["chunks"][chunk_key] = normalized_chunk

                # Update selected candidates only for new chunks or if not preserving
                if (
                    not preserve_selected_candidates
                    or chunk_key not in original_selected_candidates
                ):
                    if "best_candidate" in chunk_data:
                        metrics["selected_candidates"][chunk_key] = chunk_data[
                            "best_candidate"
                        ]

            # Restore original selected candidates if preserving
            if preserve_selected_candidates:
                for chunk_key, candidate_idx in original_selected_candidates.items():
                    metrics["selected_candidates"][chunk_key] = candidate_idx

            # Sort selected_candidates by chunk index to maintain proper order
            if "selected_candidates" in metrics:
                # Convert to int keys for sorting, then back to string keys
                sorted_selected = dict(
                    sorted(
                        metrics["selected_candidates"].items(), key=lambda x: int(x[0])
                    )
                )
                metrics["selected_candidates"] = sorted_selected

            # Update timestamp
            metrics["timestamp"] = __import__("time").time()

            # Calculate summary medians
            overall_scores = []
            similarity_scores = []
            if "chunks" in metrics:
                for chunk_data in metrics["chunks"].values():
                    if "candidates" in chunk_data:
                        for cand in chunk_data["candidates"].values():
                            if isinstance(cand, dict):
                                if "overall_quality_score" in cand:
                                    overall_scores.append(cand["overall_quality_score"])
                                if "similarity_score" in cand:
                                    similarity_scores.append(cand["similarity_score"])
            if overall_scores:
                overall_sorted = sorted(overall_scores)
                median_overall = overall_sorted[len(overall_sorted) // 2]
            else:
                median_overall = 0.0
            if similarity_scores:
                sim_sorted = sorted(similarity_scores)
                median_sim = sim_sorted[len(sim_sorted) // 2]
            else:
                median_sim = 0.0
            metrics["summary"] = {
                "median_overall_quality_score": round(median_overall, 4),
                "median_similarity_score": round(median_sim, 4)
            }

            return self.save_metrics(metrics)

        except Exception as e:
            logger.error(f"Error updating metrics selectively: {e}")
            return False
