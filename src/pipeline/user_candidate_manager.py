#!/usr/bin/env python3
"""
User Candidate Manager for interactive candidate selection editing.
Allows users to edit and select different audio candidates for final assembly.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Set

from utils.file_manager.file_manager import FileManager

logger = logging.getLogger(__name__)

@dataclass
class CandidateInfo:
    """Information about a single audio candidate."""

    candidate_id: int
    exaggeration: float
    cfg_weight: float
    temperature: float
    candidate_type: str
    similarity_score: float
    length_score: float
    quality_score: float
    validation_passed: bool
    is_selected: bool

@dataclass
class ChunkOverview:
    """Overview information for a chunk and its candidate selection."""

    chunk_id: int
    chunk_text: str
    selected_candidate_id: int
    has_changed: bool
    transcription: str = ""

class UserCandidateManager:
    """Manages user interaction for candidate selection editing."""

    def __init__(self, file_manager: FileManager, task_config: Dict):
        """
        Initialize UserCandidateManager.

        Args:
            file_manager: FileManager instance for task
            task_config: TaskConfig for the task
        """
        self.file_manager = file_manager
        self.task_config = task_config
        self.original_selections: Dict[str, int] = {}
        self.current_selections: Dict[str, int] = {}
        self._load_initial_data()

    def _load_initial_data(self) -> None:
        """Load initial candidate data and selections."""
        try:
            metrics = self.file_manager.get_metrics()
            self.original_selections = metrics.get("selected_candidates", {}).copy()
            self.current_selections = metrics.get("selected_candidates", {}).copy()
        except Exception as e:
            logger.error(f"Failed to load initial candidate data: {e}")
            self.original_selections = {}
            self.current_selections = {}

    def load_candidate_data(self) -> Dict:
        """Load complete candidate data from enhanced metrics."""
        return self.file_manager.get_metrics()

    def save_candidate_selection(self, chunk_idx: int, candidate_idx: int) -> bool:
        """
        Save candidate selection for a specific chunk.

        Args:
            chunk_idx: Chunk index (0-based)
            candidate_idx: Selected candidate index (0-based)

        Returns:
            True if save successful, False otherwise
        """
        try:
            # Update current selections
            self.current_selections[str(chunk_idx)] = candidate_idx

            # Save to enhanced metrics
            return self.file_manager._metrics_handler.update_selected_candidates(
                {int(k): v for k, v in self.current_selections.items()}
            )
        except Exception as e:
            logger.error(f"Failed to save candidate selection: {e}")
            return False

    def show_candidate_overview(self, task_info: Dict) -> None:
        """Display candidate overview prompt."""
        task_type = task_info.get("task_type", "task")
        print(
            f"\nSelected {task_type}: {task_info['job_name']} - {task_info['display_time']}"
        )
        print("\nCandidates selected as best matching for the final audio assembly:")
        print()
        print("Chunk:  Cand.:  Text:")

        chunks = self.file_manager.get_chunks()
        changed_chunks = self._get_changed_chunks()

        for i, chunk in enumerate(chunks):
            chunk_key = str(i)
            selected_candidate = (
                int(self.current_selections.get(chunk_key, 0)) + 1
            )  # Convert to 1-based

            # Truncate text for display and remove line breaks
            clean_text = chunk.text.replace("\n", " ").replace("\r", " ")
            # Replace multiple spaces with single space
            clean_text = " ".join(clean_text.split())
            display_text = (
                clean_text[:50] + "..." if len(clean_text) > 50 else clean_text
            )
            display_text = f'"{display_text}"'

            # Add (changed) marker if modified
            changed_marker = "(changed)" if i in changed_chunks else ""

            print(f"{i+1:<7} {selected_candidate}{changed_marker:<9} {display_text}")

        print()
        print("Which Chunk would you like to edit/review?:")
        print(f"1-{len(chunks):<3} - Select chunk")
        print("r       - Rerun task, re-assemble final audio from chunks")
        print("c       - Return")

    def show_candidate_selector(self, chunk_idx: int, task_info: Dict) -> int:
        """
        Display candidate selector prompt for a specific chunk.

        Args:
            chunk_idx: Chunk index (0-based)
            task_info: Task information for display

        Returns:
            Selected candidate index (0-based), or -1 for cancel
        """
        try:
            chunks = self.file_manager.get_chunks()
            if chunk_idx >= len(chunks):
                print(f"Invalid chunk index: {chunk_idx + 1}")
                return -1

            chunk = chunks[chunk_idx]
            candidate_infos = self.get_candidate_info(chunk_idx)

            if not candidate_infos:
                print(f"No candidates found for chunk {chunk_idx + 1}")
                return -1

            task_type = task_info.get("task_type", "task")
            print(
                f"\nSelected {task_type}: {task_info['job_name']} - {task_info['display_time']}"
            )
            print(
                f"\nSelect audio candidate for chunk: {chunk_idx+1:03d}/{len(chunks):03d}"
            )
            print()
            # Clean text for display - remove line breaks and normalize whitespace
            clean_text = chunk.text.replace("\n", " ").replace("\r", " ")
            clean_text = " ".join(clean_text.split())
            print(f'Text: "{clean_text}"')
            print()
            print(f"Number of candidates: {len(candidate_infos)}")

            current_selected = int(self.current_selections.get(str(chunk_idx), 0)) + 1
            print(f"Current selected Candidate: {current_selected}")
            print()

            # Display candidate table with proper alignment
            print(
                "Candidate:  exaggeration:  cfg_weight:  temp:    type:        sim_score:  length_score:  qty_score:  passed:"
            )
            for info in candidate_infos:
                selected_marker = "<- sel" if info.is_selected else ""
                passed_marker = "✅" if info.validation_passed else "❌"

                print(
                    f"{info.candidate_id+1:<11} "
                    f"{info.exaggeration:<13.2f} "
                    f"{info.cfg_weight:<12.2f} "
                    f"{info.temperature:<8.2f} "
                    f"{info.candidate_type:<12} "
                    f"{info.similarity_score:<11.2f} "
                    f"{info.length_score:<14.2f} "
                    f"{info.quality_score:<11.2f} "
                    f"{passed_marker:<7} {selected_marker}"
                )

            print()
            print("Select action:")
            print(f"1-{len(candidate_infos):<2} - Select candidate")
            print("c       - Return")

            while True:
                choice = input("\n> ").strip()

                if choice.lower() == "c":
                    return -1
                elif choice.isdigit():
                    candidate_num = int(choice)
                    if 1 <= candidate_num <= len(candidate_infos):
                        candidate_idx = candidate_num - 1  # Convert to 0-based
                        # Save selection immediately
                        if self.save_candidate_selection(chunk_idx, candidate_idx):
                            # Update display for next iteration
                            for info in candidate_infos:
                                info.is_selected = info.candidate_id == candidate_idx
                            print(
                                f"Selected candidate {candidate_num} for chunk {chunk_idx + 1}"
                            )
                            # Continue the loop to show updated display instead of recursion
                            candidate_infos = self.get_candidate_info(
                                chunk_idx
                            )  # Refresh data
                            task_type = task_info.get("task_type", "task")
                            print(
                                f"\nSelected {task_type}: {task_info['job_name']} - {task_info['display_time']}"
                            )
                            print(
                                f"\nSelect audio candidate for chunk: {chunk_idx+1:03d}/{len(chunks):03d}"
                            )
                            print()
                            # Clean text for display - remove line breaks and normalize whitespace
                            clean_text = chunk.text.replace("\n", " ").replace(
                                "\r", " "
                            )
                            clean_text = " ".join(clean_text.split())
                            print(f'Text: "{clean_text}"')
                            print()
                            print(f"Number of candidates: {len(candidate_infos)}")

                            current_selected = (
                                int(self.current_selections.get(str(chunk_idx), 0)) + 1
                            )
                            print(f"Current selected Candidate: {current_selected}")
                            print()

                            # Display candidate table with proper alignment
                            print(
                                "Candidate:  exaggeration:  cfg_weight:  temp:    type:        sim_score:  length_score:  qty_score:  passed:"
                            )
                            for info in candidate_infos:
                                selected_marker = "<- sel" if info.is_selected else ""
                                passed_marker = "✅" if info.validation_passed else "❌"

                                print(
                                    f"{info.candidate_id+1:<11} "
                                    f"{info.exaggeration:<13.2f} "
                                    f"{info.cfg_weight:<12.2f} "
                                    f"{info.temperature:<8.2f} "
                                    f"{info.candidate_type:<12} "
                                    f"{info.similarity_score:<11.2f} "
                                    f"{info.length_score:<14.2f} "
                                    f"{info.quality_score:<11.2f} "
                                    f"{passed_marker:<7} {selected_marker}"
                                )

                            print()
                            print("Select action:")
                            print(f"1-{len(candidate_infos):<2} - Select candidate")
                            print("c       - Return")
                        else:
                            print("Failed to save selection. Please try again.")
                    else:
                        print(
                            f"Invalid choice. Please enter 1-{len(candidate_infos)} or 'c'"
                        )
                else:
                    print(
                        f"Invalid choice. Please enter 1-{len(candidate_infos)} or 'c'"
                    )

        except Exception as e:
            logger.error(f"Error in candidate selector: {e}")
            return -1

    def get_candidate_info(self, chunk_idx: int) -> List[CandidateInfo]:
        """
        Get candidate information for a specific chunk.

        Args:
            chunk_idx: Chunk index (0-based)

        Returns:
            List of CandidateInfo objects
        """
        try:
            metrics = self.file_manager.get_metrics()
            chunk_key = str(chunk_idx)

            if chunk_key not in metrics.get("chunks", {}):
                return []

            chunk_data = metrics["chunks"][chunk_key]
            candidates = chunk_data.get("candidates", {})
            current_selected = int(self.current_selections.get(chunk_key, 0))

            candidate_infos = []

            # Load actual candidates to get generation parameters
            file_candidates = self.file_manager.get_candidates().get(chunk_idx, [])

            for candidate_key, candidate_data in candidates.items():
                candidate_idx = int(candidate_key)

                # Find matching file candidate for generation parameters
                file_candidate = None
                for fc in file_candidates:
                    if fc.candidate_idx == candidate_idx:
                        file_candidate = fc
                        break

                # Extract generation parameters
                if file_candidate and file_candidate.generation_params:
                    params = file_candidate.generation_params
                    exaggeration = params.get("exaggeration", 0.0)
                    cfg_weight = params.get("cfg_weight", 0.0)
                    temperature = params.get("temperature", 0.0)
                    candidate_type = params.get("type", "EXPRESSIVE")
                else:
                    exaggeration = cfg_weight = temperature = 0.0
                    candidate_type = "UNKNOWN"

                info = CandidateInfo(
                    candidate_id=candidate_idx,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    candidate_type=candidate_type,
                    similarity_score=candidate_data.get("similarity_score", 0.0),
                    length_score=0.97,  # Mock value, not stored in metrics
                    quality_score=candidate_data.get("overall_quality_score", 0.0),
                    validation_passed=candidate_data.get("overall_quality_score", 0.0)
                    > 0.6,
                    is_selected=(candidate_idx == current_selected),
                )

                candidate_infos.append(info)

            # Sort by candidate_id for consistent display
            candidate_infos.sort(key=lambda x: x.candidate_id)
            return candidate_infos

        except Exception as e:
            logger.error(f"Error getting candidate info for chunk {chunk_idx}: {e}")
            return []

    def _get_changed_chunks(self) -> Set[int]:
        """Get set of chunk indices that have been changed by user."""
        changed = set()

        for chunk_key in set(self.original_selections.keys()) | set(
            self.current_selections.keys()
        ):
            original = self.original_selections.get(chunk_key)
            current = self.current_selections.get(chunk_key)

            if original != current:
                try:
                    changed.add(int(chunk_key))
                except ValueError:
                    pass

        return changed

    def has_changes(self) -> bool:
        """Check if user has made any changes to candidate selections."""
        return len(self._get_changed_chunks()) > 0

    def get_candidate_overview(self) -> List[ChunkOverview]:
        """Get overview data for all chunks and their selected candidates."""
        overview = []
        chunks = self.file_manager.get_chunks()

        for i, chunk in enumerate(chunks):
            chunk_key = str(i)
            selected_candidate_id = int(self.current_selections.get(chunk_key, 0))
            has_changed = i in self._get_changed_chunks()

            overview.append(
                ChunkOverview(
                    chunk_id=i,
                    chunk_text=chunk.text,
                    selected_candidate_id=selected_candidate_id,
                    has_changed=has_changed,
                )
            )

        return overview
