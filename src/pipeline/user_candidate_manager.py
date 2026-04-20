#!/usr/bin/env python3
"""
User Candidate Manager for interactive candidate selection editing.
Allows users to edit and select different audio candidates for final assembly.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Set, Tuple

from utils.file_manager.file_manager import FileManager
from utils.file_manager.task_metrics_generator import TaskMetricsGenerator

logger = logging.getLogger(__name__)

# Per-model parameter columns shown in the candidate selector table.
# Each entry: (header, generation_params key, formatter, column width).
# The first column-width is also used to pad header cells.
_ParamColumn = Tuple[str, str, Callable[[Any], str], int]


def _fmt_float(width: int, decimals: int = 2) -> Callable[[Any], str]:
    def _f(v: Any) -> str:
        try:
            return f"{float(v):<{width}.{decimals}f}"
        except (TypeError, ValueError):
            return f"{'-':<{width}}"
    return _f


def _fmt_int(width: int) -> Callable[[Any], str]:
    def _f(v: Any) -> str:
        try:
            return f"{int(v):<{width}d}"
        except (TypeError, ValueError):
            return f"{'-':<{width}}"
    return _f


def _get_model_param_columns(model_type: str) -> List[_ParamColumn]:
    """Return per-model column definitions for the candidate selector table.

    The default (Chatterbox / standard / multilingual / turbo) preserves the
    historical 3-column layout (exag/cfg_w/tmp). Qwen3 and VibeVoice use 4
    columns each tailored to their most informative ramping parameters.
    """
    mt = (model_type or "").lower().strip()
    if mt == "qwen3":
        return [
            ("temp",   "temperature",           _fmt_float(6, 2), 6),
            ("topk",   "top_k",                 _fmt_int(6),      6),
            ("sub_t",  "subtalker_temperature", _fmt_float(6, 2), 6),
            ("sub_tk", "subtalker_top_k",       _fmt_int(6),      6),
        ]
    if mt in {"vibevoice", "vibevoice_1_5b", "vibevoice_q4"}:
        return [
            ("temp",   "temperature",     _fmt_float(6, 2), 6),
            ("top_p",  "top_p",           _fmt_float(6, 2), 6),
            ("cfg",    "cfg_scale",       _fmt_float(6, 2), 6),
            ("dsteps", "diffusion_steps", _fmt_int(6),      6),
        ]
    # Chatterbox family (default)
    return [
        ("exag",  "exaggeration", _fmt_float(6, 2), 6),
        ("cfg_w", "cfg_weight",   _fmt_float(6, 2), 6),
        ("tmp",   "temperature",  _fmt_float(6, 2), 6),
    ]

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
    # Prosody-related scores and final selection score (0.0..1.0)
    flow_score: float = 0.0
    liveliness_score: float = 0.0
    mos_score: float = 0.0
    wpm: float = 0.0
    prosody_score: float = 0.0
    final_selection_score: float = 0.0
    # Pre-formatted model-specific parameter cells (already padded to width).
    # Order matches the column order returned by _get_model_param_columns().
    param_cells: List[str] = field(default_factory=list)

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
        # Resolve model_type once so the candidate table can show the most
        # informative tts_params per model family (qwen3, vibevoice, chatterbox).
        try:
            self.model_type = str(
                (self.file_manager.config or {}).get("generation", {}).get("model_type", "")
            )
        except Exception:
            self.model_type = ""
        self.param_columns: List[_ParamColumn] = _get_model_param_columns(self.model_type)
        self._load_initial_data()

    def _load_initial_data(self) -> None:
        """Load initial candidate data and selections."""
        try:
            # Load selected candidates from task_metrics.json (0-based)
            selected_candidates = self.file_manager.get_selected_candidates()
            # Convert to string keys for compatibility
            self.original_selections = {str(k): v for k, v in selected_candidates.items()}
            self.current_selections = {str(k): v for k, v in selected_candidates.items()}
        except Exception as e:
            logger.error(f"Failed to load initial candidate data: {e}")
            self.original_selections = {}
            self.current_selections = {}

    def load_candidate_data(self) -> Dict:
        """Load complete candidate data from enhanced metrics."""
        return self.file_manager.get_metrics()

    def _print_candidate_table(self, candidate_infos: List[CandidateInfo]) -> None:
        """Print the candidate table using model-specific tts_param columns."""
        # Header
        param_header = " ".join(f"{h:<{w}}" for (h, _k, _f, w) in self.param_columns)
        print(
            f"{'Cand.':<6} "
            f"{param_header} "
            f"{'type':<20} "
            f"{'val_s':<7} "
            f"{'qty_s':<7} "
            f"{'wpm':<6} "
            f"{'flow':<6} "
            f"{'live':<6} "
            f"{'mos':<6} "
            f"{'pros':<6} "
            f"{'final':<7} "
            f"{'passed[s,p]':<12}"
        )
        try:
            validation_cfg = self.file_manager.config.get("validation", {})
            prosody_thr = float(
                validation_cfg.get("prosody", {}).get("thresholds", {}).get("min_prosody_score", 0.60)
            )
        except Exception:
            prosody_thr = 0.60

        for info in candidate_infos:
            selected_marker = "<- sel" if info.is_selected else ""
            sim_mark = "✅" if info.validation_passed else "❌"
            pros_mark = "✅" if float(getattr(info, "prosody_score", 0.0)) >= prosody_thr else "❌"
            passed_pair = f"{sim_mark}{pros_mark}"

            # Use pre-formatted, padded cells (or rebuild if missing for any reason).
            if info.param_cells and len(info.param_cells) == len(self.param_columns):
                params_str = " ".join(info.param_cells)
            else:
                params_str = " ".join(f"{'-':<{w}}" for (_h, _k, _f, w) in self.param_columns)

            print(
                f"{info.candidate_id+1:<6} "
                f"{params_str} "
                f"{info.candidate_type:<20} "
                f"{getattr(info, 'validation_score', 0.0):<7.2f} "
                f"{info.quality_score:<7.2f} "
                f"{getattr(info, 'wpm', 0.0):<6.0f} "
                f"{info.flow_score:<6.2f} "
                f"{info.liveliness_score:<6.2f} "
                f"{info.mos_score:<6.2f} "
                f"{info.prosody_score:<6.2f} "
                f"{info.final_selection_score:<7.2f} "
                f"{passed_pair:<12} {selected_marker}"
            )

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

            # Persist only the changed chunk and tag it as a USER selection so
            # later re-validation runs will not silently override it.
            success = self.file_manager.update_selected_candidates(
                {int(chunk_idx): int(candidate_idx)}, source="user"
            )

            return success
        except Exception as e:
            logger.error(f"Failed to save candidate selection: {e}")
            return False

    def show_candidate_overview(self, task_info: Dict) -> None:
        """Display candidate overview prompt."""
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
            Selected candidate index (0-based), or:
            -1 for cancel/return
            -2 for next chunk
            -3 for previous chunk
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

            print(
                f"\nSelect audio candidate for chunk: {chunk_idx+1:03d} / {len(chunks):03d}"
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

            self._print_candidate_table(candidate_infos)

            print()
            print("Select action:")
            print(f"1-{len(candidate_infos):<2} - Select candidate")
            if chunk_idx < len(chunks) - 1:
                print("n       - Next Chunk")
            if chunk_idx > 0:
                print("p       - Previous Chunk")
            print("c       - Return")

            while True:
                choice = input("\n> ").strip()

                if choice.lower() == "c":
                    return -1
                elif choice.lower() == "n" and chunk_idx < len(chunks) - 1:
                    return -2  # Signal next chunk
                elif choice.lower() == "p" and chunk_idx > 0:
                    return -3  # Signal previous chunk
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
                            print(
                                f"\nSelect audio candidate for chunk: {chunk_idx+1:03d} / {len(chunks):03d}"
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

                            self._print_candidate_table(candidate_infos)

                            print()
                            print("Select action:")
                            print(f"1-{len(candidate_infos):<2} - Select candidate")
                            if chunk_idx < len(chunks) - 1:
                                print("n       - Next Chunk")
                            if chunk_idx > 0:
                                print("p       - Previous Chunk")
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
                wav_exists = False
                for fc in file_candidates:
                    if fc.candidate_idx == candidate_idx:
                        # Ensure the WAV actually exists (stale metrics guard)
                        chunk_dir = self.file_manager.candidates_dir / f"chunk_{chunk_idx+1:03d}"
                        wav_path = chunk_dir / f"candidate_{candidate_idx+1:02d}.wav"
                        wav_exists = wav_path.exists()
                        if wav_exists:
                            file_candidate = fc
                            break

                # Skip candidates without existing WAV files (UI should reflect actual selectable items)
                if not wav_exists:
                    continue

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

                # Derive scores from quality_details to avoid redundancy
                quality_details = candidate_data.get("quality_details", {})
                individual_scores = quality_details.get("individual_scores", {}) if isinstance(quality_details, dict) else {}
                validation_metrics = quality_details.get("validation_metrics", {}) if isinstance(quality_details, dict) else {}

                sim_score = individual_scores.get("similarity_score", 0.0)
                val_score = validation_metrics.get("whisper_quality", 0.0)
                overall = individual_scores.get("overall_score", 0.0)

                # If quality_details contain individual length score, prefer it
                length_score = individual_scores.get("length_score", 0.0)
                if length_score is None:
                    # Fallback: approximate length_score from overall and sim weights when possible
                    # Default weights in QualityScorer: similarity 0.75, length 0.25 (updated)
                    try:
                        approx = (overall - 0.75 * sim_score) / 0.25
                        length_score = max(0.0, min(1.0, approx))
                    except Exception:
                        length_score = 0.0

                # Determine validity: prefer stored is_valid, else recompute using config thresholds (mirror WhisperValidator)
                stored_is_valid = candidate_data.get("is_valid")
                if stored_is_valid is None:
                    try:
                        validation_cfg = self.file_manager.config.get("validation", {})
                        sim_thr = float(validation_cfg.get("similarity_threshold", 0.8))
                        min_q = float(validation_cfg.get("min_quality_score", 0.75))
                        strict_validation = (sim_score >= sim_thr and val_score >= min_q)
                        flexible_validation = (
                            (val_score >= min_q + 0.02 and sim_score >= sim_thr - 0.10)
                            or (sim_score >= sim_thr + 0.02 and val_score >= min_q - 0.10)
                            or (
                                sim_score >= sim_thr - 0.05
                                and val_score >= min_q - 0.05
                                and (sim_score + val_score) >= (sim_thr + min_q) - 0.05
                            )
                        )
                        computed_is_valid = strict_validation or flexible_validation
                    except Exception:
                        computed_is_valid = overall > 0.6
                    is_valid_flag = bool(computed_is_valid)
                else:
                    is_valid_flag = bool(stored_is_valid)

                # Prosody metrics (flow, liveliness, mos, prosody) and final selection score
                prosody = candidate_data.get("prosody", {}) if isinstance(candidate_data, dict) else {}
                subscores = prosody.get("subscores", {}) if isinstance(prosody, dict) else {}
                flow = subscores.get("flow", 0.0) if isinstance(subscores, dict) else 0.0
                liveliness = subscores.get("liveliness", 0.0) if isinstance(subscores, dict) else 0.0
                mos_unit = subscores.get("mos", 0.0) if isinstance(subscores, dict) else 0.0
                prosody_score = prosody.get("prosody_score", 0.0) if isinstance(prosody, dict) else 0.0
                wpm_val = prosody.get("wpm", 0.0) if isinstance(prosody, dict) else 0.0
                final_sel = candidate_data.get("final_selection_score")
                if final_sel is None:
                    # Fallback to legacy/overall when final not present
                    final_sel = individual_scores.get("overall_score", 0.0)

                # Pre-format model-specific tts_param cells for the table
                gen_params: Dict[str, Any] = (
                    file_candidate.generation_params
                    if (file_candidate and file_candidate.generation_params)
                    else {}
                )
                param_cells = [
                    fmt(gen_params.get(key)) for (_h, key, fmt, _w) in self.param_columns
                ]

                info = CandidateInfo(
                    candidate_id=candidate_idx,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                    candidate_type=candidate_type,
                    similarity_score=sim_score,
                    length_score=float(length_score),
                    quality_score=overall,
                    validation_passed=is_valid_flag,
                    is_selected=(candidate_idx == current_selected),
                    flow_score=float(flow),
                    liveliness_score=float(liveliness),
                    mos_score=float(mos_unit),
                    wpm=float(wpm_val),
                    prosody_score=float(prosody_score),
                    final_selection_score=float(final_sel),
                    param_cells=param_cells,
                )

                # Attach validation_score dynamically for printing from val_score
                setattr(info, "validation_score", float(val_score))

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
