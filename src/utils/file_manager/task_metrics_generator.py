#!/usr/bin/env python3
"""
TaskMetricsGenerator for creating comprehensive task overview metrics.
Generates task_metrics.json with all chunks, selected candidates, and runtime information.
"""

import json
import logging
import statistics
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)

# Marker field in task_metrics.json that lists chunk-keys (1-based, as strings)
# whose ``selected_candidate`` was explicitly chosen by the user via the
# Audio User Selection Editor. Selections in this set are NEVER overwritten by
# automatic re-validation; they are only invalidated when the underlying
# candidates are re-rendered (see ``clear_user_selections``).
USER_SELECTION_FIELD = "user_selected_chunks"


class TaskMetricsGenerator:
    """Generates comprehensive task metrics overview."""

    def __init__(self, task_directory: Path, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TaskMetricsGenerator.

        Args:
            task_directory: Main task directory
            config: Task configuration (optional, will be loaded if not provided)
        """
        self.task_directory = task_directory
        self.whisper_dir = task_directory / "whisper"
        self.texts_dir = task_directory / "texts"
        self.candidates_dir = task_directory / "candidates"
        self.config = config

    def generate_task_metrics(self) -> bool:
        """
        Generate comprehensive task metrics overview.

        Returns:
            True if generation successful, False otherwise
        """
        try:
            logger.info("📊 Generating task metrics overview")

            # Load source data
            whisper_metrics = self._load_whisper_metrics()
            chunks_metadata = self._load_chunks_metadata()
            task_runtime = self._load_task_runtime()
            config = self._load_config()

            if not whisper_metrics:
                logger.warning("No whisper metrics found - cannot generate task metrics")
                return False

            # Generate task metrics structure
            task_metrics = self._build_task_metrics(
                whisper_metrics, chunks_metadata, task_runtime, config
            )

            # Save task metrics
            output_path = self.task_directory / "task_metrics.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(task_metrics, f, indent=2, ensure_ascii=False)

            logger.debug(f"✅ Task metrics saved to: {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to generate task metrics: {e}")
            return False

    def migrate_selected_candidates_from_whisper(self) -> bool:
        """
        If whisper_metrics.json contains selected_candidates (legacy),
        migrate them into task_metrics.json and remove them from whisper metrics.

        Returns:
            True if migration performed successfully or nothing to do, False on error.
        """
        try:
            whisper_path = self.whisper_dir / "whisper_metrics.json"
            if not whisper_path.exists():
                logger.info("Legacy selection migration: no whisper_metrics.json - nothing to do")
                return True

            with open(whisper_path, "r", encoding="utf-8") as f:
                whisper_metrics = json.load(f)

            legacy_sel = whisper_metrics.get("selected_candidates", {})
            if not legacy_sel:
                logger.info("Legacy selection migration: no legacy selected_candidates present")
                return True

            # Ensure task_metrics.json exists and is up-to-date
            if not self.generate_task_metrics():
                logger.debug("Legacy selection migration: task_metrics generation returned False")

            # Convert legacy selections (string->int, 0-based) and persist into task_metrics.json
            migrated_entries = 0
            invalid_entries = 0
            try:
                selections_0based = {}
                for k, v in legacy_sel.items():
                    try:
                        chunk_idx_0 = int(k)
                        cand_idx_0 = int(v)
                        selections_0based[chunk_idx_0] = cand_idx_0
                        migrated_entries += 1
                    except Exception:
                        invalid_entries += 1
                        continue
                if selections_0based:
                    if not self.update_selected_candidates(selections_0based, source="auto"):
                        logger.debug("Legacy selection migration: update_selected_candidates returned False")
            except Exception:
                # Count as invalid batch, but continue to strip legacy field below
                invalid_entries += 1

            # Strip selected_candidates from whisper metrics regardless
            try:
                whisper_metrics.pop("selected_candidates", None)
                with open(whisper_path, "w", encoding="utf-8") as f:
                    json.dump(whisper_metrics, f, indent=2, ensure_ascii=False)
            except Exception:
                # Keep noise low; file will be rewritten on next validation anyway
                pass

            logger.info(
                "Legacy selection migration: migrated %d entries, invalid %d",
                migrated_entries,
                invalid_entries,
            )
            return True

        except Exception:
            # Treat as no-op with single summary line to avoid noisy warnings in gap-filling runs
            logger.info("Legacy selection migration: skipped due to read/parse error")
            return True

    def get_selected_candidates(self) -> Dict[int, int]:
        """
        Get selected candidates from task_metrics.json (0-based indexing).
        
        Returns:
            Dictionary mapping chunk_idx to candidate_idx (0-based)
        """
        try:
            task_metrics_path = self.task_directory / "task_metrics.json"
            if not task_metrics_path.exists():
                return {}

            with open(task_metrics_path, "r", encoding="utf-8") as f:
                task_metrics = json.load(f)

            # Convert 1-based to 0-based indexing
            selected_candidates_0based = {}
            selected_candidates_1based = task_metrics.get("selected_candidates", {})
            
            for chunk_key_1based, candidate_idx_1based in selected_candidates_1based.items():
                chunk_idx_0based = int(chunk_key_1based) - 1
                candidate_idx_0based = int(candidate_idx_1based) - 1
                selected_candidates_0based[chunk_idx_0based] = candidate_idx_0based

            return selected_candidates_0based

        except Exception as e:
            logger.error(f"Failed to load selected candidates from task_metrics.json: {e}")
            return {}

    def update_selected_candidates(
        self, selections: Dict[int, int], source: str = "user"
    ) -> bool:
        """
        Update selected candidates in task_metrics.json.

        Args:
            selections: Dictionary mapping chunk_idx to candidate_idx (0-based)
            source: Origin of the selection.
                - ``"user"`` (default): explicit selection from the Audio User
                  Selection Editor. The chunk-key is added to
                  ``user_selected_chunks`` so future automatic re-validation
                  will NOT override it.
                - ``"auto"``: selection chosen by the pipeline (best_candidate
                  fallback / re-validation realignment). The chunk-key is
                  removed from ``user_selected_chunks`` if present.

        Returns:
            True if update successful, False otherwise.

        In addition to updating the ``selected_candidates`` map and the per-chunk
        ``selected_candidate`` field, this method also rebuilds the affected
        chunks' ``candidate_metrics`` (audio_filename, generation_params,
        validation, prosody, final_selection_score, gates, audio_duration) from
        the latest ``whisper/whisper_metrics.json``. This keeps task_metrics
        internally consistent after a selection change instead of leaving the
        old candidate's metrics behind.
        """
        try:
            task_metrics_path = self.task_directory / "task_metrics.json"

            if task_metrics_path.exists():
                with open(task_metrics_path, "r", encoding="utf-8") as f:
                    task_metrics = json.load(f)
            else:
                logger.warning("task_metrics.json not found - cannot update selected candidates")
                return False

            normalized_source = "user" if str(source).lower() == "user" else "auto"

            selected_candidates_1based = dict(task_metrics.get("selected_candidates", {}) or {})
            user_marked: Set[str] = set()
            for k in (task_metrics.get(USER_SELECTION_FIELD) or []):
                try:
                    user_marked.add(str(int(k)))
                except Exception:
                    continue

            for chunk_idx_0based, candidate_idx_0based in selections.items():
                chunk_key_1based = str(int(chunk_idx_0based) + 1)
                candidate_idx_1based = int(candidate_idx_0based) + 1
                selected_candidates_1based[chunk_key_1based] = candidate_idx_1based
                if normalized_source == "user":
                    user_marked.add(chunk_key_1based)
                else:
                    user_marked.discard(chunk_key_1based)

            task_metrics["selected_candidates"] = selected_candidates_1based
            task_metrics[USER_SELECTION_FIELD] = sorted(user_marked, key=lambda k: int(k))

            whisper_metrics = self._load_whisper_metrics()
            config = self._load_config() or {}
            chunks = task_metrics.get("chunks", [])

            for chunk_idx_0based, candidate_idx_0based in selections.items():
                chunk_idx_1based = int(chunk_idx_0based) + 1
                candidate_idx_1based = int(candidate_idx_0based) + 1

                chunk_entry = None
                for chunk in chunks:
                    if chunk.get("chunk_meta", {}).get("idx") == chunk_idx_1based:
                        chunk_entry = chunk
                        break
                if chunk_entry is None:
                    continue

                chunk_entry.setdefault("candidates", {})["selected_candidate"] = candidate_idx_1based
                self._refresh_candidate_metrics_inplace(
                    chunk_entry,
                    chunk_idx_0based=int(chunk_idx_0based),
                    selected_candidate_0based=int(candidate_idx_0based),
                    whisper_metrics=whisper_metrics,
                    config=config,
                )

            with open(task_metrics_path, "w", encoding="utf-8") as f:
                json.dump(task_metrics, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            logger.error(f"Failed to update selected candidates in task_metrics.json: {e}")
            return False

    def get_user_selected_chunks(self) -> Set[int]:
        """Return the set of 0-based chunk indices marked as user selections.

        Reads the ``user_selected_chunks`` field from task_metrics.json.
        Returns an empty set if the file or field is missing (legacy tasks);
        in that case all selections are treated as automatic and may be
        realigned by re-validation.
        """
        try:
            task_metrics_path = self.task_directory / "task_metrics.json"
            if not task_metrics_path.exists():
                return set()
            with open(task_metrics_path, "r", encoding="utf-8") as f:
                tm = json.load(f)
            out: Set[int] = set()
            for k in (tm.get(USER_SELECTION_FIELD) or []):
                try:
                    out.add(int(k) - 1)
                except Exception:
                    continue
            return out
        except Exception as e:
            logger.debug(f"get_user_selected_chunks failed: {e}")
            return set()

    def clear_user_selections(self, chunk_indices_0based: Iterable[int]) -> bool:
        """Remove chunks from ``user_selected_chunks`` (e.g. after re-rendering).

        After candidate WAVs are re-rendered, the previous user selection no
        longer refers to the same audio content, so we drop the user marker.
        The next re-validation will realign these chunks to the new
        ``best_candidate`` automatically.

        The ``selected_candidates`` map itself is left untouched here; the
        actual realignment to a new best candidate happens in
        ``update_selected_candidates(..., source="auto")`` or in the next
        ``generate_task_metrics`` rebuild.
        """
        try:
            task_metrics_path = self.task_directory / "task_metrics.json"
            if not task_metrics_path.exists():
                return True
            with open(task_metrics_path, "r", encoding="utf-8") as f:
                tm = json.load(f)
            existing = set()
            for k in (tm.get(USER_SELECTION_FIELD) or []):
                try:
                    existing.add(str(int(k)))
                except Exception:
                    continue
            to_drop = {str(int(i) + 1) for i in chunk_indices_0based}
            new_set = existing - to_drop
            if new_set == existing:
                return True
            tm[USER_SELECTION_FIELD] = sorted(new_set, key=lambda k: int(k))
            with open(task_metrics_path, "w", encoding="utf-8") as f:
                json.dump(tm, f, indent=2, ensure_ascii=False)
            logger.info(
                f"Cleared user selections for chunks: {sorted(int(k) for k in (existing & to_drop))}"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to clear user selections: {e}")
            return False

    def _refresh_candidate_metrics_inplace(
        self,
        chunk_entry: Dict[str, Any],
        chunk_idx_0based: int,
        selected_candidate_0based: int,
        whisper_metrics: Dict[str, Any],
        config: Dict[str, Any],
    ) -> None:
        """Rewrite ``chunk_entry["candidates"]["candidate_metrics"]`` from the
        latest whisper data so audio_filename/validation/prosody/final_score
        match the (possibly changed) selection.

        Quietly leaves the existing block intact when whisper data for the
        target candidate is missing; this avoids destroying the previous
        record before re-validation has populated new data.
        """
        whisper_chunks = whisper_metrics.get("chunks", {}) if isinstance(whisper_metrics, dict) else {}
        whisper_chunk = whisper_chunks.get(str(chunk_idx_0based), {}) or {}
        candidates_data = whisper_chunk.get("candidates", {}) or {}
        sel_data = candidates_data.get(str(selected_candidate_0based), {}) or {}
        if not isinstance(sel_data, dict) or not sel_data:
            return

        selected_1based = selected_candidate_0based + 1
        cand_block = chunk_entry.setdefault("candidates", {})
        cand_block["candidate_metrics"] = {
            "audio_duration": (
                self._get_candidate_audio_duration(chunk_idx_0based, selected_candidate_0based)
                or sel_data.get("audio_duration")
                or (sel_data.get("quality_details") or {}).get("audio_duration")
                or 0.0
            ),
            "audio_filename": f"candidate_{selected_1based:02d}.wav",
            "generation_params": self._get_selected_candidate_generation_params(
                sel_data, chunk_idx_0based, selected_candidate_0based
            ),
            "validation": self._extract_validation_data(sel_data),
            "prosody": sel_data.get("prosody"),
            "final_selection_score": sel_data.get("final_selection_score"),
            "passes_mos_gate": sel_data.get("passes_mos_gate"),
            "passes_similarity_gate": sel_data.get("passes_similarity_gate"),
        }

    def _load_whisper_metrics(self) -> Dict[str, Any]:
        """Load whisper metrics data."""
        whisper_path = self.whisper_dir / "whisper_metrics.json"
        if not whisper_path.exists():
            logger.debug(f"Whisper metrics not found: {whisper_path}")
            return {}

        try:
            with open(whisper_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load whisper metrics: {e}")
            return {}

    def _load_chunks_metadata(self) -> Dict[str, Any]:
        """Load chunks metadata."""
        chunks_path = self.texts_dir / "chunks_metadata.json"
        if not chunks_path.exists():
            logger.debug(f"Chunks metadata not found: {chunks_path}")
            return {}

        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load chunks metadata: {e}")
            return {}

    def _load_task_runtime(self) -> Dict[str, Any]:
        """Load task runtime data."""
        runtime_path = self.task_directory / "task_runtime.json"
        if not runtime_path.exists():
            logger.debug(f"Task runtime not found: {runtime_path}")
            return {}

        try:
            with open(runtime_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load task runtime: {e}")
            return {}

    def _load_config(self) -> Dict[str, Any]:
        """Load task configuration."""
        if self.config:
            return self.config
        
        # Try to load from task_config.yaml
        config_path = self.task_directory / "task_config.yaml"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load task config: {e}")
        
        logger.warning("No configuration available for parameter range calculation")
        return {}

    def _build_task_metrics(
        self,
        whisper_metrics: Dict[str, Any],
        chunks_metadata: Dict[str, Any],
        task_runtime: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build comprehensive task metrics structure."""
        # Preserve existing user selections from task_metrics.json when present.
        # Only fill in missing selections from whisper metrics (best_candidate / best score fallback).
        selected_candidates_1based: Dict[str, int] = {}

        existing_task_metrics_path = self.task_directory / "task_metrics.json"
        existing_selected_candidates_1based: Dict[str, int] = {}
        existing_user_selected_chunks: Set[str] = set()
        if existing_task_metrics_path.exists():
            try:
                with open(existing_task_metrics_path, "r", encoding="utf-8") as f:
                    existing_task_metrics = json.load(f)
                    esc = existing_task_metrics.get("selected_candidates", {})
                    if isinstance(esc, dict):
                        # ensure values are ints
                        for k, v in esc.items():
                            try:
                                selected_candidates_1based[str(int(k))] = int(v)
                            except Exception:
                                # skip invalid entries
                                pass
                        existing_selected_candidates_1based = dict(selected_candidates_1based)
                    for k in (existing_task_metrics.get(USER_SELECTION_FIELD) or []):
                        try:
                            existing_user_selected_chunks.add(str(int(k)))
                        except Exception:
                            continue
            except Exception:
                # If loading fails, proceed without existing selections
                existing_selected_candidates_1based = {}
                existing_user_selected_chunks = set()

        # Build chunks array
        chunks = []
        whisper_chunks = whisper_metrics.get("chunks", {})
        chunks_meta_list = chunks_metadata.get("chunks", [])

        # Iterate chunks in ascending numeric order of their keys for stable, readable output
        chunk_keys_sorted = sorted(whisper_chunks.keys(), key=lambda k: int(k))

        for chunk_key in chunk_keys_sorted:
            whisper_chunk_data = whisper_chunks.get(chunk_key, {})
            chunk_idx_0based = int(chunk_key)
            chunk_idx_1based = chunk_idx_0based + 1

            # Find corresponding chunk metadata
            chunk_meta = None
            for meta in chunks_meta_list:
                if meta.get("idx") == chunk_idx_0based:
                    chunk_meta = meta
                    break

            if not chunk_meta:
                logger.warning(f"No metadata found for chunk {chunk_idx_1based}")
                continue

            # Determine selected candidate: prefer existing user selection if valid; else compute fallback.
            selected_candidate_0based = 0
            # 1) Try existing selection
            existing_key_1based = str(chunk_idx_1based)
            if existing_key_1based in existing_selected_candidates_1based:
                try:
                    existing_cand_1based = int(existing_selected_candidates_1based[existing_key_1based])
                    cand_idx_0 = max(0, existing_cand_1based - 1)
                    # Validate that candidate exists in whisper data
                    candidates_map = whisper_chunk_data.get("candidates", {}) if isinstance(whisper_chunk_data, dict) else {}
                    if str(cand_idx_0) in candidates_map:
                        selected_candidate_0based = cand_idx_0
                    else:
                        # User selection no longer matches an available candidate
                        # (e.g. after re-rendering with fewer candidates). Drop
                        # the user marker so the next re-validation may realign
                        # this chunk to the new best candidate without leaving
                        # an inconsistent record behind.
                        existing_user_selected_chunks.discard(existing_key_1based)
                        raise KeyError("Existing selection not found in whisper candidates")
                except Exception:
                    # 2) Compute fallback from whisper (best_candidate or by score)
                    try:
                        if isinstance(whisper_chunk_data, dict):
                            if "best_candidate" in whisper_chunk_data and whisper_chunk_data["best_candidate"] is not None:
                                selected_candidate_0based = int(whisper_chunk_data["best_candidate"])  # already 0-based
                            else:
                                cand_map = whisper_chunk_data.get("candidates", {})
                                best_idx = 0
                                best_score = float("-inf")
                                for cand_k, cand_v in cand_map.items():
                                    try:
                                        cand_idx_int = int(cand_k)
                                    except Exception:
                                        continue
                                    score = 0.0
                                    if isinstance(cand_v, dict):
                                        # Prefer new final_selection_score, fallback to legacy scores
                                        try:
                                            raw_score = (
                                                cand_v.get("final_selection_score")
                                                or cand_v.get("overall_quality_score")
                                                or cand_v.get("final_score")
                                                or 0.0
                                            )
                                            score = float(raw_score)
                                        except Exception:
                                            score = 0.0
                                    if score > best_score:
                                        best_score = score
                                        best_idx = cand_idx_int
                                selected_candidate_0based = int(best_idx)
                    except Exception:
                        selected_candidate_0based = 0
            else:
                # No existing selection; compute fallback from whisper (best)
                try:
                    if isinstance(whisper_chunk_data, dict):
                        if "best_candidate" in whisper_chunk_data and whisper_chunk_data["best_candidate"] is not None:
                            selected_candidate_0based = int(whisper_chunk_data["best_candidate"])  # already 0-based
                        else:
                            cand_map = whisper_chunk_data.get("candidates", {})
                            best_idx = 0
                            best_score = float("-inf")
                            for cand_k, cand_v in cand_map.items():
                                try:
                                    cand_idx_int = int(cand_k)
                                except Exception:
                                    continue
                                score = 0.0
                                if isinstance(cand_v, dict):
                                    # Prefer new final_selection_score, fallback to legacy scores
                                    try:
                                        raw_score = (
                                            cand_v.get("final_selection_score")
                                            or cand_v.get("overall_quality_score")
                                            or cand_v.get("final_score")
                                            or 0.0
                                        )
                                        score = float(raw_score)
                                    except Exception:
                                        score = 0.0
                                if score > best_score:
                                    best_score = score
                                    best_idx = cand_idx_int
                            selected_candidate_0based = int(best_idx)
                except Exception:
                    selected_candidate_0based = 0

            selected_candidate_1based = selected_candidate_0based + 1
            # Record into global selected_candidates map (1-based keys/values)
            selected_candidates_1based[str(chunk_idx_1based)] = selected_candidate_1based

            # Get candidate metrics
            candidates_data = whisper_chunk_data.get("candidates", {})
            selected_candidate_data = candidates_data.get(str(selected_candidate_0based), {})

            # Calculate chunk-level metrics
            all_candidates = list(candidates_data.values())
            scores: List[float] = []
            for cand in all_candidates:
                if isinstance(cand, dict):
                    val = cand.get("overall_quality_score")
                    if val is None:
                        val = cand.get("final_score")
                    scores.append(float(0.0 if val is None else float(val)))

            chunk_lowest_score = min(scores) if scores else 0.0
            chunk_best_score = max(scores) if scores else 0.0

            # Calculate parameter ranges for this chunk based on speaker configuration
            speaker_id = chunk_meta.get("speaker_id", "")
            parameter_ranges = self._calculate_parameter_ranges(speaker_id, config)

            # Get chunk text and normalize line breaks: prefer 'text', fallback to 'chunk_text'
            chunk_text = whisper_chunk_data.get("text", whisper_chunk_data.get("chunk_text", ""))
            # Replace line breaks with \n and normalize whitespace
            normalized_text = chunk_text.replace("\r\n", "\n").replace("\r", "\n")
            # Replace multiple consecutive newlines with double newlines
            import re
            normalized_text = re.sub(r'\n{3,}', '\n\n', normalized_text)
            
            # Build chunk structure
            chunk_data = {
                "chunk_meta": {
                    "idx": chunk_idx_1based,
                    "text_length": chunk_meta.get("text_length", 0),
                    "speaker_id": chunk_meta.get("speaker_id", ""),
                    "text": normalized_text,
                    "is_paragraph_break": chunk_meta.get("is_paragraph_break", False),
                    "speaker_transition": chunk_meta.get("speaker_transition", False),
                    "speaker_transition_context": chunk_meta.get("speaker_transition_context"),
                },
                "candidates": {
                    "total_candidates": len(candidates_data),
                    "chunk_lowest_score": round(chunk_lowest_score, 4),
                    "chunk_best_score": round(chunk_best_score, 4),
                    "selected_candidate": selected_candidate_1based,
                    "exaggeration_range": parameter_ranges["exaggeration_range"],
                    "cfg_weight_range": parameter_ranges["cfg_weight_range"],
                    "temperature_range": parameter_ranges["temperature_range"],
                    "candidate_metrics": {
                        "audio_duration": (
                            # Ground truth: populated by CandidateIOHandler when the
                            # wav file is written (see candidates_metadata.json).
                            self._get_candidate_audio_duration(chunk_idx_0based, selected_candidate_0based)
                            # Backwards-compat fallbacks if metadata is stale:
                            or selected_candidate_data.get("audio_duration")
                            or (selected_candidate_data.get("quality_details") or {}).get("audio_duration")
                            or 0.0
                        ),
                        "audio_filename": f"candidate_{selected_candidate_1based:02d}.wav",
                        "generation_params": self._get_selected_candidate_generation_params(
                            selected_candidate_data, chunk_idx_0based, selected_candidate_0based
                        ),
                        "validation": self._extract_validation_data(selected_candidate_data),
                        # Prosody/MOS and selection diagnostics for the selected candidate
                        "prosody": selected_candidate_data.get("prosody"),
                        "final_selection_score": selected_candidate_data.get("final_selection_score"),
                        "passes_mos_gate": selected_candidate_data.get("passes_mos_gate"),
                        "passes_similarity_gate": selected_candidate_data.get("passes_similarity_gate"),
                    },
                },
            }

            chunks.append(chunk_data)

        # Ensure final chunks list is strictly ordered by chunk index (1-based in meta)
        try:
            chunks.sort(key=lambda c: int(c.get("chunk_meta", {}).get("idx", 0)))
        except Exception:
            # Fallback: keep existing order if sorting fails for any reason
            pass

        # Calculate summary statistics
        all_overall_scores = []
        all_similarity_scores = []
        for chunk_data in chunks:
            validation = chunk_data["candidates"]["candidate_metrics"]["validation"]
            all_overall_scores.append(validation["overall_quality_score"])
            all_similarity_scores.append(validation["whisper_similarity"])

        summary = {
            "median_overall_quality_score": round(
                statistics.median(all_overall_scores) if all_overall_scores else 0.0, 4
            ),
            "median_similarity_score": round(
                statistics.median(all_similarity_scores) if all_similarity_scores else 0.0, 4
            ),
        }

        # Derive high-level identifiers
        def _derive_job_task_info() -> Dict[str, Any]:
            job_name = self.task_directory.parent.name if self.task_directory.parent else ""
            task_name = self.task_directory.name
            run_label = ""
            timestamp = ""
            try:
                if isinstance(config, dict):
                    run_label = (
                        config.get("job", {}).get("run-label", "")
                        if isinstance(config.get("job", {}), dict)
                        else ""
                    ) or ""
            except Exception:
                run_label = ""

            try:
                m = re.search(r"_(\d{8}_\d{6})$", task_name)
                if m:
                    timestamp = m.group(1)
            except Exception:
                timestamp = ""

            return {
                "job_name": job_name,
                "task_name": task_name,
                "run_label": run_label,
                "timestamp": timestamp,
            }

        def _compute_final_audio_duration_seconds() -> float:
            try:
                final_dir = self.task_directory / "final"
                final_files = sorted(
                    final_dir.glob("*_final.wav"), key=lambda p: p.stat().st_mtime
                )
                if not final_files:
                    return 0.0
                final_file = final_files[-1]
                # Prefer torchaudio for robust metadata; fallback to wave
                try:
                    import torchaudio  # type: ignore

                    info = torchaudio.info(str(final_file))
                    num_frames = getattr(info, "num_frames", None)
                    sample_rate = getattr(info, "sample_rate", None)
                    if num_frames is not None and sample_rate:
                        return float(num_frames) / float(sample_rate)
                except Exception:
                    try:
                        import wave

                        with wave.open(str(final_file), "rb") as w:
                            frames = w.getnframes()
                            framerate = w.getframerate()
                            if framerate:
                                return float(frames) / float(framerate)
                    except Exception:
                        return 0.0
            except Exception:
                return 0.0
            return 0.0

        job_task_info = _derive_job_task_info()
        audio_duration_seconds = _compute_final_audio_duration_seconds()

        # Build final structure
        task_metrics = {
            "job_name": job_task_info.get("job_name", ""),
            "task_name": job_task_info.get("task_name", ""),
            "run_label": job_task_info.get("run_label", ""),
            "timestamp": job_task_info.get("timestamp", ""),
            "audio_duration_seconds": audio_duration_seconds,
            "total_chunks": len(chunks),
            "selected_candidates": selected_candidates_1based,
            USER_SELECTION_FIELD: sorted(
                # Only keep markers for chunks that actually have a selection
                # in the rebuilt map, so stale markers cannot survive.
                (k for k in existing_user_selected_chunks if k in selected_candidates_1based),
                key=lambda k: int(k),
            ),
            "chunks": chunks,
            "summary": summary,
            "task_runtime": task_runtime,
        }

        return task_metrics

    def _calculate_parameter_ranges(self, speaker_id: str, config: Dict[str, Any]) -> Dict[str, str]:
        """
        Calculate parameter ranges based on speaker configuration and deviations.
        
        Args:
            speaker_id: Speaker ID for this chunk
            config: Task configuration containing speaker definitions
            
        Returns:
            Dictionary with parameter range strings
        """
        try:
            # Get speaker configuration
            speakers = config.get("generation", {}).get("speakers", [])
            speaker_config = None
            
            # Find speaker config
            for speaker in speakers:
                if speaker.get("id") == speaker_id:
                    speaker_config = speaker
                    break
            
            if not speaker_config:
                logger.warning(f"No speaker configuration found for speaker_id: {speaker_id}")
                return {
                    "exaggeration_range": "0.000 – 0.000",
                    "cfg_weight_range": "0.000 – 0.000", 
                    "temperature_range": "0.000 – 0.000",
                }
            
            # Get TTS parameters
            tts_params = speaker_config.get("tts_params", {})
            
            # Get base parameters
            exaggeration = float(tts_params.get("exaggeration", 0.0))
            cfg_weight = float(tts_params.get("cfg_weight", 0.0))
            temperature = float(tts_params.get("temperature", 0.0))
            
            # Get deviation parameters
            exaggeration_max_deviation = float(tts_params.get("exaggeration_max_deviation", 0.0))
            cfg_weight_max_deviation = float(tts_params.get("cfg_weight_max_deviation", 0.0))
            temperature_max_deviation = float(tts_params.get("temperature_max_deviation", 0.0))
            
            # Calculate ranges according to RAMP strategy:
            # exaggeration: RAMP-DOWN from MAX (config) to MIN (config - max_deviation)
            # cfg_weight: RAMP-UP from MIN (config) to MAX (config + max_deviation)
            # temperature: RAMP-UP from MIN (config) to MAX (config + max_deviation)
            
            exaggeration_min = exaggeration - exaggeration_max_deviation
            exaggeration_max = exaggeration
            
            cfg_weight_min = cfg_weight
            cfg_weight_max = cfg_weight + cfg_weight_max_deviation
            
            temperature_min = temperature
            temperature_max = temperature + temperature_max_deviation
            
            return {
                "exaggeration_range": f"{exaggeration_max:.3f} – {exaggeration_min:.3f}",
                "cfg_weight_range": f"{cfg_weight_min:.3f} – {cfg_weight_max:.3f}",
                "temperature_range": f"{temperature_min:.3f} – {temperature_max:.3f}",
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate parameter ranges for speaker {speaker_id}: {e}")
            return {
                "exaggeration_range": "0.000 – 0.000",
                "cfg_weight_range": "0.000 – 0.000", 
                "temperature_range": "0.000 – 0.000",
            }

    def _get_candidate_audio_duration(
        self, chunk_idx_0based: int, candidate_idx_0based: int
    ) -> Optional[float]:
        """Ground-truth duration from ``candidates_metadata.json`` (written by
        :class:`CandidateIOHandler` at save time). Returns ``None`` when the
        metadata lacks the field (legacy jobs) so callers can fall back.
        """
        try:
            chunk_dir = self.candidates_dir / f"chunk_{chunk_idx_0based + 1:03d}"
            meta_path = chunk_dir / "candidates_metadata.json"
            if not meta_path.exists():
                return None
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for cand in data.get("candidates", []):
                if cand.get("candidate_idx") == candidate_idx_0based:
                    dur = cand.get("audio_duration")
                    if isinstance(dur, (int, float)) and dur > 0:
                        return float(dur)
                    return None
        except Exception as e:
            logger.debug(f"_get_candidate_audio_duration failed: {e}")
        return None

    def _get_selected_candidate_generation_params(
        self, candidate_data: Dict[str, Any], chunk_idx_0based: int, candidate_idx_0based: int
    ) -> Dict[str, Any]:
        """
        Get generation parameters for the selected candidate from candidates metadata.
        
        Args:
            candidate_data: Selected candidate data from whisper metrics
            chunk_idx_0based: Chunk index (0-based)
            candidate_idx_0based: Candidate index (0-based)
            
        Returns:
            Dictionary with generation parameters
        """
        try:
            # Load candidates metadata for this chunk
            chunk_dir = self.candidates_dir / f"chunk_{chunk_idx_0based + 1:03d}"
            candidates_meta_path = chunk_dir / "candidates_metadata.json"
            
            if not candidates_meta_path.exists():
                logger.warning(f"Candidates metadata not found: {candidates_meta_path}")
                return {}
            
            with open(candidates_meta_path, "r", encoding="utf-8") as f:
                candidates_metadata = json.load(f)
            
            # Find the specific candidate
            candidates = candidates_metadata.get("candidates", [])
            for candidate in candidates:
                if candidate.get("candidate_idx") == candidate_idx_0based:
                    generation_params = candidate.get("generation_params", {}) or {}
                    # Pass through the full set of generation parameters so that
                    # model-specific fields (e.g. VibeVoice's ``cfg_scale``,
                    # ``diffusion_steps``, ``voice_speed_factor``, ``use_sampling``
                    # or Chatterbox's ``exaggeration``/``min_p``/``repetition_penalty``)
                    # survive into task_metrics.json. ``seed`` and ``language_id``
                    # are filtered out to keep the section focused on sampling
                    # parameters.
                    excluded = {"seed", "language_id"}
                    return {
                        k: v
                        for k, v in generation_params.items()
                        if k not in excluded
                    }
            
            logger.warning(f"Candidate {candidate_idx_0based} not found in candidates metadata for chunk {chunk_idx_0based}")
            return self._get_fallback_generation_params(chunk_idx_0based)
            
        except Exception as e:
            logger.warning(f"Failed to get generation params for candidate {candidate_idx_0based} in chunk {chunk_idx_0based}: {e}")
            return self._get_fallback_generation_params(chunk_idx_0based)

    def _get_fallback_generation_params(self, chunk_idx_0based: int) -> Dict[str, Any]:
        """
        Get fallback generation parameters from speaker configuration.
        
        Args:
            chunk_idx_0based: Chunk index (0-based)
            
        Returns:
            Dictionary with fallback generation parameters
        """
        try:
            # Load chunks metadata to get speaker_id
            chunks_meta_path = self.texts_dir / "chunks_metadata.json"
            if not chunks_meta_path.exists():
                return {}
            
            with open(chunks_meta_path, "r", encoding="utf-8") as f:
                chunks_metadata = json.load(f)
            
            # Find speaker_id for this chunk
            chunks_meta_list = chunks_metadata.get("chunks", [])
            speaker_id = None
            for meta in chunks_meta_list:
                if meta.get("idx") == chunk_idx_0based:
                    speaker_id = meta.get("speaker_id")
                    break
            
            if not speaker_id:
                return {}
            
            # Load config
            config = self._load_config()
            if not config:
                return {}
            
            # Get speaker configuration
            speakers = config.get("generation", {}).get("speakers", [])
            speaker_config = None
            
            for speaker in speakers:
                if speaker.get("id") == speaker_id:
                    speaker_config = speaker
                    break
            
            if not speaker_config:
                return {}
            
            # Get TTS parameters from speaker config
            tts_params = speaker_config.get("tts_params", {})
            
            # Extract relevant parameters
            return {
                "exaggeration": tts_params.get("exaggeration", 0.0),
                "exaggeration_max_deviation": tts_params.get("exaggeration_max_deviation", 0.0),
                "cfg_weight": tts_params.get("cfg_weight", 0.0),
                "cfg_weight_max_deviation": tts_params.get("cfg_weight_max_deviation", 0.0),
                "temperature": tts_params.get("temperature", 0.0),
                "temperature_max_deviation": tts_params.get("temperature_max_deviation", 0.0),
                "repetition_penalty": tts_params.get("repetition_penalty", 1.0),
                "min_p": tts_params.get("min_p", 0.0),
                "top_p": tts_params.get("top_p", 0.0),
            }
            
        except Exception as e:
            logger.warning(f"Failed to get fallback generation params for chunk {chunk_idx_0based}: {e}")
            return {}

    def _extract_validation_data(self, candidate_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract validation data from candidate data structure.
        
        Args:
            candidate_data: Candidate data from whisper_metrics.json
            
        Returns:
            Dictionary with validation metrics
        """
        try:
            # Extract quality details
            quality_details = candidate_data.get("quality_details", {})
            individual_scores = quality_details.get("individual_scores", {}) if isinstance(quality_details, dict) else {}
            validation_metrics = quality_details.get("validation_metrics", {}) if isinstance(quality_details, dict) else {}
            
            # Extract validation data
            validation_data = {
                "length_score": individual_scores.get("length_score", 0.0),
                "whisper_similarity": individual_scores.get("similarity_score", 0.0),
                "whisper_quality": validation_metrics.get("whisper_quality", 0.0),
                "penalty_score": individual_scores.get("penalty_score", 0.0),
                "overall_quality_score": individual_scores.get("overall_score", candidate_data.get("final_score", 0.0)),
                "whisper_language_id": validation_metrics.get("whisper_language_id", ""),
                "passed_threshold": candidate_data.get("is_valid", False),
            }
            
            return validation_data
            
        except Exception as e:
            logger.warning(f"Failed to extract validation data: {e}")
            return {
                "length_score": 0.0,
                "whisper_similarity": 0.0,
                "whisper_quality": 0.0,
                "penalty_score": 0.0,
                "overall_quality_score": 0.0,
                "whisper_language_id": "",
                "passed_threshold": False,
            }
