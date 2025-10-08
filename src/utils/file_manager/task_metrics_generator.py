#!/usr/bin/env python3
"""
TaskMetricsGenerator for creating comprehensive task overview metrics.
Generates task_metrics.json with all chunks, selected candidates, and runtime information.
"""

import json
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


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

    def update_selected_candidates(self, selections: Dict[int, int]) -> bool:
        """
        Update selected candidates in task_metrics.json.
        
        Args:
            selections: Dictionary mapping chunk_idx to candidate_idx (0-based)
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            task_metrics_path = self.task_directory / "task_metrics.json"
            
            # Load existing task metrics
            if task_metrics_path.exists():
                with open(task_metrics_path, "r", encoding="utf-8") as f:
                    task_metrics = json.load(f)
            else:
                logger.warning("task_metrics.json not found - cannot update selected candidates")
                return False

            # Convert 0-based to 1-based indexing
            selected_candidates_1based = {}
            for chunk_idx_0based, candidate_idx_0based in selections.items():
                chunk_key_1based = str(chunk_idx_0based + 1)
                candidate_idx_1based = candidate_idx_0based + 1
                selected_candidates_1based[chunk_key_1based] = candidate_idx_1based

            # Update selected candidates
            task_metrics["selected_candidates"] = selected_candidates_1based

            # Update chunk-specific selected_candidate values
            chunks = task_metrics.get("chunks", [])
            for chunk_idx_0based, candidate_idx_0based in selections.items():
                chunk_idx_1based = chunk_idx_0based + 1
                candidate_idx_1based = candidate_idx_0based + 1
                
                # Find the chunk in the chunks array
                for chunk in chunks:
                    if chunk.get("chunk_meta", {}).get("idx") == chunk_idx_1based:
                        chunk["candidates"]["selected_candidate"] = candidate_idx_1based
                        break

            # Save updated task metrics
            with open(task_metrics_path, "w", encoding="utf-8") as f:
                json.dump(task_metrics, f, indent=2, ensure_ascii=False)

            return True

        except Exception as e:
            logger.error(f"Failed to update selected candidates in task_metrics.json: {e}")
            return False

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
        # Convert 0-based to 1-based indexing for selected_candidates
        selected_candidates_1based = {}
        whisper_selected = whisper_metrics.get("selected_candidates", {})
        for chunk_key, candidate_idx in whisper_selected.items():
            chunk_idx_1based = int(chunk_key) + 1
            selected_candidates_1based[str(chunk_idx_1based)] = candidate_idx + 1

        # Build chunks array
        chunks = []
        whisper_chunks = whisper_metrics.get("chunks", {})
        chunks_meta_list = chunks_metadata.get("chunks", [])

        for chunk_key, whisper_chunk_data in whisper_chunks.items():
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

            # Get selected candidate for this chunk
            selected_candidate_0based = whisper_selected.get(chunk_key, 0)
            selected_candidate_1based = selected_candidate_0based + 1

            # Get candidate metrics
            candidates_data = whisper_chunk_data.get("candidates", {})
            selected_candidate_data = candidates_data.get(str(selected_candidate_0based), {})

            # Calculate chunk-level metrics
            all_candidates = list(candidates_data.values())
            scores = []
            for cand in all_candidates:
                if isinstance(cand, dict):
                    score = cand.get("overall_quality_score", cand.get("final_score", 0.0))
                    scores.append(float(score))

            chunk_lowest_score = min(scores) if scores else 0.0
            chunk_best_score = max(scores) if scores else 0.0

            # Calculate parameter ranges for this chunk based on speaker configuration
            speaker_id = chunk_meta.get("speaker_id", "")
            parameter_ranges = self._calculate_parameter_ranges(speaker_id, config)

            # Get chunk text and normalize line breaks
            chunk_text = whisper_chunk_data.get("text", "")
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
                        "audio_duration": selected_candidate_data.get("audio_duration", 0.0),
                        "audio_filename": f"candidate_{selected_candidate_1based:02d}.wav",
                        "generation_params": self._get_selected_candidate_generation_params(
                            selected_candidate_data, speaker_id, config
                        ),
                        "validation": self._extract_validation_data(selected_candidate_data),
                    },
                },
            }

            chunks.append(chunk_data)

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

        # Build final structure
        task_metrics = {
            "total_chunks": len(chunks),
            "selected_candidates": selected_candidates_1based,
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

    def _get_selected_candidate_generation_params(
        self, candidate_data: Dict[str, Any], speaker_id: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get generation parameters for the selected candidate from speaker configuration.
        
        Args:
            candidate_data: Selected candidate data
            speaker_id: Speaker ID for this chunk
            config: Task configuration containing speaker definitions
            
        Returns:
            Dictionary with generation parameters
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
                return {}
            
            # Get TTS parameters from speaker config
            tts_params = speaker_config.get("tts_params", {})
            
            # Extract relevant parameters
            generation_params = {
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
            
            return generation_params
            
        except Exception as e:
            logger.warning(f"Failed to get generation params for speaker {speaker_id}: {e}")
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
