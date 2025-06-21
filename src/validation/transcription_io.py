"""
Transcription I/O operations for whisper validation results.
Handles saving transcriptions and validation data to disk.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional, Union

from utils.file_manager.io_handlers.candidate_io import AudioCandidate

# Use absolute import pattern like existing modules



class TranscriptionIO:
    """Handles I/O operations for transcription data."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def save_transcriptions_to_disk(
        self,
        transcriptions: List[str],
        chunk_index: int,
        candidate_indices: List[int],
        validation_data: Optional[List[dict]] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[str]:
        """
        Save whisper transcriptions to text files with enhanced metrics format.

        Args:
            transcriptions: List of transcription strings
            chunk_index: Index of the chunk these transcriptions belong to
            candidate_indices: List of candidate indices corresponding to transcriptions
            validation_data: List of dicts with enhanced validation metrics
            output_dir: Directory to save transcriptions in (defaults to data/output/chunks)

        Returns:
            List of saved file paths
        """
        if output_dir is None:
            project_root = Path(__file__).resolve().parents[3]
            output_dir = project_root / "data" / "output" / "chunks"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for i, (transcription, candidate_idx) in enumerate(
            zip(transcriptions, candidate_indices)
        ):
            try:
                filename = f"chunk_{chunk_index+1:03d}_candidate_{candidate_idx+1:02d}_whisper.txt"
                filepath = output_path / filename

                val_data = (
                    validation_data[i]
                    if validation_data and i < len(validation_data)
                    else {}
                )

                whisper_score = val_data.get("whisper_score", 0.0)
                fuzzy_score = val_data.get("fuzzy_score", 0.0)
                fuzzy_method = val_data.get("fuzzy_method", "unknown")
                quality_score = val_data.get("quality_score", 0.0)
                is_valid = val_data.get("is_valid", False)
                generation_params = val_data.get("generation_params", {})
                audio_duration = val_data.get("audio_duration", 0.0)
                original_wordcount = val_data.get("original_wordcount", 0)
                transcription_wordcount = val_data.get("transcription_wordcount", 0)
                word_deviation = val_data.get("word_deviation", 0.0)
                rank = val_data.get("rank", 0)
                total_candidates = val_data.get("total_candidates", 0)

                if generation_params:
                    exag = generation_params.get("exaggeration", 0.0)
                    cfg = generation_params.get("cfg_weight", 0.0)
                    temp = generation_params.get("temperature", 0.0)
                    param_type = generation_params.get("type", "UNKNOWN")
                    params_str = f"exag={exag:.2f}, cfg={cfg:.2f}, temp={temp:.2f}, type={param_type}"
                else:
                    params_str = "N/A"

                content = f"=== WHISPER TRANSCRIPTION ===\n"
                content += f"Chunk: {chunk_index:03d}\n"
                content += f"Candidate: {candidate_idx:02d}\n"
                content += f"Whisper Score: {whisper_score:.3f}\n"
                content += f"Fuzzy Score: {fuzzy_score:.3f} (method: {fuzzy_method})\n"
                content += f"Quality Score: {quality_score:.3f}\n"
                content += f"Validation Status: {'VALID' if is_valid else 'INVALID'}\n"
                content += f"Generation Params: {params_str}\n"
                content += f"Audio Duration: {audio_duration:.2f}s\n"
                content += f"Transcription Length: {len(transcription)} characters\n"
                content += f"Word Count: {transcription_wordcount} (Original: {original_wordcount}, Deviation: {word_deviation:.1f}%)\n"
                if total_candidates > 0:
                    content += f"Rank: {rank}/{total_candidates}\n"
                content += f"{'='*50}\n\n"
                content += transcription

                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)

                saved_paths.append(str(filepath))
                self.logger.debug(
                    f"Saved enhanced transcription for chunk {chunk_index}, candidate {candidate_idx} to: {filepath}"
                )

            except Exception as e:
                self.logger.error(
                    f"Failed to save transcription for chunk {chunk_index}, candidate {candidate_idx}: {e}"
                )
                continue

        if saved_paths:
            self.logger.debug(
                f"Saved {len(saved_paths)} enhanced transcriptions for chunk {chunk_index}"
            )

        return saved_paths
