"""Assembly stage handler."""

import logging
from pathlib import Path
from typing import Any, Dict, List

import torch

from utils.config_manager import TaskConfig
from utils.file_manager.file_manager import FileManager
from chunking.base_chunker import TextChunk

logger = logging.getLogger(__name__)


class AssemblyHandler:
    """Handles assembly stage (audio concatenation and post-processing)."""

    def __init__(
        self, file_manager: FileManager, config: Dict[str, Any], task_config: TaskConfig
    ):
        self.file_manager = file_manager
        self.config = config
        self.task_config = task_config

    def execute_assembly(self) -> bool:
        """Execute assembly stage (audio concatenation and post-processing)."""
        logger.info("🎵 Starting Assembly Stage")
        try:
            logger.info("Starting assembly stage")

            # Load metrics to get selected candidates
            metrics = self.file_manager.get_metrics()
            if not metrics or "selected_candidates" not in metrics:
                logger.error("No metrics or selected candidates found for assembly")
                return False

            selected_candidates = metrics["selected_candidates"]
            logger.info(
                f"Assembling audio from {len(selected_candidates)} selected candidates"
            )

            # Load audio segments
            audio_segments = self.file_manager.get_audio_segments(selected_candidates)

            if not audio_segments:
                logger.error("No audio segments loaded for assembly")
                return False

            # Load chunks for paragraph break information
            chunks = self.file_manager.get_chunks()
            has_paragraph_breaks = [chunk.has_paragraph_break for chunk in chunks]

            # Assemble audio with appropriate silences
            final_audio = self._assemble_audio_with_silences(
                audio_segments, has_paragraph_breaks
            )

            # Post-processing stub:
            # Future post-processing steps can be added here.
            # The final_audio tensor is ready for processing at this point.

            """
                final_audio = self._apply_post_processing(final_audio)
                # Example:
                def _apply_post_processing(self, audio: torch.Tensor) -> torch.Tensor:
                    processed_audio = audio.clone()
                    ...
                return processed_audio
            """
            logger.info("Skipping post-processing as it is not implemented.")

            # Create metadata
            sample_rate = self.config.get("audio", {}).get("sample_rate", 24000)
            audio_duration_seconds = len(final_audio) / sample_rate

            metadata = {
                "job_name": self.task_config.job_name,
                "task_name": self.task_config.task_name,
                "run_label": self.task_config.run_label,
                "timestamp": self.task_config.timestamp,
                "total_chunks": len(chunks),
                "selected_candidates": selected_candidates,
                "audio_duration_seconds": audio_duration_seconds,
                "sample_rate": sample_rate,
            }

            # Save final audio
            if not self.file_manager.save_final_audio(final_audio, metadata):
                logger.error("Failed to save final audio")
                return False

            logger.info("✅ Assembly stage completed successfully")
            return True

        except Exception as e:
            logger.error(f"Assembly stage failed: {e}", exc_info=True)
            return False

    def _assemble_audio_with_silences(
        self, audio_segments: List[torch.Tensor], has_paragraph_breaks: List[bool]
    ) -> torch.Tensor:
        """Assemble audio segments with appropriate silences."""
        if not audio_segments:
            return torch.tensor([])

        sample_rate = self.config.get("audio", {}).get("sample_rate", 24000)
        silence_config = self.config.get("audio", {}).get("silence_duration", {})
        normal_silence = int(sample_rate * silence_config.get("normal", 0.2))
        paragraph_silence = int(sample_rate * silence_config.get("paragraph", 0.8))

        assembled_segments = []

        for i, segment in enumerate(audio_segments):
            assembled_segments.append(segment)

            # Add silence between segments (except after the last one)
            if i < len(audio_segments) - 1:
                if i < len(has_paragraph_breaks) and has_paragraph_breaks[i]:
                    silence = torch.zeros(paragraph_silence)
                else:
                    silence = torch.zeros(normal_silence)
                assembled_segments.append(silence)

        return torch.cat(assembled_segments)
