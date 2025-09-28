"""Assembly stage handler."""

import logging
from typing import Any, Dict, List

import torch

from postprocessing.normalization import AudioNormalizer
from utils.config_manager import TaskConfig
from utils.file_manager.file_manager import FileManager

logger = logging.getLogger(__name__)

class AssemblyHandler:
    """Handles assembly stage (audio concatenation and post-processing)."""

    def __init__(
        self, file_manager: FileManager, config: Dict[str, Any], task_config: TaskConfig
    ):
        self.file_manager = file_manager
        self.config = config
        self.task_config = task_config
        
        # Initialize audio normalizer
        self.audio_normalizer = AudioNormalizer(config)

    def execute_assembly(self) -> bool:
        """Execute assembly stage (audio concatenation and post-processing)."""
        logger.info("🔧 Starting Assembly Stage")
        try:
            # Load metrics to get selected candidates
            metrics = self.file_manager.get_metrics()
            if not metrics:
                logger.error("No metrics found for assembly")
                return False

            selected_candidates = metrics.get("selected_candidates", {})
            logger.info(
                f"Assembling audio from {len(selected_candidates)} selected candidates"
            )

            # Ensure selected_candidates covers all chunks without overwriting user choices
            try:
                chunks = self.file_manager.get_chunks()
                total_chunks = len(chunks)
                if total_chunks == 0:
                    logger.error("No chunks found for assembly")
                    return False

                missing_selection_indices = [
                    idx for idx in range(total_chunks) if str(idx) not in selected_candidates
                ]

                if missing_selection_indices:
                    logger.info(
                        f"🧩 Completing missing selected_candidates for chunks: {[i+1 for i in missing_selection_indices]}"
                    )
                    chunks_metrics = metrics.get("chunks", {})
                    additions: Dict[str, int] = {}
                    for idx in missing_selection_indices:
                        key = str(idx)
                        best_idx = None
                        best_score = float("-inf")

                        if key in chunks_metrics:
                            cand_map = chunks_metrics[key].get("candidates", {})
                            for cand_key, cand_val in cand_map.items():
                                try:
                                    cand_idx_int = int(cand_key)
                                except Exception:
                                    continue
                                score = 0.0
                                if isinstance(cand_val, dict):
                                    score = float(cand_val.get("overall_quality_score", cand_val.get("final_score", 0.0)))
                                if score > best_score:
                                    best_score = score
                                    best_idx = cand_idx_int

                        if best_idx is None:
                            best_idx = 0

                        additions[key] = int(best_idx)

                    if "selected_candidates" not in metrics:
                        metrics["selected_candidates"] = {}
                    for k, v in additions.items():
                        if k not in metrics["selected_candidates"]:
                            metrics["selected_candidates"][k] = v

                    try:
                        self.file_manager.save_metrics(metrics)
                    except Exception:
                        logger.warning("Failed to persist completed selected_candidates; continuing with in-memory selections")

                    selected_candidates = metrics["selected_candidates"]

            except Exception as e:
                logger.warning(f"Failed to complete selected_candidates: {e}")

            # Load audio segments
            audio_segments = self.file_manager.get_audio_segments(selected_candidates)

            if not audio_segments:
                logger.error("No audio segments loaded for assembly")
                return False

            # Load chunks for pause boundary information
            chunks = self.file_manager.get_chunks()
            # Build boundary pause types between segments (length = len(segments) - 1)
            boundary_pause_types: List[str] = []
            for i in range(len(chunks) - 1):
                # Priority: explicit paragraph break after chunk i
                if chunks[i].has_paragraph_break:
                    # Support long paragraph breaks if marked
                    pbt = getattr(chunks[i], "paragraph_break_type", None)
                    if pbt == "long":
                        boundary_pause_types.append("long")
                    else:
                        boundary_pause_types.append("paragraph")
                    continue

                # If next chunk starts with a speaker transition, use its context
                if getattr(chunks[i + 1], "speaker_transition", False):
                    ctx = getattr(chunks[i + 1], "speaker_transition_context", None)
                    if ctx in ("paragraph", "normal", "none"):
                        boundary_pause_types.append(ctx)
                        continue

                # Check if chunk ends at a sentence boundary for intelligent pause decisions
                pause_type = self._determine_boundary_pause_type(chunks[i])
                boundary_pause_types.append(pause_type)

            # Assemble audio with appropriate silences
            final_audio = self._assemble_audio_with_silences(
                audio_segments, boundary_pause_types
            )

            # Apply post-processing (audio normalization)
            final_audio = self._apply_post_processing(final_audio)

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

    def _apply_post_processing(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply post-processing to final audio (normalization, etc.).
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Processed audio tensor
        """
        try:
            sample_rate = self.config.get("audio", {}).get("sample_rate", 24000)
            
            # Apply audio normalization
            processed_audio = self.audio_normalizer.normalize(audio, sample_rate)
            
            return processed_audio
            
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            logger.debug("Returning original audio due to post-processing failure")
            return audio

    def _assemble_audio_with_silences(
        self, audio_segments: List[torch.Tensor], boundary_pause_types: List[str]
    ) -> torch.Tensor:
        """Assemble audio segments with appropriate silences."""
        if not audio_segments:
            return torch.tensor([])

        sample_rate = self.config.get("audio", {}).get("sample_rate", 24000)
        silence_config = self.config.get("audio", {}).get("silence_duration", {})
        normal_silence = int(sample_rate * silence_config.get("normal", 0.2))
        paragraph_silence = int(sample_rate * silence_config.get("paragraph", 0.8))
        long_silence = int(sample_rate * silence_config.get("long", 1.5))

        assembled_segments = []

        for i, segment in enumerate(audio_segments):
            assembled_segments.append(segment)

            # Add silence between segments (except after the last one)
            if i < len(audio_segments) - 1:
                pause_type = boundary_pause_types[i] if i < len(boundary_pause_types) else "normal"
                if pause_type == "paragraph":
                    silence = torch.zeros(paragraph_silence)
                elif pause_type == "long":
                    silence = torch.zeros(long_silence)
                elif pause_type == "none":
                    silence = torch.zeros(0)
                else:  # "normal" or fallback
                    silence = torch.zeros(normal_silence)
                assembled_segments.append(silence)

        return torch.cat(assembled_segments)

    def _determine_boundary_pause_type(self, chunk) -> str:
        """
        Determine the appropriate pause type after a chunk based on how it ends.
        
        Args:
            chunk: TextChunk object to analyze
            
        Returns:
            Pause type: 'normal', 'none'
        """
        if not chunk or not chunk.text:
            return "normal"
            
        # Define sentence ending characters (same as ChunkValidator)
        sentence_enders = (".", "!", "?", '"', '"', "]")
        
        # Get the last non-whitespace character
        text_stripped = chunk.text.strip()
        if not text_stripped:
            return "normal"
            
        last_char = text_stripped[-1]
        
        # Check if chunk ends at a proper sentence boundary
        if last_char in sentence_enders:
            return "normal"  # Normal pause after sentence end
            
        # Check for mid-sentence punctuation that shouldn't have pauses
        mid_sentence_punct = (",", ";", ":", "—", "–", "-", "(", ")")
        if last_char in mid_sentence_punct:
            logger.debug(f"Chunk ends with mid-sentence punctuation '{last_char}' - using no pause")
            return "none"  # No pause after mid-sentence punctuation
            
        # Check if chunk ends mid-word (likely forced split)
        if last_char.isalnum():
            logger.debug(f"Chunk ends mid-word ('{last_char}') - using no pause")
            return "none"  # No pause after forced mid-word split
            
        # Default: short pause for other cases
        logger.debug(f"Chunk ends with '{last_char}' - using normal pause")
        return "normal"
