"""Assembly stage handler."""

import logging
import math
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
            # Load selected candidates from task_metrics.json
            selected_candidates = self.file_manager.get_selected_candidates()
            if not selected_candidates:
                logger.error("No selected candidates found for assembly")
                return False
            
            # Convert to string keys for compatibility
            selected_candidates = {str(k): v for k, v in selected_candidates.items()}
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
                    # Load whisper metrics to get candidate scores
                    whisper_metrics = self.file_manager.get_metrics()
                    chunks_metrics = whisper_metrics.get("chunks", {})
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

                    try:
                        # Save completed selected_candidates to task_metrics.json
                        self.file_manager.update_selected_candidates(
                            {int(k): v for k, v in additions.items()}
                        )
                    except Exception:
                        logger.warning("Failed to persist completed selected_candidates; continuing with in-memory selections")

                    # Update selected_candidates with additions
                    for k, v in additions.items():
                        selected_candidates[k] = v

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
        trim_config = self.config.get("audio", {}).get("silence_trim", {})
        trim_enabled = bool(trim_config.get("enabled", False))
        trim_threshold = float(trim_config.get("threshold", 0.01))
        min_silence_duration = float(trim_config.get("min_silence_duration", 0.05))
        min_silence_samples = int(sample_rate * min_silence_duration)
        normal_silence = int(sample_rate * silence_config.get("normal", 0.2))
        paragraph_silence = int(sample_rate * silence_config.get("paragraph", 0.8))
        long_silence = int(sample_rate * silence_config.get("long", 1.5))

        assembled_segments = []

        def _create_tone(num_samples: int, device: torch.device, dtype: torch.dtype,
                         freq_hz: float = 220.0, amplitude: float = 0.1) -> torch.Tensor:
            """Create a mono sine tone of given length, frequency and amplitude."""
            if num_samples <= 0:
                return torch.zeros(0, device=device, dtype=dtype)
            t = torch.arange(num_samples, device=device, dtype=torch.float32) / float(sample_rate)
            tone = torch.sin(2.0 * math.pi * freq_hz * t) * float(amplitude)
            return tone.to(device=device, dtype=dtype)

        def _trim_edges(wave: torch.Tensor, threshold: float, min_preserve_samples: int) -> torch.Tensor:
            """Trim leading and trailing samples with |amp| <= threshold, but preserve minimum silence."""
            if wave is None or wave.numel() == 0:
                return wave
            # Ensure 1D
            if wave.ndim > 1:
                wave_1d = wave.squeeze()
            else:
                wave_1d = wave
            try:
                mask = torch.abs(wave_1d) > threshold
                if not torch.any(mask):
                    # All silent; keep as-is to avoid removing content entirely
                    return wave
                nonzero_indices = torch.nonzero(mask).squeeze(-1)
                content_start = int(nonzero_indices[0].item())
                content_end = int(nonzero_indices[-1].item()) + 1
                
                # Preserve minimum silence at start and end
                trim_start = max(0, content_start - min_preserve_samples)
                trim_end = min(len(wave_1d), content_end + min_preserve_samples)
                
                trimmed = wave_1d[trim_start:trim_end]
                # Restore dimensionality if original had extra dims
                if wave.ndim > 1:
                    trimmed = trimmed.unsqueeze(0)
                return trimmed
            except Exception:
                # On any failure, return original
                return wave

        for i, segment in enumerate(audio_segments):
            # Boundary-aware trimming: remove leading/trailing silence from segments
            seg = segment
            if trim_enabled:
                seg = _trim_edges(seg, trim_threshold, min_silence_samples)
            assembled_segments.append(seg)

            # Add silence between segments (except after the last one)
            if i < len(audio_segments) - 1:
                pause_type = boundary_pause_types[i] if i < len(boundary_pause_types) else "normal"
                # Default behavior: insert digital silence
                if pause_type == "paragraph":
                    silence = torch.zeros(paragraph_silence)
                    # pause = _create_tone(paragraph_silence, device=segment.device, dtype=segment.dtype)
                elif pause_type == "long":
                    silence = torch.zeros(long_silence)
                    # pause = _create_tone(long_silence, device=segment.device, dtype=segment.dtype)
                elif pause_type == "none":
                    silence = torch.zeros(0)
                    # pause = torch.zeros(0, device=segment.device, dtype=segment.dtype)
                else:  # "normal" or fallback
                    silence = torch.zeros(normal_silence)
                    # pause = _create_tone(normal_silence, device=segment.device, dtype=segment.dtype)
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
