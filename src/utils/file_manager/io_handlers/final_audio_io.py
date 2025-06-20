#!/usr/bin/env python3
"""
FinalAudioIOHandler for final audio operations.
Handles saving and loading of final assembled audio and audio segments.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchaudio

logger = logging.getLogger(__name__)


class FinalAudioIOHandler:
    """Handles final audio I/O operations."""

    def __init__(self, final_dir: Path, candidates_dir: Path, config: dict):
        """
        Initialize FinalAudioIOHandler.

        Args:
            final_dir: Directory for final audio files
            candidates_dir: Directory for candidate files
            config: Configuration dictionary
        """
        self.final_dir = final_dir
        self.candidates_dir = candidates_dir
        self.config = config
        self.final_dir.mkdir(parents=True, exist_ok=True)

    def save_final_audio(self, audio: torch.Tensor, metadata: dict) -> bool:
        """Save final assembled audio with metadata."""
        try:
            text_base = Path(self.config["input"]["text_file"]).stem
            run_label = self.config["job"].get("run-label", "")
            filename = (
                f"{text_base}_{run_label}_final.wav"
                if run_label
                else f"{text_base}_final.wav"
            )

            audio_path = self.final_dir / filename
            sample_rate = self.config.get("audio", {}).get("sample_rate", 24000)
            torchaudio.save(str(audio_path), audio.unsqueeze(0), sample_rate)

            metadata_path = self.final_dir / filename.replace(".wav", "_metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as f:
                import json

                json.dump(metadata, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error saving final audio: {e}")
            return False

    def get_final_audio(self) -> Optional[torch.Tensor]:
        """Load final assembled audio."""
        final_files = list(self.final_dir.glob("*_final.wav"))
        if not final_files:
            return None

        final_file = max(final_files, key=lambda f: f.stat().st_mtime)
        try:
            audio, _ = torchaudio.load(str(final_file))
            return audio.squeeze(0)
        except Exception as e:
            logger.error(f"Error loading final audio {final_file}: {e}")
            return None

    def get_audio_segments(
        self, selected_candidates: Dict[int, int]
    ) -> List[torch.Tensor]:
        """
        Load selected audio segments for assembly.

        Args:
            selected_candidates: Dictionary mapping chunk_idx to candidate_idx

        Returns:
            List of audio tensors in chunk order
        """
        audio_segments = []

        for chunk_idx in sorted(selected_candidates.keys()):
            candidate_idx = selected_candidates[chunk_idx]

            # Ensure indices are integers (JSON loads as strings)
            chunk_idx = int(chunk_idx)
            candidate_idx = int(candidate_idx)

            # Find audio file
            chunk_dir = self.candidates_dir / f"chunk_{chunk_idx+1:03d}"
            audio_file = chunk_dir / f"candidate_{candidate_idx+1:02d}.wav"

            if audio_file.exists():
                try:
                    audio, _ = torchaudio.load(str(audio_file))

                    # VALIDATE loaded audio
                    if audio.numel() == 0:
                        raise ValueError("Empty audio file")
                    if torch.isnan(audio).any() or torch.isinf(audio).any():
                        raise ValueError("Audio contains NaN or Inf values")

                    audio_segments.append(audio.squeeze(0))  # Remove batch dimension

                except Exception as e:
                    logger.error(
                        f"🚨 CORRUPT AUDIO DETECTED in final assembly: {audio_file}"
                    )
                    logger.error(f"   Error: {e}")
                    logger.error(f"   Falling back to silence for chunk {chunk_idx}")

                    # Remove corrupt file and its validation data
                    self._remove_corrupt_candidate(chunk_idx, candidate_idx)

                    # Add silence as fallback instead of breaking the entire final audio
                    sample_rate = self.config.get("audio", {}).get("sample_rate", 24000)
                    silence_duration = 2.0  # 2 seconds of silence as fallback
                    silence = torch.zeros(int(sample_rate * silence_duration))
                    audio_segments.append(silence)
            else:
                logger.warning(f"Audio file not found: {audio_file}")
                # Add silence as fallback
                sample_rate = self.config.get("audio", {}).get("sample_rate", 24000)
                silence = torch.zeros(int(sample_rate * 0.5))
                audio_segments.append(silence)

        logger.debug(f"Loaded {len(audio_segments)} audio segments")
        return audio_segments

    def _remove_corrupt_candidate(self, chunk_idx: int, candidate_idx: int) -> bool:
        """Remove corrupt candidate file - simplified version."""
        try:
            chunk_dir = self.candidates_dir / f"chunk_{chunk_idx+1:03d}"
            audio_file = chunk_dir / f"candidate_{candidate_idx+1:02d}.wav"
            if audio_file.exists():
                audio_file.unlink()
                logger.info(f"🗑️ Removed corrupt audio file: {audio_file}")
                return True
        except Exception as e:
            logger.error(
                f"Failed to remove corrupt candidate {candidate_idx+1} for chunk {chunk_idx+1}: {e}"
            )
        return False
