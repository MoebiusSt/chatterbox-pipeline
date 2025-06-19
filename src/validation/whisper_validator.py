"""
Whisper-based audio validation module for quality assurance.
Transcribes generated audio and compares against original text.
"""

import logging
import os

# Use absolute import to avoid relative import issues
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torchaudio
import whisper

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.file_manager import AudioCandidate


@dataclass
class ValidationResult:
    """Result of audio validation process."""

    is_valid: bool
    transcription: str
    similarity_score: float
    quality_score: float
    validation_time: float
    error_message: Optional[str] = None


class WhisperValidator:
    """
    Validates audio candidates using Whisper speech-to-text.
    Transcribes audio and compares against original text for quality assurance.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        similarity_threshold: float = 0.90,
        min_quality_score: float = 0.6,
    ):
        """
        Initialize WhisperValidator.

        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
            device: Device for computation ("auto", "cuda", "cpu")
            similarity_threshold: Minimum similarity score for validation from config/default_config.yaml
            min_quality_score: Minimum quality score for validation
        """
        self.model_size = model_size
        self.similarity_threshold = similarity_threshold
        self.min_quality_score = min_quality_score
        self.logger = logging.getLogger(__name__)

        # Auto-detect device if needed
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load Whisper model."""
        try:
            self.logger.info(
                f"Loading Whisper model '{self.model_size}' on device '{self.device}'..."
            )
            self.model = whisper.load_model(self.model_size, device=self.device)
            self.logger.info(f"Whisper model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe_audio(
        self, audio: torch.Tensor, sample_rate: int = 24000, language: str = "en"
    ) -> str:
        """
        Transcribe audio tensor using Whisper.

        Args:
            audio: Audio tensor (1, num_samples) or (num_samples,)
            sample_rate: Sample rate of audio
            language: Language code for transcription

        Returns:
            Transcribed text
        """
        start_time = datetime.now()

        try:
            # Ensure audio is in correct format
            if audio.dim() == 2:
                audio = audio.squeeze(0)  # Remove channel dimension

            # Convert to numpy and ensure float32
            audio_np = audio.detach().cpu().numpy().astype("float32")

            # Resample to 16kHz if needed (Whisper requirement)
            if sample_rate != 16000:
                # Use torchaudio for resampling
                audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=16000
                )
                audio_resampled = resampler(audio_tensor).squeeze(0)
                audio_np = audio_resampled.numpy()

            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_np,
                language=language,
                task="transcribe",
                fp16=False,  # Use fp32 for better compatibility
            )

            transcription = result["text"].strip()

            duration = (datetime.now() - start_time).total_seconds()
            self.logger.debug(
                f"Transcription completed in {duration:.2f}s: '{transcription[:100]}...'"
            )

            return transcription

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise

    def validate_candidate(
        self, candidate: AudioCandidate, original_text: str, sample_rate: int = 24000
    ) -> ValidationResult:
        """
        Validate an audio candidate against original text.

        Args:
            candidate: Audio candidate to validate
            original_text: Original text for comparison
            sample_rate: Sample rate of audio

        Returns:
            ValidationResult with validation outcome
        """
        start_time = datetime.now()

        try:
            # Transcribe the audio
            transcription = self.transcribe_audio(
                candidate.audio_tensor, sample_rate=sample_rate
            )

            # Calculate similarity score (using simple approach for now)
            similarity_score = self._calculate_similarity(original_text, transcription)

            # Calculate quality score
            quality_score = self._calculate_quality_score(
                candidate, transcription, similarity_score
            )

            # Determine if validation passed - improved logic
            # Allow some flexibility: if one metric is strong, be more lenient with the other
            strict_validation = (
                similarity_score >= self.similarity_threshold 
                and quality_score >= self.min_quality_score
            )
            
            # Flexible validation: high quality score can compensate for slightly lower similarity
            flexible_validation = (
                # Strong quality score allows lower similarity (0.1 tolerance)
                (quality_score >= self.min_quality_score + 0.02 and 
                 similarity_score >= self.similarity_threshold - 0.1) or
                # Strong similarity allows lower quality score (0.1 tolerance)  
                (similarity_score >= self.similarity_threshold + 0.02 and 
                 quality_score >= self.min_quality_score - 0.1) or
                # Both are reasonably close to thresholds (within 0.05)
                (similarity_score >= self.similarity_threshold - 0.05 and 
                 quality_score >= self.min_quality_score - 0.05 and
                 (similarity_score + quality_score) >= (self.similarity_threshold + self.min_quality_score) - 0.05)
            )
            
            is_valid = strict_validation or flexible_validation

            validation_time = (datetime.now() - start_time).total_seconds()

            result = ValidationResult(
                is_valid=is_valid,
                transcription=transcription,
                similarity_score=similarity_score,
                quality_score=quality_score,
                validation_time=validation_time,
            )

            # Note: Final validation decision is made in main.py after fuzzy matching improvements
            # No logging here to avoid confusion with preliminary Whisper-only scores

            return result

        except Exception as e:
            validation_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Validation failed: {e}")

            return ValidationResult(
                is_valid=False,
                transcription="",
                similarity_score=0.0,
                quality_score=0.0,
                validation_time=validation_time,
                error_message=str(e),
            )

    def _calculate_similarity(self, original: str, transcription: str) -> float:
        """
        Calculate similarity between original text and transcription.
        This is a simplified implementation - will be enhanced by FuzzyMatcher.

        Args:
            original: Original text
            transcription: Transcribed text

        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            # Simple token-based similarity for now
            original_tokens = set(original.lower().split())
            transcription_tokens = set(transcription.lower().split())

            if not original_tokens and not transcription_tokens:
                return 1.0
            if not original_tokens or not transcription_tokens:
                return 0.0

            intersection = original_tokens.intersection(transcription_tokens)
            union = original_tokens.union(transcription_tokens)

            similarity = len(intersection) / len(union) if union else 0.0
            return min(1.0, max(0.0, similarity))

        except Exception as e:
            self.logger.warning(f"Similarity calculation failed: {e}")
            return 0.0

    def _calculate_quality_score(
        self, candidate: AudioCandidate, transcription: str, similarity_score: float
    ) -> float:
        """
        Calculate overall quality score for the candidate.

        Args:
            candidate: Audio candidate
            transcription: Transcribed text
            similarity_score: Text similarity score

        Returns:
            Quality score (0.0 to 1.0)
        """
        try:
            # Length score - ratio of transcription length to original text length
            if len(candidate.chunk_text) > 0:
                length_score = min(1.0, len(transcription) / len(candidate.chunk_text))
            else:
                length_score = 1.0 if transcription else 0.0

            # Combined quality score (removed unreliable audio duration length_score)
            quality_score = (
                similarity_score * 0.7  # 70% similarity (increased from 60%)
                + length_score * 0.30   # 30% text length comparison
            )

            return min(1.0, max(0.0, quality_score))

        except Exception as e:
            self.logger.warning(f"Quality score calculation failed: {e}")
            return similarity_score  # Fallback to similarity only

    def batch_validate(
        self,
        candidates: list[AudioCandidate],
        original_texts: list[str],
        sample_rate: int = 24000,
    ) -> list[ValidationResult]:
        """
        Validate multiple candidates in batch.

        Args:
            candidates: List of audio candidates
            original_texts: List of original texts (same length as candidates)
            sample_rate: Sample rate of audio

        Returns:
            List of validation results
        """
        if len(candidates) != len(original_texts):
            raise ValueError("Number of candidates must match number of original texts")

        results = []
        for i, (candidate, original_text) in enumerate(zip(candidates, original_texts)):
            self.logger.info(f"Validating candidate {i+1}/{len(candidates)}...")
            result = self.validate_candidate(candidate, original_text, sample_rate)
            results.append(result)

        return results

    def save_transcriptions_to_disk(
        self,
        transcriptions: List[str],
        chunk_index: int,
        candidate_indices: List[int],
        validation_data: Optional[List[dict]] = None,
        output_dir: Optional[str] = None,
    ) -> List[str]:
        """
        Save whisper transcriptions to text files with enhanced metrics format.

        RECOVERY SYSTEM DEPENDENCY WARNING:
        ===================================
        This method defines the transcription file naming scheme and content structure
        that multiple Recovery System modules depend on. If you modify:

        1. FILENAME PATTERN: "chunk_{chunk_index+1:03d}_candidate_{candidate_idx+1:02d}_whisper.txt"
           -> Update: src/recovery/gap_analyzer.py (_analyze_transcriptions method)
           -> Update: src/recovery/state_detector.py (_analyze_transcriptions_directory)
           -> Update: src/recovery/metrics_loader.py (MetricsLoader._parse_transcription_file)
           -> Update: src/recovery/validation_recovery.py (save_enhanced_metrics method)

        2. FILE CONTENT FORMAT: The "=== WHISPER TRANSCRIPTION ===" header and metadata structure
           -> Update: src/recovery/metrics_loader.py (_extract_metrics_from_content method)
           -> Update: src/recovery/gap_analyzer.py (_validate_transcription_file method)
           -> Update: src/recovery/validation_recovery.py (_format_enhanced_transcription)

        3. METADATA FIELDS: Whisper Score, Fuzzy Score, Quality Score, Generation Params, etc.
           -> Update: All recovery modules that parse these enhanced metrics
           -> Update: src/recovery/metrics_loader.py field extraction methods

        4. DIRECTORY STRUCTURE: output_dir (typically texts/ subdirectory)
           -> Update: All recovery modules that expect transcriptions in texts/ directory

        The Recovery System uses these files for intelligent analysis, candidate selection,
        and enhanced validation recovery. Changes here affect recovery capabilities!

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
            # Default to project data/output/chunks directory (same as chunks)
            project_root = Path(__file__).resolve().parents[2]
            output_dir = project_root / "data" / "output" / "chunks"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for i, (transcription, candidate_idx) in enumerate(
            zip(transcriptions, candidate_indices)
        ):
            try:
                # Create filename matching the candidate format (without timestamp)
                filename = (
                    f"chunk_{chunk_index+1:03d}_candidate_{candidate_idx+1:02d}_whisper.txt"
                )
                filepath = output_path / filename

                # Get validation data for this candidate
                val_data = (
                    validation_data[i]
                    if validation_data and i < len(validation_data)
                    else {}
                )

                # Extract enhanced metrics
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

                # Format generation parameters
                if generation_params:
                    exag = generation_params.get("exaggeration", 0.0)
                    cfg = generation_params.get("cfg_weight", 0.0)
                    temp = generation_params.get("temperature", 0.0)
                    param_type = generation_params.get("type", "UNKNOWN")
                    params_str = f"exag={exag:.2f}, cfg={cfg:.2f}, temp={temp:.2f}, type={param_type}"
                else:
                    params_str = "N/A"

                # Create enhanced content format
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

                # Save to file
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
