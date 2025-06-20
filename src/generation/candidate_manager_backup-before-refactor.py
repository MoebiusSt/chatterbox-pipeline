import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torchaudio

from chunking.base_chunker import TextChunk
from utils.file_manager import AudioCandidate

from .tts_generator import TTSGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
import logging

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Represents the result of candidate generation for a text chunk."""

    chunk: TextChunk
    candidates: List[AudioCandidate]
    selected_candidate: Optional[AudioCandidate]
    generation_attempts: int
    success: bool


class CandidateManager:
    """
    Manages the generation and selection of audio candidates for text chunks.
    Implements retry logic and candidate selection strategies.
    """

    def __init__(
        self,
        tts_generator: Optional[TTSGenerator] = None,
        config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[Path] = None,
        max_candidates: int = 3,
        max_retries: int = 2,
        min_successful_candidates: int = 1,
        save_candidates: bool = True,
        candidates_dir: Optional[str] = None,
    ):
        """
        Initializes the CandidateManager.

        Args:
            tts_generator: TTSGenerator instance
            config: Pipeline configuration
            output_dir: Output directory
            max_candidates: Maximum number of candidates to generate per chunk.
            max_retries: Maximum number of retry attempts for failed generations.
            min_successful_candidates: Minimum number of successful candidates required.
            save_candidates: Whether to save candidates to disk for debugging.
            candidates_dir: Directory to save candidates in.
        """
        # Store components for generation
        self.tts_generator = tts_generator
        self.config = config or {}
        self.output_dir = (
            output_dir  # Store output_dir as instance variable for whisper deletion
        )

        # Extract parameters from config if provided
        generation_config = self.config.get("generation", {})
        self.max_candidates = generation_config.get("num_candidates", max_candidates)
        self.max_retries = max_retries
        self.min_successful_candidates = min_successful_candidates
        self.save_candidates = save_candidates

        # Set up candidates directory
        if output_dir:
            candidates_dir = output_dir / "candidates"
        elif candidates_dir is None:
            candidates_dir = (
                Path(__file__).resolve().parents[2] / "data" / "output" / "candidates"
            )
        self.candidates_dir = Path(candidates_dir)

        if self.save_candidates:
            self.candidates_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"💻 Candidates will be saved to: {self.candidates_dir}")

        logger.info(
            f"CandidateManager initialized: max_candidates={self.max_candidates}, max_retries={max_retries}"
        )

    def generate_candidates(
        self, text_chunk, chunk_index: int, output_dir: Path
    ) -> List[AudioCandidate]:
        """
        Generate all candidates for a chunk.

        Args:
            text_chunk: TextChunk object
            chunk_index: Index of the chunk
            output_dir: Output directory for saving

        Returns:
            List of AudioCandidate objects
        """
        if not self.tts_generator:
            raise RuntimeError("TTS generator not initialized")

        # Extract generation parameters from config
        generation_config = self.config.get("generation", {})

        # Generate candidates using the existing method
        result = self.generate_candidates_for_chunk(
            chunk=text_chunk,
            tts_generator=self.tts_generator,
            generation_params=generation_config,
        )

        # Save candidates to disk
        if result.candidates:
            self.save_candidates_to_disk(
                candidates=result.candidates,
                chunk_index=chunk_index,
                sample_rate=self.config.get("audio", {}).get(
                    "sample_rate", 24000
                ),  # ChatterboxTTS native sample rate
            )

        return result.candidates

    def generate_specific_candidates(
        self,
        text_chunk,
        chunk_index: int,
        candidate_indices: List[int],
        output_dir: Path,
    ) -> List[AudioCandidate]:
        """
        Generate only specific candidate indices for a chunk (selective recovery).
        Uses TTSGenerator.generate_specific_candidates to generate ONLY the missing candidates.

        Args:
            text_chunk: TextChunk object
            chunk_index: Index of the chunk
            candidate_indices: List of specific candidate indices to generate (1-based)
            output_dir: Output directory for saving

        Returns:
            List of AudioCandidate objects
        """
        if not self.tts_generator:
            raise RuntimeError("TTS generator not initialized")

        logger.debug(
            f"🔧 generate_specific_candidates(): Generating candidates {candidate_indices} for chunk {chunk_index + 1}"
        )

        # Extract generation parameters from config
        generation_config = self.config.get("generation", {})
        tts_params = generation_config.get("tts_params", {})
        conservative_config = generation_config.get("conservative_candidate", {})
        total_candidates = generation_config.get("num_candidates", 5)

        # Convert 1-based candidate indices to 0-based for TTSGenerator
        zero_based_indices = [idx - 1 for idx in candidate_indices]

        # Generate ONLY the specific candidates using the new method
        specific_candidates = self.tts_generator.generate_specific_candidates(
            text=text_chunk.text,
            candidate_indices=zero_based_indices,
            exaggeration=tts_params.get("exaggeration", 0.5),
            cfg_weight=tts_params.get("cfg_weight", 0.4),
            temperature=tts_params.get("temperature", 0.8),
            conservative_config=conservative_config,
            total_candidates=total_candidates,
        )

        # Save the specific candidates using FileManager structure (chunk_XXX/candidate_YY.wav)
        saved_candidates = []
        for candidate in specific_candidates:
            try:
                # Delete corresponding whisper file BEFORE saving new candidate (ensures re-validation)
                self._delete_whisper_file(
                    output_dir, chunk_index, candidate.candidate_idx + 1
                )

                # Update candidate metadata for FileManager compatibility
                candidate.chunk_idx = chunk_index
                saved_candidates.append(candidate)

                logger.debug(
                    f"✅ Generated candidate {candidate.candidate_idx+1} for chunk {chunk_index+1}"
                )

            except Exception as e:
                logger.error(
                    f"❌ Failed to prepare candidate {candidate.candidate_idx+1} for chunk {chunk_index+1}: {e}"
                )
                continue

        # Use FileManager to save candidates in correct structure (chunk_XXX/candidate_YY.wav)
        if saved_candidates and hasattr(self, "file_manager"):
            self.file_manager.save_candidates(chunk_index, saved_candidates)
        elif saved_candidates:
            # Fallback: save manually in correct structure if file_manager not available
            self._save_candidates_in_correct_structure(saved_candidates, chunk_index)

        logger.debug(
            f"✅ Successfully generated {len(saved_candidates)}/{len(candidate_indices)} specific candidates for chunk {chunk_index+1}"
        )
        return saved_candidates

    def _delete_whisper_file(
        self, output_dir: Path, chunk_index: int, candidate_idx: int
    ):
        """
        Delete corresponding whisper validation file for a candidate (ensures re-validation).

        Args:
            output_dir: Output directory containing whisper subdirectory
            chunk_index: Chunk index (0-based)
            candidate_idx: Candidate index (1-based)
        """
        # CORRECTED: Whisper files are in whisper/ directory, not texts/
        whisper_dir = output_dir / "whisper"
        whisper_file = (
            whisper_dir
            / f"chunk_{chunk_index+1:03d}_candidate_{candidate_idx:02d}_whisper.json"
        )

        if whisper_file.exists():
            whisper_file.unlink()
            logger.debug(f"🗑️ Deleted old whisper file: {whisper_file.name}")

        # Also try alternative naming patterns (in case of inconsistencies)
        alt_whisper_file = (
            whisper_dir
            / f"chunk_{chunk_index+1:03d}_candidate_{candidate_idx:02d}_whisper.txt"
        )
        if alt_whisper_file.exists():
            alt_whisper_file.unlink()
            logger.debug(f"🗑️ Deleted old whisper TXT file: {alt_whisper_file.name}")

    def generate_candidates_for_chunk(
        self,
        chunk: TextChunk,
        tts_generator: TTSGenerator,
        generation_params: Dict[str, Any],
    ) -> GenerationResult:
        """
        Generates multiple candidates for a single text chunk with proper retry logic.

        RETRY LOGIC:
        1. Generate num_candidates normal candidates first
        2. If all are invalid after validation, generate max_retries additional conservative candidates
        3. Select best from all candidates (num_candidates + max_retries total)
        4. On quality ties, prefer shorter audio duration

        Args:
            chunk: The TextChunk to generate audio for.
            tts_generator: The TTSGenerator instance to use.
            generation_params: Parameters for TTS generation.

        Returns:
            GenerationResult containing candidates and metadata.
        """
        logger.info(
            f"Starting candidate generation for chunk (length: {len(chunk.text)} chars)"
        )
        logger.debug(
            f"Chunk text preview: '{chunk.text[:100]}{'...' if len(chunk.text) > 100 else ''}'"
        )

        all_candidates = []
        generation_attempts = 0

        # PHASE 1: Generate normal candidates
        generation_attempts += 1
        logger.info(f"Phase 1: Generating {self.max_candidates} normal candidates")

        try:
            params_for_attempt = generation_params.copy()
            conservative_config_for_call = params_for_attempt.pop(
                "conservative_config", None
            )

            # Generate normal candidates
            normal_candidates = tts_generator.generate_candidates(
                text=chunk.text,
                num_candidates=self.max_candidates,
                conservative_config=conservative_config_for_call,
                **params_for_attempt,
            )

            # Filter out failed candidates (those with very short audio)
            valid_normal_candidates = [
                c
                for c in normal_candidates
                if c.audio is not None and c.audio.numel() > 100
            ]

            all_candidates.extend(valid_normal_candidates)
            logger.info(
                f"Generated {len(valid_normal_candidates)}/{len(normal_candidates)} valid normal candidates"
            )

        except Exception as e:
            logger.error(f"Normal generation failed: {e}")

        # Store normal candidates for validation check
        normal_candidates_for_validation = all_candidates.copy()

        # NOTE: Conservative retry logic is now handled in main.py after validation
        # This allows for smarter retry decisions based on validation results

        # Select the best candidate (for backward compatibility, but validation will be done in main.py)
        selected_candidate = all_candidates[0] if all_candidates else None

        success = len(all_candidates) >= self.min_successful_candidates

        result = GenerationResult(
            chunk=chunk,
            candidates=all_candidates,
            selected_candidate=selected_candidate,
            generation_attempts=generation_attempts,
            success=success,
        )

        logger.info(
            f"Generation completed: {len(all_candidates)} total candidates ({generation_attempts} attempts), success={success}"
        )
        return result

    def select_best_candidate(
        self, candidates: List[AudioCandidate], selection_strategy: str = "shortest"
    ) -> Optional[AudioCandidate]:
        """
        Selects the best candidate from a list of candidates.

        Args:
            candidates: List of AudioCandidate objects.
            selection_strategy: Strategy for selection ("shortest", "random", "first").

        Returns:
            The selected AudioCandidate or None if list is empty.
        """
        if not candidates:
            logger.warning("No candidates provided for selection")
            return None

        if selection_strategy == "shortest":
            # Select candidate with shortest audio duration (often indicates fewer artifacts)
            selected = min(
                candidates,
                key=lambda c: (
                    c.audio_tensor.shape[-1]
                    if c.audio_tensor is not None
                    else float("inf")
                ),
            )
            logger.debug(
                f"Selected shortest candidate with length: {selected.audio_tensor.shape[-1] if selected.audio_tensor is not None else 0}"
            )

        elif selection_strategy == "first":
            selected = candidates[0]
            logger.debug(f"Selected first candidate")

        elif selection_strategy == "random":
            import random

            selected = random.choice(candidates)
            logger.debug(f"Selected random candidate")

        else:
            logger.warning(
                f"Unknown selection strategy: {selection_strategy}, using first"
            )
            selected = candidates[0]

        return selected

    def process_chunks(
        self,
        chunks: List[TextChunk],
        tts_generator: TTSGenerator,
        generation_params: Dict[str, Any],
    ) -> List[GenerationResult]:
        """
        Processes multiple chunks and generates candidates for each.

        Args:
            chunks: List of TextChunk objects to process.
            tts_generator: The TTSGenerator instance to use.
            generation_params: Parameters for TTS generation.

        Returns:
            List of GenerationResult objects.
        """
        logger.info(f"Processing {len(chunks)} chunks for candidate generation")

        results = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")

            try:
                result = self.generate_candidates_for_chunk(
                    chunk=chunk,
                    tts_generator=tts_generator,
                    generation_params=generation_params,
                )
                results.append(result)

                if not result.success:
                    logger.warning(
                        f"Failed to generate sufficient candidates for chunk {i+1}"
                    )

            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                # Create a failed result
                failed_result = GenerationResult(
                    chunk=chunk,
                    candidates=[],
                    selected_candidate=None,
                    generation_attempts=self.max_retries + 1,
                    success=False,
                )
                results.append(failed_result)

        successful_chunks = sum(1 for r in results if r.success)
        logger.info(
            f"Candidate generation completed: {successful_chunks}/{len(chunks)} chunks successful"
        )

        return results

    def save_candidates_to_disk(
        self,
        candidates: List[AudioCandidate],
        chunk_index: int,
        sample_rate: int = 24000,  # ChatterboxTTS native sample rate
    ) -> List[str]:
        """
        Save candidates to disk using CORRECT FileManager structure.

        CORRECTED STRUCTURE:
        ===================================
        This method now uses the correct directory structure that FileManager expects:
        - candidates/chunk_001/candidate_01.wav (NOT flat structure with timestamps)
        - whisper/chunk_001_candidate_01_whisper.json (in whisper/ directory)

        This ensures consistency with FileManager.save_candidates() and proper recovery support.

        Args:
            candidates: List of audio candidates to save
            chunk_index: Index of the chunk these candidates belong to
            sample_rate: Audio sample rate

        Returns:
            List of saved file paths
        """
        if not self.save_candidates or not candidates:
            return []

        # Use FileManager structure: candidates/chunk_XXX/candidate_YY.wav
        chunk_dir = self.candidates_dir / f"chunk_{chunk_index+1:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for candidate in candidates:
            try:
                # Use simple filename without timestamp (FileManager structure)
                filename = f"candidate_{candidate.candidate_idx+1:02d}.wav"
                filepath = chunk_dir / filename

                # Delete corresponding whisper file if it exists (ensures re-validation)
                if self.output_dir:
                    self._delete_whisper_file(
                        self.output_dir, chunk_index, candidate.candidate_idx + 1
                    )

                # Save audio to file
                # Ensure audio tensor is 2D for torchaudio.save (channels, samples)
                audio_tensor = candidate.audio_tensor.cpu()
                if audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)  # Add channel dimension

                torchaudio.save(str(filepath), audio_tensor, sample_rate)

                # Update candidate metadata with correct path
                candidate.audio_path = filepath

                saved_paths.append(str(filepath))
                logger.debug(f"Saved candidate to: {filepath}")

            except Exception as e:
                logger.error(
                    f"Failed to save candidate {candidate.candidate_idx+1} for chunk {chunk_index}: {e}"
                )
                continue

        # Save candidate metadata (consistent with FileManager)
        if saved_paths:
            self._save_candidate_metadata(candidates, chunk_index, chunk_dir)

        return saved_paths

    def _save_candidates_in_correct_structure(
        self, candidates: List[AudioCandidate], chunk_index: int
    ):
        """
        Fallback method to save candidates in correct FileManager structure.
        """
        chunk_dir = self.candidates_dir / f"chunk_{chunk_index+1:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        sample_rate = self.config.get("audio", {}).get("sample_rate", 24000)

        for candidate in candidates:
            try:
                filename = f"candidate_{candidate.candidate_idx+1:02d}.wav"
                filepath = chunk_dir / filename

                # Save audio file
                audio_tensor = candidate.audio_tensor.cpu()
                if audio_tensor.ndim == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)

                torchaudio.save(str(filepath), audio_tensor, sample_rate)
                candidate.audio_path = filepath

                logger.debug(f"Saved candidate to correct structure: {filepath}")

            except Exception as e:
                logger.error(
                    f"Failed to save candidate {candidate.candidate_idx+1}: {e}"
                )

    def _save_candidate_metadata(
        self, candidates: List[AudioCandidate], chunk_index: int, chunk_dir: Path
    ):
        """
        Save candidate metadata consistent with FileManager format.
        """
        try:
            candidate_metadata = {
                "chunk_idx": chunk_index,
                "total_candidates": len(candidates),
                "candidates": [
                    {
                        "candidate_idx": c.candidate_idx,
                        "audio_filename": f"candidate_{c.candidate_idx+1:02d}.wav",
                        "generation_params": c.generation_params,
                    }
                    for c in candidates
                ],
            }

            metadata_path = chunk_dir / "candidates_metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                import json

                json.dump(candidate_metadata, f, indent=2)

            logger.debug(f"Saved candidate metadata: {metadata_path}")

        except Exception as e:
            logger.error(f"Failed to save candidate metadata: {e}")

    def select_best_candidate_with_validation(
        self,
        candidates_with_validation: List[
            tuple
        ],  # [(candidate, validation_result, quality_score), ...]
        prefer_valid: bool = True,
    ) -> Optional[tuple]:
        """
        Selects the best candidate from validated candidates with proper tie-breaking.

        SELECTION LOGIC:
        1. If prefer_valid=True, prioritize valid candidates over invalid ones
        2. Among candidates of same validity, select highest quality score
        3. On quality score ties, select candidate with shortest audio duration

        Args:
            candidates_with_validation: List of (candidate, validation_result, quality_score) tuples
            prefer_valid: Whether to prefer valid candidates over invalid ones

        Returns:
            The selected (candidate, validation_result, quality_score) tuple or None
        """
        if not candidates_with_validation:
            logger.warning("No candidates provided for selection")
            return None

        logger.debug(
            f"🔍 Selecting best from {len(candidates_with_validation)} validated candidates..."
        )

        # Separate valid and invalid candidates if prefer_valid is True
        if prefer_valid:
            valid_candidates = [
                (c, v, q) for c, v, q in candidates_with_validation if v.is_valid
            ]
            invalid_candidates = [
                (c, v, q) for c, v, q in candidates_with_validation if not v.is_valid
            ]

            # Try valid candidates first
            candidates_to_consider = (
                valid_candidates if valid_candidates else invalid_candidates
            )
            selection_pool = "valid" if valid_candidates else "invalid (fallback)"
        else:
            candidates_to_consider = candidates_with_validation
            selection_pool = "all"

        logger.debug(
            f"🎯 Considering {len(candidates_to_consider)} {selection_pool} candidates"
        )

        # Sort by quality score (descending), then by audio duration (ascending) for tie-breaking
        def sort_key(item):
            candidate, validation_result, quality_score = item
            audio_duration = (
                candidate.audio_tensor.shape[-1]
                if candidate.audio_tensor is not None
                else float("inf")
            )
            return (
                -quality_score,
                audio_duration,
            )  # Negative quality for descending, positive duration for ascending

        candidates_to_consider.sort(key=sort_key)

        # Select the best candidate
        best_candidate_tuple = candidates_to_consider[0]
        candidate, validation_result, quality_score = best_candidate_tuple

        audio_duration = (
            candidate.audio_tensor.shape[-1]
            if candidate.audio_tensor is not None
            else 0
        )

        logger.debug(
            f"✅ Selected candidate with quality={quality_score:.3f}, "
            f"valid={validation_result.is_valid}, audio_duration={audio_duration} samples"
        )

        return best_candidate_tuple
