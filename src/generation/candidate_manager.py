import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from chunking.base_chunker import TextChunk
from utils.file_manager.io_handlers.candidate_io import (
    AudioCandidate,
    CandidateIOHandler,
)

from .batch_processor import BatchChunkProcessor, GenerationResult
from .selection_strategies import SelectionStrategies
from .tts_generator import TTSGenerator

# Use the centralized logging configuration from main.py
logger = logging.getLogger(__name__)


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
            self.candidates_dir = output_dir / "candidates"
        elif candidates_dir is None:
            self.candidates_dir = (
                Path(__file__).resolve().parents[2] / "data" / "output" / "candidates"
            )
        else:
            self.candidates_dir = Path(candidates_dir)

        # Initialize components
        self.batch_processor = BatchChunkProcessor(max_retries=max_retries)
        self.selection_strategies = SelectionStrategies()

        # Initialize candidate IO handler
        if self.save_candidates:
            self.candidate_io = CandidateIOHandler(self.candidates_dir, self.config)
            logger.info(f"💻 Candidates will be saved to: {self.candidates_dir}")

        logger.info(
            f"CandidateManager initialized: max_candidates={self.max_candidates}, max_retries={max_retries}"
        )

    def generate_candidates(
        self, text_chunk, chunk_index: int, output_dir: Path
    ) -> List[AudioCandidate]:
        """
        Generate all candidates for a chunk.

        Returns:
            List of AudioCandidate objects
        """
        if not self.tts_generator:
            raise RuntimeError("TTS generator not initialized")

        # Extract generation parameters from config
        generation_config = self.config.get("generation", {})

        # Check if chunk has speaker information for speaker-aware generation
        if hasattr(text_chunk, "speaker_id") and text_chunk.speaker_id:
            logger.info(
                f"🎭 Using speaker '{text_chunk.speaker_id}' in chunk {chunk_index + 1}"
            )

            # Generate candidates using speaker-aware method
            config_manager = getattr(self, "file_manager", None)

            candidates = self.tts_generator.generate_candidates_with_speaker(
                text=text_chunk.text,
                speaker_id=text_chunk.speaker_id,
                num_candidates=self.max_candidates,
                config_manager=config_manager,
            )

            # Create a mock result object for backward compatibility
            from .batch_processor import GenerationResult

            result = GenerationResult(
                chunk=text_chunk,
                candidates=candidates,
                selected_candidate=candidates[0] if candidates else None,
                generation_attempts=1,
                success=len(candidates) > 0,
            )
        else:
            # No speaker ID found, use default speaker
            logger.info(
                f"🔧 No speaker ID found, using default speaker for chunk {chunk_index + 1}"
            )
            config_manager = getattr(self, "file_manager", None)
            
            # Get default speaker ID from configuration
            default_speaker_id = "default"
            if config_manager and hasattr(config_manager, "get_default_speaker_id"):
                default_speaker_id = config_manager.get_default_speaker_id()
            
            candidates = self.tts_generator.generate_candidates_with_speaker(
                text=text_chunk.text,
                speaker_id=default_speaker_id,
                num_candidates=self.max_candidates,
                config_manager=config_manager,
            )

            # Create a mock result object for backward compatibility
            from .batch_processor import GenerationResult

            result = GenerationResult(
                chunk=text_chunk,
                candidates=candidates,
                selected_candidate=candidates[0] if candidates else None,
                generation_attempts=1,
                success=len(candidates) > 0,
            )

        # Save candidates to disk
        if result.candidates and self.save_candidates:
            self.candidate_io.save_candidates_to_disk(
                candidates=result.candidates,
                chunk_index=chunk_index,
                sample_rate=self.config.get("audio", {}).get("sample_rate", 24000),
                output_dir=output_dir,
            )

        return result.candidates

    def generate_specific_candidates(
        self,
        text_chunk,
        chunk_index: int,
        candidate_indices: List[int],
        output_dir: Path,
        reference_audio_path: Optional[str] = None,
    ) -> List[AudioCandidate]:
        """
        Generate only specific candidate indices for a chunk (selective recovery).
        Uses TTSGenerator.generate_specific_candidates to generate ONLY the missing candidates.

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

        # Check if chunk has speaker information
        if hasattr(text_chunk, "speaker_id") and text_chunk.speaker_id:
            logger.info(f"🎭 Using speaker for speaker '{text_chunk.speaker_id}'")

            # Use speaker-specific generation
            config_manager = getattr(self, "file_manager", None)

            # Generate ONLY the specific candidates using speaker-aware method
            specific_candidates = self.tts_generator.generate_candidates_with_speaker(
                text=text_chunk.text,
                speaker_id=text_chunk.speaker_id,
                num_candidates=len(candidate_indices),
                config_manager=config_manager,
            )

            # Map the generated candidates to the correct indices
            for i, candidate in enumerate(specific_candidates):
                if i < len(zero_based_indices):
                    candidate.candidate_idx = zero_based_indices[i]
        else:
            # No speaker ID found, use default speaker
            logger.info("🔧 No speaker ID found, using default speaker")
            config_manager = getattr(self, "file_manager", None)
            
            # Get default speaker ID from configuration
            default_speaker_id = "default"
            if config_manager and hasattr(config_manager, "get_default_speaker_id"):
                default_speaker_id = config_manager.get_default_speaker_id()
            
            specific_candidates = self.tts_generator.generate_candidates_with_speaker(
                text=text_chunk.text,
                speaker_id=default_speaker_id,
                num_candidates=len(candidate_indices),
                config_manager=config_manager,
            )

            # Map the generated candidates to the correct indices
            for i, candidate in enumerate(specific_candidates):
                if i < len(zero_based_indices):
                    candidate.candidate_idx = zero_based_indices[i]

        # Save the specific candidates using FileManager structure (chunk_XXX/candidate_YY.wav)
        saved_candidates = []
        for candidate in specific_candidates:
            try:
                # Delete corresponding whisper file BEFORE saving new candidate (ensures re-validation)
                if self.save_candidates:
                    self.candidate_io._delete_whisper_file(
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
        if saved_candidates and self.save_candidates:
            self.candidate_io._save_candidates_in_correct_structure(
                saved_candidates, chunk_index
            )

        logger.debug(
            f"✅ Successfully generated {len(saved_candidates)}/{len(candidate_indices)} specific candidates for chunk {chunk_index+1}"
        )
        return saved_candidates

    def generate_candidates_for_chunk(
        self,
        chunk: TextChunk,
        tts_generator: TTSGenerator,
        generation_params: Dict[str, Any],
        reference_audio_path: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generates multiple candidates for a single text chunk with proper retry logic.

        RETRY LOGIC:
        1. Generate num_candidates normal candidates first
        2. If all are invalid after validation, generate max_retries additional conservative candidates
        3. Select best from all candidates (num_candidates + max_retries total)
        4. On quality ties, prefer shorter audio duration

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
            # Extract only TTS-relevant parameters from generation_params
            tts_params = generation_params.get("tts_params", {})
            conservative_config_for_call = generation_params.get(
                "conservative_candidate", None
            )

            # Generate normal candidates with correct parameter passing
            normal_candidates = tts_generator.generate_candidates(
                text=chunk.text,
                num_candidates=self.max_candidates,
                exaggeration=tts_params.get("exaggeration"),
                cfg_weight=tts_params.get("cfg_weight"),
                temperature=tts_params.get("temperature"),
                conservative_config=conservative_config_for_call,
                tts_params=tts_params,  # Pass complete tts_params for repetition_penalty etc.
                reference_audio_path=reference_audio_path,
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
        # normal_candidates_for_validation = all_candidates.copy()

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

    # Delegate methods to components
    def select_best_candidate(
        self, candidates: List[AudioCandidate], selection_strategy: str = "shortest"
    ) -> Optional[AudioCandidate]:
        return self.selection_strategies.select_best_candidate(
            candidates, selection_strategy
        )

    def select_best_candidate_with_validation(
        self,
        candidates_with_validation: List[tuple],
        prefer_valid: bool = True,
    ) -> Optional[tuple]:
        return self.selection_strategies.select_best_candidate_with_validation(
            candidates_with_validation, prefer_valid
        )

    def process_chunks(
        self,
        chunks: List[TextChunk],
        tts_generator: TTSGenerator,
        generation_params: Dict[str, Any],
    ) -> List[GenerationResult]:
        return self.batch_processor.process_chunks(
            chunks, tts_generator, generation_params, self
        )
