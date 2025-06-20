import logging
from typing import Any, Dict, List

from chunking.base_chunker import TextChunk
from utils.file_manager import AudioCandidate

from .tts_generator import TTSGenerator

logger = logging.getLogger(__name__)


class GenerationResult:
    """Represents the result of candidate generation for a text chunk."""

    def __init__(
        self,
        chunk: TextChunk,
        candidates: List[AudioCandidate],
        selected_candidate=None,
        generation_attempts: int = 0,
        success: bool = False,
    ):
        self.chunk = chunk
        self.candidates = candidates
        self.selected_candidate = selected_candidate
        self.generation_attempts = generation_attempts
        self.success = success


class BatchProcessor:
    """Handles batch processing of text chunks for candidate generation."""

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def process_chunks(
        self,
        chunks: List[TextChunk],
        tts_generator: TTSGenerator,
        generation_params: Dict[str, Any],
        candidate_manager,
    ) -> List[GenerationResult]:
        """Processes a list of text chunks, generating and managing audio candidates for each."""
        logger.info(f"Processing {len(chunks)} chunks for candidate generation")

        results = []

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")

            try:
                result = candidate_manager.generate_candidates_for_chunk(
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
