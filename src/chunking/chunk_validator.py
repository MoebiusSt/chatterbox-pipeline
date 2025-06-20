import logging
from typing import List

from .base_chunker import TextChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkValidator:
    """
    Validates the quality and integrity of text chunks.
    """

    def __init__(self, max_limit: int, min_length: int):
        """
        Initializes the ChunkValidator.

        Args:
            max_limit: The maximum character limit for any chunk.
            min_length: The minimum character length for a chunk (not strictly enforced on the last chunk).
        """
        self.max_limit = max_limit
        self.min_length = min_length
        self.sentence_enders = (
            ".",
            "!",
            "?",
            '"',
            '"',
            "]",
        )  # Added ] for TTS annotations

    def validate_chunk_length(self, chunk: TextChunk) -> bool:
        """
        Validates if the chunk's length is within the acceptable limits.

        """
        chunk_len = len(chunk.text)
        if chunk_len > self.max_limit:
            logger.warning(
                f"Chunk exceeds max length ({chunk_len}/{self.max_limit}): '{chunk.text[:80]}...'"
            )
            return False
        # Min length check is less critical, can be a warning.
        if chunk_len < self.min_length:
            logger.info(
                f"Chunk is shorter than min length ({chunk_len}/{self.min_length}): '{chunk.text[:80]}...'"
            )
        return True

    def validate_sentence_boundaries(self, chunk: TextChunk) -> bool:
        """
        Validates that the chunk ends on what looks like a sentence boundary.
        This is a heuristic and may not cover all cases perfectly.
        """
        if not chunk.text:
            return True  # An empty chunk is valid in this context

        last_char = chunk.text.strip()[-1]

        # Be more lenient with fallback split chunks that may end with secondary delimiters
        if getattr(chunk, "is_fallback_split", False):
            # Allow fallback splits to end with secondary delimiters used for splitting
            fallback_delimiters = [";", "—", "–", '"', '"', ":", ","]
            if last_char in fallback_delimiters:
                logger.debug(
                    f"Fallback split chunk ends with secondary delimiter '{last_char}' - this is acceptable"
                )
                return True

        if last_char not in self.sentence_enders:
            logger.warning(
                f"Chunk does not appear to end on a sentence boundary (ends with '{last_char}'): '{chunk.text[-80:]}'"
            )
            return False
        return True

    def run_all_validations(self, chunks: List[TextChunk]) -> bool:

        all_valid = True
        for i, chunk in enumerate(chunks):
            logger.info(f"Validating chunk {i+1}/{len(chunks)}...")

            # Check length validation
            if not self.validate_chunk_length_with_context(chunk, i + 1, len(chunks)):
                all_valid = False

            # Check sentence boundaries
            if not self.validate_sentence_boundaries(chunk):
                all_valid = False

        if all_valid:
            logger.info("All chunks passed validation.")
        else:
            logger.error("One or more chunks failed validation.")

        return all_valid

    def validate_chunk_length_with_context(
        self, chunk: TextChunk, chunk_number: int, total_chunks: int
    ) -> bool:
        """
        Validates if the chunk's length is within the acceptable limits, considering its position.

        """
        chunk_len = len(chunk.text)
        is_last_chunk = chunk_number == total_chunks

        # Max length check (more lenient for fallback splits)
        if chunk_len > self.max_limit:
            if getattr(chunk, "is_fallback_split", False):
                logger.info(
                    f"Chunk exceeds max length ({chunk_len}/{self.max_limit}) but was created by fallback splitting - this is acceptable: '{chunk.text[:80]}...'"
                )
                # Don't fail validation for fallback split chunks
            else:
                logger.warning(
                    f"Chunk exceeds max length ({chunk_len}/{self.max_limit}): '{chunk.text[:80]}...'"
                )
                return False

        # Min length check (more lenient for last chunk)
        if chunk_len < self.min_length:
            if is_last_chunk:
                logger.info(
                    f"Last chunk is shorter than min length ({chunk_len}/{self.min_length}) - this is acceptable: '{chunk.text[:80]}...'"
                )
                # Don't fail validation for short last chunk
            else:
                logger.warning(
                    f"Chunk is shorter than min length ({chunk_len}/{self.min_length}): '{chunk.text[:80]}...'"
                )
                # For non-last chunks, still return True but log as warning

        return True
