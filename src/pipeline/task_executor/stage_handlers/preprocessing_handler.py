"""Preprocessing stage handler."""

import logging
from typing import Any, Dict

from chunking.chunk_validator import ChunkValidator
from chunking.spacy_chunker import SpaCyChunker
from utils.file_manager.file_manager import FileManager

logger = logging.getLogger(__name__)


class PreprocessingHandler:
    """Handles text preprocessing stage."""

    def __init__(
        self, file_manager: FileManager, config: Dict[str, Any], chunker: SpaCyChunker
    ):
        self.file_manager = file_manager
        self.config = config
        self.chunker = chunker

    def execute_preprocessing(self) -> bool:
        """Execute the text preprocessing stage."""
        logger.info("🚀 Starting Preprocessing Stage")
        try:
            logger.info("Starting preprocessing stage")

            # Set available speakers in chunker for speaker-aware chunking
            try:
                available_speakers = self.file_manager.get_all_speaker_ids()
                if hasattr(self.chunker, 'set_available_speakers'):
                    self.chunker.set_available_speakers(available_speakers)
                    logger.debug(f"Set available speakers in chunker: {available_speakers}")
            except Exception as e:
                logger.debug(f"Could not set speakers in chunker (not critical): {e}")

            # Early validation of input text existence
            if not self.file_manager.check_input_text_exists():
                logger.error("❌ Input text validation failed")
                try:
                    # Try to get detailed error information
                    self.file_manager.get_input_text()
                except FileNotFoundError as e:
                    logger.error(str(e))
                logger.error("⚠️  The preprocessing stage cannot proceed without input text.")
                return False

            # Load input text with graceful error handling
            try:
                input_text = self.file_manager.get_input_text()
                logger.debug(f"Loaded input text: {len(input_text)} characters")
            except FileNotFoundError as e:
                logger.error(f"❌ Input text file not found: {e}")
                logger.error("⚠️  preprocessing cannot proceed.Please ensure the input text file exists in the correct location.")
                return False
            except Exception as e:
                logger.error(f"❌ Failed to load input text: {e}")
                logger.error("⚠️  The preprocessing stage cannot proceed without valid input text.")
                return False

            # Chunk text
            text_chunks = self.chunker.chunk_text(input_text)
            logger.info(f"Generated {len(text_chunks)} text chunks")

            # Update indices for TextChunk objects
            for i, chunk in enumerate(text_chunks):
                chunk.idx = i

            # Validate chunks
            chunking_config = self.config["chunking"]
            chunk_validator = ChunkValidator(
                max_limit=chunking_config.get("max_chunk_limit", 600),
                min_length=chunking_config.get("min_chunk_length", 50),
            )
            if not chunk_validator.run_all_validations(text_chunks):
                logger.warning("Chunk validation reported issues – proceeding anyway")

            # Save chunks
            if not self.file_manager.save_chunks(text_chunks):
                logger.error("Failed to save chunks")
                return False

            logger.info("Preprocessing stage completed successfully")
            return True

        except Exception as e:
            logger.error(f"Preprocessing stage failed: {e}", exc_info=True)
            return False
