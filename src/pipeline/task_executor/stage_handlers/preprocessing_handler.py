"""Preprocessing stage handler."""

import logging
from typing import Any, Dict

from chunking.chunk_validator import ChunkValidator
from chunking.spacy_chunker import SpaCyChunker
from utils.file_manager import FileManager

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

            # Load input text
            input_text = self.file_manager.get_input_text()
            logger.debug(f"Loaded input text: {len(input_text)} characters")

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
