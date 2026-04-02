"""
Text preprocessing module for TTS pipeline.
Handles text normalization and preparation before chunking.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Preprocesses input text before chunking.
    Handles text normalization, cleanup, and file management.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize text preprocessor.
        """
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)

    def process_text_file(
        self, input_text_path: Path, output_dir: Path, text_base_name: str
    ) -> Dict[str, Path]:
        """
        Process text file and save results to output directory.

        Args:
            text_base_name: Base name for output files (without extension)

        Returns:
            Dict with paths to input_copy and processed_text files
        """
        logger.info("🔄 PHASE 0: TEXT PREPROCESSING")
        logger.info("=" * 80)

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Define output file paths
        input_copy_path = output_dir / f"{text_base_name}_input.txt"
        processed_text_path = output_dir / f"{text_base_name}_processed.txt"

        # Step 1: Copy original input text to job directory
        logger.info("📝 Copying input text to job directory...")
        shutil.copy2(input_text_path, input_copy_path)
        logger.info(f"✅ Input text copied to: {input_copy_path.name}")

        # Step 2: Load and process the text
        logger.info("🔄 Processing text...")

        with open(input_text_path, "r", encoding="utf-8") as f:
            original_text = f.read()

        if not self.enabled:
            logger.info("⚠️ Preprocessing disabled, using original text")
            processed_text = original_text
        else:
            processed_text = self._process_text_content(original_text)

        # Step 3: Save processed text
        with open(processed_text_path, "w", encoding="utf-8") as f:
            f.write(processed_text)

        logger.info(f"✅ Processed text saved to: {processed_text_path.name}")
        logger.info(f"📊 Original length: {len(original_text)} characters")
        logger.info(f"📊 Processed length: {len(processed_text)} characters")

        return {"input_copy": input_copy_path, "processed_text": processed_text_path}

    def _process_text_content(self, text: str) -> str:
        """
        Apply text processing transformations.

        Returns:
            Processed text content
        """
        processed_text = text

        # Normalize line endings (moved from SpaCy chunker)
        if self.config.get("normalize_line_endings", True):
            original_length = len(processed_text)
            processed_text = processed_text.replace("\r\n", "\n").replace("\r", "\n")
            if len(processed_text) != original_length:
                logger.info("✅ Normalized line endings")

        # Optional punctuation normalization (simulates Chatterbox internal processing).
        # IMPORTANT: No replacement here must touch square-bracket patterns such as [laugh]
        # or [cough] because these are paralinguistic tags consumed by the Turbo TTS model.
        if self.config.get("normalize_punct", True):
            original_length = len(processed_text)
            replacements = [
                ("...", ", "),
                ("…", ", "),
                (" - ", ", "),
                (";", "."),
                ("—", "-"),
                ("–", "-"),
                (" ,", ","),
                ("“", "\""),
                ("„", "\""),
                ("”", "\""),
                ("‘", "'"),
                ("’", "'"),
                ("‚", "'"),
                ("»", "\""),
                ("«", "\""),
            ]
            for old, new in replacements:
                processed_text = processed_text.replace(old, new)
            if len(processed_text) != original_length:
                logger.info("✅ Normalized punctuation (including Chatterbox standarts)")

        # Future preprocessing options can be added here:
        # - Extra whitespace removal
        # - Encoding issue fixes
        # - Special character handling

        return processed_text