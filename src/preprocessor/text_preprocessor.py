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
        logger.info(f"📝 Copying input text to job directory...")
        shutil.copy2(input_text_path, input_copy_path)
        logger.info(f"✅ Input text copied to: {input_copy_path.name}")

        # Step 2: Load and process the text
        logger.info(f"🔄 Processing text...")

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
                logger.info(f"✅ Normalized line endings")

        # Future preprocessing options can be added here:
        # - Quote normalization
        # - Extra whitespace removal
        # - Encoding issue fixes
        # - Special character handling

        return processed_text

    def validate_processed_text(
        self, processed_text_path: Path, run_config_path: Optional[Path] = None
    ) -> bool:
        """
        Validate that processed text exists and is valid.
        Regenerate if needed using run configuration.

        """
        if not processed_text_path.exists():
            logger.warning(f"⚠️ Processed text file missing: {processed_text_path}")
            return False

        try:
            with open(processed_text_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                logger.warning(f"⚠️ Processed text file is empty: {processed_text_path}")
                return False

            logger.info(
                f"✅ Processed text validation passed: {len(content)} characters"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Error validating processed text: {e}")
            return False

    def regenerate_processed_text(
        self,
        output_dir: Path,
        text_base_name: str,
        run_config_path: Optional[Path] = None,
    ) -> bool:
        """
        Regenerate processed text from input copy using run configuration.

        Returns:
            True if regeneration successful
        """
        logger.info("🔄 Regenerating processed text...")

        # Check if input copy exists
        input_copy_path = output_dir / f"{text_base_name}_input.txt"
        if not input_copy_path.exists():
            logger.error(f"❌ Input copy not found for regeneration: {input_copy_path}")
            return False

        # Load preprocessing config from run config if available
        preprocessing_config = self.config
        if run_config_path and run_config_path.exists():
            try:
                import yaml

                with open(run_config_path, "r") as f:
                    run_config = yaml.safe_load(f)
                preprocessing_config = run_config.get("preprocessing", self.config)
                logger.info(f"📄 Loaded preprocessing config from run configuration")
            except Exception as e:
                logger.warning(f"⚠️ Could not load run config, using default: {e}")

        # Create new preprocessor with run-specific config
        regeneration_preprocessor = TextPreprocessor(preprocessing_config)

        # Load input text and reprocess
        with open(input_copy_path, "r", encoding="utf-8") as f:
            input_text = f.read()

        processed_text = regeneration_preprocessor._process_text_content(input_text)

        # Save regenerated processed text
        processed_text_path = output_dir / f"{text_base_name}_processed.txt"
        with open(processed_text_path, "w", encoding="utf-8") as f:
            f.write(processed_text)

        logger.info(f"✅ Regenerated processed text: {len(processed_text)} characters")
        return True
