"""Preprocessing stage handler."""

import logging
from pathlib import Path
from typing import Any, Dict, List

from chunking.chunk_validator import ChunkValidator
from chunking.spacy_chunker import SpaCyChunker, SpeakerMarkupParser
from preprocessor.text_preprocessor import TextPreprocessor
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

    def _validate_speaker_markup(
        self, text: str, available_speakers: List[str], default_speaker_id: str
    ) -> bool:
        """
        Strict validation of <speaker:id> markup against the configured speakers.

        When ``chunking.strict_speaker_validation`` is true (default), the pipeline
        aborts with a verbose error message if the source text references a
        speaker ID that is not configured in the merged job/task YAML. The
        default-speaker aliases ``0``, ``default`` and ``reset`` are always
        accepted.

        Returns:
            True if validation passes (or is disabled), False if it fails and the
            stage must abort.
        """
        chunking_cfg = self.config.get("chunking", {}) or {}
        strict = bool(chunking_cfg.get("strict_speaker_validation", True))

        parser = SpeakerMarkupParser()
        unknown = parser.find_unknown_speakers(text, available_speakers)

        if not unknown:
            return True

        unique_unknown = sorted({sid for _, sid in unknown})
        configured = list(available_speakers) + list(parser.DEFAULT_SPEAKER_ALIASES)

        if not strict:
            for line, sid in unknown:
                logger.warning(
                    f"⚠️  Unknown speaker '<speaker:{sid}>' at line {line} – "
                    f"falling back to default speaker '{default_speaker_id}' "
                    "(strict_speaker_validation=false)"
                )
            return True

        logger.info("=" * 60)
        logger.error("❌ SPEAKER MARKUP VALIDATION FAILED")
        logger.info("=" * 60)
        logger.error(
            f"The source text references {len(unique_unknown)} speaker ID(s) "
            "that are not defined in the merged job/task configuration:"
        )
        logger.error("")
        for sid in unique_unknown:
            occurrences = [str(line) for line, s in unknown if s == sid]
            logger.error(
                f"   • <speaker:{sid}>  (line(s): {', '.join(occurrences)})"
            )
        logger.error("")
        logger.error(
            f"📋 Configured speaker IDs ({len(available_speakers)}): "
            f"{available_speakers}"
        )
        logger.error(
            f"📋 Accepted default-speaker aliases: "
            f"{list(parser.DEFAULT_SPEAKER_ALIASES)}"
        )
        logger.error(f"📋 Resolved default_speaker: '{default_speaker_id}'")
        logger.error("")
        logger.error("🛠  Resolution options:")
        logger.error(
            "   1. Add the missing speaker(s) to generation.speakers in the job "
            "YAML (id, reference_audio, language [, tts_params])."
        )
        logger.error(
            "   2. Or replace the unknown <speaker:…> markup in the source text "
            "with an existing ID or one of the default aliases."
        )
        logger.error(
            "   3. Or set chunking.strict_speaker_validation: false in the job "
            "YAML to silently fall back to the default speaker (legacy behaviour)."
        )
        logger.error("")
        logger.error(
            f"🔎 Known IDs (incl. aliases) for cross-check: {configured}"
        )
        logger.info("=" * 60)
        return False

    def execute_preprocessing(self) -> bool:
        """Execute the text preprocessing stage."""
        try:
            logger.info("Starting preprocessing stage")

            # Set available speakers in chunker for speaker-aware chunking
            try:
                available_speakers = self.file_manager.get_all_speaker_ids()
                if hasattr(self.chunker, "set_available_speakers"):
                    self.chunker.set_available_speakers(available_speakers)
                    logger.debug(
                        f"Set available speakers in chunker: {available_speakers}"
                    )
                
                # Set default speaker ID in chunker
                default_speaker_id = self.file_manager.get_default_speaker_id()
                if hasattr(self.chunker, "set_default_speaker_id"):
                    self.chunker.set_default_speaker_id(default_speaker_id)
                    logger.debug(
                        f"Set default speaker ID in chunker: {default_speaker_id}"
                    )
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
                logger.error(
                    "⚠️  The preprocessing stage cannot proceed without input text."
                )
                return False

            # Apply text preprocessing if enabled
            preprocessing_config = self.config.get("preprocessing", {})
            preprocessing_enabled = preprocessing_config.get("enabled", True)
            
            if preprocessing_enabled:
                logger.info("🔄 Text preprocessing enabled - applying transformations")
                text_preprocessor = TextPreprocessor(preprocessing_config)
                
                # Load raw input text
                try:
                    input_text = self.file_manager.get_input_text()
                    logger.debug(f"Loaded raw input text: {len(input_text)} characters")
                except FileNotFoundError as e:
                    logger.error(f"❌ Input text file not found: {e}")
                    logger.error(
                        "⚠️  preprocessing cannot proceed. Please ensure the input text file exists in the correct location."
                    )
                    return False
                except Exception as e:
                    logger.error(f"❌ Failed to load input text: {e}")
                    logger.error(
                        "⚠️  The preprocessing stage cannot proceed without valid input text."
                    )
                    return False
                
                # Apply preprocessing transformations
                processed_text = text_preprocessor._process_text_content(input_text)
                logger.info(f"✅ Text preprocessing applied: {len(input_text)} → {len(processed_text)} characters")
                
                # Use the processed text for chunking
                chunking_input_text = processed_text

                # Persist processed text for traceability next to original backup
                try:
                    text_file = self.file_manager.config["input"]["text_file"]
                    text_stem = Path(text_file).stem
                    processed_out_path = self.file_manager.texts_dir / f"original_{text_stem}_processed.txt"
                    with open(processed_out_path, "w", encoding="utf-8") as f:
                        f.write(processed_text)
                    logger.info(f"✅ Persisted processed text to: {processed_out_path.name}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not persist processed text copy: {e}")
            else:
                logger.info("⚠️ Text preprocessing disabled - using original text")
                # Load input text with graceful error handling
                try:
                    chunking_input_text = self.file_manager.get_input_text()
                    logger.debug(f"Loaded original input text: {len(chunking_input_text)} characters")
                except FileNotFoundError as e:
                    logger.error(f"❌ Input text file not found: {e}")
                    logger.error(
                        "⚠️  preprocessing cannot proceed. Please ensure the input text file exists in the correct location."
                    )
                    return False
                except Exception as e:
                    logger.error(f"❌ Failed to load input text: {e}")
                    logger.error(
                        "⚠️  The preprocessing stage cannot proceed without valid input text."
                    )
                    return False

            # Strict validation: every <speaker:id> in the source text must
            # resolve to a configured speaker (or a default-speaker alias).
            try:
                available_speakers = self.file_manager.get_all_speaker_ids()
                default_speaker_id = self.file_manager.get_default_speaker_id()
            except Exception as e:
                logger.error(
                    f"❌ Could not resolve speaker configuration for markup "
                    f"validation: {e}"
                )
                return False

            if not self._validate_speaker_markup(
                chunking_input_text, available_speakers, default_speaker_id
            ):
                logger.error(
                    "⚠️  Aborting preprocessing stage due to undefined speaker "
                    "markup. See details above."
                )
                return False

            # Chunk text (using processed or original text based on preprocessing config)
            text_chunks = self.chunker.chunk_text(chunking_input_text)
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
