"""Generation stage handler."""

import logging
from typing import Any, Dict, List

from chunking.base_chunker import TextChunk
from generation.candidate_manager import CandidateManager
from generation.tts_generator import TTSGenerator
from utils.file_manager.file_manager import FileManager
from utils.file_manager.io_handlers.candidate_io import AudioCandidate

from ..retry_logic import RetryLogic

logger = logging.getLogger(__name__)


class GenerationHandler:
    """Handles candidate generation stage."""

    def __init__(
        self,
        file_manager: FileManager,
        config: Dict[str, Any],
        tts_generator: TTSGenerator,
        candidate_manager: CandidateManager,
    ):
        self.file_manager = file_manager
        self.config = config
        self.tts_generator = tts_generator
        self.candidate_manager = candidate_manager
        self.retry_logic = RetryLogic(config, tts_generator)

    def execute_generation(self) -> bool:
        """Execute the candidate generation stage with multi-speaker support."""
        logger.info("🎙️ Starting Generation Stage")
        try:
            logger.info("")
            logger.info("▶️  Starting generation stage")

            # 1. Validate speaker configuration
            if not self._validate_speakers():
                return False

            chunks = self.file_manager.get_chunks()
            if not chunks:
                logger.error("No chunks found for generation")
                return False

            # 2. Set available speakers in chunker (if applicable)
            available_speakers = self.file_manager.get_all_speaker_ids()

            # Try to reach the chunker via file_manager (if used)
            # This is for future chunking operations with speaker support
            try:
                if hasattr(self.file_manager, "_chunk_handler") and hasattr(
                    self.file_manager._chunk_handler, "chunker"
                ):
                    chunker = self.file_manager._chunk_handler.chunker
                    if hasattr(chunker, "set_available_speakers"):
                        chunker.set_available_speakers(available_speakers)
                        logger.debug(
                            f"Set available speakers in chunker: {available_speakers}"
                        )
            except Exception as e:
                logger.debug(f"Could not set speakers in chunker (not critical): {e}")

            # 3. Legacy support: Validate reference_audio if still used
            if self.config.get("input", {}).get("reference_audio"):
                logger.debug("Legacy reference_audio detected - checking existence")
                if not self.file_manager.check_reference_audio_exists():
                    logger.error("❌ Legacy reference audio validation failed")
                    try:
                        self.file_manager.get_reference_audio()
                    except FileNotFoundError as e:
                        logger.error(str(e))
                    logger.error(
                        "⚠️  The generation stage cannot proceed without reference audio."
                    )
                    return False

                try:
                    reference_audio_path = self.file_manager.get_reference_audio()
                    self.tts_generator.load_reference_audio(str(reference_audio_path))
                    logger.info(
                        f"✅ Legacy reference audio loaded: {reference_audio_path.name}"
                    )
                    self.reference_audio_path = str(reference_audio_path)
                except FileNotFoundError as e:
                    logger.error(f"❌ Reference audio file not found: {e}")
                    logger.error(
                        "⚠️  The generation stage cannot proceed without reference audio."
                    )
                    return False
                except Exception as e:
                    logger.error(f"❌ Failed to load reference audio: {e}")
                    logger.error(
                        "⚠️  The generation stage cannot proceed without valid reference audio."
                    )
                    return False
            else:
                # Speaker system: Initialize with default speaker
                default_speaker_id = self.file_manager.get_default_speaker_id()
                try:
                    reference_audio_path = (
                        self.file_manager.get_reference_audio_for_speaker(
                            default_speaker_id
                        )
                    )
                    self.tts_generator.load_reference_audio(str(reference_audio_path))
                    logger.info(
                        f"🎭 Default speaker '{default_speaker_id}' loaded: {reference_audio_path.name}"
                    )
                    self.reference_audio_path = str(reference_audio_path)
                except Exception as e:
                    logger.error(
                        f"❌ Failed to load default speaker '{default_speaker_id}': {e}"
                    )
                    return False

            total_chunks = len(chunks)
            generation_config = self.config["generation"]
            num_candidates = generation_config["num_candidates"]

            # Pre-analyze chunks to determine which need generation
            chunks_to_generate = []
            complete_chunks = []

            for chunk in chunks:
                chunk_dir = (
                    self.file_manager.candidates_dir / f"chunk_{chunk.idx+1:03d}"
                )
                existing_file_count = 0
                if chunk_dir.exists():
                    candidate_files = list(chunk_dir.glob("candidate_*.wav"))
                    existing_file_count = len(candidate_files)

                if existing_file_count >= num_candidates:
                    complete_chunks.append((chunk.idx + 1, existing_file_count))
                else:
                    chunks_to_generate.append((chunk, existing_file_count))

            # Log summary
            logger.info(f"⚡ GENERATION PHASE: Processing {total_chunks} chunks")
            logger.info("=" * 50)

            # Log complete chunks compactly
            if complete_chunks:
                logger.info(f"📋 {len(complete_chunks)} chunks already complete:")
                for chunk_num, file_count in complete_chunks:
                    logger.info(
                        f"CHUNK {chunk_num:02d}/{total_chunks} complete ({file_count}/{num_candidates} candidates)"
                    )
                if chunks_to_generate:
                    logger.info("-" * 25)

            # Process chunks that need generation
            if chunks_to_generate:
                logger.info(
                    f"🔄 Processing {len(chunks_to_generate)} chunks requiring generation:"
                )

                for chunk, existing_file_count in chunks_to_generate:
                    logger.info("")
                    chunk_num = chunk.idx + 1
                    logger.info(f"🎯 CHUNK {chunk_num}/{total_chunks}")
                    logger.debug(f"Text length: {len(chunk.text)} characters")
                    if len(chunk.text) > 80:
                        preview = chunk.text[:80] + "..."
                    else:
                        preview = chunk.text
                    logger.debug(f'Preview: "{preview}"')
                    logger.info("-" * 50)

                    if existing_file_count > 0:
                        logger.info(
                            f"⚡ Found {existing_file_count}/{num_candidates} candidates - generating {num_candidates - existing_file_count} missing candidates"
                        )

                    missing_count = num_candidates - existing_file_count

                    if missing_count > 0:
                        chunk_dir = (
                            self.file_manager.candidates_dir
                            / f"chunk_{chunk.idx+1:03d}"
                        )
                        existing_indices = set()
                        if chunk_dir.exists():
                            candidate_files = list(chunk_dir.glob("candidate_*.wav"))
                            for candidate_file in candidate_files:
                                try:
                                    candidate_num = int(
                                        candidate_file.stem.split("_")[1]
                                    )
                                    candidate_idx = candidate_num - 1
                                    existing_indices.add(candidate_idx)
                                except (IndexError, ValueError):
                                    continue

                        missing_indices = []
                        for i in range(num_candidates):
                            if i not in existing_indices:
                                missing_indices.append(i)

                        new_candidates = self._generate_missing_candidates(
                            chunk, missing_indices
                        )

                        if not new_candidates:
                            logger.error(
                                f"❌ Failed to generate missing candidates for chunk {chunk_num}"
                            )
                            return False

                        logger.debug(
                            f"✅ Successfully generated {len(new_candidates)} missing candidates"
                        )
                    else:
                        logger.info("⚡ Generating candidates...")
                        candidates = self._generate_candidates_for_chunk(chunk)

                        if not candidates:
                            logger.error(
                                f"❌ Failed to generate candidates for chunk {chunk_num}"
                            )
                            return False

                        if not self.file_manager.save_candidates(
                            chunk.idx, candidates, overwrite_existing=True
                        ):
                            logger.error(
                                f"❌ Failed to save candidates for chunk {chunk_num}"
                            )
                            return False

                        logger.info(
                            f"✅ Successfully generated {len(candidates)} candidates"
                        )
            else:
                logger.info("✅ All chunks already have complete candidates")

            logger.info("✅ Generation stage completed successfully")
            return True

        except Exception as e:
            logger.error(f"Generation stage failed: {e}", exc_info=True)
            return False

    def _generate_candidates_for_chunk(self, chunk: TextChunk) -> List[AudioCandidate]:
        """Generate candidates for a single text chunk with speaker-aware generation."""
        logger.debug(
            f"Generating {self.candidate_manager.max_candidates} candidates for chunk '{chunk.text[:50]}...'"
        )
        try:
            generation_config = self.config["generation"]
            num_candidates = generation_config["num_candidates"]

            # Speaker-aware generation
            if hasattr(chunk, "speaker_id") and chunk.speaker_id:
                logger.debug(f"Using speaker '{chunk.speaker_id}'")

                # Switch to appropriate speaker if needed
                if hasattr(chunk, "speaker_transition") and chunk.speaker_transition:
                    logger.info(
                        f"🎭 Speaker transition detected to '{chunk.speaker_id}'"
                    )

                # Use speaker-specific generation
                candidates = self.tts_generator.generate_candidates_with_speaker(
                    text=chunk.text,
                    speaker_id=chunk.speaker_id,
                    num_candidates=num_candidates,
                    config_manager=self.file_manager,
                )
            else:
                # Legacy generation without speaker system
                logger.debug("Using legacy generation (no speaker information)")
                tts_params = generation_config.get("tts_params", {})
                candidates = self.tts_generator.generate_candidates(
                    text=chunk.text,
                    num_candidates=num_candidates,
                    exaggeration=tts_params.get("exaggeration"),
                    cfg_weight=tts_params.get("cfg_weight"),
                    temperature=tts_params.get("temperature"),
                    conservative_config=generation_config.get(
                        "conservative_candidate", None
                    ),
                    tts_params=tts_params,
                    reference_audio_path=getattr(self, "reference_audio_path", None),
                )

            # Set chunk-specific metadata
            for candidate in candidates:
                candidate.chunk_idx = chunk.idx
                candidate.chunk_text = chunk.text
                # Add speaker ID to metadata
                if hasattr(chunk, "speaker_id"):
                    if (
                        hasattr(candidate, "generation_params")
                        and candidate.generation_params
                    ):
                        candidate.generation_params["speaker_id"] = chunk.speaker_id

            return candidates

        except Exception as e:
            logger.error(f"Error generating candidates for chunk {chunk.idx+1}: {e}")
            return []

    def _generate_missing_candidates(
        self, chunk: TextChunk, missing_indices: List[int]
    ) -> List[AudioCandidate]:
        """Generate specific missing candidates for a chunk."""
        logger.info(f"Generating candidates for chunk {chunk.idx+1}")
        try:
            logger.debug(
                f"starting _generate_missing_candidates(): Generating {len(missing_indices)} candidates for indices: {missing_indices}"
            )

            one_based_indices = [idx + 1 for idx in missing_indices]

            missing_candidates = self.candidate_manager.generate_specific_candidates(
                text_chunk=chunk,
                chunk_index=chunk.idx,
                candidate_indices=one_based_indices,
                output_dir=self.file_manager.task_directory,
                reference_audio_path=getattr(self, "reference_audio_path", None),
            )

            logger.debug(
                f"Returning from candidate manager: generated {len(missing_candidates)}/{len(missing_indices)} missing candidates"
            )
            return missing_candidates

        except Exception as e:
            logger.error(f"Error in missing candidate generation: {e}")
            return []

    def generate_retry_candidates(
        self, chunk: TextChunk, max_retries: int, start_candidate_idx: int
    ) -> List[AudioCandidate]:
        """Generate additional conservative candidates if initial generation fails quality."""
        return self.retry_logic.generate_retry_candidates(
            chunk, max_retries, start_candidate_idx
        )

    def _validate_speakers(self) -> bool:
        """
        Validate speaker configuration and reference_audio files.

        Returns:
            True if all speakers are valid, False otherwise
        """
        try:
            # Validate speaker configuration structure
            speakers = self.config.get("generation", {}).get("speakers", [])
            if not speakers:
                logger.error("❌ No speakers defined in configuration")
                return False

            # Validate reference_audio files for all speakers
            validation_results = self.file_manager.validate_speakers_reference_audio()

            if not validation_results["valid"]:
                # Create detailed error message
                failed_speakers = validation_results["failed_speakers"]
                missing_files = validation_results["missing_files"]
                available_files = validation_results["available_files"]
                configured_speakers = validation_results["configured_speakers"]

                logger.info("="*60)
                logger.error("❌ SPEAKER VALIDATION FAILED")
                logger.info("="*60)
                logger.info(f"Failed speakers: {len(failed_speakers)}")
                logger.error("")
                
                # List each failed speaker with its missing file
                for speaker_id in failed_speakers:
                    missing_file = missing_files.get(speaker_id, "unknown")
                    logger.error(f"   • Speaker '{speaker_id}' → Missing file: {missing_file}")
                
                logger.error("")
                logger.error(f"📂 Available reference audio files ({len(available_files)}):")
                if available_files:
                    for i, file in enumerate(sorted(available_files), 1):
                        logger.info(f"   {i:2d}. {file}")
                else:
                    logger.error("   (No .wav files found in reference_audio directory)")
                
                logger.error("")
                logger.error(f"⚙️  Configured speakers ({len(configured_speakers)}):")
                for i, speaker_id in enumerate(configured_speakers, 1):
                    status = "✅" if speaker_id not in failed_speakers else "❌"
                    logger.info(f"   {i:2d}. {speaker_id} {status}")
                
                logger.error("")
                logger.error("💡 To fix this issue:")
                logger.error("   1. Restore the missing reference audio files to data/input/reference_audio/")
                logger.error("   2. Or update the speaker configurations to use available files")
                logger.error("   3. Or remove the invalid speakers from your configuration")
                logger.error("="*60)
                
                return False

            logger.info(
                f"✅ All {len(validation_results['configured_speakers'])} speakers validated: {validation_results['configured_speakers']}"
            )
            return True

        except Exception as e:
            logger.error(f"Speaker validation error: {e}")
            return False
