"""Generation stage handler."""

import logging
from typing import Any, Dict, List, Set

from chunking.base_chunker import TextChunk
from generation.candidate_manager import CandidateManager
from generation.tts_generator import TTSGenerator
from utils.file_manager.file_manager import FileManager
from utils.file_manager.io_handlers.candidate_io import AudioCandidate
from utils.language_registry import validate_languages_for_model

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

    def _validate_languages_for_model(self) -> None:
        """
        Check whether all speaker languages defined in the config are supported by the
        active model type, and emit warnings for any unsupported language codes.

        This is a soft check: warnings are logged but execution continues.
        """
        model_type = self.config.get("generation", {}).get("model_type", "standard")
        speakers: List[Dict[str, Any]] = self.config.get("generation", {}).get("speakers", [])
        default_speaker_id = self.config.get("generation", {}).get("default_speaker", "")

        # Collect language codes that will actually be used by configured speakers
        used_languages: Set[str] = set()
        for speaker in speakers:
            lang = speaker.get("language")
            if lang:
                used_languages.add(str(lang).strip().lower())

        # Also include the default_speaker language explicitly (should be in speakers already,
        # but keep as belt-and-suspenders)
        for speaker in speakers:
            if speaker.get("id") == default_speaker_id:
                lang = speaker.get("language")
                if lang:
                    used_languages.add(str(lang).strip().lower())
                break

        if not used_languages:
            return

        warnings = validate_languages_for_model(model_type, used_languages)
        for msg in warnings:
            logger.warning(f"⚠️ Language validation: {msg}")

    def execute_generation(self) -> bool:
        """Execute the candidate generation stage with multi-speaker support."""
        logger.info("🎙️ Starting Generation Stage")
        try:
            logger.info("")
            logger.info("▶️  Starting generation stage")

            # 1. Validate speaker configuration
            if not self._validate_speakers():
                return False

            # 1b. Soft language check: warn if any speaker language is not supported by model
            self._validate_languages_for_model()

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

            # 3. Initialize with default speaker using switch_speaker for proper state tracking
            default_speaker_id = self.file_manager.get_default_speaker_id()
            try:
                # Use switch_speaker to properly initialize and track current speaker
                self.tts_generator.switch_speaker(default_speaker_id, self.file_manager)
                
                # Get reference_audio_path for legacy compatibility
                reference_audio_path = (
                    self.file_manager.get_reference_audio_for_speaker(
                        default_speaker_id
                    )
                )
                self.reference_audio_path = str(reference_audio_path)
                
                logger.info(
                    f"🎭 Default speaker '{default_speaker_id}' loaded: {reference_audio_path.name}"
                )
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

                        # Re-rendering replaces the audio content of every
                        # candidate slot, so any prior USER selection no
                        # longer refers to the same audio. Drop the marker
                        # so the next validation may realign automatically.
                        try:
                            self.file_manager.clear_user_selections([chunk.idx])
                        except Exception as e:
                            logger.debug(
                                f"clear_user_selections failed for chunk {chunk_num}: {e}"
                            )

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
        """Generate candidates for a single text chunk with speaker-aware generation.

        Installs a ``candidate_ready`` hook on the TTS generator so each
        freshly rendered candidate is persisted to disk immediately. If the
        process crashes mid-batch, earlier candidates are retained and the
        resume path in :meth:`execute_generation` (which counts existing wav
        files per chunk) fills only the missing indices.
        """
        logger.debug(
            f"Generating {self.candidate_manager.max_candidates} candidates for chunk '{chunk.text[:50]}...'"
        )

        def _save_hook(cand: AudioCandidate) -> None:
            # The TTS paths create candidates with ``chunk_idx=0``; the closure
            # fixes the index so the CandidateIOHandler writes to the right
            # ``chunk_XXX`` directory and ``candidates_metadata.json`` entry.
            cand.chunk_idx = chunk.idx
            cand.chunk_text = chunk.text
            if hasattr(chunk, "speaker_id") and chunk.speaker_id:
                if cand.generation_params is not None:
                    cand.generation_params.setdefault("speaker_id", chunk.speaker_id)
            self.file_manager.save_candidate(chunk.idx, cand, overwrite_existing=True)

        self.tts_generator.set_candidate_ready_hook(_save_hook)
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
                # Use default speaker when no speaker ID is set
                logger.debug("No speaker ID found, using default speaker")
                default_speaker_id = self.file_manager.get_default_speaker_id()
                candidates = self.tts_generator.generate_candidates_with_speaker(
                    text=chunk.text,
                    speaker_id=default_speaker_id,
                    num_candidates=num_candidates,
                    config_manager=self.file_manager,
                )

            # Final safety net: ensure chunk metadata is set even when the
            # hook was skipped for any reason (e.g. the hook raised and was
            # swallowed by _emit_candidate_ready).
            for candidate in candidates:
                candidate.chunk_idx = chunk.idx
                candidate.chunk_text = chunk.text
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
        finally:
            # Ensure the hook never leaks into later (retry/specific) paths
            # where candidate indices get remapped after generation.
            self.tts_generator.set_candidate_ready_hook(None)

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
        try:
            # Ensure the correct speaker is active before generating retry candidates
            speaker_id = getattr(chunk, "speaker_id", None)
            if not speaker_id and hasattr(self.file_manager, "get_default_speaker_id"):
                speaker_id = self.file_manager.get_default_speaker_id()

            if speaker_id:
                self.tts_generator.switch_speaker(speaker_id, self.file_manager)
                logger.debug(
                    f"Retry generation: ensured speaker '{speaker_id}' is active for chunk {chunk.idx+1}"
                )
        except Exception as e:
            logger.error(f"Failed to ensure correct speaker before retry generation: {e}")

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
