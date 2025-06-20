"""Generation stage handler."""

import logging
from typing import Any, Dict, List

from generation.candidate_manager import CandidateManager
from generation.tts_generator import TTSGenerator
from utils.file_manager import AudioCandidate, FileManager, TextChunk

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
        """Execute the candidate generation stage."""
        logger.info("🎙️ Starting Generation Stage")
        try:
            logger.info("")
            logger.info("Starting generation stage")

            chunks = self.file_manager.get_chunks()
            if not chunks:
                logger.error("No chunks found for generation")
                return False

            reference_audio_path = self.file_manager.get_reference_audio()
            self.tts_generator.load_reference_audio(str(reference_audio_path))

            total_chunks = len(chunks)
            logger.info(f"⚡ GENERATION PHASE: Processing {total_chunks} chunks")
            logger.info("=" * 50)

            for chunk in chunks:
                chunk_num = chunk.idx + 1
                logger.info("")
                logger.info(f"🎯 CHUNK {chunk_num}/{total_chunks}")
                logger.debug(f"Text length: {len(chunk.text)} characters")
                if len(chunk.text) > 80:
                    preview = chunk.text[:80] + "..."
                else:
                    preview = chunk.text
                logger.debug(f'Preview: "{preview}"')
                logger.info("-" * 50)

                existing_candidates = self.file_manager.get_candidates(chunk.idx)
                chunk_candidates = existing_candidates.get(chunk.idx, [])

                generation_config = self.config["generation"]
                num_candidates = generation_config["num_candidates"]

                chunk_dir = (
                    self.file_manager.candidates_dir / f"chunk_{chunk.idx+1:03d}"
                )
                existing_file_count = 0
                if chunk_dir.exists():
                    candidate_files = list(chunk_dir.glob("candidate_*.wav"))
                    existing_file_count = len(candidate_files)

                if existing_file_count >= num_candidates:
                    logger.debug(
                        f"✓ Candidates already exist for chunk {chunk_num} ({existing_file_count}/{num_candidates}), skipping"
                    )
                    continue
                elif existing_file_count > 0:
                    logger.info(
                        f"⚡ Found {existing_file_count}/{num_candidates} candidates - generating {num_candidates - existing_file_count} missing candidates"
                    )

                missing_count = num_candidates - existing_file_count

                if missing_count > 0:
                    existing_indices = set()
                    if chunk_dir.exists():
                        candidate_files = list(chunk_dir.glob("candidate_*.wav"))
                        for candidate_file in candidate_files:
                            try:
                                candidate_num = int(candidate_file.stem.split("_")[1])
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
                            f"❌ Failed to generate missing candidates for chunk {chunk_num+1}"
                        )
                        return False

                    logger.debug(
                        f"✅ Successfully generated {len(new_candidates)} missing candidates"
                    )
                else:
                    logger.info(f"⚡ Generating candidates...")
                    candidates = self._generate_candidates_for_chunk(chunk)

                    if not candidates:
                        logger.error(
                            f"❌ Failed to generate candidates for chunk {chunk_num+1}"
                        )
                        return False

                    if not self.file_manager.save_candidates(
                        chunk.idx, candidates, overwrite_existing=True
                    ):
                        logger.error(
                            f"❌ Failed to save candidates for chunk {chunk_num+1}"
                        )
                        return False

                    logger.info(
                        f"✅ Successfully generated {len(candidates)} candidates"
                    )
            logger.info("✅ Generation stage completed successfully")
            return True

        except Exception as e:
            logger.error(f"Generation stage failed: {e}", exc_info=True)
            return False

    def _generate_candidates_for_chunk(self, chunk: TextChunk) -> List[AudioCandidate]:
        """Generate candidates for a single text chunk."""
        logger.debug(
            f"Generating {self.candidate_manager.max_candidates} candidates for chunk '{chunk.text[:50]}...'"
        )
        try:
            generation_config = self.config["generation"]
            num_candidates = generation_config["num_candidates"]

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
            )

            for candidate in candidates:
                candidate.chunk_idx = chunk.idx
                candidate.chunk_text = chunk.text

            return candidates

        except Exception as e:
            logger.error(f"Error generating candidates for chunk {chunk.idx+1}: {e}")
            return []

    def _generate_missing_candidates(
        self, chunk: TextChunk, missing_indices: List[int]
    ) -> List[AudioCandidate]:
        """Generate specific missing candidates for a chunk."""
        logger.info(
            f"Generating missing candidates {missing_indices} for chunk {chunk.idx+1}"
        )
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
            )

            logger.debug(
                f"Returning from candidate manager: generated {len(missing_candidates)}/{len(missing_indices)} missing candidates"
            )
            return missing_candidates

        except Exception as e:
            logger.error(f"Error in missing candidate generation: {e}")
            return []

    def delete_whisper_file(self, chunk_index: int, candidate_idx: int):
        """Delete a specific whisper validation file."""
        self.candidate_manager._delete_whisper_file(
            self.file_manager.task_directory, chunk_index, candidate_idx
        )

    def generate_retry_candidates(
        self, chunk: TextChunk, max_retries: int, start_candidate_idx: int
    ) -> List[AudioCandidate]:
        """Generate additional conservative candidates if initial generation fails quality."""
        return self.retry_logic.generate_retry_candidates(
            chunk, max_retries, start_candidate_idx
        )
