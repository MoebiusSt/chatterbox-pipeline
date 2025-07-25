"""Retry logic for candidate generation."""

import logging
from pathlib import Path
from typing import List

import torch

from chunking.base_chunker import TextChunk
from generation.tts_generator import TTSGenerator
from utils.file_manager.io_handlers.candidate_io import AudioCandidate

logger = logging.getLogger(__name__)


class RetryLogic:
    """Handles retry logic for candidate generation."""

    def __init__(self, config: dict, tts_generator: TTSGenerator):
        self.config = config
        self.tts_generator = tts_generator

    def generate_retry_candidates(
        self, chunk: TextChunk, max_retries: int, start_candidate_idx: int
    ) -> List[AudioCandidate]:
        """Generate additional conservative candidates if initial generation fails quality."""
        retry_candidates = []
        try:
            generation_config = self.config["generation"]

            # Get speaker-specific conservative config
            conservative_config = self._get_conservative_config_for_chunk(
                chunk, generation_config
            )

            if not conservative_config.get("enabled", False):
                logger.warning(
                    f"⚠️ Conservative candidate not enabled for speaker '{getattr(chunk, 'speaker_id', 'unknown')}', using default values for retries"
                )
                base_exaggeration = 0.45
                base_cfg_weight = 0.4
                base_temperature = 0.8
            else:
                base_exaggeration = conservative_config.get("exaggeration", 0.45)
                base_cfg_weight = conservative_config.get("cfg_weight", 0.4)
                base_temperature = conservative_config.get("temperature", 0.8)

            logger.debug(
                f"Generating {max_retries} retry candidates with conservative base values: "
                f"exag={base_exaggeration:.2f}, cfg={base_cfg_weight:.2f}, temp={base_temperature:.2f}"
            )

            for i in range(max_retries):
                try:
                    if i == 0:
                        variation_factor = 0.0
                    else:
                        variation_factor = (
                            (i - 1) / max(1, max_retries - 2)
                        ) * 2.0 - 1.0
                        variation_factor *= 0.05

                    retry_exaggeration = max(
                        0.1, min(1.0, base_exaggeration + variation_factor)
                    )
                    retry_cfg_weight = max(
                        0.1, min(1.0, base_cfg_weight + variation_factor)
                    )
                    retry_temperature = max(
                        0.1, min(2.0, base_temperature + variation_factor)
                    )

                    retry_seed = (
                        self.tts_generator.seed
                        + (chunk.idx * 1000)
                        + (start_candidate_idx + i) * 100
                    )

                    logger.debug(
                        f"Retry {i+1}/{max_retries}: exag={retry_exaggeration:.3f}, "
                        f"cfg={retry_cfg_weight:.3f}, temp={retry_temperature:.3f}, seed={retry_seed}"
                    )

                    torch.manual_seed(retry_seed)

                    retry_audio = self.tts_generator.generate_single(
                        text=chunk.text,
                        exaggeration=retry_exaggeration,
                        cfg_weight=retry_cfg_weight,
                        temperature=retry_temperature,
                    )

                    candidate_idx = start_candidate_idx + i
                    generation_params = {
                        "exaggeration": retry_exaggeration,
                        "cfg_weight": retry_cfg_weight,
                        "temperature": retry_temperature,
                        "seed": retry_seed,
                        "type": "RETRY_CONSERVATIVE",
                        "variation_factor": variation_factor,
                        "retry_attempt": i + 1,
                    }

                    retry_candidate = AudioCandidate(
                        chunk_idx=chunk.idx,
                        candidate_idx=candidate_idx,
                        audio_path=Path(),
                        audio_tensor=retry_audio,
                        generation_params=generation_params,
                        chunk_text=chunk.text,
                    )

                    retry_candidates.append(retry_candidate)

                    logger.debug(
                        f"✅ Generated retry candidate {i+1} (idx={candidate_idx}) with duration={retry_audio.shape[-1]/24000:.2f}s\n"
                    )

                except Exception as e:
                    logger.error(f"Failed to generate retry candidate {i+1}: {e}")
                    continue

            logger.debug(
                f"Successfully generated {len(retry_candidates)}/{max_retries} retry candidates"
            )
            return retry_candidates

        except Exception as e:
            logger.error(f"Error in retry candidate generation: {e}")
            return []

    def _get_conservative_config_for_chunk(
        self, chunk: TextChunk, generation_config: dict
    ) -> dict:
        """
        Get conservative candidate configuration for a specific chunk's speaker.

        Args:
            chunk: TextChunk with speaker information
            generation_config: Generation configuration

        Returns:
            Conservative candidate configuration dict
        """
        # Get speaker ID from chunk, fallback to default
        speaker_id = getattr(chunk, "speaker_id", "default")

        # Look for speaker-specific config
        speakers = generation_config.get("speakers", [])

        # Find speaker configuration
        for speaker in speakers:
            if speaker.get("id") == speaker_id:
                return speaker.get("conservative_candidate", {})

        # Fallback to explicit default speaker
        default_speaker = generation_config.get("default_speaker")
        if default_speaker and speakers:
            logger.debug(
                f"Speaker '{speaker_id}' not found, using default speaker '{default_speaker}' for conservative config"
            )
            for speaker in speakers:
                if speaker.get("id") == default_speaker:
                    return speaker.get("conservative_candidate", {})

        # Final fallback to first speaker
        if speakers:
            logger.debug(
                "Default speaker not found, using first speaker for conservative config"
            )
            return speakers[0].get("conservative_candidate", {})

        # Legacy fallback: check if conservative_candidate exists at generation level
        legacy_config = generation_config.get("conservative_candidate", {})
        if legacy_config:
            logger.debug("Using legacy conservative_candidate configuration")
            return legacy_config

        # No conservative configuration found
        logger.warning(
            f"No conservative_candidate configuration found for speaker '{speaker_id}'"
        )
        return {}
