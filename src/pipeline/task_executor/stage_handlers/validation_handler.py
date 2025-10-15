"""Validation stage handler."""

import logging
import time
from typing import Any, Dict, List, Optional

from chunking.base_chunker import TextChunk
from utils.file_manager.file_manager import FileManager
from utils.file_manager.io_handlers.candidate_io import AudioCandidate
from validation.quality_scorer import QualityScorer
from validation.preprocessors.tail_trimmer import TailTrimmer
from validation.prosody_scorer import ProsodyScorer
from validation.mos_providers.utmos import UTMOSProvider
from validation.mos_providers.nisqa import NISQAProvider
from validation.mos_providers.combined import CombinedMOSProvider
from validation.whisper_validator import ValidationResult, WhisperValidator

logger = logging.getLogger(__name__)


class ValidationHandler:

    def __init__(
        self,
        file_manager: FileManager,
        config: Dict[str, Any],
        whisper_validator: WhisperValidator,
        quality_scorer: QualityScorer,
        generation_handler,  # Import cycle avoidance
    ):
        self.file_manager = file_manager
        self.config = config
        self.whisper_validator = whisper_validator
        self.quality_scorer = quality_scorer
        self.generation_handler = generation_handler
        # Ensure validator sees configuration (for prompt, normalization, etc.)
        try:
            setattr(self.whisper_validator, "_config", self.config)
        except Exception:
            pass
        # Initialize tail trimmer (optional, controlled by config)
        try:
            self.tail_trimmer = TailTrimmer(config=self.config, sample_rate=self.config.get("audio", {}).get("sample_rate", 24000))
        except Exception:
            self.tail_trimmer = None
        # Initialize MOS provider (optional) and Prosody scorer (should not fail if MOS missing)
        mos_provider = None
        try:
            mos_cfg = self.config.get("validation", {}).get("mos", {})
            provider_name = str(mos_cfg.get("provider", "combined")).lower()
            # enforce de/en gate per requirement, even if config wider
            enabled_langs = mos_cfg.get("enabled_languages", ["de", "en"]) or ["de", "en"]
            if provider_name == "utmos":
                mos_provider = UTMOSProvider(
                    enabled_languages=enabled_langs,
                    prefer_utmosv2=bool(mos_cfg.get("prefer_utmosv2", False)),
                )
            elif provider_name == "nisqa":
                weights_path = mos_cfg.get("nisqa_weights_path")
                mos_provider = NISQAProvider(enabled_languages=enabled_langs, weights_path=weights_path)
            elif provider_name == "combined":
                weights_path = mos_cfg.get("nisqa_weights_path")
                nisqa = NISQAProvider(enabled_languages=enabled_langs, weights_path=weights_path)
                utmos = UTMOSProvider(
                    enabled_languages=enabled_langs,
                    prefer_utmosv2=bool(mos_cfg.get("prefer_utmosv2", False)),
                )
                mos_provider = CombinedMOSProvider([nisqa, utmos], enabled_languages=enabled_langs)
            else:
                weights_path = mos_cfg.get("nisqa_weights_path")
                nisqa = NISQAProvider(enabled_languages=enabled_langs, weights_path=weights_path)
                utmos = UTMOSProvider(
                    enabled_languages=enabled_langs,
                    prefer_utmosv2=bool(mos_cfg.get("prefer_utmosv2", False)),
                )
                mos_provider = CombinedMOSProvider([nisqa, utmos], enabled_languages=enabled_langs)
        except Exception:
            mos_provider = None
        try:
            self.prosody_scorer = ProsodyScorer(
                config=self.config,
                mos_provider=mos_provider,
                sample_rate=self.config.get("audio", {}).get("sample_rate", 24000),
            )
        except Exception:
            self.prosody_scorer = None

    def execute_validation(self) -> bool:
        try:
            logger.info("=" * 50)
            logger.info("Starting validation stage")

            chunks = self.file_manager.get_chunks()
            all_candidates = self.file_manager.get_candidates()

            if not chunks or not all_candidates:
                logger.error("No chunks or candidates found for validation")
                return False
            
            # Enhance chunks with language_id from speaker configuration
            self._enhance_chunks_with_language_id(chunks)

            validation_results = {}
            logger.info(f"🚦 VALIDATION PHASE: Processing {len(chunks)} chunks")

            for chunk in chunks:
                chunk_candidates = all_candidates.get(chunk.idx, [])
                if not chunk_candidates:
                    logger.warning(f"No candidates found for chunk {chunk.idx}")
                    continue

                chunk_num = chunk.idx + 1
                logger.info("")
                logger.info(f"🎯 CHUNK {chunk_num}/{len(chunks)} - {len(chunk_candidates)} candidates")
                logger.debug(f"Candidates to validate: {len(chunk_candidates)}")
                logger.info("-" * 40)

                chunk_results = {}

                # Inline incremental candidate progress line (single line updated step by step)
                progress_line = []

                for candidate in chunk_candidates:
                    candidate_num = candidate.candidate_idx + 1
                    logger.debug(f"🔍 Validating candidate {candidate_num}...")
                    try:
                        progress_line.append(str(candidate_num))
                        # Use carriage return to update the same line in supported terminals
                        logger.info("\rCandidates: " + ", ".join(progress_line))
                    except Exception:
                        pass

                    # Resolve language for this candidate
                    try:
                        gen_params = getattr(candidate, "generation_params", None) or {}
                        base_lang = getattr(chunk, "language_id", None)
                        cand_language = (
                            gen_params.get("language_id")
                            or gen_params.get("language")
                            or base_lang
                            or "en"
                        )
                        logger.debug(
                            f"🌐 Using language '{cand_language}' for chunk {chunk.idx + 1}, candidate {candidate_num}"
                        )
                    except Exception:
                        cand_language = getattr(chunk, "language_id", "en")

                    # Check if whisper result already exists
                    existing_whisper = self.file_manager.get_whisper(
                        chunk.idx, candidate.candidate_idx
                    )
                    if candidate.candidate_idx in existing_whisper:
                        logger.debug(
                            f"✓ Whisper result already exists for candidate {candidate_num}"
                        )
                        # Backfill prosody/final score/gates if missing in prior runs
                        prev = existing_whisper[candidate.candidate_idx]
                        needs_backfill = not isinstance(prev.get("prosody"), dict) or ("final_selection_score" not in prev)
                        if needs_backfill and candidate.audio_tensor is not None:
                            try:
                                # Recompute prosody
                                prosody_details = None
                                prosody_score = 0.0
                                mos_unit = 0.0
                                if hasattr(self, "prosody_scorer") and self.prosody_scorer:
                                    prosody_details = self.prosody_scorer.score(
                                        candidate.audio_tensor,
                                        language=getattr(chunk, 'language_id', 'en'),
                                        original_text=chunk.text,
                                        asr_transcription=str(prev.get("transcription") or ""),
                                    )
                                    prosody_score = float(prosody_details.get("prosody_score", 0.0))
                                    mos_unit = float(prosody_details.get("subscores", {}).get("mos", 0.0))
                                # Compute selection score using existing overall_quality_score
                                sel_cfg = self.config.get("validation", {}).get("prosody", {}).get("select", {})
                                alpha = float(sel_cfg.get("alpha_quality", 0.50))
                                beta = float(sel_cfg.get("beta_prosody", 0.35))
                                gamma = float(sel_cfg.get("gamma_mos", 0.15))
                                overall = float(prev.get("overall_quality_score", 0.0))
                                final_selection_score = alpha * overall + beta * prosody_score + gamma * mos_unit
                                # Gates
                                gating = self.config.get("validation", {}).get("selection", {}).get("gating", {})
                                require_mos = bool(gating.get("require_mos", True))
                                require_similarity = bool(gating.get("require_similarity", True))
                                passes_mos = True
                                if require_mos and isinstance(prosody_details, dict):
                                    min_mos = float(self.config.get("validation", {}).get("mos", {}).get("min_mos", 3.5))
                                    raw_mos = prosody_details.get("raw_mos")
                                    passes_mos = (raw_mos is None) or (float(raw_mos) >= min_mos)
                                passes_similarity = True
                                if require_similarity:
                                    passes_similarity = bool(prev.get("is_valid", True))
                                # Merge and save
                                prev["prosody"] = prosody_details
                                prev["final_selection_score"] = final_selection_score
                                prev["passes_mos_gate"] = passes_mos
                                prev["passes_similarity_gate"] = passes_similarity
                                self.file_manager.save_whisper(chunk.idx, candidate.candidate_idx, prev)
                            except Exception as e:
                                logger.debug(f"Backfill prosody failed: {e}")
                        chunk_results[candidate.candidate_idx] = existing_whisper[candidate.candidate_idx]
                        continue

                    candidate.chunk_text = chunk.text

                    # Tail-trim candidate audio before validation to remove trailing non-speech
                    try:
                        if hasattr(self, "tail_trimmer") and self.tail_trimmer and candidate.audio_tensor is not None:
                            trimmed_audio, removed_tail, trim_meta = self.tail_trimmer.trim(
                                candidate.audio_tensor,
                                language=cand_language,
                                original_text=chunk.text,
                            )
                            if trimmed_audio is not None and trimmed_audio.numel() > 0:
                                candidate.audio_tensor = trimmed_audio
                                logger.debug(f"✂️  Tail-trim applied (method={trim_meta.get('method')}, cut={trim_meta.get('cut_sample')}) for candidate {candidate_num}")
                                # Optional: persist removed tail for inspection if enabled in config
                                try:
                                    tt_cfg = self.config.get("validation", {}).get("preprocessing", {}).get("tail_trim", {})
                                    sample_rate = self.config.get("audio", {}).get("sample_rate", 24000)
                                    # Save removed tail for debugging
                                    if removed_tail is not None and bool(tt_cfg.get("debug_save_removed_tail", False)):
                                        from pathlib import Path
                                        import torchaudio as ta
                                        out_dir = self.file_manager.task_directory / "debug_tail_trimmer"
                                        out_dir.mkdir(parents=True, exist_ok=True)
                                        out_path = out_dir / f"chunk_{chunk.idx+1:03d}_cand_{candidate.candidate_idx+1:02d}_removed_tail.wav"
                                        ta.save(str(out_path), removed_tail.unsqueeze(0).cpu(), sample_rate)

                                    # Persist kept (trimmed) candidate alongside original if enabled
                                    if bool(tt_cfg.get("persist_trimmed_candidate", True)) and trim_meta.get("cut_sample") is not None:
                                        from pathlib import Path
                                        import torchaudio as ta
                                        cand_dir = self.file_manager.candidates_dir / f"chunk_{chunk.idx+1:03d}"
                                        cand_dir.mkdir(parents=True, exist_ok=True)
                                        kept_path = cand_dir / f"candidate_{candidate.candidate_idx+1:02d}_trimmed.wav"
                                        ta.save(str(kept_path), trimmed_audio.unsqueeze(0).cpu(), sample_rate)
                                        trim_meta["trimmed_path"] = str(kept_path)
                                except Exception as e:
                                    logger.debug(f"Failed persisting tail-trim artifacts: {e}")
                    except Exception as e:
                        logger.debug(f"Tail-trim failed for candidate {candidate_num}: {e}")

                    # Perform Whisper validation with language-specific model
                    chunk_language = cand_language
                    whisper_result = self.whisper_validator.validate_candidate(
                        candidate, chunk.text, language=chunk_language
                    )

                    if whisper_result:
                        # Perform quality scoring
                        quality_result = self.quality_scorer.score_candidate(
                            candidate, whisper_result
                        )

                        # Prosody + MOS scoring (optional)
                        prosody_details = None
                        prosody_score = 0.0
                        mos_value = None
                        if candidate.audio_tensor is not None and getattr(candidate.audio_tensor, "numel", lambda: 0)() > 0:
                            # Ensure scorer exists even if init failed earlier
                            if not getattr(self, "prosody_scorer", None):
                                try:
                                    self.prosody_scorer = ProsodyScorer(
                                        config=self.config,
                                        mos_provider=None,
                                        sample_rate=self.config.get("audio", {}).get("sample_rate", 24000),
                                    )
                                except Exception:
                                    self.prosody_scorer = None
                            if getattr(self, "prosody_scorer", None):
                                try:
                                    prosody_details = self.prosody_scorer.score(
                                        candidate.audio_tensor,
                                        language=cand_language,
                                        original_text=chunk.text,
                                        asr_transcription=whisper_result.transcription,
                                    )
                                    prosody_score = float(prosody_details.get("prosody_score", 0.0))
                                    mos_value = prosody_details.get("raw_mos")
                                except Exception as e:
                                    logger.debug(f"Prosody scoring failed (chunk {chunk.idx+1}, cand {candidate_num}): {e}")
                                    prosody_details = {
                                        "enabled": False,
                                        "subscores": {
                                            "semantic_alignment": 0.0,
                                            "flow": 0.0,
                                            "liveliness": 0.0,
                                            "intelligibility": 0.0,
                                            "mos": 0.0,
                                        },
                                        "prosody_score": 0.0,
                                    }
                                    prosody_score = 0.0

                        # Final selection score
                        sel_cfg = self.config.get("validation", {}).get("prosody", {}).get("select", {})
                        alpha = float(sel_cfg.get("alpha_quality", 0.50))
                        beta = float(sel_cfg.get("beta_prosody", 0.35))
                        gamma = float(sel_cfg.get("gamma_mos", 0.15))
                        # Derive mos_unit from prosody_details (already unitized) if available
                        mos_unit = 0.0
                        if isinstance(prosody_details, dict):
                            mos_unit = float(prosody_details.get("subscores", {}).get("mos", 0.0))
                        final_selection_score = (
                            alpha * float(quality_result.overall_score)
                            + beta * float(prosody_score)
                            + gamma * float(mos_unit)
                        )

                        # Apply gating
                        gating = self.config.get("validation", {}).get("selection", {}).get("gating", {})
                        require_mos = bool(gating.get("require_mos", True))
                        require_similarity = bool(gating.get("require_similarity", True))
                        passes_mos = True
                        if require_mos and isinstance(prosody_details, dict):
                            min_mos = float(self.config.get("validation", {}).get("mos", {}).get("min_mos", 3.5))
                            raw_mos = prosody_details.get("raw_mos")
                            # If MOS is unavailable (None), do not block selection
                            passes_mos = (raw_mos is None) or (float(raw_mos) >= min_mos)
                        passes_similarity = True
                        if require_similarity:
                            passes_similarity = bool(whisper_result.is_valid)

                        combined_result = {
                            "is_valid": whisper_result.is_valid,
                            "transcription": whisper_result.transcription,
                            "similarity_score": whisper_result.similarity_score,
                            "quality_score": whisper_result.quality_score,
                            "validation_time": whisper_result.validation_time,
                            "error_message": whisper_result.error_message,
                            "overall_quality_score": quality_result.overall_score,
                            "quality_details": quality_result.details,
                            "speaker_id": chunk.speaker_id,
                            "language_id": cand_language,
                            # Prosody/MOS
                            "prosody": prosody_details,
                            # Debug transparency for MOS provider chain
                            "mos_provider": (
                                getattr(getattr(self, "prosody_scorer", None), "mos_provider", None).__class__.__name__
                                if (getattr(self, "prosody_scorer", None) is not None and getattr(getattr(self, "prosody_scorer", None), "mos_provider", None) is not None)
                                else None
                            ),
                            "mos_details": (
                                getattr(getattr(getattr(self, "prosody_scorer", None), "mos_provider", None), "_last_details", None)
                                if (getattr(self, "prosody_scorer", None) is not None and getattr(getattr(self, "prosody_scorer", None), "mos_provider", None) is not None)
                                else None
                            ),
                            "final_selection_score": final_selection_score,
                            "passes_mos_gate": passes_mos,
                            "passes_similarity_gate": passes_similarity,
                        }
                        # Attach tail_trim metadata if available
                        try:
                            if 'trim_meta' in locals() and isinstance(trim_meta, dict):
                                combined_result["tail_trim"] = {
                                    "method": trim_meta.get("method"),
                                    "match_type": trim_meta.get("match_type"),
                                    "cut_sample": trim_meta.get("cut_sample"),
                                    "kept_post_silence_ms": trim_meta.get("kept_post_silence_ms"),
                                    "fade_out_ms": trim_meta.get("fade_out_ms"),
                                    "matched_words": trim_meta.get("matched_words"),
                                    "match_ratio": trim_meta.get("match_ratio"),
                                    "language_gated": trim_meta.get("language_gated", False),
                                    "trimmed_path": trim_meta.get("trimmed_path"),
                                }
                        except Exception:
                            pass

                        # Save whisper result
                        self.file_manager.save_whisper(
                            chunk.idx, candidate.candidate_idx, combined_result
                        )
                        chunk_results[candidate.candidate_idx] = combined_result

                        # Log validation result
                        status = "✅ Valid" if whisper_result.is_valid else "❌ Invalid"
                        logger.debug(
                            f"{status} - candidate {candidate_num} (similarity: {whisper_result.similarity_score:.3f}, quality: {whisper_result.quality_score:.3f}, overall: {quality_result.overall_score:.3f})"
                        )
                    else:
                        logger.warning(
                            f"❌ Whisper validation failed for candidate {candidate_num}"
                        )

                validation_results[chunk.idx] = chunk_results
                valid_count = sum(
                    1
                    for result in chunk_results.values()
                    if result.get("is_valid", False)
                )

                # Log summary
                if chunk_results:
                    overall_scores = [
                        result.get("overall_quality_score", 0.0)
                        for result in chunk_results.values()
                    ]
                    selection_scores = [
                        result.get("final_selection_score", result.get("overall_quality_score", 0.0))
                        for result in chunk_results.values()
                    ]
                    min_overall = min(overall_scores)
                    max_overall = max(overall_scores)
                    min_select = min(selection_scores)
                    max_select = max(selection_scores)
                    logger.info(
                        f"✅ Validation complete: {valid_count}/{len(chunk_candidates)} candidates valid (overall: {min_overall:.3f}..{max_overall:.3f}, final_selection: {min_select:.3f}..{max_select:.3f})"
                    )
                else:
                    logger.info(
                        f"⚠️ Validation complete: {valid_count}/{len(chunk_candidates)} candidates valid"
                    )

                # Retry logic if no valid candidates
                if not self._handle_retry_logic(
                    chunk, chunk_results, all_candidates, validation_results
                ):
                    return False

            # Create and save consolidated whisper metrics (without selections)
            metrics = self._create_enhanced_metrics(
                chunks, all_candidates, validation_results
            )

            if not self.file_manager.save_metrics(metrics):
                logger.error("Failed to save validation metrics")
                return False

            logger.info("")
            logger.info("✅ Validation stage completed successfully")
            return True

        except Exception as e:
            logger.error(f"Validation stage failed: {e}", exc_info=True)
            return False

    def _handle_retry_logic(
        self,
        chunk: TextChunk,
        chunk_results: dict,
        all_candidates: dict,
        validation_results: dict,
    ) -> bool:
        """Handle retry logic for chunks with no valid candidates."""
        valid_count = sum(
            1 for result in chunk_results.values() if result.get("is_valid", False)
        )

        if valid_count > 0:
            return True

        generation_config = self.config.get("generation", {})
        max_retries = generation_config.get("max_retries", 0)
        num_candidates = generation_config.get("num_candidates", 5)
        max_total_candidates = num_candidates + max_retries

        # Check actual file count
        chunk_dir = self.file_manager.candidates_dir / f"chunk_{chunk.idx+1:03d}"
        highest_candidate_idx = -1
        if chunk_dir.exists():
            candidate_files = list(chunk_dir.glob("candidate_*.wav"))
            for candidate_file in candidate_files:
                try:
                    candidate_num = int(candidate_file.stem.split("_")[1])
                    candidate_idx = candidate_num - 1
                    highest_candidate_idx = max(highest_candidate_idx, candidate_idx)
                except (IndexError, ValueError):
                    continue

        max_candidate_idx = max_total_candidates - 1
        already_at_max = highest_candidate_idx >= max_candidate_idx

        if already_at_max:
            logger.warning(
                f"⚠️ All candidates invalid but maximum retry limit reached (max: {max_total_candidates} candidates)"
            )
            return True

        if max_retries > 0:
            next_candidate_idx = highest_candidate_idx + 1
            remaining_slots = max_total_candidates - (highest_candidate_idx + 1)
            actual_retries = min(max_retries, remaining_slots)

            logger.info(
                f"⚠️ All candidates invalid - generating {actual_retries} retry candidates"
            )

            # Generate retry candidates
            retry_candidates = self.generation_handler.generate_retry_candidates(
                chunk, actual_retries, next_candidate_idx
            )

            # Delete whisper files for retry candidates before re-validation
            for retry_candidate in retry_candidates:
                self.file_manager.delete_whisper(
                    chunk.idx, retry_candidate.candidate_idx
                )

            if retry_candidates:
                logger.info(
                    f"🔁 Validating {len(retry_candidates)} retry candidates..."
                )

                # Validate retry candidates
                for retry_candidate in retry_candidates:
                    candidate_num = retry_candidate.candidate_idx + 1
                    logger.debug(f"🔍 Validating retry candidate {candidate_num}...")

                    retry_candidate.chunk_text = chunk.text

                    # Resolve language for this retry candidate (same logic as main path)
                    try:
                        gen_params = getattr(retry_candidate, "generation_params", None) or {}
                        base_lang = getattr(chunk, "language_id", None)
                        cand_language = (
                            gen_params.get("language_id")
                            or gen_params.get("language")
                            or base_lang
                            or "en"
                        )
                        logger.debug(
                            f"🌐 Using language '{cand_language}' for chunk {chunk.idx + 1}, retry candidate {candidate_num}"
                        )
                    except Exception:
                        cand_language = getattr(chunk, "language_id", "en")

                    # Tail-trim retry candidate before validation
                    try:
                        if hasattr(self, "tail_trimmer") and self.tail_trimmer and retry_candidate.audio_tensor is not None:
                            trimmed_audio, removed_tail, trim_meta = self.tail_trimmer.trim(
                                retry_candidate.audio_tensor,
                                language=cand_language,
                                original_text=chunk.text,
                            )
                            if trimmed_audio is not None and trimmed_audio.numel() > 0:
                                retry_candidate.audio_tensor = trimmed_audio
                                logger.debug(f"✂️  Tail-trim (retry) applied (method={trim_meta.get('method')}, cut={trim_meta.get('cut_sample')}) for retry candidate {candidate_num}")
                                try:
                                    tt_cfg = self.config.get("validation", {}).get("preprocessing", {}).get("tail_trim", {})
                                    sample_rate = self.config.get("audio", {}).get("sample_rate", 24000)
                                    if removed_tail is not None and bool(tt_cfg.get("debug_save_removed_tail", False)):
                                        from pathlib import Path
                                        import torchaudio as ta
                                        out_dir = self.file_manager.task_directory / "debug_tail_trimmer"
                                        out_dir.mkdir(parents=True, exist_ok=True)
                                        out_path = out_dir / f"chunk_{chunk.idx+1:03d}_cand_{retry_candidate.candidate_idx+1:02d}_removed_tail.wav"
                                        ta.save(str(out_path), removed_tail.unsqueeze(0).cpu(), sample_rate)
                                    if bool(tt_cfg.get("persist_trimmed_candidate", True)) and trim_meta.get("cut_sample") is not None:
                                        from pathlib import Path
                                        import torchaudio as ta
                                        cand_dir = self.file_manager.candidates_dir / f"chunk_{chunk.idx+1:03d}"
                                        cand_dir.mkdir(parents=True, exist_ok=True)
                                        kept_path = cand_dir / f"candidate_{retry_candidate.candidate_idx+1:02d}_trimmed.wav"
                                        ta.save(str(kept_path), trimmed_audio.unsqueeze(0).cpu(), sample_rate)
                                        trim_meta["trimmed_path"] = str(kept_path)
                                except Exception as e:
                                    logger.debug(f"Failed persisting tail-trim artifacts (retry): {e}")
                    except Exception as e:
                        logger.debug(f"Tail-trim (retry) failed for retry candidate {candidate_num}: {e}")

                    # Use language-specific validation for retry candidates
                    chunk_language = cand_language
                    whisper_result = self.whisper_validator.validate_candidate(
                        retry_candidate, chunk.text, language=chunk_language
                    )

                    if whisper_result:
                        quality_result = self.quality_scorer.score_candidate(
                            retry_candidate, whisper_result
                        )

                        # Prosody + MOS scoring (same as main path)
                        prosody_details = None
                        prosody_score = 0.0
                        mos_value = None
                        if retry_candidate.audio_tensor is not None and getattr(retry_candidate.audio_tensor, "numel", lambda: 0)() > 0:
                            if not getattr(self, "prosody_scorer", None):
                                try:
                                    self.prosody_scorer = ProsodyScorer(
                                        config=self.config,
                                        mos_provider=None,
                                        sample_rate=self.config.get("audio", {}).get("sample_rate", 24000),
                                    )
                                except Exception:
                                    self.prosody_scorer = None
                            if getattr(self, "prosody_scorer", None):
                                try:
                                    prosody_details = self.prosody_scorer.score(
                                        retry_candidate.audio_tensor,
                                        language=cand_language,
                                        original_text=chunk.text,
                                        asr_transcription=whisper_result.transcription,
                                    )
                                    prosody_score = float(prosody_details.get("prosody_score", 0.0))
                                    mos_value = prosody_details.get("raw_mos")
                                except Exception as e:
                                    logger.debug(f"Prosody scoring failed (retry, chunk {chunk.idx+1}, cand {candidate_num}): {e}")
                                    prosody_details = {
                                        "enabled": False,
                                        "subscores": {
                                            "semantic_alignment": 0.0,
                                            "flow": 0.0,
                                            "liveliness": 0.0,
                                            "intelligibility": 0.0,
                                            "mos": 0.0,
                                        },
                                        "prosody_score": 0.0,
                                    }
                                    prosody_score = 0.0

                        # Final selection score (same weights as main path)
                        sel_cfg = self.config.get("validation", {}).get("prosody", {}).get("select", {})
                        alpha = float(sel_cfg.get("alpha_quality", 0.50))
                        beta = float(sel_cfg.get("beta_prosody", 0.35))
                        gamma = float(sel_cfg.get("gamma_mos", 0.15))
                        mos_unit = 0.0
                        if isinstance(prosody_details, dict):
                            mos_unit = float(prosody_details.get("subscores", {}).get("mos", 0.0))
                        final_selection_score = (
                            alpha * float(quality_result.overall_score)
                            + beta * float(prosody_score)
                            + gamma * float(mos_unit)
                        )

                        # Apply gating flags
                        gating = self.config.get("validation", {}).get("selection", {}).get("gating", {})
                        require_mos = bool(gating.get("require_mos", True))
                        require_similarity = bool(gating.get("require_similarity", True))
                        passes_mos = True
                        if require_mos and isinstance(prosody_details, dict):
                            min_mos = float(self.config.get("validation", {}).get("mos", {}).get("min_mos", 3.5))
                            raw_mos = prosody_details.get("raw_mos")
                            passes_mos = (raw_mos is None) or (float(raw_mos) >= min_mos)
                        passes_similarity = True
                        if require_similarity:
                            passes_similarity = bool(whisper_result.is_valid)

                        combined_result = {
                            "is_valid": whisper_result.is_valid,
                            "transcription": whisper_result.transcription,
                            "similarity_score": whisper_result.similarity_score,
                            "quality_score": whisper_result.quality_score,
                            "validation_time": whisper_result.validation_time,
                            "error_message": whisper_result.error_message,
                            "overall_quality_score": quality_result.overall_score,
                            "quality_details": quality_result.details,
                            "speaker_id": chunk.speaker_id,
                            "language_id": cand_language,
                            "prosody": prosody_details,
                            "final_selection_score": final_selection_score,
                            "passes_mos_gate": passes_mos,
                            "passes_similarity_gate": passes_similarity,
                        }

                        # Attach tail_trim metadata if available (retry)
                        try:
                            if 'trim_meta' in locals() and isinstance(trim_meta, dict):
                                combined_result["tail_trim"] = {
                                    "method": trim_meta.get("method"),
                                    "match_type": trim_meta.get("match_type"),
                                    "cut_sample": trim_meta.get("cut_sample"),
                                    "kept_post_silence_ms": trim_meta.get("kept_post_silence_ms"),
                                    "fade_out_ms": trim_meta.get("fade_out_ms"),
                                    "matched_words": trim_meta.get("matched_words"),
                                    "match_ratio": trim_meta.get("match_ratio"),
                                    "language_gated": trim_meta.get("language_gated", False),
                                    "trimmed_path": trim_meta.get("trimmed_path"),
                                }
                        except Exception:
                            pass

                        self.file_manager.save_whisper(
                            chunk.idx, retry_candidate.candidate_idx, combined_result
                        )
                        chunk_results[retry_candidate.candidate_idx] = combined_result

                        status = "✅ Valid" if whisper_result.is_valid else "❌ Invalid"
                        logger.debug(
                            f"{status} - retry candidate {candidate_num} (similarity: {whisper_result.similarity_score:.3f}, quality: {whisper_result.quality_score:.3f}, overall: {quality_result.overall_score:.3f})"
                        )

                # Update all_candidates and save
                all_candidates[chunk.idx].extend(retry_candidates)

                if not self.file_manager.save_candidates(
                    chunk.idx, all_candidates[chunk.idx], overwrite_existing=False
                ):
                    logger.warning(
                        f"Failed to save retry candidates for chunk {chunk.idx+1}"
                    )
                else:
                    logger.debug(
                        f"✓ Saved {len(retry_candidates)} retry candidates to disk"
                    )

                validation_results[chunk.idx] = chunk_results

                # Log results
                new_valid_count = sum(
                    1
                    for result in chunk_results.values()
                    if result.get("is_valid", False)
                )
                if chunk_results:
                    overall_scores = [
                        result.get("overall_quality_score", 0.0)
                        for result in chunk_results.values()
                    ]
                    min_score = min(overall_scores)
                    max_score = max(overall_scores)
                    score_summary = (
                        f" (overall scores: {min_score:.3f} to {max_score:.3f})"
                    )
                else:
                    score_summary = ""

                if new_valid_count > valid_count:
                    logger.info(
                        f"🎉 Retry success: {new_valid_count-valid_count} additional valid candidates found!{score_summary}"
                    )
                else:
                    logger.info(
                        f"😞 Retry complete: Still no valid candidates{score_summary}"
                    )
            else:
                logger.warning(
                    f"Failed to generate retry candidates for chunk {chunk.idx+1}"
                )

        return True

    def _create_enhanced_metrics(
        self,
        chunks: List[TextChunk],
        candidates: Dict[int, List[AudioCandidate]],
        validation_results: Dict[int, Dict[int, dict]],
    ) -> Dict[str, Any]:
        """Create enhanced metrics for all chunks and candidates."""
        metrics: Dict[str, Any] = {
            "timestamp": time.time(),
            "total_chunks": len(chunks),
            "chunks": {},
        }
        logger.info("")
        logger.info("🔍 Create enhanced metrics")

        for chunk_idx, chunk in enumerate(chunks):
            chunk_key = str(chunk_idx)
            chunk_metrics = {
                "text": chunk.text,
                "chunk_text": (
                    chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
                ),
                "speaker_id": chunk.speaker_id,
                "start_pos": chunk.start_pos,
                "end_pos": chunk.end_pos,
                "candidates": {},
            }
            candidates_list = candidates.get(chunk_idx, [])
            chunk_validation = validation_results.get(chunk_idx, {})

            if not chunk_validation or not candidates_list:
                continue

            validation_results_list = []
            filtered_candidates_list = []

            for candidate in candidates_list:
                if candidate.candidate_idx in chunk_validation:
                    result_dict = chunk_validation[candidate.candidate_idx]

                    validation_result = ValidationResult(
                        is_valid=result_dict.get("is_valid", False),
                        transcription=result_dict.get("transcription", ""),
                        similarity_score=result_dict.get("similarity_score", 0.0),
                        quality_score=result_dict.get("quality_score", 0.0),
                        validation_time=result_dict.get("validation_time", 0.0),
                        error_message=result_dict.get("error_message"),
                    )

                    validation_results_list.append(validation_result)
                    filtered_candidates_list.append(candidate)

            if not filtered_candidates_list:
                continue

            try:
                # Find best candidate (prefer final_selection_score if present)
                best_candidate_idx = None
                best_score_value = float("-inf")

                for candidate in filtered_candidates_list:
                    result_dict = chunk_validation[candidate.candidate_idx]
                    quality_details = result_dict.get("quality_details", {})
                    individual_scores = quality_details.get("individual_scores", {})
                    # Prefer our combined score; fallback to legacy overall
                    candidate_score = float(
                        result_dict.get(
                            "final_selection_score",
                            individual_scores.get("overall_score", 0.0),
                        )
                    )

                    # Respect gating flags if present
                    passes_mos = result_dict.get("passes_mos_gate", True)
                    passes_similarity = result_dict.get("passes_similarity_gate", True)
                    if not (passes_mos and passes_similarity):
                        # Demote gated-out candidates
                        candidate_effective = float("-inf")
                    else:
                        candidate_effective = candidate_score

                    if candidate_effective > best_score_value:
                        best_score_value = candidate_effective
                        best_candidate_idx = candidate.candidate_idx

                # Fallback: if all were gated out, ignore gates and choose best by candidate_score
                if best_candidate_idx is None or best_score_value == float("-inf"):
                    best_score_value = float("-inf")
                    for candidate in filtered_candidates_list:
                        result_dict = chunk_validation[candidate.candidate_idx]
                        quality_details = result_dict.get("quality_details", {})
                        individual_scores = quality_details.get("individual_scores", {})
                        candidate_score = float(
                            result_dict.get(
                                "final_selection_score",
                                individual_scores.get("overall_score", 0.0),
                            )
                        )
                        if candidate_score > best_score_value:
                            best_score_value = candidate_score
                            best_candidate_idx = candidate.candidate_idx

                chunk_metrics = {
                    "text": chunk.text,
                    "chunk_text": (
                        chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
                    ),
                    "speaker_id": chunk.speaker_id,
                    "language_id": getattr(chunk, 'language_id', 'en'),
                    "candidates": {},
                    "best_candidate": best_candidate_idx,
                    "best_score": (0.0 if best_candidate_idx is None else float(best_score_value)),
                }

                candidate_scores = []
                for candidate in filtered_candidates_list:
                    result_dict = chunk_validation[candidate.candidate_idx]
                    quality_details = result_dict.get("quality_details", {})
                    individual_scores = quality_details.get("individual_scores", {})
                    final_sel = float(
                        result_dict.get(
                            "final_selection_score",
                            individual_scores.get("overall_score", 0.0),
                        )
                    )
                    candidate_scores.append(final_sel)

                    # Use string keys for candidate indices to avoid int/str duplicates in JSON
                    cand_key = str(candidate.candidate_idx)
                    chunk_metrics["candidates"][cand_key] = {
                        "transcription": result_dict.get("transcription", ""),
                        "quality_details": quality_details,
                        "overall_quality_score": float(individual_scores.get("overall_score", 0.0)),
                        "final_selection_score": final_sel,
                        "prosody": result_dict.get("prosody"),
                        "is_valid": result_dict.get("is_valid", False),
                        "passes_mos_gate": result_dict.get("passes_mos_gate", True),
                        "passes_similarity_gate": result_dict.get("passes_similarity_gate", True),
                    }

                # Log results
                if candidate_scores:
                    # min_score = min(candidate_scores)  # Calculated but not used
                    # max_score = max(candidate_scores)  # Calculated but not used
                    best_candidate_display = (
                        best_candidate_idx + 1 if best_candidate_idx is not None else 0
                    )

                    # Get TTS parameters from best candidate
                    if best_candidate_idx is not None:
                        best_candidate_obj = next(
                            (
                                c
                                for c in filtered_candidates_list
                                if c.candidate_idx == best_candidate_idx
                            ),
                            None,
                        )
                        if best_candidate_obj and best_candidate_obj.generation_params:
                            best_params = best_candidate_obj.generation_params
                            exaggeration = best_params.get("exaggeration", 0.0)
                            cfg_weight = best_params.get("cfg_weight", 0.0)
                            temperature = best_params.get("temperature", 0.0)
                            min_p = best_params.get("min_p", 0.05)
                            top_p = best_params.get("top_p", 0.95)
                        else:
                            exaggeration = cfg_weight = temperature = min_p = top_p = 0.0
                    else:
                        exaggeration = cfg_weight = temperature = min_p = top_p = 0.0

                    logger.info(
                        f"Chunk_{chunk.idx + 1:02d} - "
                        f"Best candidate: {best_candidate_display} of {len(filtered_candidates_list)} (score: {best_score_value:.3f}) "
                        f"– exaggeration: {exaggeration:.2f}, cfg_weight: {cfg_weight:.2f}, temperature: {temperature:.2f}, min_p: {min_p:.2f}, top_p: {top_p:.2f}"
                    )

                # Type-safe dictionary access
                chunks_dict = metrics["chunks"]
                chunks_dict[chunk.idx] = chunk_metrics

            except Exception as e:
                logger.warning(
                    f"Failed to select best candidate for chunk {chunk.idx}: {e}"
                )

        # Calculate median overall_quality_score and similarity_score across all candidates
        all_overall_scores = []
        all_similarity_scores = []
        for chunk_validation in validation_results.values():
            for result_dict in chunk_validation.values():
                all_overall_scores.append(result_dict.get("overall_quality_score", 0.0))
                all_similarity_scores.append(result_dict.get("similarity_score", 0.0))

        median_overall_score = round(float(sum(all_overall_scores) / len(all_overall_scores)), 4) if all_overall_scores else 0.0
        median_similarity_score = round(float(sum(all_similarity_scores) / len(all_similarity_scores)), 4) if all_similarity_scores else 0.0

        metrics["summary"] = {
            "median_overall_quality_score": median_overall_score,
            "median_similarity_score": median_similarity_score,
        }

        return metrics

    def execute_selective_validation(
        self, chunks_to_validate: Optional[List[int]] = None
    ) -> bool:
        """
        Execute validation only for specific chunks (for gap-filling scenarios).
        Preserves existing selected_candidates and only adds/updates new validation data.

        Args:
            chunks_to_validate: List of chunk indices to validate (0-based).
                              If None, determines automatically based on missing validation data.

        Returns:
            True if validation successful, False otherwise
        """
        logger.info("▶️ Starting Selective Validation Stage")
        try:
            chunks = self.file_manager.get_chunks()
            all_candidates = self.file_manager.get_candidates()

            if not chunks or not all_candidates:
                logger.error("No chunks or candidates found for validation")
                return False

            # Enhance chunks with language_id from speaker configuration (was missing in selective mode)
            self._enhance_chunks_with_language_id(chunks)

            # Determine which chunks need validation
            if chunks_to_validate is None:
                # Auto-determine based on missing validation data (legacy behavior)
                chunks_to_validate = []
                existing_metrics = self.file_manager.get_metrics()
                existing_chunks = (
                    existing_metrics.get("chunks", {}) if existing_metrics else {}
                )

                for chunk in chunks:
                    chunk_candidates = all_candidates.get(chunk.idx, [])
                    if not chunk_candidates:
                        continue

                    chunk_key = str(chunk.idx)
                    existing_chunk_data = existing_chunks.get(chunk_key, {})
                    existing_candidate_data = existing_chunk_data.get("candidates", {})

                    # Check if any candidates are missing validation data
                    needs_validation = False
                    for candidate in chunk_candidates:
                        candidate_key = str(candidate.candidate_idx)
                        if candidate_key not in existing_candidate_data:
                            needs_validation = True
                            break

                        # Also check if validation data exists but is incomplete
                        candidate_data = existing_candidate_data[candidate_key]
                        if not all(
                            key in candidate_data
                            for key in [
                                "transcription",
                                "similarity_score",
                                "overall_quality_score",
                            ]
                        ):
                            needs_validation = True
                            break

                    if needs_validation:
                        chunks_to_validate.append(chunk.idx)
            else:
                # Use explicitly provided chunk indices for gap-filling - NO AUTO-DETECTION
                logger.info(
                    f"📋 Using provided chunk indices for gap-filling validation: {[idx+1 for idx in chunks_to_validate]}"
                )

            if not chunks_to_validate:
                logger.info(
                    "✅ No chunks require validation - all validation data is complete"
                )
                return True

            logger.info(
                f"🚦 SELECTIVE VALIDATION: Processing {len(chunks_to_validate)} chunks: {[idx+1 for idx in chunks_to_validate]}"
            )

            validation_results = {}

            for chunk_idx in chunks_to_validate:
                if chunk_idx >= len(chunks):
                    logger.warning(f"Invalid chunk index {chunk_idx}, skipping")
                    continue

                chunk = chunks[chunk_idx]
                chunk_candidates = all_candidates.get(chunk.idx, [])
                if not chunk_candidates:
                    logger.warning(f"No candidates found for chunk {chunk.idx}")
                    continue

                chunk_num = chunk.idx + 1
                logger.info("-" * 40)
                logger.info(f"🎯 CHUNK {chunk_num}/{len(chunks)} (selective) - {len(chunk_candidates)} candidates")
                logger.debug(f"Candidates to validate: {len(chunk_candidates)}")

                chunk_results = {}

                progress_line = []
                for candidate in chunk_candidates:
                    candidate_num = candidate.candidate_idx + 1
                    logger.debug(f"🔍 Validating candidate {candidate_num}...")
                    try:
                        progress_line.append(str(candidate_num))
                        # Fallback for terminals that do not support inline updates: concise per-candidate line
                        logger.info(f"Candidate {candidate_num}")
                        logger.info("\rCandidates: " + ", ".join(progress_line))
                    except Exception:
                        pass

                    # Check if whisper result already exists
                    existing_whisper = self.file_manager.get_whisper(
                        chunk.idx, candidate.candidate_idx
                    )
                    if candidate.candidate_idx in existing_whisper:
                        logger.debug(
                            f"✓ Whisper result already exists for candidate {candidate_num}"
                        )
                        # Backfill prosody/final score/gates if missing
                        prev = existing_whisper[candidate.candidate_idx]
                        needs_backfill = not isinstance(prev.get("prosody"), dict) or ("final_selection_score" not in prev)
                        if needs_backfill and candidate.audio_tensor is not None and getattr(candidate.audio_tensor, "numel", lambda: 0)() > 0:
                            try:
                                # Ensure scorer exists
                                if not getattr(self, "prosody_scorer", None):
                                    try:
                                        self.prosody_scorer = ProsodyScorer(
                                            config=self.config,
                                            mos_provider=None,
                                            sample_rate=self.config.get("audio", {}).get("sample_rate", 24000),
                                        )
                                    except Exception:
                                        self.prosody_scorer = None
                                prosody_details = None
                                prosody_score = 0.0
                                mos_unit = 0.0
                                if getattr(self, "prosody_scorer", None):
                                    prosody_details = self.prosody_scorer.score(
                                        candidate.audio_tensor,
                                        language=getattr(chunk, 'language_id', 'en'),
                                        original_text=chunk.text,
                                        asr_transcription=str(prev.get("transcription") or ""),
                                    )
                                    prosody_score = float(prosody_details.get("prosody_score", 0.0))
                                    mos_unit = float(prosody_details.get("subscores", {}).get("mos", 0.0))
                                else:
                                    prosody_details = {
                                        "enabled": False,
                                        "subscores": {
                                            "semantic_alignment": 0.0,
                                            "flow": 0.0,
                                            "liveliness": 0.0,
                                            "intelligibility": 0.0,
                                            "mos": 0.0,
                                        },
                                        "prosody_score": 0.0,
                                    }
                                sel_cfg = self.config.get("validation", {}).get("prosody", {}).get("select", {})
                                alpha = float(sel_cfg.get("alpha_quality", 0.50))
                                beta = float(sel_cfg.get("beta_prosody", 0.35))
                                gamma = float(sel_cfg.get("gamma_mos", 0.15))
                                overall = float(prev.get("overall_quality_score", 0.0))
                                final_selection_score = alpha * overall + beta * prosody_score + gamma * mos_unit
                                # Gates
                                gating = self.config.get("validation", {}).get("selection", {}).get("gating", {})
                                require_mos = bool(gating.get("require_mos", True))
                                require_similarity = bool(gating.get("require_similarity", True))
                                passes_mos = True
                                if require_mos and isinstance(prosody_details, dict):
                                    min_mos = float(self.config.get("validation", {}).get("mos", {}).get("min_mos", 3.5))
                                    raw_mos = prosody_details.get("raw_mos")
                                    passes_mos = (raw_mos is None) or (float(raw_mos) >= min_mos)
                                passes_similarity = True
                                if require_similarity:
                                    passes_similarity = bool(prev.get("is_valid", True))
                                prev["prosody"] = prosody_details
                                prev["final_selection_score"] = final_selection_score
                                prev["passes_mos_gate"] = passes_mos
                                prev["passes_similarity_gate"] = passes_similarity
                                self.file_manager.save_whisper(chunk.idx, candidate.candidate_idx, prev)
                            except Exception as e:
                                logger.debug(f"Selective backfill prosody failed: {e}")
                        chunk_results[candidate.candidate_idx] = self.file_manager.get_whisper(
                            chunk.idx, candidate.candidate_idx
                        ).get(candidate.candidate_idx, prev)
                        continue

                    # Set chunk text for validation compatibility
                    candidate.chunk_text = chunk.text

                    # Tail-trim candidate audio before validation to remove trailing non-speech (selective)
                    try:
                        if hasattr(self, "tail_trimmer") and self.tail_trimmer and candidate.audio_tensor is not None:
                            trimmed_audio, removed_tail, trim_meta = self.tail_trimmer.trim(
                                candidate.audio_tensor,
                                language=getattr(chunk, 'language_id', 'en'),
                                original_text=chunk.text,
                            )
                            if trimmed_audio is not None and trimmed_audio.numel() > 0:
                                candidate.audio_tensor = trimmed_audio
                                logger.debug(f"✂️  Tail-trim (selective) applied (method={trim_meta.get('method')}, cut={trim_meta.get('cut_sample')}) for candidate {candidate_num}")
                                try:
                                    tt_cfg = self.config.get("validation", {}).get("preprocessing", {}).get("tail_trim", {})
                                    sample_rate = self.config.get("audio", {}).get("sample_rate", 24000)
                                    # Save removed tail for debugging if enabled
                                    if removed_tail is not None and bool(tt_cfg.get("debug_save_removed_tail", False)):
                                        from pathlib import Path
                                        import torchaudio as ta
                                        out_dir = self.file_manager.task_directory / "debug_tail_trimmer"
                                        out_dir.mkdir(parents=True, exist_ok=True)
                                        out_path = out_dir / f"chunk_{chunk.idx+1:03d}_cand_{candidate.candidate_idx+1:02d}_removed_tail.wav"
                                        ta.save(str(out_path), removed_tail.unsqueeze(0).cpu(), sample_rate)
                                    # Persist kept (trimmed) candidate if enabled
                                    if bool(tt_cfg.get("persist_trimmed_candidate", True)) and trim_meta.get("cut_sample") is not None:
                                        from pathlib import Path
                                        import torchaudio as ta
                                        cand_dir = self.file_manager.candidates_dir / f"chunk_{chunk.idx+1:03d}"
                                        cand_dir.mkdir(parents=True, exist_ok=True)
                                        kept_path = cand_dir / f"candidate_{candidate.candidate_idx+1:02d}_trimmed.wav"
                                        ta.save(str(kept_path), trimmed_audio.unsqueeze(0).cpu(), sample_rate)
                                        trim_meta["trimmed_path"] = str(kept_path)
                                except Exception as e:
                                    logger.debug(f"Failed persisting tail-trim artifacts (selective): {e}")
                    except Exception as e:
                        logger.debug(f"Tail-trim (selective) failed for candidate {candidate_num}: {e}")

                    # Perform Whisper validation with language-specific model
                    chunk_language = getattr(chunk, 'language_id', 'en')
                    whisper_result = self.whisper_validator.validate_candidate(
                        candidate, chunk.text, language=chunk_language
                    )

                    if whisper_result:
                        # Perform quality scoring
                        quality_result = self.quality_scorer.score_candidate(
                            candidate, whisper_result
                        )

                        # Prosody + MOS scoring (same as full validation)
                        prosody_details = None
                        prosody_score = 0.0
                        mos_unit = 0.0
                        if candidate.audio_tensor is not None and getattr(candidate.audio_tensor, "numel", lambda: 0)() > 0:
                            if not getattr(self, "prosody_scorer", None):
                                try:
                                    self.prosody_scorer = ProsodyScorer(
                                        config=self.config,
                                        mos_provider=None,
                                        sample_rate=self.config.get("audio", {}).get("sample_rate", 24000),
                                    )
                                except Exception:
                                    self.prosody_scorer = None
                            if getattr(self, "prosody_scorer", None):
                                try:
                                    prosody_details = self.prosody_scorer.score(
                                        candidate.audio_tensor,
                                        language=chunk_language,
                                        original_text=chunk.text,
                                        asr_transcription=whisper_result.transcription,
                                    )
                                    prosody_score = float(prosody_details.get("prosody_score", 0.0))
                                    mos_unit = float(prosody_details.get("subscores", {}).get("mos", 0.0))
                                except Exception as e:
                                    logger.debug(f"Prosody scoring failed (selective, chunk {chunk.idx+1}, cand {candidate_num}): {e}")
                                    prosody_details = {
                                        "enabled": False,
                                        "subscores": {
                                            "semantic_alignment": 0.0,
                                            "flow": 0.0,
                                            "liveliness": 0.0,
                                            "intelligibility": 0.0,
                                            "mos": 0.0,
                                        },
                                        "prosody_score": 0.0,
                                    }
                                    prosody_score = 0.0
                                    mos_unit = 0.0

                        # Final selection score (same weights)
                        sel_cfg = self.config.get("validation", {}).get("prosody", {}).get("select", {})
                        alpha = float(sel_cfg.get("alpha_quality", 0.50))
                        beta = float(sel_cfg.get("beta_prosody", 0.35))
                        gamma = float(sel_cfg.get("gamma_mos", 0.15))
                        final_selection_score = (
                            alpha * float(quality_result.overall_score)
                            + beta * float(prosody_score)
                            + gamma * float(mos_unit)
                        )

                        # Gates (same logic)
                        gating = self.config.get("validation", {}).get("selection", {}).get("gating", {})
                        require_mos = bool(gating.get("require_mos", True))
                        require_similarity = bool(gating.get("require_similarity", True))
                        passes_mos = True
                        if require_mos and isinstance(prosody_details, dict):
                            min_mos = float(self.config.get("validation", {}).get("mos", {}).get("min_mos", 3.5))
                            raw_mos = prosody_details.get("raw_mos")
                            passes_mos = (raw_mos is None) or (float(raw_mos) >= min_mos)
                        passes_similarity = True
                        if require_similarity:
                            passes_similarity = bool(whisper_result.is_valid)

                        # Combine results
                        combined_result = {
                            "is_valid": whisper_result.is_valid,
                            "transcription": whisper_result.transcription,
                            "similarity_score": whisper_result.similarity_score,
                            "quality_score": whisper_result.quality_score,
                            "validation_time": whisper_result.validation_time,
                            "error_message": whisper_result.error_message,
                            "overall_quality_score": quality_result.overall_score,
                            "quality_details": quality_result.details,
                            "speaker_id": chunk.speaker_id,
                            "language_id": getattr(chunk, 'language_id', 'en'),
                            "prosody": prosody_details,
                            "final_selection_score": final_selection_score,
                            "passes_mos_gate": passes_mos,
                            "passes_similarity_gate": passes_similarity,
                        }

                        # Attach tail_trim metadata if available (selective)
                        try:
                            if 'trim_meta' in locals() and isinstance(trim_meta, dict):
                                combined_result["tail_trim"] = {
                                    "method": trim_meta.get("method"),
                                    "match_type": trim_meta.get("match_type"),
                                    "cut_sample": trim_meta.get("cut_sample"),
                                    "kept_post_silence_ms": trim_meta.get("kept_post_silence_ms"),
                                    "fade_out_ms": trim_meta.get("fade_out_ms"),
                                    "matched_words": trim_meta.get("matched_words"),
                                    "match_ratio": trim_meta.get("match_ratio"),
                                    "language_gated": trim_meta.get("language_gated", False),
                                    "trimmed_path": trim_meta.get("trimmed_path"),
                                }
                        except Exception:
                            pass

                        # Save whisper result
                        self.file_manager.save_whisper(
                            chunk.idx, candidate.candidate_idx, combined_result
                        )
                        chunk_results[candidate.candidate_idx] = combined_result

                        # Log validation result
                        status = "✅ Valid" if whisper_result.is_valid else "❌ Invalid"
                        logger.debug(
                            f"{status} - candidate {candidate_num} (similarity: {whisper_result.similarity_score:.3f}, quality: {whisper_result.quality_score:.3f}, overall: {quality_result.overall_score:.3f})"
                        )
                    else:
                        logger.warning(
                            f"❌ Whisper validation failed for candidate {candidate_num}"
                        )

                validation_results[chunk.idx] = chunk_results
                valid_count = sum(
                    1
                    for result in chunk_results.values()
                    if result.get("is_valid", False)
                )

                # Log summary
                if chunk_results:
                    overall_scores = [
                        result.get("overall_quality_score", 0.0)
                        for result in chunk_results.values()
                    ]
                    selection_scores = [
                        result.get("final_selection_score", result.get("overall_quality_score", 0.0))
                        for result in chunk_results.values()
                    ]
                    min_overall = min(overall_scores)
                    max_overall = max(overall_scores)
                    min_select = min(selection_scores)
                    max_select = max(selection_scores)
                    logger.info(
                        f"✅ Validation complete: {valid_count}/{len(chunk_candidates)} candidates valid (overall: {min_overall:.3f}..{max_overall:.3f}, final_selection: {min_select:.3f}..{max_select:.3f})"
                    )
                else:
                    logger.info(
                        f"✅ Validation complete: {valid_count}/{len(chunk_candidates)} candidates valid"
                    )

                # Retry logic if no valid candidates
                if not self._handle_retry_logic(
                    chunk, chunk_results, all_candidates, validation_results
                ):
                    return False

            # Update enhanced metrics selectively (preserve existing selected_candidates)
            if not self._update_enhanced_metrics_selectively(
                chunks, all_candidates, validation_results
            ):
                logger.error("Failed to update validation metrics selectively")
                return False

            logger.info("")
            logger.info("✅ Selective validation stage completed successfully")
            return True

        except Exception as e:
            logger.error(f"Selective validation stage failed: {e}", exc_info=True)
            return False

    def _update_enhanced_metrics_selectively(
        self,
        all_chunks: List[TextChunk],
        all_candidates: Dict[int, List[AudioCandidate]],
        validation_results: Dict[int, Dict[int, dict]],
    ) -> bool:
        """
        Update enhanced metrics selectively, preserving existing selected_candidates.
        Only updates the chunk data for validated chunks.
        """
        try:
            # Create chunk data for validated chunks only
            new_chunk_data = {}

            for chunk_idx, chunk_validation in validation_results.items():
                if chunk_idx >= len(all_chunks):
                    continue

                chunk = all_chunks[chunk_idx]
                chunk_candidates = all_candidates.get(chunk.idx, [])

                if not chunk_validation or not chunk_candidates:
                    continue

                validation_results_list = []
                candidates_list = []

                for candidate in chunk_candidates:
                    if candidate.candidate_idx in chunk_validation:
                        result_dict = chunk_validation[candidate.candidate_idx]

                        validation_result = ValidationResult(
                            is_valid=result_dict.get("is_valid", False),
                            transcription=result_dict.get("transcription", ""),
                            similarity_score=result_dict.get("similarity_score", 0.0),
                            quality_score=result_dict.get("quality_score", 0.0),
                            validation_time=result_dict.get("validation_time", 0.0),
                            error_message=result_dict.get("error_message"),
                        )

                        validation_results_list.append(validation_result)
                        candidates_list.append(candidate)

                if not candidates_list:
                    continue

                try:
                    # Find best candidate for new chunk data (prefer final_selection_score; respect gates).
                    best_candidate_idx = None
                    best_score_value = float("-inf")

                    for candidate in candidates_list:
                        result_dict = chunk_validation[candidate.candidate_idx]
                        quality_details = result_dict.get("quality_details", {})
                        individual_scores = quality_details.get("individual_scores", {})
                        # Prefer final_selection_score; fallback to legacy overall
                        candidate_score = float(
                            result_dict.get(
                                "final_selection_score",
                                individual_scores.get("overall_score", 0.0),
                            )
                        )
                        # Respect gating flags if present
                        passes_mos = result_dict.get("passes_mos_gate", True)
                        passes_similarity = result_dict.get("passes_similarity_gate", True)
                        candidate_effective = candidate_score if (passes_mos and passes_similarity) else float("-inf")

                        if candidate_effective > best_score_value:
                            best_score_value = candidate_effective
                            best_candidate_idx = candidate.candidate_idx

                    # Fallback: if all were gated out, ignore gates and choose best by candidate_score
                    if best_candidate_idx is None or best_score_value == float("-inf"):
                        best_score_value = float("-inf")
                        for candidate in candidates_list:
                            result_dict = chunk_validation[candidate.candidate_idx]
                            quality_details = result_dict.get("quality_details", {})
                            individual_scores = quality_details.get("individual_scores", {})
                            candidate_score = float(
                                result_dict.get(
                                    "final_selection_score",
                                    individual_scores.get("overall_score", 0.0),
                                )
                            )
                            if candidate_score > best_score_value:
                                best_score_value = candidate_score
                                best_candidate_idx = candidate.candidate_idx

                    chunk_metrics = {
                        "text": chunk.text,
                        "chunk_text": (
                            chunk.text[:100] + "..."
                            if len(chunk.text) > 100
                            else chunk.text
                        ),
                        "candidates": {},
                        "best_candidate": best_candidate_idx,
                        "best_score": best_score_value,
                    }

                    candidate_scores = []
                    for candidate in candidates_list:
                        result_dict = chunk_validation[candidate.candidate_idx]
                        quality_details = result_dict.get("quality_details", {})
                        individual_scores = quality_details.get("individual_scores", {})
                        final_sel = float(
                            result_dict.get(
                                "final_selection_score",
                                individual_scores.get("overall_score", 0.0),
                            )
                        )
                        candidate_scores.append(final_sel)

                        chunk_metrics["candidates"][candidate.candidate_idx] = {
                            "transcription": result_dict.get("transcription", ""),
                            "quality_details": quality_details,
                            "overall_quality_score": float(individual_scores.get("overall_score", 0.0)),
                            "final_selection_score": final_sel,
                            "prosody": result_dict.get("prosody"),
                            "is_valid": result_dict.get("is_valid", False),
                            "passes_mos_gate": result_dict.get("passes_mos_gate", True),
                            "passes_similarity_gate": result_dict.get("passes_similarity_gate", True),
                        }

                    # Log results
                    if candidate_scores:
                        min_score = min(candidate_scores)
                        max_score = max(candidate_scores)
                        best_candidate_display = (
                            best_candidate_idx + 1
                            if best_candidate_idx is not None
                            else 0
                        )

                        # Get TTS parameters from best candidate
                        if best_candidate_idx is not None:
                            best_candidate_obj = next(
                                (
                                    c
                                    for c in candidates_list
                                    if c.candidate_idx == best_candidate_idx
                                ),
                                None,
                            )
                            if (
                                best_candidate_obj
                                and best_candidate_obj.generation_params
                            ):
                                best_params = best_candidate_obj.generation_params
                                exaggeration = best_params.get("exaggeration", 0.0)
                                cfg_weight = best_params.get("cfg_weight", 0.0)
                                temperature = best_params.get("temperature", 0.0)
                                min_p = best_params.get("min_p", 0.05)
                                top_p = best_params.get("top_p", 0.95)
                            else:
                                exaggeration = cfg_weight = temperature = min_p = top_p = 0.0
                        else:
                            exaggeration = cfg_weight = temperature = min_p = top_p = 0.0

                        logger.info(
                            f"Chunk_{chunk.idx + 1:02d}: score {min_score:.3f} to {max_score:.3f}. "
                            f"New best candidate: {best_candidate_display} of {len(candidates_list)} (score: {best_score_value:.3f}) "
                            f"– exaggeration: {exaggeration:.2f}, cfg_weight: {cfg_weight:.2f}, temperature: {temperature:.2f}, min_p: {min_p:.2f}, top_p: {top_p:.2f}"
                        )

                    new_chunk_data[chunk.idx] = chunk_metrics

                except Exception as e:
                    logger.warning(
                        f"Failed to select best candidate for chunk {chunk.idx}: {e}"
                    )
                    if candidates_list:
                        chunk_metrics = {
                            "chunk_text": (
                                chunk.text[:100] + "..."
                                if len(chunk.text) > 100
                                else chunk.text
                            ),
                            "candidates": {},
                            "best_candidate": candidates_list[0].candidate_idx,
                            "best_score": 0.0,
                        }
                        new_chunk_data[chunk.idx] = chunk_metrics

            # Update metrics selectively with preserved selected_candidates
            return self.file_manager._metrics_handler.update_metrics_selectively(
                new_chunk_data, preserve_selected_candidates=True
            )

        except Exception as e:
            logger.error(f"Failed to update enhanced metrics selectively: {e}")
            return False
    
    def _enhance_chunks_with_language_id(self, chunks: List[TextChunk]) -> None:
        """
        Enhance chunks with language_id from speaker configuration.
        
        Args:
            chunks: List of chunks to enhance (modified in-place)
        """
        try:
            gen_cfg = self.config.get("generation", {})
            default_language = gen_cfg.get("default_language", "en")
            speakers_config = gen_cfg.get("speakers", [])

            # Create speaker_id -> language mapping with safe defaulting
            speaker_language_map = {}
            for speaker in speakers_config:
                speaker_id = speaker.get("id")
                language = speaker.get("language")
                if speaker_id:
                    # Use speaker-specific language if truthy, otherwise fall back to default_language
                    speaker_language_map[speaker_id] = language or default_language

            # Enhance chunks with language_id using safe fallback
            for chunk in chunks:
                mapped_lang = speaker_language_map.get(chunk.speaker_id)
                chunk_language = mapped_lang or default_language or "en"
                chunk.language_id = chunk_language
                logger.debug(f"Chunk {chunk.idx + 1} (speaker: {chunk.speaker_id}) → language: {chunk_language}")

        except Exception as e:
            logger.warning(f"Failed to enhance chunks with language_id: {e}")
            # Set fallback language for all chunks
            default_language = self.config.get("generation", {}).get("default_language", "en")
            for chunk in chunks:
                chunk.language_id = default_language
