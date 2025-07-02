"""Validation stage handler."""

import logging
import time
from typing import Any, Dict, List, Optional

from utils.file_manager.file_manager import FileManager
from utils.file_manager.io_handlers.candidate_io import AudioCandidate
from chunking.base_chunker import TextChunk
from validation.quality_scorer import QualityScorer
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

    def execute_validation(self) -> bool:
        try:
            logger.info("=" * 50)
            logger.info("Starting validation stage")

            chunks = self.file_manager.get_chunks()
            all_candidates = self.file_manager.get_candidates()

            if not chunks or not all_candidates:
                logger.error("No chunks or candidates found for validation")
                return False

            validation_results = {}
            logger.info(f"🚦 VALIDATION PHASE: Processing {len(chunks)} chunks")
            logger.info("=" * 50)

            for chunk in chunks:
                chunk_candidates = all_candidates.get(chunk.idx, [])
                if not chunk_candidates:
                    logger.warning(f"No candidates found for chunk {chunk.idx}")
                    continue

                chunk_num = chunk.idx + 1
                logger.info("")
                logger.info(f"🎯 CHUNK {chunk_num}/{len(chunks)}")
                logger.debug(f"Candidates to validate: {len(chunk_candidates)}")
                logger.info("-" * 40)

                chunk_results = {}

                for candidate in chunk_candidates:
                    candidate_num = candidate.candidate_idx + 1
                    logger.debug(f"🔍 Validating candidate {candidate_num}...")

                    # Check if whisper result already exists
                    existing_whisper = self.file_manager.get_whisper(
                        chunk.idx, candidate.candidate_idx
                    )
                    if candidate.candidate_idx in existing_whisper:
                        logger.debug(
                            f"✓ Whisper result already exists for candidate {candidate_num}"
                        )
                        chunk_results[candidate.candidate_idx] = existing_whisper[
                            candidate.candidate_idx
                        ]
                        continue

                    candidate.chunk_text = chunk.text

                    # Perform Whisper validation
                    whisper_result = self.whisper_validator.validate_candidate(
                        candidate, chunk.text
                    )

                    if whisper_result:
                        # Perform quality scoring
                        quality_result = self.quality_scorer.score_candidate(
                            candidate, whisper_result
                        )

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
                        }

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
                    min_score = min(overall_scores)
                    max_score = max(overall_scores)
                    logger.info(
                        f"✅ Validation complete: {valid_count}/{len(chunk_candidates)} candidates valid (overall scores: {min_score:.3f} to {max_score:.3f})"
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

            # Create and save enhanced metrics
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

                    whisper_result = self.whisper_validator.validate_candidate(
                        retry_candidate, chunk.text
                    )

                    if whisper_result:
                        quality_result = self.quality_scorer.score_candidate(
                            retry_candidate, whisper_result
                        )

                        combined_result = {
                            "is_valid": whisper_result.is_valid,
                            "transcription": whisper_result.transcription,
                            "similarity_score": whisper_result.similarity_score,
                            "quality_score": whisper_result.quality_score,
                            "validation_time": whisper_result.validation_time,
                            "error_message": whisper_result.error_message,
                            "overall_quality_score": quality_result.overall_score,
                            "quality_details": quality_result.details,
                        }

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
            "selected_candidates": {},
        }

        for chunk in chunks:
            chunk_candidates = candidates.get(chunk.idx, [])
            chunk_validation = validation_results.get(chunk.idx, {})

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
                # Find best candidate
                best_candidate_idx = None
                best_score_value = -1.0

                for candidate in candidates_list:
                    result_dict = chunk_validation[candidate.candidate_idx]
                    candidate_score = result_dict.get("overall_quality_score", 0.0)

                    if candidate_score > best_score_value:
                        best_score_value = candidate_score
                        best_candidate_idx = candidate.candidate_idx

                chunk_metrics = {
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
                    candidate_score = result_dict.get("overall_quality_score", 0.0)
                    candidate_scores.append(candidate_score)

                    chunk_metrics["candidates"][candidate.candidate_idx] = {
                        "transcription": result_dict.get("transcription", ""),
                        "similarity_score": result_dict.get("similarity_score", 0.0),
                        "validation_score": result_dict.get("quality_score", 0.0),
                        "overall_quality_score": candidate_score,
                        "quality_details": result_dict.get("quality_details", {}),
                        "final_score": candidate_score,
                    }

                # Log results
                if candidate_scores:
                    min_score = min(candidate_scores)
                    max_score = max(candidate_scores)
                    best_candidate_display = (
                        best_candidate_idx + 1 if best_candidate_idx is not None else 0
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
                        if best_candidate_obj and best_candidate_obj.generation_params:
                            best_params = best_candidate_obj.generation_params
                            exaggeration = best_params.get("exaggeration", 0.0)
                            cfg_weight = best_params.get("cfg_weight", 0.0)
                            temperature = best_params.get("temperature", 0.0)
                        else:
                            exaggeration = cfg_weight = temperature = 0.0
                    else:
                        exaggeration = cfg_weight = temperature = 0.0

                    logger.info(
                        f"Chunk_{chunk.idx + 1:02d}: score {min_score:.3f} to {max_score:.3f}. "
                        f"Best candidate: {best_candidate_display} of {len(candidates_list)} (score: {best_score_value:.3f}) "
                        f"– exaggeration: {exaggeration:.2f}, cfg_weight: {cfg_weight:.2f}, temperature: {temperature:.2f}"
                    )

                # Type-safe dictionary access
                chunks_dict = metrics["chunks"] 
                selected_candidates_dict = metrics["selected_candidates"]
                chunks_dict[chunk.idx] = chunk_metrics
                selected_candidates_dict[chunk.idx] = best_candidate_idx

            except Exception as e:
                logger.warning(
                    f"Failed to select best candidate for chunk {chunk.idx}: {e}"
                )
                if candidates_list:
                    selected_candidates_dict = metrics["selected_candidates"]
                    selected_candidates_dict[chunk.idx] = candidates_list[0].candidate_idx

        return metrics

    def execute_selective_validation(self, chunks_to_validate: Optional[List[int]] = None) -> bool:
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

            # Determine which chunks need validation
            if chunks_to_validate is None:
                # Auto-determine based on missing validation data (legacy behavior)
                chunks_to_validate = []
                existing_metrics = self.file_manager.get_metrics()
                existing_chunks = existing_metrics.get("chunks", {}) if existing_metrics else {}
                
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
                        if not all(key in candidate_data for key in ["transcription", "similarity_score", "overall_quality_score"]):
                            needs_validation = True
                            break
                    
                    if needs_validation:
                        chunks_to_validate.append(chunk.idx)
            else:
                # Use explicitly provided chunk indices for gap-filling - NO AUTO-DETECTION
                logger.info(f"📋 Using provided chunk indices for gap-filling validation: {[idx+1 for idx in chunks_to_validate]}")

            if not chunks_to_validate:
                logger.info("✅ No chunks require validation - all validation data is complete")
                return True

            logger.info(f"🚦 SELECTIVE VALIDATION: Processing {len(chunks_to_validate)} chunks: {[idx+1 for idx in chunks_to_validate]}")

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
                logger.info("")
                logger.info(f"🎯 CHUNK {chunk_num}/{len(chunks)} (selective)")
                logger.debug(f"Candidates to validate: {len(chunk_candidates)}")
                logger.info("-" * 40)

                chunk_results = {}

                for candidate in chunk_candidates:
                    candidate_num = candidate.candidate_idx + 1
                    logger.debug(f"🔍 Validating candidate {candidate_num}...")

                    # Check if whisper result already exists
                    existing_whisper = self.file_manager.get_whisper(
                        chunk.idx, candidate.candidate_idx
                    )
                    if candidate.candidate_idx in existing_whisper:
                        logger.debug(
                            f"✓ Whisper result already exists for candidate {candidate_num}"
                        )
                        chunk_results[candidate.candidate_idx] = existing_whisper[
                            candidate.candidate_idx
                        ]
                        continue

                    # Set chunk text for validation compatibility
                    candidate.chunk_text = chunk.text

                    # Perform Whisper validation
                    whisper_result = self.whisper_validator.validate_candidate(
                        candidate, chunk.text
                    )

                    if whisper_result:
                        # Perform quality scoring
                        quality_result = self.quality_scorer.score_candidate(
                            candidate, whisper_result
                        )

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
                        }

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
                    min_score = min(overall_scores)
                    max_score = max(overall_scores)
                    logger.info(
                        f"✅ Validation complete: {valid_count}/{len(chunk_candidates)} candidates valid (overall scores: {min_score:.3f} to {max_score:.3f})"
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
            if not self._update_enhanced_metrics_selectively(chunks, all_candidates, validation_results):
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
                    # Find best candidate for new chunk data (but don't update selected_candidates)
                    best_candidate_idx = None
                    best_score_value = -1.0

                    for candidate in candidates_list:
                        result_dict = chunk_validation[candidate.candidate_idx]
                        candidate_score = result_dict.get("overall_quality_score", 0.0)

                        if candidate_score > best_score_value:
                            best_score_value = candidate_score
                            best_candidate_idx = candidate.candidate_idx

                    chunk_metrics = {
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
                        candidate_score = result_dict.get("overall_quality_score", 0.0)
                        candidate_scores.append(candidate_score)

                        chunk_metrics["candidates"][candidate.candidate_idx] = {
                            "transcription": result_dict.get("transcription", ""),
                            "similarity_score": result_dict.get("similarity_score", 0.0),
                            "validation_score": result_dict.get("quality_score", 0.0),
                            "overall_quality_score": candidate_score,
                            "quality_details": result_dict.get("quality_details", {}),
                            "final_score": candidate_score,
                        }

                    # Log results
                    if candidate_scores:
                        min_score = min(candidate_scores)
                        max_score = max(candidate_scores)
                        best_candidate_display = (
                            best_candidate_idx + 1 if best_candidate_idx is not None else 0
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
                            if best_candidate_obj and best_candidate_obj.generation_params:
                                best_params = best_candidate_obj.generation_params
                                exaggeration = best_params.get("exaggeration", 0.0)
                                cfg_weight = best_params.get("cfg_weight", 0.0)
                                temperature = best_params.get("temperature", 0.0)
                            else:
                                exaggeration = cfg_weight = temperature = 0.0
                        else:
                            exaggeration = cfg_weight = temperature = 0.0

                        logger.info(
                            f"Chunk_{chunk.idx + 1:02d}: score {min_score:.3f} to {max_score:.3f}. "
                            f"New best candidate: {best_candidate_display} of {len(candidates_list)} (score: {best_score_value:.3f}) "
                            f"– exaggeration: {exaggeration:.2f}, cfg_weight: {cfg_weight:.2f}, temperature: {temperature:.2f}"
                        )

                    new_chunk_data[chunk.idx] = chunk_metrics

                except Exception as e:
                    logger.warning(
                        f"Failed to select best candidate for chunk {chunk.idx}: {e}"
                    )
                    if candidates_list:
                        chunk_metrics = {
                            "chunk_text": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text,
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
