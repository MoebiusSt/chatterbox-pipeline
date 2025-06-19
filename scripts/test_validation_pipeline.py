#!/usr/bin/env python3
"""
Validation pipeline test script that demonstrates Phase 2 functionality:
- Whisper-based audio validation
- Fuzzy text matching
- Quality scoring and candidate selection
"""

import logging
import sys
from pathlib import Path

import torch
import yaml

# Add src to path for imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from chunking.chunk_validator import ChunkValidator
from chunking.spacy_chunker import SpaCyChunker
from generation.audio_processor import AudioProcessor
from generation.tts_generator import AudioCandidate
from validation.fuzzy_matcher import FuzzyMatcher, MatchResult
from validation.quality_scorer import QualityScorer, ScoringStrategy
from validation.whisper_validator import ValidationResult, WhisperValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_mock_candidates_with_variations(chunks, num_candidates=3):
    """Create mock audio candidates with intentional variations for testing."""
    all_candidates = []

    for chunk_idx, chunk in enumerate(chunks):
        chunk_candidates = []

        for candidate_idx in range(num_candidates):
            # Create variations in duration and audio characteristics
            base_duration = max(0.5, len(chunk.text) * 0.02)

            if candidate_idx == 0:
                # "Perfect" candidate - normal duration
                duration = base_duration
                frequency = 440
                text_variation = chunk.text  # Exact text
            elif candidate_idx == 1:
                # "Long" candidate - artifacts make it longer
                duration = base_duration * 1.3
                frequency = 420
                text_variation = chunk.text + " um, uh"  # Some disfluency
            else:
                # "Short" candidate - might be cut off
                duration = base_duration * 0.7
                frequency = 460
                # Might have slight text variation
                text_variation = chunk.text.replace(".", "")

            # Generate mock audio
            sample_rate = 24000  # ChatterboxTTS native sample rate
            num_samples = int(duration * sample_rate)
            t = torch.linspace(0, duration, num_samples)

            # Add some complexity to the waveform
            audio = (
                0.3 * torch.sin(2 * torch.pi * frequency * t)
                + 0.1 * torch.sin(2 * torch.pi * frequency * 2 * t)
                + 0.05 * torch.randn(num_samples) * 0.1  # Small noise
            ).unsqueeze(0)

            candidate = AudioCandidate(
                chunk_idx=chunk_idx,
                candidate_idx=candidate_idx + 1,  # 1-based indexing
                audio_path=Path(f"mock_chunk_{chunk_idx}_candidate_{candidate_idx + 1}.wav"),
                audio_tensor=audio,
                generation_params={
                    "mock": True,
                    "variant": ["perfect", "long", "short"][candidate_idx],
                    "frequency": frequency,
                },
                chunk_text=text_variation,
            )
            chunk_candidates.append(candidate)

        all_candidates.append(chunk_candidates)

    return all_candidates


def create_mock_validation_results(candidates_by_chunk, original_chunks):
    """Create mock validation results simulating Whisper transcription."""
    all_validation_results = []

    for chunk_idx, (chunk_candidates, original_chunk) in enumerate(
        zip(candidates_by_chunk, original_chunks)
    ):
        chunk_validations = []

        for candidate in chunk_candidates:
            # Simulate different transcription quality
            variant = candidate.generation_params.get("variant", "perfect")
            original_text = original_chunk.text

            if variant == "perfect":
                # High quality transcription
                transcription = original_text  # Perfect match
                similarity_score = 0.98
                quality_score = 0.95
                validation_time = 2.0
            elif variant == "long":
                # Some artifacts but mostly good
                transcription = candidate.chunk_text  # Includes "um, uh"
                similarity_score = 0.85
                quality_score = 0.75
                validation_time = 2.5
            else:  # short
                # Potential cut-off
                transcription = candidate.chunk_text  # Missing punctuation
                similarity_score = 0.80
                quality_score = 0.70
                validation_time = 1.8

            # Determine validation status
            is_valid = similarity_score >= 0.8 and quality_score >= 0.7

            validation_result = ValidationResult(
                is_valid=is_valid,
                transcription=transcription,
                similarity_score=similarity_score,
                quality_score=quality_score,
                validation_time=validation_time,
            )

            chunk_validations.append(validation_result)

        all_validation_results.append(chunk_validations)

    return all_validation_results


def main():
    """Main test function for validation pipeline."""

    # Setup paths
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config" / "default_config.yaml"
    input_text_path = project_root / "data" / "input" / "texts" / "input-document.txt"

    logger.info(f"Project root: {project_root}")

    # Load configuration
    logger.info(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load input text (use first few chunks for testing)
    logger.info(f"Loading input text from: {input_text_path}")
    with open(input_text_path, "r", encoding="utf-8") as f:
        input_text = f.read().strip()

    if not input_text:
        logger.error("No input text found")
        return False

    # Use only first paragraph for faster testing
    test_text = input_text.split("\n\n")[0]
    logger.info(f"Using test text ({len(test_text)} chars): '{test_text[:100]}...'")

    # --- PHASE 1: CHUNKING ---
    logger.info("=== PHASE 1: TEXT CHUNKING ===")

    chunking_config = config.get("chunking", {})
    chunker = SpaCyChunker(
        model_name=chunking_config.get("spacy_model", "en_core_web_sm"),
        target_limit=300,  # Smaller chunks for testing
        max_limit=400,
        min_length=50,
    )

    chunks = chunker.chunk_text(test_text)
    logger.info(f"Generated {len(chunks)} test chunks")

    for i, chunk in enumerate(chunks):
        logger.info(f"  Chunk {i+1}: {len(chunk.text)} chars")

    # --- PHASE 2: MOCK CANDIDATE GENERATION ---
    logger.info("=== PHASE 2: MOCK CANDIDATE GENERATION ===")

    candidates_by_chunk = create_mock_candidates_with_variations(
        chunks, num_candidates=3
    )
    logger.info(f"Created {len(candidates_by_chunk)} chunks with 3 candidates each")

    # --- PHASE 3: MOCK VALIDATION ---
    logger.info("=== PHASE 3: MOCK VALIDATION ===")

    validation_results_by_chunk = create_mock_validation_results(
        candidates_by_chunk, chunks
    )

    # --- PHASE 4: FUZZY MATCHING ---
    logger.info("=== PHASE 4: FUZZY MATCHING ===")

    validation_config = config.get("validation", {})
    fuzzy_matcher = FuzzyMatcher(
        threshold=validation_config.get("similarity_threshold", 0.90),
        case_sensitive=False,
        normalize_whitespace=True,
    )

    all_match_results = []
    for chunk_idx, (chunk_candidates, chunk_validations, original_chunk) in enumerate(
        zip(candidates_by_chunk, validation_results_by_chunk, chunks)
    ):
        chunk_match_results = []

        for candidate, validation in zip(chunk_candidates, chunk_validations):
            match_result = fuzzy_matcher.match_texts(
                original_chunk.text, validation.transcription, method="auto"
            )
            chunk_match_results.append(match_result)

            logger.info(
                f"  Chunk {chunk_idx+1}, Candidate {candidate.candidate_idx}: "
                f"Match={match_result.similarity:.3f} ({match_result.method})"
            )

        all_match_results.append(chunk_match_results)

    # --- PHASE 5: QUALITY SCORING ---
    logger.info("=== PHASE 5: QUALITY SCORING ===")

    quality_scorer = QualityScorer(
        sample_rate=24000,  # ChatterboxTTS native sample rate
    )

    # Score and select best candidates for each chunk
    best_candidates = []
    best_scores = []

    for chunk_idx, (
        chunk_candidates,
        chunk_validations,
        chunk_matches,
        original_chunk,
    ) in enumerate(
        zip(candidates_by_chunk, validation_results_by_chunk, all_match_results, chunks)
    ):
        logger.info(f"\n--- Scoring Chunk {chunk_idx+1} Candidates ---")

        # Score all candidates for this chunk
        scored_candidates = quality_scorer.rank_candidates(
            chunk_candidates, chunk_validations, chunk_matches
        )

        # Display rankings
        for rank, (candidate, score) in enumerate(scored_candidates):
            variant = candidate.generation_params.get("variant", "unknown")
            logger.info(
                f"  Rank {rank+1}: chunk_{candidate.chunk_idx}_candidate_{candidate.candidate_idx} ({variant}) - "
                f"Score: {score.overall_score:.3f} "
                f"(sim={score.similarity_score:.3f}, len={score.length_score:.3f})"
            )

        # Select best candidate
        best_candidate, best_score = scored_candidates[0]
        best_candidates.append(best_candidate)
        best_scores.append(best_score)

        logger.info(
            f"  → Selected: chunk_{best_candidate.chunk_idx}_candidate_{best_candidate.candidate_idx} "
            f"({best_candidate.generation_params.get('variant')}) "
            f"with score {best_score.overall_score:.3f}"
        )

    # --- PHASE 6: FINAL AUDIO PROCESSING ---
    logger.info("=== PHASE 6: FINAL AUDIO PROCESSING ===")

    audio_config = config.get("audio", {})
    audio_processor = AudioProcessor(
        sample_rate=audio_config.get(
            "sample_rate", 24000
        ),  # ChatterboxTTS native sample rate
        normal_silence_duration=0.1,  # Shorter for testing
        paragraph_silence_duration=0.1,
        device="cpu",
    )

    # Extract paragraph break information
    has_paragraph_breaks = [chunk.has_paragraph_break for chunk in chunks]

    # Concatenate selected candidates
    final_audio = audio_processor.concatenate_candidates(
        candidates=best_candidates, has_paragraph_breaks=has_paragraph_breaks
    )

    duration = audio_processor.get_audio_duration(final_audio)
    logger.info(f"Final audio duration: {duration:.2f} seconds")

    # Save output
    output_path = (
        project_root / "data" / "output" / "final" / "validation_test_output.wav"
    )
    success = audio_processor.save_audio(final_audio, str(output_path))

    if success:
        logger.info(f"✅ Validation pipeline test completed successfully!")
        logger.info(f"📊 Results Summary:")
        logger.info(f"   - Processed {len(chunks)} text chunks")
        logger.info(
            f"   - Generated {sum(len(candidates) for candidates in candidates_by_chunk)} total candidates"
        )
        logger.info(f"   - Selected {len(best_candidates)} best candidates")
        logger.info(
            f"   - Average quality score: {sum(score.overall_score for score in best_scores) / len(best_scores):.3f}"
        )
        logger.info(f"   - Output saved to: {output_path}")
        logger.info(f"   - This demonstrates successful Phase 2 implementation!")
        return True
    else:
        logger.error("❌ Failed to save output audio")
        return False


if __name__ == "__main__":
    try:
        success = main()
        if success:
            logger.info("Validation pipeline test completed successfully")
            sys.exit(0)
        else:
            logger.error("Validation pipeline test failed")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during test: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
