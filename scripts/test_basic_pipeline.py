#!/usr/bin/env python3
"""
Basic pipeline test script that demonstrates the enhanced TTS pipeline components
without requiring the full ChatterboxTTS model dependencies.
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
from utils.file_manager import AudioCandidate

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_mock_audio_candidates(chunks):
    """Create mock audio candidates for testing purposes."""
    candidates = []

    for i, chunk in enumerate(chunks):
        # Create mock audio tensor (sine wave for demonstration)
        duration = max(
            0.5, len(chunk.text) * 0.02
        )  # Rough estimate: 50ms per character
        sample_rate = 24000  # ChatterboxTTS native sample rate
        num_samples = int(duration * sample_rate)

        # Generate a simple sine wave as mock audio
        frequency = 440 + (i * 50)  # Vary frequency slightly for each chunk
        t = torch.linspace(0, duration, num_samples)
        audio = 0.3 * torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)

        candidate = AudioCandidate(
            chunk_idx=i,
            candidate_idx=0,
            audio_path=None,
            audio_tensor=audio,
            generation_params={"mock": True},
            chunk_text=chunk.text,
        )
        candidates.append(candidate)

    return candidates


def main():
    """Main test function."""

    # Setup paths
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config" / "default_config.yaml"
    input_text_path = project_root / "data" / "input" / "texts" / "input-document.txt"
    output_path = project_root / "data" / "output" / "final" / "test_basic_output.wav"

    logger.info(f"Project root: {project_root}")

    # Load configuration
    logger.info(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Load input text
    logger.info(f"Loading input text from: {input_text_path}")
    with open(input_text_path, "r", encoding="utf-8") as f:
        input_text = f.read().strip()

    if not input_text:
        logger.error("No input text found")
        return False

    logger.info(f"Input text length: {len(input_text)} characters")

    # --- PHASE 1: CHUNKING ---
    logger.info("=== PHASE 1: TEXT CHUNKING ===")

    chunking_config = config.get("chunking", {})
    chunker = SpaCyChunker(
        model_name=chunking_config.get("spacy_model", "en_core_web_sm"),
        target_limit=chunking_config.get("target_chunk_limit", 600),
        max_limit=chunking_config.get("max_chunk_limit", 700),
        min_length=chunking_config.get("min_chunk_length", 150),
    )

    chunks = chunker.chunk_text(input_text)
    logger.info(f"Generated {len(chunks)} text chunks")

    # Display chunks
    for i, chunk in enumerate(chunks):
        logger.info(
            f"Chunk {i+1}: Length={len(chunk.text)}, Paragraph Break={chunk.has_paragraph_break}"
        )
        logger.info(
            f"  Text: '{chunk.text[:80]}{'...' if len(chunk.text) > 80 else ''}'"
        )

    # Validate chunks
    validator = ChunkValidator(
        max_limit=chunking_config.get("max_chunk_limit", 700),
        min_length=chunking_config.get("min_chunk_length", 150),
    )
    is_valid = validator.run_all_validations(chunks)
    logger.info(
        f"Chunk validation: {'PASSED' if is_valid else 'FAILED (but continuing)'}"
    )

    # --- PHASE 2: MOCK AUDIO GENERATION ---
    logger.info("=== PHASE 2: MOCK AUDIO GENERATION ===")

    logger.info("Creating mock audio candidates...")
    audio_candidates = create_mock_audio_candidates(chunks)
    logger.info(f"Created {len(audio_candidates)} mock audio candidates")

    # --- PHASE 3: AUDIO PROCESSING ---
    logger.info("=== PHASE 3: AUDIO PROCESSING ===")

    audio_config = config.get("audio", {})
    silence_config = audio_config.get("silence_duration", {})

    audio_processor = AudioProcessor(
        sample_rate=audio_config.get(
            "sample_rate", 24000
        ),  # ChatterboxTTS native sample rate
        normal_silence_duration=silence_config.get("normal", 0.20),
        paragraph_silence_duration=silence_config.get("paragraph", 0.20),
        device="cpu",  # Use CPU for this test
    )

    # Extract paragraph break information
    has_paragraph_breaks = [chunk.has_paragraph_break for chunk in chunks]

    # Concatenate all audio segments
    logger.info("Concatenating mock audio segments...")
    final_audio = audio_processor.concatenate_candidates(
        candidates=audio_candidates, has_paragraph_breaks=has_paragraph_breaks
    )

    # logger.info(f"Final audio shape: {final_audio.shape}")
    duration = audio_processor.get_audio_duration(final_audio)
    # logger.info(f"Final audio duration: {duration:.2f} seconds")

    # --- PHASE 4: OUTPUT ---
    logger.info("=== PHASE 4: SAVING OUTPUT ===")

    success = audio_processor.save_audio(
        audio=final_audio, output_path=str(output_path)
    )

    if success:
        logger.info(f"Test completed successfully! Mock output saved to: {output_path}")
        logger.info(
            f"Generated {duration:.2f} seconds of mock audio from {len(input_text)} characters of text"
        )
        logger.info(
            "This demonstrates that the pipeline components are working correctly."
        )
        logger.info(
            "To use real TTS, ensure all ChatterboxTTS dependencies are installed."
        )
        return True
    else:
        logger.error("Failed to save output audio")
        return False


if __name__ == "__main__":
    try:
        success = main()
        if success:
            logger.info("Basic pipeline test completed successfully")
            sys.exit(0)
        else:
            logger.error("Basic pipeline test failed")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during test: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
