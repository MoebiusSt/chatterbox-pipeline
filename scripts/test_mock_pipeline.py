#!/usr/bin/env python3
"""
Mock TTS Pipeline Test Script
Provides mock audio generation for CI/CD testing and development without requiring heavy TTS models.
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from chunking.chunk_validator import ChunkValidator
from chunking.spacy_chunker import SpaCyChunker
from generation.audio_processor import AudioProcessor
from utils.progress_tracker import ProgressTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def detect_device() -> str:
    """Automatically detect the best available device."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"Using device: {device}")
    return device


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    logger.info(f"Loading configuration from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.1f}s"


def main():
    """Mock TTS pipeline execution for testing and CI/CD."""
    # Track overall pipeline start time
    pipeline_start_time = time.time()

    # Setup paths
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config" / "default_config.yaml"

    logger.info(f"🧪 MOCK TTS PIPELINE TEST")
    logger.info(f"Project root: {project_root}")

    # Load configuration
    config = load_config(config_path)

    # Get input configuration
    input_config = config.get("input", {})
    text_filename = input_config.get("text_file", "input-document.txt")

    # Create timestamped output directories for this mock run
    start_datetime = datetime.fromtimestamp(pipeline_start_time)
    text_base_name = Path(text_filename).stem  # Remove .txt extension
    timestamp_folder = (
        f"MOCK_{text_base_name}_{start_datetime.strftime('%Y%m%d_%H%M%S')}"
    )

    # Setup output directories for this specific mock run
    run_output_dir = project_root / "data" / "output" / timestamp_folder
    run_texts_dir = run_output_dir / "texts"  # Combined chunks and transcriptions
    run_final_dir = run_output_dir / "final"

    # Create output directories
    for output_dir in [run_output_dir, run_texts_dir, run_final_dir]:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Setup basic dual logging for mock pipeline
    log_file_path = run_output_dir / "log.txt"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Add file handler to existing logger
    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)

    logger.info(f"📁 Created mock run directory: {timestamp_folder}")
    logger.info(f"📄 Mock log file: {log_file_path.relative_to(project_root)}")

    # Setup paths
    input_text_path = project_root / "data" / "input" / "texts" / text_filename

    # Create final output file in timestamped directory
    output_path = run_final_dir / f"{text_base_name}_mock.wav"

    logger.info(f"Using text file: {text_filename}")

    # Setup device
    device = detect_device()

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
        target_limit=chunking_config.get("target_chunk_limit"),
        max_limit=chunking_config.get("max_chunk_limit"),
        min_length=chunking_config.get("min_chunk_length"),
    )

    chunks = chunker.chunk_text(input_text)
    logger.info(f"Generated {len(chunks)} text chunks")

    # Save chunks to disk for analysis
    chunk_paths = chunker.save_chunks_to_disk(chunks, output_dir=str(run_texts_dir))
    logger.info(f"Saved {len(chunk_paths)} chunk text files for analysis")

    # Validate chunks
    validator = ChunkValidator(
        max_limit=chunking_config.get("max_chunk_limit", 600),
        min_length=chunking_config.get("min_chunk_length", 150),
    )
    is_valid = validator.run_all_validations(chunks)
    logger.info(
        f"Chunk validation: {'PASSED' if is_valid else 'FAILED (but continuing)'}"
    )

    # --- PHASE 2: MOCK AUDIO GENERATION ---
    logger.info("=== PHASE 2: MOCK AUDIO GENERATION ===")
    logger.warning("🎭 RUNNING IN MOCK TTS MODE!")
    logger.info("Generating mock audio (sine waves) instead of real TTS.")

    audio_segments = []
    has_paragraph_breaks = []

    # Initialize progress tracker for mock TTS generation
    progress_tracker = ProgressTracker(len(chunks), "Mock TTS Generation")

    for i, chunk in enumerate(chunks):
        progress_tracker.update(i + 1, f"Generating mock audio for chunk {i+1}")

        # Create a sine wave as mock audio
        duration_seconds = max(1.0, len(chunk.text) / 15)  # Estimate duration
        sample_rate = config.get("audio", {}).get(
            "sample_rate", 24000
        )  # ChatterboxTTS native sample rate
        num_samples = int(duration_seconds * sample_rate)
        time_tensor = torch.linspace(0, duration_seconds, num_samples, device=device)
        frequency = 440 + (i * 20)  # Vary frequency slightly per chunk
        amplitude = 0.3
        mock_audio = (
            torch.sin(2 * torch.pi * frequency * time_tensor) * amplitude
        ).unsqueeze(0)

        audio_segments.append(mock_audio)
        has_paragraph_breaks.append(chunk.has_paragraph_break)

    progress_tracker.finish()
    logger.info(f"Generated {len(audio_segments)} mock audio segments.")

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
        device=device,
    )

    # Concatenate all audio segments
    logger.info("Concatenating audio segments...")
    final_audio = audio_processor.concatenate_segments(
        audio_segments=audio_segments, has_paragraph_breaks=has_paragraph_breaks
    )

    duration = audio_processor.get_audio_duration(final_audio)

    # --- PHASE 4: OUTPUT ---
    logger.info("=== PHASE 4: SAVING OUTPUT ===")

    success = audio_processor.save_audio(
        audio=final_audio, output_path=str(output_path)
    )

    if success:
        # --- Log total pipeline time ---
        pipeline_end_time = time.time()
        total_pipeline_time = pipeline_end_time - pipeline_start_time

        logger.info(f"{'='*60}")
        logger.info(f"🎭 MOCK TTS PIPELINE COMPLETED")
        logger.info(f"{'='*60}")
        logger.info(
            f"Start time: {datetime.fromtimestamp(pipeline_start_time).strftime('%H:%M:%S')}"
        )
        logger.info(
            f"End time: {datetime.fromtimestamp(pipeline_end_time).strftime('%H:%M:%S')}"
        )
        logger.info(f"- Total execution time: {_format_duration(total_pipeline_time)}")
        logger.info(
            f"Generated {duration:.2f} seconds of mock audio from {len(input_text)} characters of text"
        )
        logger.info(f"- Mock output saved to: {output_path}")
        logger.info(f"✅ Mock pipeline test successful - ready for CI/CD integration")
        logger.info(f"{'='*60}")

        return True
    else:
        logger.error("❌ Failed to save mock audio output")
        return False


if __name__ == "__main__":
    try:
        success = main()
        if success:
            logger.info("🎭 Mock pipeline execution completed successfully")
            sys.exit(0)
        else:
            logger.error("❌ Mock pipeline execution failed")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("🛑 Mock pipeline execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"💥 Unexpected error during mock pipeline execution: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
