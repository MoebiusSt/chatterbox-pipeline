#!/usr/bin/env python3
"""
Integration tests for the TTS pipeline using mock audio generation.
Suitable for CI/CD environments without heavy TTS model dependencies.
"""

import sys
from pathlib import Path

import pytest
import torch
import yaml

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from chunking.chunk_validator import ChunkValidator
from chunking.spacy_chunker import SpaCyChunker
from generation.audio_processor import AudioProcessor
from utils.progress_tracker import ProgressTracker


class TestPipelineIntegration:
    """Integration tests for the TTS pipeline using mock components."""

    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return """
        This is the first paragraph with multiple sentences. It should be processed correctly by the SpaCy chunker. 
        The chunker should respect sentence boundaries.


        This is a second paragraph. It contains different content and should be handled as a separate section.
        The paragraph break should be preserved in the chunk metadata.
        """

    @pytest.fixture
    def config(self):
        """Test configuration."""
        return {
            "chunking": {
                "target_chunk_limit": 200,
                "max_chunk_limit": 300,
                "min_chunk_length": 50,
                "spacy_model": "en_core_web_sm",
            },
            "audio": {
                "sample_rate": 24000,  # ChatterboxTTS native sample rate
                "silence_duration": {"normal": 0.1, "paragraph": 0.2},
            },
        }

    def test_chunking_integration(self, sample_text, config):
        """Test SpaCy chunker integration."""
        chunker = SpaCyChunker(
            model_name=config["chunking"]["spacy_model"],
            target_limit=config["chunking"]["target_chunk_limit"],
            max_limit=config["chunking"]["max_chunk_limit"],
            min_length=config["chunking"]["min_chunk_length"],
        )

        chunks = chunker.chunk_text(sample_text)

        assert len(chunks) > 0
        assert all(
            len(chunk.text) <= config["chunking"]["max_chunk_limit"] for chunk in chunks
        )
        assert all(
            len(chunk.text) >= config["chunking"]["min_chunk_length"]
            for chunk in chunks
        )

        # Check paragraph break detection
        paragraph_chunks = [chunk for chunk in chunks if chunk.has_paragraph_break]
        assert len(paragraph_chunks) > 0

    def test_chunk_validation_integration(self, sample_text, config):
        """Test chunk validator integration."""
        chunker = SpaCyChunker(
            model_name=config["chunking"]["spacy_model"],
            target_limit=config["chunking"]["target_chunk_limit"],
            max_limit=config["chunking"]["max_chunk_limit"],
            min_length=config["chunking"]["min_chunk_length"],
        )

        chunks = chunker.chunk_text(sample_text)

        validator = ChunkValidator(
            max_limit=config["chunking"]["max_chunk_limit"],
            min_length=config["chunking"]["min_chunk_length"],
        )

        # All chunks should pass validation
        for chunk in chunks:
            assert validator.validate_chunk_length(chunk)
            assert validator.validate_sentence_boundaries(chunk)

        # Overall validation should pass
        assert validator.run_all_validations(chunks)

    def test_mock_audio_generation_integration(self, sample_text, config):
        """Test mock audio generation and processing."""
        # Generate chunks
        chunker = SpaCyChunker(
            model_name=config["chunking"]["spacy_model"],
            target_limit=config["chunking"]["target_chunk_limit"],
            max_limit=config["chunking"]["max_chunk_limit"],
            min_length=config["chunking"]["min_chunk_length"],
        )

        chunks = chunker.chunk_text(sample_text)

        # Generate mock audio segments
        device = "cpu"
        audio_segments = []
        has_paragraph_breaks = []

        for i, chunk in enumerate(chunks):
            # Create sine wave mock audio
            duration_seconds = max(1.0, len(chunk.text) / 15)
            sample_rate = config["audio"]["sample_rate"]
            num_samples = int(duration_seconds * sample_rate)
            time_tensor = torch.linspace(
                0, duration_seconds, num_samples, device=device
            )
            frequency = 440 + (i * 20)  # Vary frequency per chunk
            amplitude = 0.3
            mock_audio = (
                torch.sin(2 * torch.pi * frequency * time_tensor) * amplitude
            ).unsqueeze(0)

            audio_segments.append(mock_audio)
            has_paragraph_breaks.append(chunk.has_paragraph_break)

        # Process audio
        audio_processor = AudioProcessor(
            sample_rate=config["audio"]["sample_rate"],
            normal_silence_duration=config["audio"]["silence_duration"]["normal"],
            paragraph_silence_duration=config["audio"]["silence_duration"]["paragraph"],
            device=device,
        )

        final_audio = audio_processor.concatenate_segments(
            audio_segments=audio_segments, has_paragraph_breaks=has_paragraph_breaks
        )

        # Verify final audio
        assert final_audio is not None
        assert final_audio.shape[0] == 1  # Mono audio
        assert final_audio.shape[1] > 0  # Has samples

        duration = audio_processor.get_audio_duration(final_audio)
        assert duration > 0

    def test_progress_tracker_integration(self, sample_text, config):
        """Test progress tracker integration."""
        chunker = SpaCyChunker(
            model_name=config["chunking"]["spacy_model"],
            target_limit=config["chunking"]["target_chunk_limit"],
            max_limit=config["chunking"]["max_chunk_limit"],
            min_length=config["chunking"]["min_chunk_length"],
        )

        chunks = chunker.chunk_text(sample_text)

        # Test progress tracker
        progress_tracker = ProgressTracker(len(chunks), "Test Progress")

        for i in range(len(chunks)):
            progress_tracker.update(i + 1, f"Processing chunk {i+1}")

        progress_tracker.finish()

        # Progress tracker should complete without errors
        assert progress_tracker.current_item == len(chunks)
        assert progress_tracker.total_items == len(chunks)

    def test_full_mock_pipeline_integration(self, sample_text, config):
        """Test complete mock pipeline integration."""
        device = "cpu"

        # Phase 1: Chunking
        chunker = SpaCyChunker(
            model_name=config["chunking"]["spacy_model"],
            target_limit=config["chunking"]["target_chunk_limit"],
            max_limit=config["chunking"]["max_chunk_limit"],
            min_length=config["chunking"]["min_chunk_length"],
        )

        chunks = chunker.chunk_text(sample_text)
        assert len(chunks) > 0

        # Phase 2: Mock Audio Generation
        progress_tracker = ProgressTracker(len(chunks), "Mock TTS Generation")
        audio_segments = []
        has_paragraph_breaks = []

        for i, chunk in enumerate(chunks):
            progress_tracker.update(i + 1, f"Generating mock audio for chunk {i+1}")

            # Generate mock audio
            duration_seconds = max(1.0, len(chunk.text) / 15)
            sample_rate = config["audio"]["sample_rate"]
            num_samples = int(duration_seconds * sample_rate)
            time_tensor = torch.linspace(
                0, duration_seconds, num_samples, device=device
            )
            frequency = 440 + (i * 20)
            amplitude = 0.3
            mock_audio = (
                torch.sin(2 * torch.pi * frequency * time_tensor) * amplitude
            ).unsqueeze(0)

            audio_segments.append(mock_audio)
            has_paragraph_breaks.append(chunk.has_paragraph_break)

        progress_tracker.finish()

        # Phase 3: Audio Processing
        audio_processor = AudioProcessor(
            sample_rate=config["audio"]["sample_rate"],
            normal_silence_duration=config["audio"]["silence_duration"]["normal"],
            paragraph_silence_duration=config["audio"]["silence_duration"]["paragraph"],
            device=device,
        )

        final_audio = audio_processor.concatenate_segments(
            audio_segments=audio_segments, has_paragraph_breaks=has_paragraph_breaks
        )

        # Verify complete pipeline
        assert final_audio is not None
        assert len(audio_segments) == len(chunks)

        duration = audio_processor.get_audio_duration(final_audio)
        assert duration > 0

        # Should have processed all chunks
        assert len(audio_segments) == len(chunks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
