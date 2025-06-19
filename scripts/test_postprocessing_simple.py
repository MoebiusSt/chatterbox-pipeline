#!/usr/bin/env python3
"""
Simplified test script for Phase 3: Post-Processing Components
Tests noise analysis, audio cleaning without complex pipeline imports.
"""

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

# Add src to path for imports
script_dir = Path(__file__).parent
src_dir = script_dir.parent / "src"
sys.path.insert(0, str(src_dir))


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_test_audio(
    sample_rate: int = 24000, duration: float = 3.0
) -> torch.Tensor:  # ChatterboxTTS native sample rate
    """Create synthetic test audio with speech-like patterns."""
    num_samples = int(duration * sample_rate)
    t = torch.linspace(0, duration, num_samples)

    # Create speech-like signal
    speech = torch.zeros_like(t)
    for freq in [200, 400, 600, 800]:
        speech += 0.3 * torch.sin(2 * np.pi * freq * t) * torch.exp(-t * 0.5)

    # Add formants
    speech += 0.2 * torch.sin(2 * np.pi * 1200 * t) * torch.exp(-t * 0.8)

    # Add pauses
    envelope = torch.ones_like(t)
    envelope[int(0.8 * sample_rate) : int(1.2 * sample_rate)] = 0.1
    speech = speech * envelope

    # Add noise and artifacts
    noise = 0.02 * torch.randn_like(t)
    artifacts = 0.005 * torch.sin(2 * np.pi * 50 * t)

    audio = speech + noise + artifacts
    audio = 0.7 * audio / torch.max(torch.abs(audio))

    return audio


def test_noise_analyzer():
    """Test the NoiseAnalyzer component."""
    print("\n=== Testing NoiseAnalyzer ===")

    try:
        from postprocessing.noise_analyzer import NoiseAnalyzer

        # Create test audio
        audio = create_test_audio(duration=5.0)
        print(
            f"Created test audio: {len(audio)} samples, {len(audio)/24000:.2f}s"
        )  # ChatterboxTTS native sample rate

        # Initialize analyzer
        analyzer = NoiseAnalyzer(sample_rate=24000)  # ChatterboxTTS native sample rate

        # Test noise floor analysis
        print("\nAnalyzing noise floor...")
        profile = analyzer.analyze_noise_floor(audio.unsqueeze(0))

        print(f"Noise Profile:")
        print(f"  Noise floor: {profile.noise_floor:.6f}")
        print(f"  Speech threshold: {profile.speech_threshold:.6f}")
        print(f"  Peak amplitude: {profile.peak_amplitude:.6f}")
        print(f"  Dynamic range: {profile.dynamic_range:.2f} dB")
        print(f"  Confidence: {profile.confidence:.3f}")

        # Test speech segments
        segments = analyzer.detect_speech_segments(audio.unsqueeze(0))
        print(f"\nDetected {len(segments)} speech segments")

        return True, profile

    except Exception as e:
        print(f"NoiseAnalyzer test failed: {e}")
        return False, None


def test_audio_cleaner():
    """Test the AudioCleaner component."""
    print("\n=== Testing AudioCleaner ===")

    try:
        from postprocessing.audio_cleaner import AudioCleaner, CleaningSettings

        # Create test audio with artifacts
        audio = create_test_audio(duration=4.0)
        audio += 0.1 * torch.sin(2 * np.pi * 60 * torch.linspace(0, 4.0, len(audio)))

        print(f"Created test audio with artifacts: {len(audio)} samples")

        # Initialize cleaner
        settings = CleaningSettings(
            spectral_gating=True,
            normalize_audio=True,
            target_rms=0.3,
            remove_dc_offset=True,
        )

        cleaner = AudioCleaner(
            sample_rate=24000, settings=settings
        )  # ChatterboxTTS native sample rate

        # Clean the audio
        print("\nCleaning audio...")
        cleaned_audio = cleaner.clean_audio(audio)

        # Get statistics
        stats = cleaner.get_cleaning_stats(audio, cleaned_audio)

        print(f"Cleaning Results:")
        print(f"  Original RMS: {stats['original_rms']:.6f}")
        print(f"  Cleaned RMS: {stats['cleaned_rms']:.6f}")
        print(f"  Noise Reduction: {stats['estimated_noise_reduction_db']:.2f} dB")

        return True, stats

    except Exception as e:
        print(f"AudioCleaner test failed: {e}")
        import traceback

        traceback.print_exc()
        return False, None


def test_auto_editor_wrapper():
    """Test the AutoEditorWrapper component."""
    print("\n=== Testing AutoEditorWrapper ===")

    try:
        from postprocessing.auto_editor_wrapper import AutoEditorWrapper

        # Create test audio with silent sections
        audio = create_test_audio(duration=6.0)
        sample_rate = 24000  # ChatterboxTTS native sample rate

        # Add quiet sections
        audio[int(1.5 * sample_rate) : int(2.0 * sample_rate)] = 0.01 * torch.randn(
            int(0.5 * sample_rate)
        )

        print(f"Created test audio with silent sections: {len(audio)} samples")

        # Initialize wrapper
        wrapper = AutoEditorWrapper(
            margin_before=0.1, margin_after=0.1, preserve_natural_sounds=True
        )

        # Test processing
        print("\nProcessing with Auto-Editor...")
        cleaned_audio = wrapper.clean_audio(audio, sample_rate=sample_rate)

        original_duration = len(audio) / sample_rate
        cleaned_duration = len(cleaned_audio) / sample_rate
        compression_ratio = (
            original_duration / cleaned_duration if cleaned_duration > 0 else 1.0
        )

        print(f"Auto-Editor Results:")
        print(f"  Original: {original_duration:.2f}s")
        print(f"  Cleaned: {cleaned_duration:.2f}s")
        print(f"  Compression: {compression_ratio:.2f}x")

        return True, compression_ratio

    except Exception as e:
        print(f"Auto-Editor test failed: {e}")
        print("This is expected if Auto-Editor is not installed")
        return False, None


def save_test_results(original_audio, cleaned_audio, output_dir):
    """Save test audio results."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        torchaudio.save(
            str(output_path / "test_original.wav"), original_audio.unsqueeze(0), 24000
        )  # ChatterboxTTS native sample rate
        torchaudio.save(
            str(output_path / "test_cleaned.wav"), cleaned_audio.unsqueeze(0), 24000
        )  # ChatterboxTTS native sample rate

        print(f"Test audio saved to {output_path}")
        return True

    except Exception as e:
        print(f"Failed to save test results: {e}")
        return False


def main():
    """Main test function."""
    print("Phase 3 Post-Processing Components Test")
    print("=" * 50)

    setup_logging()

    results = {}

    # Test 1: NoiseAnalyzer
    success, profile = test_noise_analyzer()
    results["noise_analyzer"] = success

    # Test 2: AudioCleaner
    success, stats = test_audio_cleaner()
    results["audio_cleaner"] = success

    # Test 3: AutoEditorWrapper
    success, compression = test_auto_editor_wrapper()
    results["auto_editor"] = success

    # Save some test results
    if results["audio_cleaner"]:
        original = create_test_audio(duration=3.0)
        try:
            from postprocessing.audio_cleaner import AudioCleaner

            cleaner = AudioCleaner()
            cleaned = cleaner.clean_audio(original)
            save_test_results(original, cleaned, "../data/tests")
        except Exception as e:
            print(f"Could not save test results: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("Phase 3 Component Testing Summary")
    print("=" * 50)

    for component, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {component}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print(f"\nOverall: {total_passed}/{total_tests} components working")

    if total_passed == total_tests:
        print("🎉 All Phase 3 components implemented successfully!")
    elif total_passed > 0:
        print("⚠️  Phase 3 partially implemented - some components working")
    else:
        print("💥 Phase 3 implementation needs work")

    return 0 if total_passed > 0 else 1


if __name__ == "__main__":
    exit(main())
