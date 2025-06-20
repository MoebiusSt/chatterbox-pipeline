"""
Auto-Editor wrapper for automated audio post-processing.
Removes silent periods and artifacts while preserving natural speech sounds.
"""

import logging
import os
import shutil
import subprocess

# Use absolute imports
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torchaudio

from postprocessing.noise_analyzer import NoiseAnalyzer, NoiseProfile


@dataclass
class ProcessingResult:
    """Result of Auto-Editor processing."""

    success: bool
    input_path: str
    output_path: str
    original_duration: float
    processed_duration: float
    compression_ratio: float
    error_message: Optional[str] = None
    auto_editor_output: Optional[str] = None


class AutoEditorWrapper:
    """
    Wrapper for Auto-Editor that provides intelligent audio artifact removal.
    Automatically determines optimal parameters and handles file I/O.
    """

    def __init__(
        self,
        margin_before: float = 0.1,
        margin_after: float = 0.1,
        preserve_natural_sounds: bool = True,
        temp_dir: Optional[str] = None,
    ):
        """
        Initialize AutoEditorWrapper.

        Args:
            margin_before: Margin before cuts (seconds)
            margin_after: Margin after cuts (seconds)
            preserve_natural_sounds: Whether to preserve breathing/lip sounds
            temp_dir: Directory for temporary files (None = system temp)
        """
        self.margin_before = margin_before
        self.margin_after = margin_after
        self.preserve_natural_sounds = preserve_natural_sounds
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.logger = logging.getLogger(__name__)

        # Initialize noise analyzer
        self.noise_analyzer = NoiseAnalyzer()

        # Verify Auto-Editor is available
        self._verify_auto_editor()

    def _verify_auto_editor(self):
        """Verify that Auto-Editor is available and working."""
        try:
            result = subprocess.run(
                ["auto-editor", "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                self.logger.info(f"Auto-Editor available: {version}")
            else:
                raise RuntimeError("Auto-Editor not working properly")
        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            FileNotFoundError,
        ) as e:
            self.logger.error(f"Auto-Editor verification failed: {e}")
            raise RuntimeError(
                "Auto-Editor not available. Please install with: pip install auto-editor"
            )

    def clean_audio(
        self,
        audio: torch.Tensor,
        sample_rate: int = 24000,  # ChatterboxTTS native sample rate
        reference_audio_path: Optional[str] = None,
        custom_threshold: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Clean audio using Auto-Editor with intelligent parameter detection.

        Args:
            audio: Input audio tensor (1, num_samples) or (num_samples,)
            sample_rate: Sample rate of audio
            reference_audio_path: Optional reference audio for threshold calculation
            custom_threshold: Optional custom threshold (overrides auto-detection)

        Returns:
            Cleaned audio tensor
        """
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as input_file:
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
            output_path = output_file.name

        try:
            # Save input audio
            if audio.dim() == 2:
                audio_for_save = audio
            else:
                audio_for_save = audio.unsqueeze(0)

            torchaudio.save(input_path, audio_for_save.cpu(), sample_rate)

            # Process with Auto-Editor
            result = self._process_file(
                input_path, output_path, reference_audio_path, custom_threshold
            )

            if result.success:
                # Load processed audio
                processed_audio, _ = torchaudio.load(output_path)

                # Match original dimensions
                if audio.dim() == 1:
                    processed_audio = processed_audio.squeeze(0)

                self.logger.info(
                    f"Audio cleaned successfully: "
                    f"{result.original_duration:.2f}s → {result.processed_duration:.2f}s "
                    f"(compression: {result.compression_ratio:.2f}x)"
                )

                return processed_audio
            else:
                self.logger.warning(
                    f"Auto-Editor processing failed: {result.error_message}"
                )
                self.logger.warning("Returning original audio")
                return audio

        except Exception as e:
            self.logger.error(f"Audio cleaning failed: {e}")
            return audio
        finally:
            # Clean up temporary files
            for path in [input_path, output_path]:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up {path}: {e}")

    def clean_audio_file(
        self,
        input_path: str,
        output_path: str,
        reference_audio_path: Optional[str] = None,
        custom_threshold: Optional[float] = None,
    ) -> ProcessingResult:
        """
        Clean audio file using Auto-Editor.

        Args:
            input_path: Path to input audio file
            output_path: Path for output audio file
            reference_audio_path: Optional reference audio for threshold calculation
            custom_threshold: Optional custom threshold

        Returns:
            ProcessingResult with operation details
        """
        return self._process_file(
            input_path, output_path, reference_audio_path, custom_threshold
        )

    def _process_file(
        self,
        input_path: str,
        output_path: str,
        reference_audio_path: Optional[str] = None,
        custom_threshold: Optional[float] = None,
    ) -> ProcessingResult:
        """Internal file processing method."""
        try:
            # Get original duration
            original_audio, sr = torchaudio.load(input_path)
            original_duration = original_audio.shape[-1] / sr

            # Determine processing parameters
            if custom_threshold is not None:
                threshold_db = (
                    20 * torch.log10(torch.tensor(max(custom_threshold, 1e-8))).item()
                )
            else:
                # Analyze audio to determine optimal threshold
                if reference_audio_path and Path(reference_audio_path).exists():
                    self.logger.info(
                        f"Using reference audio for threshold: {reference_audio_path}"
                    )
                    profile = self.noise_analyzer.analyze_reference_audio(
                        reference_audio_path
                    )
                else:
                    self.logger.info("Analyzing input audio for threshold")
                    profile = self.noise_analyzer.analyze_noise_floor(original_audio)

                threshold_db = (
                    20
                    * torch.log10(
                        torch.tensor(max(profile.recommended_threshold, 1e-8))
                    ).item()
                )

            # Build Auto-Editor command
            cmd = self._build_command(input_path, output_path, threshold_db)

            self.logger.info(f"Running Auto-Editor with threshold {threshold_db:.1f}dB")
            self.logger.debug(f"Command: {' '.join(cmd)}")

            # Run Auto-Editor
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                # Check if output file was created
                if Path(output_path).exists():
                    # Get processed duration
                    processed_audio, _ = torchaudio.load(output_path)
                    processed_duration = processed_audio.shape[-1] / sr
                    compression_ratio = (
                        original_duration / processed_duration
                        if processed_duration > 0
                        else 1.0
                    )

                    return ProcessingResult(
                        success=True,
                        input_path=input_path,
                        output_path=output_path,
                        original_duration=original_duration,
                        processed_duration=processed_duration,
                        compression_ratio=compression_ratio,
                        auto_editor_output=result.stdout,
                    )
                else:
                    return ProcessingResult(
                        success=False,
                        input_path=input_path,
                        output_path=output_path,
                        original_duration=original_duration,
                        processed_duration=0.0,
                        compression_ratio=1.0,
                        error_message="Output file not created",
                        auto_editor_output=result.stdout,
                    )
            else:
                error_msg = f"Auto-Editor failed (exit code {result.returncode}): {result.stderr}"
                return ProcessingResult(
                    success=False,
                    input_path=input_path,
                    output_path=output_path,
                    original_duration=original_duration,
                    processed_duration=0.0,
                    compression_ratio=1.0,
                    error_message=error_msg,
                    auto_editor_output=result.stdout,
                )

        except subprocess.TimeoutExpired:
            error_msg = "Auto-Editor timed out"
            self.logger.error(error_msg)
            return ProcessingResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                original_duration=0.0,
                processed_duration=0.0,
                compression_ratio=1.0,
                error_message=error_msg,
            )
        except Exception as e:
            error_msg = f"Auto-Editor processing failed: {e}"
            self.logger.error(error_msg)
            return ProcessingResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                original_duration=0.0,
                processed_duration=0.0,
                compression_ratio=1.0,
                error_message=error_msg,
            )

    def _build_command(
        self, input_path: str, output_path: str, threshold_db: float
    ) -> List[str]:
        """Build Auto-Editor command with appropriate parameters."""
        # Auto-Editor >= 24 uses the generic "--edit audio:threshold=<value>dB" syntax
        # to specify silence-detection thresholds. The old --silent-threshold flag
        # has been removed (see blog post "Why It's Time to Remove --silent-threshold").
        # We therefore rely on the default "audio" edit mode and pass an explicit
        # threshold. Keeping --margin for padding around cuts and disabling any UI.

        cmd = [
            "auto-editor",
            input_path,
            "--output",
            output_path,
            "--edit",
            f"audio:threshold={threshold_db:.1f}dB",
            "--no-open",  # Don't open the output file
            # Auto-Editor ≥ 28 hat strikte Prüfungen für doppelte --margin-Angaben.
            # Selbst eine einzelne Angabe führt in manchen Umgebungen zu
            # "Option --margin may not be used more than once". Wir verzichten
            # daher vollständig auf das Flag und nutzen den Default-Wert (0.2 s).
        ]

        # Add natural sound preservation settings
        if self.preserve_natural_sounds:
            # Use smaller frame margin to preserve subtle sounds
            cmd.extend(["--frame-margin", "1"])

            # Conservative cutting settings - REMOVED deprecated --min-clip-length and --min-cut-length
            # Auto-Editor v28+ no longer supports these flags
            # The new --edit syntax handles minimum durations automatically
        else:
            # More aggressive settings
            cmd.extend(
                [
                    "--frame-margin",
                    "3",
                    # REMOVED deprecated min-clip-length and min-cut-length flags
                ]
            )

        return cmd

    def batch_process(
        self,
        input_files: List[str],
        output_dir: str,
        reference_audio_path: Optional[str] = None,
    ) -> List[ProcessingResult]:
        """
        Process multiple audio files in batch.

        Args:
            input_files: List of input file paths
            output_dir: Directory for output files
            reference_audio_path: Optional reference audio for threshold

        Returns:
            List of ProcessingResults
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []

        for input_file in input_files:
            input_path = Path(input_file)
            output_path = output_dir / f"cleaned_{input_path.name}"

            self.logger.info(f"Processing {input_path.name}...")
            result = self.clean_audio_file(
                str(input_path), str(output_path), reference_audio_path
            )
            results.append(result)

            if result.success:
                self.logger.info(f"✅ {input_path.name} processed successfully")
            else:
                self.logger.warning(
                    f"❌ {input_path.name} processing failed: {result.error_message}"
                )

        successful = sum(1 for r in results if r.success)
        self.logger.info(
            f"Batch processing complete: {successful}/{len(results)} successful"
        )

        return results

    def calculate_threshold_from_reference(self, reference_path: str) -> float:
        """
        Calculate optimal threshold from reference audio.

        Args:
            reference_path: Path to reference audio file

        Returns:
            Recommended threshold value
        """
        try:
            profile = self.noise_analyzer.analyze_reference_audio(reference_path)
            return profile.recommended_threshold
        except Exception as e:
            self.logger.error(f"Reference threshold calculation failed: {e}")
            return 0.01  # Fallback threshold
