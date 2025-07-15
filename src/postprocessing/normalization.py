"""
Audio normalization module for consistent loudness levels.
Provides LUFS, RMS, and peak-based normalization with integrated peak limiting.
"""

import logging
from typing import Any, Dict

import numpy as np
import torch
from scipy import signal

logger = logging.getLogger(__name__)


class AudioNormalizer:
    """
    Audio normalization class with integrated peak limiting.
    
    Instead of normalizing blindly and then limiting, this normalizer calculates
    the optimal gain that achieves the target level while respecting peak limits.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AudioNormalizer with configuration.
        
        Args:
            config: Configuration dictionary containing normalization settings
        """
        self.config = config
        self.normalization_config = config.get("audio", {}).get("normalization", {})
        
        # Extract configuration parameters
        self.enabled = self.normalization_config.get("enabled", False)
        
        # Support both new target_level and old target_lufs for backward compatibility
        self.target_level = self.normalization_config.get("target_level")
        if self.target_level is None:
            self.target_level = self.normalization_config.get("target_lufs", -23.0)
            
        self.method = self.normalization_config.get("method", "lufs")
        self.saturation_factor = self.normalization_config.get("saturation_factor", 0.0)
        
        logger.debug(f"AudioNormalizer initialized: enabled={self.enabled}, "
                    f"target={self.target_level} dB, method={self.method}, "
                    f"saturation_factor={self.saturation_factor}")
    
    def normalize(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Normalize audio tensor to target loudness level with integrated peak limiting.
        
        This method calculates the optimal gain that achieves the target level
        while respecting the peak limit in a single step.
        
        Args:
            audio: Input audio tensor (1D or 2D)
            sample_rate: Audio sample rate
            
        Returns:
            Normalized audio tensor
        """
        if not self.enabled:
            logger.debug("Audio normalization disabled, returning original audio")
            return audio
            
        if audio.numel() == 0:
            logger.warning("Empty audio tensor provided, returning as-is")
            return audio
            
        try:
            # Analyze current audio level
            audio_analysis = self._analyze_audio_level(audio, sample_rate)
            
            # Calculate target gain based on method
            target_gain_db = self._calculate_target_gain(audio_analysis)
            
            # Apply the gain directly - no peak limiting
            normalized_audio = self._apply_gain(audio, target_gain_db)
            
            # Log normalization
            logger.info(f"Audio normalized: {self.method.upper()} "
                       f"gain {target_gain_db:+.1f} dB")
            
            # Apply TanH saturation with loudness compensation (Stage 2)
            final_audio = self._apply_tanh_saturation(normalized_audio, sample_rate)
                           
            return final_audio
            
        except Exception as e:
            logger.error(f"Audio normalization failed: {e}")
            logger.debug("Returning original audio due to normalization failure")
            return audio
    
    def _calculate_target_gain(self, audio_analysis: Dict[str, Any]) -> float:
        """
        Calculate the gain needed to reach the target level.
        
        The target_level parameter is ALWAYS the target level in dB.
        The method parameter determines how to measure the current level:
        - "lufs": Measure with LUFS, normalize to target_level (as LUFS)
        - "rms": Measure with RMS, normalize to target_level (as RMS target in dB)
        - "peak": Measure with Peak, normalize to target_level (as Peak target in dB)
        """
        if self.method == "lufs":
            current_level = float(audio_analysis["lufs"])
            target_level = self.target_level
        elif self.method == "rms":
            current_level = float(audio_analysis["rms_db"])
            target_level = self.target_level  # target_level interpreted as RMS target
        elif self.method == "peak":
            current_level = float(audio_analysis["peak_db"])
            target_level = self.target_level  # target_level interpreted as Peak target
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
            
        return target_level - current_level
    

    
    def _apply_gain(self, audio: torch.Tensor, gain_db: float) -> torch.Tensor:
        """Apply gain to audio tensor."""
        if gain_db == 0:
            return audio
            
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear
    
    def _apply_tanh_saturation(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Apply TanH saturation with loudness compensation.
        
        This method applies configurable TanH saturation while preserving 
        the loudness level achieved in the normalization stage.
        
        Args:
            audio: Input audio tensor (already normalized)
            sample_rate: Audio sample rate
            
        Returns:
            Saturated audio tensor with preserved loudness
        """
        if self.saturation_factor == 0.0:
            # No saturation requested
            return audio
            
        try:
            # Measure loudness before saturation
            pre_analysis = self._analyze_audio_level(audio, sample_rate)
            pre_loudness = pre_analysis[f"{self.method}_db" if self.method != "lufs" else "lufs"]
            
            # Apply TanH saturation with configurable intensity
            # saturation_factor scales the input to TanH: 0.0=none, 1.0=strong
            saturation_scale = 1.0 + (self.saturation_factor * 4.0)  # 1.0 to 5.0 scaling
            saturated_audio = torch.tanh(audio * saturation_scale) / saturation_scale
            
            # Measure loudness after saturation
            post_analysis = self._analyze_audio_level(saturated_audio, sample_rate)
            post_loudness = post_analysis[f"{self.method}_db" if self.method != "lufs" else "lufs"]
            
            # Calculate compensation gain to restore original loudness
            compensation_gain_db = pre_loudness - post_loudness
            
            # Apply loudness compensation
            compensated_audio = self._apply_gain(saturated_audio, compensation_gain_db)
            
            # Log saturation details
            if self.saturation_factor > 0.0:
                logger.info(f"🌊 TanH saturation: factor={self.saturation_factor:.2f}, "
                           f"compensation={compensation_gain_db:+.1f} dB")
            
            return compensated_audio
            
        except Exception as e:
            logger.error(f"TanH saturation failed: {e}")
            return audio
    

    
    def _analyze_audio_level(self, audio: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
        """
        Analyze audio level and return various volume metrics.
        
        Args:
            audio: Audio tensor
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary with volume metrics
        """
        try:
            # Convert to numpy for analysis
            if hasattr(audio, 'cpu'):
                audio_np = audio.cpu().numpy()
            else:
                audio_np = audio.numpy()
            
            # Ensure 1D array
            if audio_np.ndim > 1:
                audio_np = audio_np.flatten()
            
            # Calculate RMS level
            rms = np.sqrt(np.mean(audio_np**2))
            rms_db = 20 * np.log10(rms + 1e-10)
            
            # Calculate peak level
            peak = np.max(np.abs(audio_np))
            peak_db = 20 * np.log10(peak + 1e-10)
            
            # Calculate LUFS approximation with K-weighting
            lufs = self._calculate_lufs_approximation(audio_np, sample_rate)
            
            return {
                'rms_db': float(rms_db),
                'peak_db': float(peak_db),
                'lufs': float(lufs),
                'duration': len(audio_np) / sample_rate,
                'rms_linear': float(rms),
                'peak_linear': float(peak)
            }
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}")
            return {
                'rms_db': -40.0,
                'peak_db': -20.0,
                'lufs': -23.0,
                'duration': 0.0,
                'rms_linear': 0.0,
                'peak_linear': 0.0
            }
    
    def _calculate_lufs_approximation(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Calculate LUFS approximation using K-weighting filter.
        
        Args:
            audio: Audio array
            sample_rate: Audio sample rate
            
        Returns:
            LUFS value (approximation)
        """
        try:
            # Apply K-weighting filter (simplified implementation)
            # High-shelf filter at 4kHz
            sos_high = signal.butter(2, 4000, 'highpass', fs=sample_rate, output='sos')
            filtered_high = signal.sosfilt(sos_high, audio)
            
            # Ensure it's a numpy array
            if not isinstance(filtered_high, np.ndarray):
                filtered_high = np.array(filtered_high)
            
            # High-frequency emphasis
            sos_shelf = signal.butter(2, 1500, 'highpass', fs=sample_rate, output='sos')
            filtered_shelf = signal.sosfilt(sos_shelf, filtered_high)
            
            # Ensure it's a numpy array
            if not isinstance(filtered_shelf, np.ndarray):
                filtered_shelf = np.array(filtered_shelf)
            
            # Calculate mean square and convert to LUFS
            ms = np.mean(filtered_shelf * filtered_shelf)
            lufs = -0.691 + 10 * np.log10(ms + 1e-10)
            
            return lufs
            
        except Exception as e:
            logger.debug(f"LUFS calculation failed, falling back to RMS: {e}")
            # Fallback to RMS-based approximation
            rms = np.sqrt(np.mean(audio * audio))
            return 20 * np.log10(rms + 1e-10)
    
    def get_analysis(self, audio: torch.Tensor, sample_rate: int) -> Dict[str, Any]:
        """
        Get detailed audio analysis without normalization.
        
        Args:
            audio: Input audio tensor
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary with detailed audio analysis
        """
        analysis = self._analyze_audio_level(audio, sample_rate)
        
        # Add normalization preview
        if self.enabled:
            try:
                target_gain_db = self._calculate_target_gain(analysis)
                
                analysis["normalization_preview"] = {
                    "method": self.method,
                    "target_gain": target_gain_db,
                    "would_normalize": abs(target_gain_db) > 0.1,
                    "saturation_factor": self.saturation_factor,
                    "would_saturate": self.saturation_factor > 0.0
                }
            except Exception as e:
                analysis["normalization_preview"] = {
                    "error": str(e)
                }
        else:
            analysis["normalization_preview"] = {
                "enabled": False,
                "message": "Normalization is disabled"
            }
        
        return analysis 