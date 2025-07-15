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
        self.peak_limit = self.normalization_config.get("peak_limit", -1.0)
        
        # Aggressive mode for brickwall limiting
        self.aggressive_mode = self.normalization_config.get("aggressive_mode", False)
        
        # Scale factor for TanH limiting curve
        self.tanh_scale_factor = self.normalization_config.get("tanh_scale_factor", 1.5)
        
        logger.debug(f"AudioNormalizer initialized: enabled={self.enabled}, "
                    f"target={self.target_level} dB, method={self.method}, "
                    f"aggressive_mode={self.aggressive_mode}, tanh_scale_factor={self.tanh_scale_factor}")
    
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
            
            if self.aggressive_mode:
                # Aggressive mode: Apply target gain first, then limit
                logger.debug(f"Aggressive mode: applying {target_gain_db:+.1f} dB gain, then limiting")
                
                # Apply target gain without peak consideration
                gained_audio = self._apply_gain(audio, target_gain_db)
                
                # Apply brickwall limiting
                normalized_audio = self._apply_brickwall_limiting(gained_audio, self.peak_limit)
                
                # Log what happened
                logger.info(f"Audio normalized (AGGRESSIVE): {self.method.upper()} "
                           f"gain {target_gain_db:+.1f} dB + TanH limiting at {self.peak_limit:.1f} dB")
                           
            else:
                # Standard mode: Calculate peak-limited gain (the key improvement!)
                limited_gain_db = self._calculate_peak_limited_gain(
                    audio, target_gain_db, self.peak_limit
                )
                
                # Apply the optimal gain in one step
                normalized_audio = self._apply_gain(audio, limited_gain_db)
                
                # Log what happened
                if abs(limited_gain_db - target_gain_db) > 0.1:
                    logger.info(f"Audio normalized: {self.method.upper()} "
                               f"target gain {target_gain_db:+.1f} dB → "
                               f"peak-limited gain {limited_gain_db:+.1f} dB")
                else:
                    logger.info(f"Audio normalized: {self.method.upper()} "
                               f"gain {limited_gain_db:+.1f} dB")
                           
            return normalized_audio
            
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
        
        peak_limit is ALWAYS only used as a safety limit to prevent clipping.
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
    
    def _calculate_peak_limited_gain(
        self, 
        audio: torch.Tensor, 
        target_gain_db: float,
        peak_limit_db: float
    ) -> float:
        """
        Calculate the actual gain considering peak limits.
        
        This is the key improvement: we calculate what the peak would be
        AFTER applying the target gain, and limit the gain if necessary.
        """
        # Convert to numpy for analysis
        if hasattr(audio, 'cpu'):
            audio_np = audio.cpu().numpy()
        else:
            audio_np = audio.numpy()
            
        if audio_np.ndim > 1:
            audio_np = audio_np.flatten()
        
        # Calculate current peak
        current_peak = np.max(np.abs(audio_np))
        
        if current_peak == 0:
            return 0.0  # Silent audio
            
        # Calculate what the peak would be after applying target gain
        target_gain_linear = 10 ** (target_gain_db / 20)
        predicted_peak = current_peak * target_gain_linear
        
        # Convert peak limit to linear
        peak_limit_linear = 10 ** (peak_limit_db / 20)
        
        # If predicted peak would exceed limit, reduce the gain
        if predicted_peak > peak_limit_linear:
            # Calculate the maximum gain we can apply
            max_gain_linear = peak_limit_linear / current_peak
            max_gain_db = 20 * np.log10(max_gain_linear)
            
            logger.debug(f"Peak limiting: target gain {target_gain_db:+.1f} dB → "
                        f"limited gain {max_gain_db:+.1f} dB")
            
            return max_gain_db
        else:
            # No limiting needed
            return target_gain_db
    
    def _apply_gain(self, audio: torch.Tensor, gain_db: float) -> torch.Tensor:
        """Apply gain to audio tensor."""
        if gain_db == 0:
            return audio
            
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear
    
    def _apply_brickwall_limiting(self, audio: torch.Tensor, threshold_db: float) -> torch.Tensor:
        """
        Apply soft limiting using TanH function for musical clipping.
        
        TanH limiting provides smooth, musical distortion instead of harsh clipping.
        The function is scaled so that its maximum exactly matches the threshold.
        
        Args:
            audio: Input audio tensor
            threshold_db: Threshold in dB for soft limiting
            
        Returns:
            TanH-limited audio tensor
        """
        try:
            # Convert threshold to linear scale
            threshold_linear = 10 ** (threshold_db / 20)
            
            # Scale factor for TanH limiting curve
            scale_factor = self.tanh_scale_factor  # Configurable saturation curve
            
            # Apply TanH limiting with correct scaling
            # We want the output to reach exactly threshold_linear at high inputs
            
            # Step 1: Scale audio for TanH processing
            # We scale relative to the threshold so that threshold input → scale_factor
            scaled_audio = audio / threshold_linear * scale_factor
            
            # Step 2: Apply TanH
            tanh_limited = torch.tanh(scaled_audio)
            
            # Step 3: Scale the result to reach the threshold
            # Since TanH approaches 1 asymptotically, we scale by threshold_linear
            # This ensures that TanH(∞) → threshold_linear
            limited_audio = tanh_limited * threshold_linear
            
            # Count samples that were affected by limiting
            affected_samples = torch.sum(torch.abs(audio - limited_audio) > 0.001)
            total_samples = audio.numel()
            
            if affected_samples > 0:
                affected_percent = (affected_samples.float() / total_samples) * 100
                max_output = torch.max(torch.abs(limited_audio))
                max_output_db = 20 * torch.log10(max_output + 1e-10)
                logger.info(f"🌊 TanH limiting: {affected_samples}/{total_samples} samples affected ({affected_percent:.2f}%), max output: {max_output_db:.1f} dB")
            else:
                logger.debug("🌊 TanH limiting: no limiting needed")
                
            return limited_audio
            
        except Exception as e:
            logger.error(f"TanH limiting failed: {e}")
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
                limited_gain_db = self._calculate_peak_limited_gain(
                    audio, target_gain_db, self.peak_limit
                )
                
                analysis["normalization_preview"] = {
                    "method": self.method,
                    "target_gain": target_gain_db,
                    "limited_gain": limited_gain_db,
                    "peak_limiting_active": abs(limited_gain_db - target_gain_db) > 0.1,
                    "would_normalize": abs(limited_gain_db) > 0.1
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