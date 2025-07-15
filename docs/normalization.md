# Audio Normalization Documentation

## Overview

This normalization system provides simple, direct audio level adjustment to achieve consistent loudness across all audio files.

## Configuration

```yaml
audio:
  normalization:
    enabled: true            # Enable/disable audio normalization
    target_level: -16.0      # Target loudness level in dB (unit depends on method)
    method: "lufs"           # Normalization method: "lufs", "rms", or "peak"
    saturation_factor: 0.0   # TanH saturation intensity: 0.0=none, 1.0=strong (with loudness compensation)
```

## Parameters

- **`enabled`**: Enable/disable normalization
- **`target_level`**: Target loudness level in dB (unit depends on method)
- **`method`**: Normalization method
  - `"lufs"`: Perceptual loudness (EBU R128 standard) - **Recommended**
  - `"rms"`: Average energy level
  - `"peak"`: Maximum peak level
- **`saturation_factor`**: TanH saturation intensity (0.0 to 1.0)
  - `0.0`: No saturation (transparent)
  - `0.3`: Light saturation (subtle harmonic enhancement)
  - `0.6`: Medium saturation (noticeable warmth)
  - `1.0`: Strong saturation (aggressive compression)

## How It Works

The normalization system works in two stages:

### Stage 1: Loudness Normalization
1. **Analyze current audio level** using the selected method
2. **Calculate required gain** to reach target_level
3. **Apply gain directly** to the audio

### Stage 2: TanH Saturation (Optional)
1. **Measure loudness** of the normalized audio
2. **Apply TanH saturation** based on saturation_factor
3. **Apply loudness compensation** to maintain target_level

**Key Benefits:**
- **Loudness preserved**: Target level is maintained regardless of saturation
- **Musical saturation**: TanH provides smooth, harmonic distortion
- **Configurable intensity**: From transparent (0.0) to aggressive (1.0)
- **No harsh clipping**: Soft limiting with natural compression

### Methods Explained

#### LUFS (Recommended)
- Measures perceptual loudness using EBU R128 standard
- Best for consistent listening experience
- Recommended for speech and music content

#### RMS
- Measures average energy level
- Good for consistent volume perception
- Useful for technical applications

#### Peak
- Measures maximum peak level
- Useful for preventing clipping
- Simple but less perceptually accurate

## Example Configurations

### Speech Content (Conservative)
```yaml
audio:
  normalization:
    enabled: true
    target_level: -20.0      # Conservative loudness
    method: "lufs"
    saturation_factor: 0.0   # No saturation for natural speech
```

### Music Content (Broadcast Safe)
```yaml
audio:
  normalization:
    enabled: true
    target_level: -16.0      # Broadcast standard
    method: "lufs"
    saturation_factor: 0.3   # Light saturation for warmth
```

### Maximum Loudness with Saturation
```yaml
audio:
  normalization:
    enabled: true
    target_level: -12.0      # High loudness
    method: "lufs"
    saturation_factor: 0.6   # Medium saturation for presence
```

### Aggressive Mastering
```yaml
audio:
  normalization:
    enabled: true
    target_level: -12.0      # High loudness
    method: "rms"
    saturation_factor: 1.0   # Strong saturation for maximum impact
```

## Important Notes

- **Two-stage processing**: Normalization first, then optional saturation
- **Loudness preservation**: Target level is maintained even with saturation
- **No harsh clipping**: TanH provides smooth, musical saturation
- **Configurable intensity**: saturation_factor from 0.0 (off) to 1.0 (strong)
- **Performance**: Fast processing with optional saturation stage

## Troubleshooting

### Audio sounds distorted after normalization
- Check if saturation_factor is too high - try reducing it
- Your target_level may be too high for the input audio
- Try a lower target_level (e.g., -20.0 instead of -12.0)
- Consider the dynamic range of your input audio

### Inconsistent loudness between files
- Make sure all files use the same normalization method
- LUFS method provides the most consistent perceptual loudness
- Check that normalization is enabled for all files
- Ensure saturation_factor is consistent across files

### No volume change after normalization
- Check that `enabled: true` in your configuration
- Verify that the input audio level is different from target_level
- Check logs for normalization gain values

### Saturation not working as expected
- Make sure saturation_factor > 0.0 for saturation to be applied
- Check logs for "🌊 TanH saturation" messages
- Remember: loudness is preserved, but dynamics are reduced
- Try different saturation_factor values (0.3, 0.6, 1.0)
