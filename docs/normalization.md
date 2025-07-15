### Features:

- **Optimal Gain Calculation**: Calculates the best gain that achieves target level while respecting peak limit
- **Tensor-based Processing**: Works directly with torch tensors (no file I/O)
- **Multiple Normalization Methods**: LUFS, RMS, and peak-based


### Configuration:

```yaml
audio:
  normalization:
    enabled: false            # Enable/disable audio normalization
    target_level: -23.0      # Target loudness level in dB (unit depends on method)
    method: "lufs"           # Normalization method: "lufs", "rms", or "peak"
    peak_limit: -1.0         # Peak limit in dB to prevent clipping
    aggressive_mode: false   # Enable TanH limiting for soft saturating compression (apply gain first, then soft limit)
    tanh_scale_factor: 0.8   # TanH saturation curve: 0.6=subtle, 0.8=balanced, 1.2=strong
```

The normalization is applied automatically to the final assembled audio before saving, with integrated peak limiting that prevents clipping while achieving the target loudness level in a single processing step.

### Parameter Explanation:

- **`target_level`**: The target loudness level in dB. The unit depends on the method:
  - `method: "lufs"` → target_level is interpreted as LUFS
  - `method: "rms"` → target_level is interpreted as RMS level in dB
  - `method: "peak"` → target_level is interpreted as Peak level in dB
- **`method`**: Determines how to measure the current audio level
- **`peak_limit`**: Safety limit to prevent clipping (always in dB)
- **`aggressive_mode`**: Enable TanH limiting for maximum loudness
- **`tanh_scale_factor`**: Controls TanH saturation curve (only used when aggressive_mode=true):
  - `0.2` = Very subtle, transparent saturation
  - `0.5` = Gentle, musical saturation (recommended) 
  - `0.8` = Strong, noticeable saturation
  - `1.2` = Overly strong saturation (use with caution)

### Common Configuration Examples:

#### Audiobooks (Recommended)
```yaml
audio:
  normalization:
    enabled: true
    target_level: -18.0      # Professional audiobook standard
    method: "lufs"           # Best for perceived loudness
    peak_limit: -1.0         # Safe headroom for digital distribution
    aggressive_mode: false   # Conservative approach
```

#### Broadcast/TV Standard
```yaml
audio:
  normalization:
    enabled: true
    target_level: -23.0      # EBU R128 broadcast standard
    method: "lufs"           # Industry standard measurement
    peak_limit: -1.0         # Broadcast compliance
    aggressive_mode: false   # Conservative approach
```

#### Speech/Voice (RMS-based)
```yaml
audio:
  normalization:
    enabled: true
    target_level: -20.0      # Clear speech level
    method: "rms"            # Good for consistent speech energy
    peak_limit: -1.0         # Prevents clipping
    aggressive_mode: false   # Conservative approach
```

#### Conservative/Safe Settings
```yaml
audio:
  normalization:
    enabled: true
    target_level: -20.0      # Moderate level
    method: "lufs"           # Perceptual standard
    peak_limit: -3.0         # Extra headroom for safety
    aggressive_mode: false   # Conservative approach
```

#### High-Impact Content (Aggressive Mastering)
```yaml
audio:
  normalization:
    enabled: true
    target_level: -12.0      # Loud for impact
    method: "rms"            # Consistent energy
    peak_limit: -0.5         # Minimal headroom
    aggressive_mode: true    # Maximum loudness
    tanh_scale_factor: 0.5   # Strong but controlled saturation
```

#### Maximum Loudness (Streaming/Radio)
```yaml
audio:
  normalization:
    enabled: true
    target_level: -10.0      # Very loud
    method: "rms"            # Energy-based
    peak_limit: -0.1         # Aggressive headroom
    aggressive_mode: true    # Maximum loudness
    tanh_scale_factor: 0.9   # Strong saturation (use with caution)
```

#### Gentle Mastering (Musical Content)
```yaml
audio:
  normalization:
    enabled: true
    target_level: -18.0      # Moderate loudness
    method: "lufs"           # Perceived loudness
    peak_limit: -0.5         # Some headroom
    aggressive_mode: true    # Controlled limiting
    tanh_scale_factor: 0.2   # Gentle, musical saturation
```
