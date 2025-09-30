# Chatterbox TTS Tester

A lightweight standalone GUI tool for quickly testing and tuning Chatterbox TTS parameters.

## Purpose

This tool allows you to rapidly experiment with different:
- Reference audio files
- TTS models (classic vs multilanguage)
- Languages (for multilanguage model)
- TTS parameters (exaggeration, cfg_weight, temperature, etc.)

Perfect for finding optimal voice settings before integrating them into your cbpipe configurations.

## Features

- **Reference Audio Selection**: Choose from available WAV files in `data/input/reference_audio/`
- **Model Switching**: Toggle between classic and multilanguage models
- **Language Selection**: Pick target language when using multilanguage model (24 languages: Arabic, Danish, German, Dutch, English, Spanish, Finnish, French, Greek, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Norwegian, Polish, Portuguese, Russian, Swahili, Swedish, Turkish, Chinese)
- **Seed Control**: Set random seed for reproducibility (with 🎲 Random button)
- **Text Input**: Enter up to 500 characters of text to synthesize
- **Parameter Sliders**: Adjust TTS parameters:
  - exaggeration (0.0 - 2.0)
  - cfg_weight (0.0 - 2.0)
  - temperature (0.0 - 2.0)
  - repetition_penalty (1.0 - 3.0)
  - min_p (0.01 - 0.5)
  - top_p (0.5 - 1.0)
- **Manual Generation**: REFRESH button triggers audio generation and playback
- **Audio Playback**: Play, pause, and scrub through generated audio
- **Save Audio**: Save generated audio to `output/` directory with timestamp and metadata YAML
- **Settings Persistence**: All settings (voice, model, language, text, parameters) are automatically saved
- **Copy Settings**: Export current parameters as YAML for cbpipe configs (clipboard)

## Performance

This tool uses **Chatterbox TTS v0.1.3** (stable release):
- **Reliable performance**: ~40-50 it/s generation speed
- **No CUDA graph crashes**: Stable with varying parameters
- **All parameters work correctly**: exaggeration, cfg_weight, temperature, etc.

**Note**: Experimental faster/faster-multi branches offer ~3x speedup but have CUDA graph stability issues when parameters change during testing. For a testing tool with frequent parameter adjustments, stability is more important than raw speed.

## Usage

### Starting the Tool

```bash
# Install required system package (one-time)
sudo apt-get install python3-tk

# Optional: Install ffplay for better audio playback quality in WSL
sudo apt-get install ffmpeg

# Run the tester
source venv/bin/activate
python chatterbox_tester.py
```

**Note on Audio Playback:**
- The tool automatically uses `ffplay` or `aplay` if available (better quality)
- Falls back to pygame mixer if external players not found
- If playback has crackling/distortion, use the "Save Audio" button - saved WAV files are clean

### Workflow

1. **Select Reference Audio**: Choose your target voice from the dropdown
2. **Choose Model**: Select classic or multilanguage
3. **Select Language**: (multilanguage only) Pick target language
4. **Set Seed**: Enter a seed value for reproducibility (or use 🎲 Random button)
5. **Enter Text**: Type or paste text to synthesize (max 500 chars)
6. **Adjust Parameters**: Move sliders to tune the voice
7. **Click REFRESH**: Press the green REFRESH button to generate audio
8. **Listen**: Generated audio plays automatically
9. **Fine-tune**: Adjust parameters/seed and click REFRESH again to hear changes
10. **Copy Settings**: When satisfied, click "copy parameter JSON to clipboard" to export YAML

### Playback Controls

- **Timeline**: Click anywhere to jump to that position
- **Play (▶)**: Start/resume playback (restarts from beginning if at end)
- **Pause (⏸)**: Pause playback at current position

### Settings Persistence

**Automatic Saving**: All settings are automatically saved when:
- You click the REFRESH button
- You close the application

Settings are stored in `.chatterbox_tester_settings.json` and restored on next launch.

**Saved Settings Include**:
- Reference audio selection
- Model type (classic/multilanguage)
- Language selection
- Seed value
- Input text
- All TTS parameters

### Exporting Settings

The "📋 Copy YAML to Clipboard" button copies settings in this format:

```yaml
tts_params:
        exaggeration: 0.70
        exaggeration_max_deviation: 0.0
        cfg_weight: 0.45
        cfg_weight_max_deviation: 0.0
        temperature: 0.80
        temperature_max_deviation: 0.0
        repetition_penalty: 2.20
        min_p: 0.05
        top_p: 0.95
```

Simply paste this into your cbpipe YAML configuration.

### Audio Metadata

When you save audio using the "💾 Save Audio" button, two files are created:

1. **WAV file**: `test_[reference]_[timestamp].wav`
2. **YAML metadata**: `test_[reference]_[timestamp].yaml`

Example YAML metadata (ready for cbpipe):
```yaml
# Chatterbox TTS Test - 20250130_143022
# Reference Audio: stephan_moebius_1.wav
# Model: multilanguage
# Language: Deutsch (de)
# Seed: 12345

# Text:
# Your input text here...

tts_params:
        exaggeration: 0.70
        exaggeration_max_deviation: 0.0
        cfg_weight: 0.45
        cfg_weight_max_deviation: 0.0
        temperature: 0.80
        temperature_max_deviation: 0.0
        repetition_penalty: 2.20
        min_p: 0.05
        top_p: 0.95
```

This YAML metadata can be directly copied into your cbpipe configuration. It includes all settings needed to reproduce the exact generation.

## Dependencies

The tool requires minimal additional dependencies beyond the main pipeline:
- `pygame>=2.6.0` - for audio playback (already included)
- `tkinter` - GUI framework (system package: `python3-tk`)

Optional for better audio quality in WSL:
- `ffmpeg` - provides ffplay for high-quality audio playback

All other dependencies (torch, chatterbox-tts, etc.) are already part of the project.

## Technical Details

- **Independent**: Runs standalone, not integrated with cbpipe
- **Lightweight**: Minimal UI using tkinter (Python stdlib)
- **Direct API**: Uses Chatterbox models directly via `ChatterboxModelCache`
- **No Logging**: Suppresses Chatterbox log output for clean UI
- **No Persistence**: Settings are not saved (use copy-to-clipboard feature)

## Limitations

- Maximum text length: 500 characters
- No file saving (use clipboard export)
- No generation history
- Single audio output at a time (no candidate comparison)
