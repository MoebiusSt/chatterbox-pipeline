# Chatterbox TTS Tester

A lightweight standalone GUI tool for quickly testing and tuning TTS parameters across all five supported models.

## Purpose

Rapidly experiment with different:
- Reference audio files
- TTS models (classic / multilanguage / turbo / qwen3 / vibevoice)
- Languages (multilanguage: 24 languages; qwen3: 10 languages; vibevoice: auto)
- TTS parameters — with model-specific sliders shown/hidden automatically

Perfect for finding optimal voice settings before integrating them into cbpipe configurations.

## Features

- **Reference Audio Selection**: Choose from available WAV files in `data/input/reference_audio/`
- **Five TTS Models**: classic, multilanguage, turbo, qwen3, vibevoice — selected via dropdown
- **Adaptive Sliders**: Parameters automatically shown or hidden based on the active model
- **Language Selection**: Available for multilanguage (24 languages) and qwen3 (10 languages)
- **ref_text Indicator** (qwen3): Shows whether a `.txt` sidecar transcript exists next to the reference WAV
- **Seed Control**: Set random seed for reproducibility (with 🎲 Random button)
- **Text Input**: Enter up to 500 characters (classic/multilanguage/turbo/qwen3) or 10,000 characters (vibevoice)
- **VibeVoice Sampling Toggle**: Enable/disable sampling mode (`use_sampling`) for vibevoice
- **Manual Generation**: REFRESH button triggers audio generation and playback
- **Audio Playback**: Play, pause, and scrub through generated audio
- **A/B Comparison**: Two independent parameter states (1 / 2) with copy arrows
- **Undo/Redo**: Full history of parameter changes (Ctrl+Z / Ctrl+Y)
- **Presets**: Save and load per-audio, per-model parameter presets (YAML file)
- **Copy YAML**: Export current parameters as cbpipe-ready YAML to clipboard
- **Insert from Clipboard**: Paste a cbpipe YAML block back to restore all sliders
- **Save Audio**: Save generated audio to `output/` with timestamp and metadata YAML
- **Settings Persistence**: All settings automatically saved/restored on restart

## Model Comparison

| Feature                 | classic | multilanguage | turbo   | qwen3        | vibevoice |
|-------------------------|---------|---------------|---------|--------------|-----------|
| Language                | EN only | 24 languages  | EN only | 10 languages | auto      |
| exaggeration            | ✓       | ✓             | ✓       | —            | —         |
| cfg_weight              | ✓       | ✓             | ✓       | —            | —         |
| cfg_scale               | —       | —             | —       | —            | ✓         |
| temperature             | ✓       | ✓             | ✓       | ✓            | ✓         |
| repetition_penalty      | ✓       | ✓             | ✓       | ✓            | —         |
| min_p                   | ✓       | ✓             | ✓       | —            | —         |
| top_p                   | ✓       | ✓             | ✓       | ✓            | ✓         |
| top_k                   | —       | —             | 0–2000  | 0–300        | —         |
| diffusion_steps         | —       | —             | —       | —            | ✓         |
| voice_speed_factor      | —       | —             | —       | —            | ✓         |
| ref_text sidecar        | —       | —             | —       | ✓            | —         |

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

### Workflow

1. **Select Reference Audio**: Choose your target voice from the dropdown
2. **Choose Model**: Select from the model dropdown (classic / multilanguage / turbo / qwen3 / vibevoice)
3. **Select Language**: (multilanguage / qwen3 only) Pick target language
4. **Set Seed**: Enter a seed value for reproducibility (or use 🎲 Random)
5. **Enter Text**: Type or paste text to synthesize (max 500 chars; vibevoice max 10,000)
6. **Adjust Parameters**: Move the visible sliders — irrelevant sliders are hidden per model
7. **Click REFRESH**: Generate audio (button turns bright green when parameters changed)
8. **Listen**: Generated audio plays automatically
9. **Fine-tune & Compare**: Use states 1/2 for A/B comparison
10. **Copy Settings**: Click "📋 Copy YAML to Clipboard" to export for cbpipe

### Qwen3 Voice Cloning

Qwen3 requires a reference audio clip for voice cloning. For best quality (ICL mode), place a `.txt` transcript next to your reference `.wav`:

```
data/input/reference_audio/my_speaker.wav
data/input/reference_audio/my_speaker.txt   ← transcript of the WAV content
```

The tester shows a status indicator below the reference audio dropdown:
- **✓ ref_text: my_speaker.txt** — ICL mode active (full voice cloning)
- **⚠ no .txt sidecar – x_vector only** — reduced quality, but still works

If no transcript is found, the model falls back to `x_vector_only_mode=True` (speaker embedding only).

### VibeVoice Voice Cloning

VibeVoice uses the selected reference audio directly and does not require a `.txt` sidecar transcript.

- Recommended reference length: **20-60 seconds**
- Supported tester controls: `cfg_scale`, `temperature`, `top_p`, `diffusion_steps`, `voice_speed_factor`, `use_sampling`
- Default mode is deterministic (`use_sampling=false`)

### Parameter Defaults by Model

| Parameter          | classic / multilanguage / turbo | qwen3  | vibevoice |
|--------------------|---------------------------------|--------|-----------|
| exaggeration       | 0.70                            | —      | —         |
| cfg_weight         | 0.45                            | —      | —         |
| cfg_scale          | —                               | —      | 1.30      |
| temperature        | 0.80                            | 0.80*  | 0.80      |
| repetition_penalty | 2.20                            | 2.20*  | —         |
| min_p              | 0.05                            | —      | —         |
| top_p              | 0.98                            | 0.98*  | 0.98      |
| top_k              | — / — / 1000                    | 50     | —         |
| diffusion_steps    | —                               | —      | 20        |
| voice_speed_factor | —                               | —      | 1.00      |

*Qwen3 native defaults differ (temperature 0.9, repetition_penalty 1.05, top_p 1.0, top_k 50). The sliders start at the Chatterbox defaults on first run; adjust manually to match Qwen3's typical operating range.

### Presets

Presets are stored in `.chatterbox_tester_presets.yaml`. They are keyed per reference audio and model:

- classic / multilanguage: key = `filename.wav` (backward compatible with old presets)
- turbo: key = `filename.wav::turbo`
- qwen3: key = `filename.wav::qwen3`
- vibevoice: key = `filename.wav::vibevoice`

This means you can have separate presets for the same voice file across different models.

### Exporting Settings

The "📋 Copy YAML to Clipboard" button copies model-aware settings:

**classic / multilanguage:**
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

**turbo** (adds `top_k`):
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
        top_k: 1000
```

**qwen3** (no exaggeration / cfg_weight / min_p):
```yaml
tts_params:
        temperature: 0.90
        temperature_max_deviation: 0.0
        repetition_penalty: 1.05
        top_k: 50
        top_p: 1.00
        subtalker_temperature: 0.90
        subtalker_top_k: 50
        subtalker_top_p: 1.00
```

**vibevoice**:
```yaml
tts_params:
        cfg_scale: 1.30
        temperature: 0.95
        top_p: 0.95
        diffusion_steps: 20
        voice_speed_factor: 1.00
        use_sampling: false
```

Paste directly into any cbpipe speaker definition.

### Audio Metadata

When you save audio with "💾 Save Audio", two files are created:

1. **WAV file**: `output/test_[reference]_[timestamp].wav`
2. **YAML metadata**: `output/test_[reference]_[timestamp].yaml` — includes model, language/ref_text (if applicable), seed, and full `tts_params` block

## Audio Playback

- Uses `ffplay` (from ffmpeg) if available — best quality with seeking support
- Falls back to `aplay` if ffplay is not found
- If playback has issues, use "💾 Save Audio" and open the WAV directly

## Dependencies

- `tkinter` — GUI framework (system package: `sudo apt-get install python3-tk`)
- `pygame>=2.6.0` — bundled with project
- All TTS dependencies already part of the project (`chatterbox-tts`, `qwen-tts`, etc.)

Optional for better audio quality in WSL:
- `ffmpeg` — provides `ffplay` for high-quality audio playback with seeking

## Technical Details

- Uses `ChatterboxModelCache` directly for all five model types
- Qwen3 calls `generate_voice_clone()` directly (no prompt caching in the tester — acceptable latency for single generations)
- VibeVoice uses locally vendored inference code from `src/third_party/vibevoice` (no `trust_remote_code`)
- Model-specific slider visibility controlled via tkinter `grid` / `grid_remove()`
- Settings stored in `.chatterbox_tester_settings.json` (JSON)
- Presets stored in `.chatterbox_tester_presets.yaml` (YAML)
