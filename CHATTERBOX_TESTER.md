# Chatterbox TTS Tester

A lightweight standalone GUI tool for quickly testing and tuning TTS parameters across all built-in models plus optional **DramaBox** (tester-only).

## Purpose

Rapidly experiment with different:
- Reference audio files
- TTS models (classic / multilanguage / turbo / qwen3 / vibevoice / dramabox)
- Languages (multilanguage: 24 languages; qwen3: 10 languages; vibevoice: auto; dramabox: English)
- TTS parameters — with model-specific sliders shown/hidden automatically

Perfect for finding optimal voice settings before integrating them into cbpipe configurations.

## Features

- **Reference Audio Selection**: Choose from available WAV files in `data/input/reference_audio/`
- **Six model entries** (dropdown): classic, multilanguage, turbo, qwen3, three VibeVoice variants, and **dramabox** (loads from a separate [DramaBox](https://github.com/resemble-ai/DramaBox) clone when configured)
- **Adaptive Sliders**: Parameters automatically shown or hidden based on the active model
- **Language Selection**: Available for multilanguage (24 languages) and qwen3 (10 languages)
- **ref_text Indicator** (qwen3): Shows whether a `.txt` sidecar transcript exists next to the reference WAV
- **Seed Control**: Set random seed for reproducibility (with 🎲 Random button)
- **Text Input**: Up to 500 characters (classic / multilanguage / turbo); up to 10,000 for qwen3, all VibeVoice variants, and **dramabox** (scene-style prompts — see below)
- **VibeVoice Sampling Toggle**: Visible for VibeVoice models only (`use_sampling`)
- **DramaBox-only**: Reference-audio toggle, Perth watermark toggle, CFG auto-rescale toggle, sliders for `cfg_scale`, `stg_scale`, `duration_multiplier`, `gen_duration`, `ref_duration`, and manual `rescale_scale` when auto rescale is off
- **Manual Generation**: REFRESH button triggers audio generation and playback
- **Audio Playback**: Play, pause, and scrub through generated audio
- **A/B Comparison**: Two independent parameter states (1 / 2) with copy arrows
- **Undo/Redo**: Full history of parameter changes (Ctrl+Z / Ctrl+Y)
- **Presets**: Save and load per-audio, per-model parameter presets (YAML file)
- **Copy YAML**: Export current parameters as YAML to clipboard (cbpipe-ready for Chatterbox / Qwen3 / VibeVoice; DramaBox block is **tester-only** and not consumed by cbpipe)
- **Insert from Clipboard**: Paste a cbpipe YAML block back to restore all sliders
- **Save Audio**: Save generated audio to `output/` with timestamp and metadata YAML
- **Settings Persistence**: All settings automatically saved/restored on restart

## Model Comparison

| Feature                 | classic | multilanguage | turbo   | qwen3        | vibevoice | dramabox   |
|-------------------------|---------|---------------|---------|--------------|-----------|------------|
| Language                | EN only | 24 languages  | EN only | 10 languages | auto      | English    |
| exaggeration            | ✓       | ✓             | ✓       | —            | —         | —          |
| cfg_weight              | ✓       | ✓             | ✓       | —            | —         | —          |
| cfg_scale               | —       | —             | —       | —            | ✓ (VV)    | ✓ (DB)    |
| temperature             | ✓       | ✓             | ✓       | ✓            | ✓         | —          |
| repetition_penalty      | ✓       | ✓             | ✓       | ✓            | —         | —          |
| min_p                   | ✓       | ✓             | ✓       | —            | —         | —          |
| top_p                   | ✓       | ✓             | ✓       | ✓            | ✓         | —          |
| top_k                   | —       | —             | 0–2000  | 0–300        | —         | —          |
| diffusion_steps         | —       | —             | —       | —            | ✓         | —          |
| voice_speed_factor      | —       | —             | —       | —            | ✓         | —          |
| ref_text sidecar (qwen) | —       | —             | —       | ✓            | —         | —          |
| optional voice ref      | ✓       | ✓             | ✓       | ✓            | ✓         | ✓ optional |
| stg_scale / rescale …   | —       | —             | —       | —            | —         | ✓ (§ below)|

## DramaBox (tester-only)

DramaBox is Resemble’s expressive prompt-driven TTS on LTX-2.3 ([Hugging Face: ResembleAI/Dramabox](https://huggingface.co/ResembleAI/Dramabox), [GitHub](https://github.com/resemble-ai/DramaBox)). It is **not** wired into cbpipe — only into this tester.

**Requirements**

- NVIDIA GPU (~24 GB VRAM peak cited on the model card; CUDA required).

**Isolation (recommended — does not alter the cbpipe venv)**

Never install DramaBox `requirements.txt` into `chatterbox-pipeline/venv` if you want a stable chatterbox Torch stack.

1. Once: create DramaBox-local venv and install upstream deps:

   ```bash
   cd /path/to/chatterbox-pipeline
   ./scripts/dramabox_venv_install.sh /path/to/DramaBox
   ```

   This writes `DramaBox/.venv-drama/`.

2. Point the tester at that interpreter (**required** when using isolation):

   ```bash
   export CHATTERBOX_TESTER_DRAMABOX_PYTHON=/path/to/DramaBox/.venv-drama/bin/python
   ```

3. Run GUI from the **pipeline** venv plus the env var:

   ```bash
   cd /path/to/chatterbox-pipeline
   source venv/bin/activate
   ./scripts/run_chatterbox_tester_dramabox.sh
   ```

   The DramaBox subprocess keeps one warm `TTSServer` alive (stdin/stdout worker); chatterbox‑pipeline code stays on its own Torch/Gradio.

4. Repo path (`CHATTERBOX_TESTER_DRAMABOX_ROOT` / `dramabox_repo_root.txt`) stays as documented below.

If the pipeline venv was already polluted after a mixed install, recreate it cleanly from `requirements.txt` or run:

```bash
./scripts/cbpipe_venv_restore.sh
```

(Optionally reinstall CUDA‑matched Torch 2.6 wheels per [pytorch.org](https://pytorch.org/get-started/locally/) if pip still resolves to the wrong build.)

**Legacy (single interpreter — discouraged)**

Leaving `CHATTERBOX_TESTER_DRAMABOX_PYTHON` unset imports DramaBox inside the tester process (`Torch 2.8` upstream stack). Conflicts with `chatterbox-tts`.

**Checkout path**

- `export CHATTERBOX_TESTER_DRAMABOX_ROOT=/absolute/path/to/DramaBox` (alias `DRAMABOX_REPO`)

  Or put that path **on one line** in `dramabox_repo_root.txt` at the root of chatterbox‑pipeline.

**Upstream `requirements.txt`**

Adds `torchvision==0.23.0` next to Torch 2.8 so `torchvision`/Torch stay ABI‑matched — required for transformers image imports during Gemma encoding.

**Prompts**

The main text field is the DramaBox *prompt* (stage directions outside double quotes, spoken content inside `"quotes"`). See the upstream README on the GitHub repo for examples (laughs as `"Hahaha"`, etc.).

**VRAM**

Selecting DramaBox calls `ChatterboxModelCache.free_vram()` in the **pipeline process** before starting the DramaBox worker. Inference runs in an **optional subprocess** (`CHATTERBOX_TESTER_DRAMABOX_PYTHON`): that process holds GPU VRAM for DramaBox. Switching away terminates the subprocess and clears the chatterbox tester’s references.

## Usage

### Starting the Tool

```bash
# Install required system package (one-time)
sudo apt-get install python3-tk

# Optional: Install ffplay for better audio playback quality in WSL
sudo apt-get install ffmpeg

# Run tester (classic / multilingual / turbo / …) without DramaBox in-process:
source venv/bin/activate
python chatterbox_tester.py

# Recommended when using DramaBox: pipeline venv GUI + DramaBox‑only subprocess
./scripts/run_chatterbox_tester_dramabox.sh
```

### Workflow

1. **Select Reference Audio**: Choose your target voice from the dropdown
2. **Choose Model**: Select from the model dropdown (including `dramabox` once the DramaBox path is configured — env var or `dramabox_repo_root.txt`)
3. **Select Language**: (multilanguage / qwen3 only) Pick target language
4. **Set Seed**: Enter a seed value for reproducibility (or use 🎲 Random)
5. **Enter Text**: Type or paste input (max 500 chars for short-form models; 10,000 for qwen3, VibeVoice, and dramabox prompts)
6. **Adjust Parameters**: Move the visible sliders — irrelevant sliders are hidden per model
7. **Click REFRESH**: Generate audio (button turns bright green when parameters changed)
8. **Listen**: Generated audio plays automatically
9. **Fine-tune & Compare**: Use states 1/2 for A/B comparison
10. **Copy Settings**: Click "📋 Copy YAML to Clipboard"; for DramaBox the block documents `dramabox_params` for your notes only — not cbpipe.

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

### DramaBox reference and options

- Optional **reference audio** (≈10+ s recommended upstream): enable “use reference audio” and pick a WAV from `data/input/reference_audio/`, or disable cloning and rely on the prompt alone.
- **Watermark**: On by default (Resemble Perth); turn off only for local debugging.
- **Auto rescale**: Matches upstream `rescale_scale="auto"`; when off, the `db_rescale_scale` slider is shown (0–1).

### Parameter Defaults by Model

| Parameter          | classic / multilanguage / turbo | qwen3  | vibevoice | dramabox (UI keys) |
|--------------------|---------------------------------|--------|-----------|---------------------|
| exaggeration       | 0.70                            | —      | —         | —                   |
| cfg_weight         | 0.45                            | —      | —         | —                   |
| cfg_scale          | —                               | —      | 1.30      | `db_cfg_scale` → 2.5 |
| temperature        | 0.80                            | 0.80*  | 0.80      | —                   |
| repetition_penalty | 2.20                            | 2.20*  | —         | —                   |
| min_p              | 0.05                            | —      | —         | —                   |
| top_p              | 0.98                            | 0.98*  | 0.98      | —                   |
| top_k              | — / — / 1000                    | 50     | —         | —                   |
| diffusion_steps    | —                               | —      | 20        | —                   |
| voice_speed_factor | —                               | —      | 1.00      | —                   |
| stg_scale          | —                               | —      | —         | `db_stg_scale` → 1.5 |
| duration_multiplier | —                            | —      | —         | `db_duration_multiplier` → 1.1 |
| gen_duration (0=auto) | —                         | —      | —         | `db_gen_duration` → 0 |
| ref_duration       | —                               | —      | —         | `db_ref_duration` → 10 |
| rescale_scale      | —                               | —      | —         | auto or `db_rescale_scale` |

*Qwen3 native defaults differ (temperature 0.9, repetition_penalty 1.05, top_p 1.0, top_k 50). The sliders start at the Chatterbox defaults on first run; adjust manually to match Qwen3's typical operating range.

### Presets

Presets are stored in `.chatterbox_tester_presets.yaml`. They are keyed per reference audio and model:

- classic / multilanguage: key = `filename.wav` (backward compatible with old presets)
- turbo: key = `filename.wav::turbo`
- qwen3: key = `filename.wav::qwen3`
- vibevoice / `vibevoice_1_5b` / `vibevoice_q4`: key = `filename.wav::<exact dropdown name>`
- dramabox: key = `filename.wav::dramabox`

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

Paste directly into any cbpipe speaker definition (except DramaBox — use for notes / clipboard round-trip in the tester only).

**dramabox** (excerpt — full block includes booleans and `rescale_scale: auto` or a float):

```yaml
# DramaBox — tester-only (not used by cbpipe)
dramabox_params:
  cfg_scale: 2.500
  stg_scale: 1.500
  duration_multiplier: 1.100
  gen_duration: 0
  ref_duration: 10.000
  rescale_scale: auto
  use_reference_audio: true
  watermark: true
  auto_rescale: true
```

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
- **DramaBox**: separate venv (`DramaBox/.venv-drama`); see scripts above — **never** merges into chatterbox‑pipeline `/venv` unless you accept pin conflicts.

Optional for better audio quality in WSL:
- `ffmpeg` — provides `ffplay` for high-quality audio playback with seeking

## Technical Details

- Uses `ChatterboxModelCache` for Chatterbox / Qwen3 / VibeVoice (pipeline venv)
- DramaBox: optional worker subprocess driven by [`src/tools/dramabox_worker_main.py`](src/tools/dramabox_worker_main.py) when `CHATTERBOX_TESTER_DRAMABOX_PYTHON` points at DramaBox‑only venv [`src/tools/dramabox_tester_loader.py`](src/tools/dramabox_tester_loader.py)
- Qwen3 calls `generate_voice_clone()` directly (no prompt caching in the tester — acceptable latency for single generations)
- VibeVoice uses locally vendored inference code from `src/third_party/vibevoice` (no `trust_remote_code`)
- Model-specific slider visibility controlled via tkinter `grid` / `grid_remove()`
- Settings stored in `.chatterbox_tester_settings.json` (JSON)
- Presets stored in `.chatterbox_tester_presets.yaml` (YAML)
