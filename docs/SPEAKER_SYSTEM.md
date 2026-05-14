# 🎭 Multi-Speaker System Documentation

The TTS pipeline supports a complete multi-speaker system with dynamic speaker switching within documents.

## 📋 Overview

### Core Features
- **🎯 Dynamic speaker switching** within a single document
- **🎭 Speaker markup syntax** for simple text annotation
- **🔧 Easy configuration** with speaker list in default_config.yaml

### Architecture
- **Speaker-specific `reference_audio` files**
- **Markup parser for `<speaker:id>` tags**
- **Speaker-aware chunk creation**


---

## 🎪 Speaker Markup Syntax

### Basic Syntax
```text
Default speaker text without markup.

<speaker:narrator>
From here on, the narrator speaks.
This text is also from the narrator.

<speaker:character>
"Hello!", said the character.

<speaker:default>
Back to the default speaker.
```

### Supported Tags
- `<speaker:id>` - Switches to speaker with corresponding ID
- `<speaker:default>` - Back to default speaker (configured via default_speaker key)
- `<speaker:0>` - Alternative for default speaker
- `<speaker:reset>` - Another alternative for default speaker

### Rules
- ✅ **Only start tags required** - no end tags needed
- ✅ **Speaker changes have highest chunking priority**
- ✅ **Redundant IDs** → with `chunking.collapse_redundant_speaker_tags: true`
  (default), `<speaker:id>` is treated as a no-op when it resolves to the
  speaker that is already active; no extra hard chunk break and no artificial
  speaker-transition pause are inserted. Set the flag to `false` to preserve
  legacy behaviour where every speaker tag forces a boundary.
- ✅ **Unknown IDs** → with `chunking.strict_speaker_validation: true` (default)
  the preprocessing stage aborts with a verbose error listing every offending
  `<speaker:id>` and its line number; set the flag to `false` to fall back
  silently to the default speaker (legacy behaviour).
- ✅ **Syntax errors** → Ignore + warning
- ✅ **Default speaker** → Explicitly configured via `default_speaker` key
- ✅ **Aliases** `<speaker:0>`, `<speaker:default>`, `<speaker:reset>` are always
  accepted and resolve to the configured `default_speaker`.

---

## ⚙️ Configuration

### New Speaker Structure
```yaml
generation:
  num_candidates: 3
  max_retries: 2
  default_speaker: david                    # Explicit default speaker configuration
  # No global default_language - language is defined per speaker
  
  # Speaker definitions - Define all your speakers here
  speakers:
    - id: david                          # Speaker referenced by default_speaker
      reference_audio: david_barnes_1.wav
      language: en                    # Required for default speaker: Language ID (e.g., en, de, fr); others inherit
      tts_params:
        exaggeration: 0.55
        exaggeration_max_deviation: 0.20
        cfg_weight: 0.2
        cfg_weight_max_deviation: 0.40
        temperature: 0.9
        temperature_max_deviation: 0.3
        repetition_penalty: 1.3
      conservative_candidate:
        enabled: true
        exaggeration: 0.4
        cfg_weight: 0.5
        temperature: 0.7
    
    - id: narrator                       # Additional speaker
      reference_audio: cori_samuel_1.wav
      language: en                    # Optional: Inherits from default speaker if omitted
      tts_params:
        exaggeration: 0.65
        exaggeration_max_deviation: 0.25
        cfg_weight: 0.3
        cfg_weight_max_deviation: 0.35
        temperature: 1.0
        temperature_max_deviation: 0.3
        repetition_penalty: 1.2
      conservative_candidate:
        enabled: true
        exaggeration: 0.45
        cfg_weight: 0.5
        temperature: 0.75
    
    - id: character                      # Additional speaker
      reference_audio: mike_kamp_1.wav
      language: en                    # Optional: Can differ for multilingual; inherits if omitted
      tts_params:
        exaggeration: 0.45
        exaggeration_max_deviation: 0.20
        cfg_weight: 0.4
        cfg_weight_max_deviation: 0.30
        temperature: 0.85
        temperature_max_deviation: 0.25
        repetition_penalty: 1.3
      conservative_candidate:
        enabled: true
        exaggeration: 0.35
        cfg_weight: 0.6
        temperature: 0.8
```

### Simple Configuration
The speaker system uses a clear and intuitive structure:

```yaml
generation:
  default_speaker: david                  # Explicit default speaker – must exist in speakers-list or default_config.yaml
  # Language is configured per speaker (required for default_speaker, optional for others with inheritance)
  speakers:
    - id: david                         
      reference_audio: voice.wav
      language: en                    # Required for default speaker
      tts_params: {...}
      conservative_candidate: {...}
    - id: narrator                      # Additional speakers
      reference_audio: narrator.wav
      language: en                    # Optional: Inherits from default (can be de for multilingual)
      tts_params: {...}
```

### Important Notes
- **Default Speaker**: Explicitly configured via `default_speaker` key in generation config
- **Speaker Aliases**: `<speaker:0>`, `<speaker:default>`, `<speaker:reset>` all refer to the configured default speaker
- **Fallback Behavior**: Unknown speaker IDs automatically fallback to the CONFIGURED default_speaker
- **Validation**: The `default_speaker` value must match an existing speaker ID in the speakers list
- **Note**: Text without markup uses the configured default speaker
- **Languages**: Arabic (ar) • Danish (da) • German (de) • Greek (el) • English (en) • Spanish (es) • Finnish (fi) • French (fr) • Hebrew (he) • Hindi (hi) • Italian (it) • Japanese (ja) • Korean (ko) • Malay (ms) • Dutch (nl) • Norwegian (no) • Polish (pl) • Portuguese (pt) • Russian (ru) • Swedish (sv) • Swahili (sw) • Turkish (tr) • Chinese (zh)
**Language Requirement**: The default_speaker must define a `language` field (e.g., en, de). Other speakers can omit it and inherit via cascading from the default. For multilingual mode, different speakers can have different languages.

---

## 🔧 API Reference

### ConfigManager
```python
# Get speaker configuration
speaker_config = config_manager.get_speaker_config(config, "narrator")

# Available speaker IDs
speaker_ids = config_manager.get_available_speaker_ids(config)

# Get default speaker ID
default_speaker = config_manager.get_default_speaker_id(config)

# Speaker validation
is_valid = config_manager.validate_speakers_config(config)
```

### FileManager
```python
# Speaker-specific reference_audio
audio_path = file_manager.get_reference_audio_for_speaker("narrator")

# All speaker IDs
speaker_ids = file_manager.get_all_speaker_ids()

# Speaker validation
validation_results = file_manager.validate_speakers_reference_audio()

# Default speaker
default_id = file_manager.get_default_speaker_id()
```

### TTSGenerator
```python
# Speaker switching
tts_generator.switch_speaker("narrator", config_manager)

# Speaker-specific generation
candidates = tts_generator.generate_candidates_with_speaker(
    text="Hello world",
    speaker_id="narrator",
    num_candidates=3,
    config_manager=file_manager
)
```

### SpaCyChunker
```python
# Enable speaker support
chunker.set_available_speakers(["david", "narrator", "character"])

# Chunking with speaker markup
chunks = chunker.chunk_text(text_with_markup)

# Each chunk now has:
# - chunk.speaker_id
# - chunk.speaker_transition
# - chunk.original_markup
```

---

## 🧪 Testing

### Unit Tests
```bash
# Run all speaker tests
python -m pytest tests/test_speaker_system.py -v

# Individual components
python -m pytest tests/test_speaker_markup_parser.py
python -m pytest tests/test_config_manager_speakers.py
python -m pytest tests/test_chunker_speakers.py
```

### Example Test
```python
def test_speaker_chunking():
    chunker = SpaCyChunker()
    # Set available speakers (default determined by config)
    chunker.set_available_speakers(["david", "narrator", "character"])
    
    text = "Default text. <speaker:narrator>Narrator here."
    chunks = chunker.chunk_text(text)
    
    assert len(chunks) == 2
    assert chunks[0].speaker_id == "david"        # Configured default speaker
    assert chunks[1].speaker_id == "narrator"
    assert chunks[1].speaker_transition == True
```

### Integration Tests
```bash
# End-to-end pipeline test
python scripts/test_speaker_pipeline.py

# Performance tests
python scripts/test_speaker_performance.py
```

---

## 🔍 Troubleshooting

### Common Issues

**Problem: "❌ SPEAKER MARKUP VALIDATION FAILED" (preprocessing aborts)**
```bash
Cause:    Source text contains <speaker:id> markup whose id is not in
          generation.speakers and is not one of the default aliases
          ("0", "default", "reset").
Solution: 1. Add the missing speaker to generation.speakers in the job YAML
             (id, reference_audio, language [, tts_params]).
          2. Or replace the unknown markup in the source text with an
             existing ID or a default alias.
          3. Or set chunking.strict_speaker_validation: false in the job
             YAML to silently fall back to the default speaker (legacy).
```

**Problem: "Reference audio not found"**
```bash
Solution: Make sure all .wav files exist in 
          data/input/reference_audio/
```

**Problem: "No speaker transitions detected"**
```bash
Solution: Check markup syntax: <speaker:id>
          Make sure there are no typos
```

### Debugging
```python
# Enable debug mode
import logging
logging.getLogger().setLevel(logging.DEBUG)

# Analyze speaker chunking
chunks = chunker.chunk_text(text)
for chunk in chunks:
    print(f"Chunk {chunk.idx}: Speaker {chunk.speaker_id}, "
          f"Transition: {chunk.speaker_transition}")
```

### Validation
```python
# Validate speaker configuration
config_manager = ConfigManager(Path.cwd())
config = config_manager.load_default_config()
is_valid = config_manager.validate_speakers_config(config)

# FileManager validation
file_manager = FileManager(task_config, config)
validation_results = file_manager.validate_speakers_reference_audio()
print(f"Speaker validation: {validation_results}")
```

---