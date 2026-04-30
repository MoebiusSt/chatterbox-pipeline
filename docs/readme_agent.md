# AI Agent Reference: Chatterbox Pipeline

**Context**: Enhanced Text-to-Speech pipeline wrapper around resemble-ai/chatterbox  
**Purpose**: Production TTS with intelligent chunking, candidate generation, quality validation  
**Language**: Python 3.9+, PyTorch

## Project Architecture

### Core Pipeline Flow
```
Text File → Preprocessor → SpaCyChunker → TTSGenerator → WhisperValidator → QualityScorer → AudioProcessor → Final WAV
```

### Job/Task Management System
```
Job (Template) → Task (Execution Instance) → Output Directory
```
- **Jobs**: YAML configuration templates in `config/`
- **Tasks**: Runtime execution instances in `data/output/{job_name}/`
- **State Management**: File-based, resumable execution with gap analysis

> **See mermaid_diagram.md** for detailed visual workflow diagrams and **TECHNICAL_OVERVIEW.md** for comprehensive architecture documentation.

## Directory Structure

```
chatterbox-pipeline/
├── src/
│   ├── cbpipe.py                    # Main entry point
│   ├── chunking/                    # SpaCyChunker - intelligent text segmentation
│   ├── generation/                  # TTSGenerator, CandidateManager, AudioProcessor
│   ├── validation/                  # WhisperValidator, FuzzyMatcher, QualityScorer
│   ├── pipeline/                    # JobManager, TaskOrchestrator, TaskExecutor
│   ├── preprocessor/                # Text normalization
│   └── utils/                       # ConfigManager, FileManager, logging
├── config/
│   ├── default_config.yaml         # Base configuration template
│   └── *.yaml                      # Job-specific configurations
├── data/
│   ├── input/
│   │   ├── reference_audio/         # Speaker voice samples (.wav)
│   │   └── texts/                   # Input text files
│   └── output/{job_name}/           # Task execution results
└── scripts/                        # Testing and debugging tools
```

## Key Components

### 1. Text Processing Chain
```python
# Preprocessing
TextPreprocessor.normalize() → normalized text

# Chunking  
SpaCyChunker.chunk_text() → List[TextChunk]
# Chatterbox defaults:        target_chunk_limit=380, max_chunk_limit=460,
#                             force_paragraph_chunks=true
#                             → every \n\n is a HARD chunk boundary (paragraph-break silence guaranteed)
# Qwen3 defaults:             target_chunk_limit=1200, max_chunk_limit=2000,
#                             force_paragraph_chunks=false (longform, see VV semantics below)
# VibeVoice defaults:         target_chunk_limit=2800, max_chunk_limit=4800,
#                             force_paragraph_chunks=false
# Longform (Qwen3 / VibeVoice): HARD boundaries only at \n\n\n+ (2+ empty lines); last chunk
#   before such a break gets paragraph_break_type="long" → Assembly inserts
#   audio.silence_duration.long (default 1.30 s).
#   SOFT boundaries: single \n\n is a preferred but not mandatory split-point
#   (greedy packing still applies within a hard section).
#   Micro-chunks are merged into neighbors, never across hard breaks.
# Features: sentence-boundary aware, speaker-transition priority (HIGHEST),
#           paragraph-break awareness, micro-chunk merge

# Speaker Support
<speaker:id> markup → automatic speaker switching.
# Speaker splits always win, independent of force_paragraph_chunks.
# <speaker:0>, <speaker:default>, <speaker:reset> all return to the default speaker.

# Finalization (_finalize_chunks, src/chunking/spacy_chunker.py)
# _merge_micro_chunks: merge short chunks (≤ micro_chunk_max_chars = min_chunk_length) into neighbors.
# Hard-break stops: speaker_transition OR paragraph_break_type=="long".
# With force_paragraph_chunks=true (Chatterbox), plain paragraph breaks also stop the merge.
```

### 2. TTS Generation System
```python
# Core Generation
TTSGenerator.generate_candidates() → List[AudioCandidate]

# Supported model types (generation.model_type):
# - "standard"        ChatterboxTTS              EN-only, full speaker switching
# - "multilingual"    ChatterboxMultilingualTTS  23 languages, full speaker switching
# - "turbo"           ChatterboxTurboTTS         EN-only, full speaker switching, norm_loudness, paralinguistic tags
# - "qwen3"           Qwen3TTSModel 1.7B Base    10 languages, full speaker switching, voice clone, requires .txt sidecar
# - "vibevoice"       VibeVoice-Large Q8 (7B)    officially EN/ZH, longform, speaker switching via pipeline
# - "vibevoice_1_5b"  VibeVoice 1.5B             lighter VRAM variant of vibevoice, same tts_params
# - "vibevoice_q4"    VibeVoice 7B 4-bit         lowest-VRAM variant (DevParker repo, 4bit subfolder),
#                                                same tts_params; flash_attention_2 incompatible → use sdpa

# Speaker switching: works for ALL model types
# - Each speaker may have its own reference_audio + tts_params (prosody/kwargs)
# - Speaker transitions change voice, exaggeration, cfg_weight, temperature etc.
# - Language restrictions (standard/turbo are EN-only) are soft WARNINGs at startup only;
#   they do NOT block speaker switching

# Language validation (generation_handler.execute_generation)
# - Collects all speaker languages from config
# - Warns if any language code is unsupported for the active model_type
# - Soft check: warns but does not abort

# RAMP Strategy (N candidates per chunk)
# - Candidate 1: Exact config parameters (baseline)
# - Candidates 2-N: Linear interpolation from config to deviation limits
# - Last candidate: Conservative parameters (optional, for stability)
# - conservative_candidate + num_candidates == 2 → Ramping disabled (only 1 expressive slot; no cfg/temp/diffusion_steps ramp across expressives)

# Unified ramp (all model types, all ramped params in tts_generator.py):
#   candidate_value = base + (max_deviation * ramp_position),  ramp_position in [0, 1]
#   across expressive candidates (candidate 1 = ramp_position 0 → always *base*).
#   Positive *max_deviation* → ramp UP; negative → ramp DOWN.
#
# Parameter behavior (Chatterbox models: standard / multilingual / turbo)
#   exaggeration:  use negative *exaggeration_max_deviation* for a downward ramp (legacy used MAX+subtraction)
#   cfg_weight / temperature:  positive *max_deviation* ramps UP from the configured start value
#
# Parameter behavior (Qwen3)
#   temperature, top_k, subtalker_top_k:  positive *max_deviation* usually ramps UP
#   subtalker_temperature:  sign of *subtalker_temperature_max_deviation* sets direction (negative = DOWN)

# Parameter behavior (VibeVoice: vibevoice / vibevoice_1_5b / vibevoice_q4)
# Implemented in TTSGenerator.generate_vibevoice_candidates (src/generation/tts_generator.py).
cfg_scale:      same unified formula; base = first expressive value
                (higher = calmer/more stable; too low < ~1.2 risks artefacts)
temperature:    same; only applied to the LM when use_sampling=true; with use_sampling=false the
                LM runs greedy and temperature/top_p are ignored.
top_p:          nucleus sampling threshold; applied only when use_sampling=true (not ramped)
diffusion_steps: int; ramp via *diffusion_steps_max_deviation* (non-zero; same ramp_pos; clamp 5–60).
                 Default deviation 0 leaves steps at base for all expressives.
                 Last candidate may override via conservative_candidate.diffusion_steps.
voice_speed_factor: resamples the *reference* audio by this factor (1.00 = unchanged,
                    ~0.98..1.05 for subtle speed tweaks); does NOT time-stretch the output.
use_sampling:   false = deterministic greedy decoding, true = sampling with temperature/top_p.
                Deviation keys (cfg_scale_max_deviation, temperature_max_deviation,
                diffusion_steps_max_deviation) are consumed internally and never forwarded
                to the model.

# Parameter filtering (all models)
# - Unsupported params in tts_params are gracefully skipped with a one-time info log
# - SUPPORTED_TTS_PARAMS per model type defined in src/utils/language_registry.py
# - conservative_candidate unsupported keys also gracefully skipped
# - Internal-only keys (all *_max_deviation, enabled, type, seed) are never forwarded

# Turbo-specific
# - norm_loudness: bool in tts_params → enables built-in loudness normalization
#   When active, the external AudioNormalizer in assembly is automatically skipped
# - top_k: int (default 1000)
# - Paralinguistic tags [laugh], [cough], etc. are preserved through chunking and preprocessing

# Qwen3-specific
# - Voice clone requires reference audio + reference text transcript
# - Reference text auto-loaded from .txt sidecar file (same stem as reference .wav)
#   or via speaker config field: reference_text: "path/to/transcript.txt" or inline text
# - Reference audio duration warning if > 3s (recommended max for optimal cloning quality)
# - x_vector_only_mode=True used as fallback when no ref_text is available
# - Voice clone prompts are cached per speaker_id (no recomputation across chunks)
# - Seed: torch.manual_seed(seed) called before each generation (Qwen3 has no native seed param)
#
# VibeVoice-specific
# - Uses local vendored VibeVoice inference code under src/third_party/vibevoice (no trust_remote_code)
# - Voice clone uses full reference audio (recommended up to ~60s, no .txt sidecar required)
# - Supported tts_params: cfg_scale, temperature, top_p, diffusion_steps, voice_speed_factor, use_sampling
# - Deterministic by default via use_sampling: false (greedy LM); set true to enable temperature/top_p
# - Official language support: EN and ZH. Other languages work via reference audio but may
#   degrade; controlled by generation.vibevoice.language_strict (default false = warn only,
#   true = raise ValueError for unsupported codes).  See src/generation/tts_generator.py.
# - Reference audio is resampled to 24 kHz mono; voice_speed_factor is applied to this ref.
# - Per-candidate seed: base_seed(speaker) + i*1000 + hash(text)%10000
#   (torch.manual_seed only, VibeVoice has no native seed parameter)
# - Variant attention defaults (src/generation/model_cache.py):
#     vibevoice      → None (HF auto)
#     vibevoice_1_5b → sdpa
#     vibevoice_q4   → sdpa (FA2 incompatible with 4-bit quant)
# - When switching between variants in the same process the previous VibeVoice model
#   is evicted to avoid >15 GB VRAM pile-up.
```

### 3. Quality Validation Pipeline
```python
# Validation Chain
ASRBackend.transcribe() → transcription (Whisper or VibeVoice-ASR)
WhisperValidator.validate() → transcription + similarity_score
ProsodyScorer.score() → prosody subscores (flow, liveliness, mos, …)
QualityScorer.score_candidate() → QualityScore (combined metrics)

# ASR backend selection (validation.asr_backend)
#   auto          → Whisper for ALL model types incl. VibeVoice (default)
#   whisper       → force Whisper
#   vibevoice_asr → opt-in VibeVoice-ASR; falls back to Whisper on failure
# Rationale: VibeVoice-ASR (~16 GB weights) spills to CPU RAM on ≤16 GB GPUs
# after the TTS model is loaded, and the vendored long-form path has an
# off-by-one mask bug on ≥4-min clips. Whisper is faster and stable.
# See src/validation/asr/factory.py and docs/TESTRUN_FINDINGS.md.

# Robust Similarity
# - Lowercasing, unicode punctuation normalization, whitespace/hyphen unification,
#   German ß→ss tolerance
# - RapidFuzz token ratios if available, otherwise token Jaccard

# Numbers Normalization (non-EN only)
# - validation.numbers_normalization_mode: off|placeholder|digits|words
# - Similarity computed on normalized texts; length uses normalized original

# Dynamic Thresholding
# - Similarity threshold is adjusted by text length and punctuation density
#   (no new config flags)

# Prosody + MOS scoring (ProsodyScorer)
# - Weights: validation.prosody.weights.{semantic_alignment, flow, liveliness,
#            intelligibility, mos}
# - Targets: validation.prosody.targets.{wpm_min, wpm_max, liveliness_target}
# - Select:  validation.prosody.select.{alpha_quality, beta_prosody, gamma_mos}
# - MOS windowing for long audio (motivated by VibeVoice longform outputs):
#     validation.mos.segmentation.max_unsegmented_seconds (default 20)
#     validation.mos.segmentation.window_s / hop_s / aggregator (median|mean|min)
#   When duration > max_unsegmented_seconds AND the MOS provider has
#   score_segmented, windows are MOS-scored and aggregated – keeps short-
#   utterance models (NISQA/UTMOS) in-distribution on ~1-minute VibeVoice chunks.
# - MOS language gate: validation.mos.enabled_languages (default [de, en]).
#   Other languages skip MOS (but prosody/flow/liveliness still apply).

# Selection Logic (3-stage fallback)
# Stage 1: Best valid candidate (similarity > effective_threshold)
# Stage 2: Best invalid candidate (highest quality score)
# Stage 3: Emergency fallback (first candidate)
```

### Smart Tail-Trim (Developer Quick Reference)
```python
# Location
src/validation/preprocessors/tail_trimmer.py
src/pipeline/task_executor/stage_handlers/validation_handler.py  # integration
src/utils/file_manager/io_handlers/final_audio_io.py             # prefers *_trimmed.wav
src/utils/file_manager/io_handlers/whisper_io.py                 # persists tail_trim meta

# Config (default_config.yaml)
validation.preprocessing.tail_trim:
  enabled: true
  lookback_seconds: 6.0
  post_speech_silence_ms: 200
  fade_out_ms: 120
  prefer_whisperx: true
  whisperx_device: cpu
  persist_trimmed_candidate: true
  debug_save_removed_tail: true
  smart_match:
    enabled: true
    last_n_words: 2
    fuzzy_ratio: 0.85
    search_window_words: 30
    language_gate: ["en","de","fr","es","it"]

# Behavior
# 1) Smart-Trim: find last N input-words in WhisperX-aligned words near the end
#    - if match ends before final aligned word → content_cut at last matched word end
#    - if match aligns exactly at the last word → exact_final (no content cut)
# 2) Fallbacks: whisperx last word end → VAD → energy
# 3) Post-processing: add post_speech_silence_ms (capped by last voiced frame via VAD), apply fade_out_ms
# 4) Persistence: candidate_YY_trimmed.wav (if enabled), optional *_removed_tail.wav
# 5) Metrics: result["tail_trim"] persisted into whisper_metrics.json

# Notes
# - Smart-Trim shares post_speech_silence_ms and fade_out_ms with VAD/Energy trim
# - search_window_words focuses on the last K alignment tokens for efficiency/precision
# - language_gate skips Smart-Trim for unsupported languages

# Test Script
scripts/test_smart_tail_trim.py
```

### 4. Multi-Speaker System
```python
# Configuration
generation:
  default_speaker: david
  speakers:
    - id: david
      reference_audio: voice.wav
      language: en                  # Optional: Speaker-specific language
      tts_params: {...}

# Text Markup
"Default text. <speaker:narrator>Narrator speaks. <speaker:default>Back to default."

# Internal Processing
SpaCyChunker.set_available_speakers() → speaker-aware chunking
TTSGenerator.switch_speaker() → dynamic reference_audio + language switching
TTSGenerator.generate_candidates_with_speaker() → automatic language_id from speaker config
```

> **See SPEAKER_SYSTEM.md** for complete multi-speaker documentation including markup syntax, configuration examples, and API reference.

## Configuration System

### Cascading Configuration (N-Level Inheritance via `parent:`)
```yaml
default_config.yaml                    # Root base template (no parent)
    ↑
config/defaults/chatterbox.yaml        # Model-specific defaults (parent: default_config.yaml)
    ↑
job_config.yaml                        # Job-specific overrides (parent: defaults/chatterbox.yaml)
    ↑
task_config.yaml                       # Runtime snapshot (no parent resolution)
```

Any YAML file (except `default_config.yaml` itself and task snapshots) may declare an
optional `parent:` key at the top level, pointing to another config file relative to the
`config/` directory:

```yaml
parent: defaults/qwen3.yaml    # optional; path relative to config/

job:
  name: my-qwen3-job
  ...
```

When loading a job config, `ConfigManager` resolves the full parent chain recursively
until a file without a `parent:` key is reached, which then falls back to
`default_config.yaml`.  Circular references and chains longer than 10 hops are detected
at load time and raise a clear error.

Without a `parent:` field the behavior is identical to the previous 2-level system
(job → default_config.yaml).

**Config Cascade**: any job- or task.yaml should reference as parent one of the Pre-built model defaults:
**Pre-built model defaults** in `config/defaults/`:
- `defaults/chatterbox.yaml` — standard / multilingual Chatterbox models
- `defaults/turbo.yaml`      — ChatterboxTurboTTS
- `defaults/qwen3.yaml`      — Qwen3 TTS (1.7B)
- `defaults/vibevoice.yaml`  — VibeVoice (all three size variants; longform chunking,
                                language_strict=false)

### Minimum Viable Job YAML (per model type)

Each job only needs to override what differs from the chosen parent. All other
sections (`chunking`, `validation`, `audio`, …) are inherited via the parent
chain. Each speaker needs at minimum `id` and `reference_audio`; `language` and
`tts_params` inherit from the parent if omitted.

```yaml
# Chatterbox (standard / multilingual / turbo)
parent: defaults/chatterbox.yaml        # or defaults/turbo.yaml
job:
  name: my-job
input:
  text_file: some-text.txt              # relative to data/input/texts/
generation:
  default_speaker: alice
  speakers:
    - id: alice
      reference_audio: alice.wav        # relative to data/input/reference_audio/

# Qwen3
parent: defaults/qwen3.yaml
job: { name: my-qwen3-job }
input: { text_file: my.txt }
generation:
  default_speaker: alice
  speakers:
    - id: alice
      reference_audio: alice.wav        # needs .txt sidecar (or reference_text field)

# VibeVoice – minimum, inherits everything incl. longform chunking
parent: defaults/vibevoice.yaml
job: { name: my-vibevoice-job }
input: { text_file: long-text.txt }
generation:
  model_type: vibevoice                 # or vibevoice_1_5b / vibevoice_q4
  default_speaker: alice
  global_seed: 0                        # 0 = random seed per generation
  speakers:
    - id: alice
      reference_audio: alice.wav        # up to ~60s, 24 kHz is fine
      language: de
      tts_params:
        cfg_scale: 1.30                 # MIN – ramps UP
        cfg_scale_max_deviation: 0.20
        temperature: 0.95               # MIN – ramps UP (needs use_sampling: true)
        temperature_max_deviation: 0.40
        top_p: 0.98
        diffusion_steps: 20
        voice_speed_factor: 1.00
        use_sampling: true
```

For a speaker-switched VibeVoice job with inline foreign-language sections, add
a second speaker (e.g. `id: it`) with its own reference audio and use
`<speaker:it>…<speaker:default>` markers in the input text.

### Critical Parameters
```yaml
# Text Processing
chunking:
  # Chatterbox / Qwen3 – many small paragraphs with explicit break silence
  target_chunk_limit: 380        # Balance: quality vs context window
  max_chunk_limit: 460
  min_chunk_length: 80
  spacy_model: en_core_web_sm
  force_paragraph_chunks: true   # split at every blank line (Chatterbox longform is weaker)

  # VibeVoice – longform (values from defaults/vibevoice.yaml)
  # target_chunk_limit: 2800
  # max_chunk_limit: 4800
  # min_chunk_length: 120
  # force_paragraph_chunks: false  # only \n\n\n+ is a hard boundary (long-pause silence)

# Generation Quality
generation:
  num_candidates: 3              # More candidates = better quality, higher compute
  max_retries: 1
  model_type: "standard"         # standard|multilingual|turbo|qwen3|vibevoice|vibevoice_1_5b|vibevoice_q4
  default_speaker: alice
  global_seed: 12345             # 0 = random seed per generation

  # VibeVoice-specific subsection (ignored by other model types)
  vibevoice:
    language_strict: false       # true = raise ValueError on non-en/zh, false = warn once
  
# TTS Parameters (per speaker) – unsupported params for the active model are gracefully skipped
tts_params:
  # Chatterbox (standard/multilingual/turbo)
  exaggeration: 0.40             # MAX value for RAMP-DOWN
  exaggeration_max_deviation: 0.20
  cfg_weight: 0.20               # MIN value for RAMP-UP
  cfg_weight_max_deviation: 0.20
  temperature: 0.90              # MIN value for RAMP-UP (all models)
  temperature_max_deviation: 0.30
  min_p: 0.03                    # Stability (Chatterbox only)
  top_p: 0.99                    # Creativity (Chatterbox + Qwen3 + VibeVoice-sampling)
  # turbo extras:    top_k: 1000, norm_loudness: false
  # qwen3 extras:    top_k, top_k_max_deviation, repetition_penalty, max_new_tokens,
  #                  do_sample, subtalker_top_k, subtalker_top_k_max_deviation,
  #                  subtalker_temperature, subtalker_temperature_max_deviation,
  #                  subtalker_top_p, subtalker_dosample
  # vibevoice keys:  cfg_scale (ramp UP), cfg_scale_max_deviation,
  #                  temperature (ramp UP, needs use_sampling),
  #                  temperature_max_deviation, top_p, diffusion_steps (ramp UP),
  #                  diffusion_steps_max_deviation, voice_speed_factor, use_sampling

# Quality Gates
validation:
  similarity_threshold: 0.64     # Text similarity requirement (dynamic adjustment applied)
  min_quality_score: 0.72        # Combined quality threshold
  whisper_model: small           # base/small/medium/large
  numbers_normalization_mode: placeholder   # off|placeholder|digits|words (non-EN)
  asr_backend: auto              # auto|whisper|vibevoice_asr (auto = whisper everywhere)

  prosody:
    weights: { semantic_alignment: 0.0, flow: 0.30, liveliness: 0.30,
               intelligibility: 0.0, mos: 0.40 }
    targets: { wpm_min: 125, wpm_max: 145, liveliness_target: 0.44 }
    select:  { alpha_quality: 0.50, beta_prosody: 0.30, gamma_mos: 0.20 }
  mos:
    enabled_languages: [de, en]
    min_mos: 3.0
    segmentation:                 # windowed MOS for long audio (VibeVoice-friendly)
      max_unsegmented_seconds: 20.0
      window_s: 12.0
      hop_s: 10.0
      aggregator: median

# Audio Assembly
audio:
  silence_duration:
    normal: 0.40                 # Inter-sentence pause (no paragraph break)
    paragraph: 0.90              # After chunk with has_paragraph_break (single blank line)
    long: 1.30                   # After >= 2 blank lines (paragraph_break_type="long")
  sample_rate: 24000             # Fixed – matches Chatterbox / Qwen3 / VibeVoice output rate
```

## Command Line Interface

### Execution Patterns
```bash
# Basic Usage
python src/cbpipe.py                                    # Open interactive menu
python src/cbpipe.py create my_job.yaml                 # Create new task from job config
python src/cbpipe.py resume my_job.yaml                 # Fill gaps in latest task
python src/cbpipe.py resume --job "pattern*"            # Pattern matching jobs

# Execution Verbs
python src/cbpipe.py create my_job.yaml                 # Create new task
python src/cbpipe.py resume my_job.yaml                 # Resume latest task
python src/cbpipe.py reassemble my_job.yaml             # Regenerate final audio
python src/cbpipe.py resume my_job.yaml --all           # Process all tasks
python src/cbpipe.py rebuild my_job.yaml                # Rerender candidates and final audio

# Options
python src/cbpipe.py --verbose                         # Debug output
python src/cbpipe.py --device cuda                     # Force GPU
```
More command line argument details in `basic-usage_CLI-arguments.md`

### Task State Management
```
Task Lifecycle:
1. Job Config (YAML) → Create Task Config → Execute Task
2. State Detection → Gap Analysis → Resume/Complete
3. Missing Files → Automatic Recovery → Fill Gaps

State Files:
- task_config.yaml: Task configuration snapshot
- chunks.json: Text segments with speaker info
- candidates/: Generated audio files per chunk
- transcriptions.json: Whisper validation results  
- quality_scores.json: Candidate selection metrics
- final_audio.wav: Assembled output
```

## Performance Considerations

### Model Cache System
```python
# Cache Behavior (per process)
First Call: "🔄 Loading ChatterboxTTS model (cache miss)" → 8-12s loading
Same Process: "♻️ Using cached model (cache hit)" → instant

# Optimization Strategy
python src/cbpipe.py job1.yaml job2.yaml job3.yaml  # Process multiple jobs in one run
```

> **See PERSISTENT_CACHE.md** for detailed explanation of cache behavior and why cache miss on every run is normal.

### Memory Management
```python
# High Memory Operations
num_candidates: 5+               # More candidates = more GPU memory
whisper_model: large            # Larger model = more memory

# Optimization
num_candidates: 3               # Balance quality/performance
whisper_model: small           # Adequate for most use cases
```

## Development Guidelines

### Dependencies
```python
# Core Dependencies
torch>=2.0.0                    # TTS model backend
chatterbox-tts>=0.1.2           # TTS generation engine (includes multilingual support)
spacy>=3.7.0                    # Text processing
openai-whisper>=20231117        # Validation
fuzzywuzzy>=0.18.0             # Text similarity
librosa>=0.10.0                 # Audio processing
num2words>=0.5.13               # Digits → words conversion
word2num-de>=1.1                # German word → number conversion

# Development Tools  
black>=23.0.0                   # Code formatting
pytest>=7.0.0                   # Testing
mypy>=1.0.0                     # Type checking
```

> **See CONTRIBUTING.md** for complete development setup, code standards, and contribution workflow.

### Code Standards
```python
# Type Hints Required
def process_chunk(chunk: TextChunk, params: Dict[str, Any]) -> AudioCandidate:
    pass

# Logging Patterns
logger.info("📝 User-facing progress")
logger.debug("🔍 Debug information") 
log_info_file_only("💾 File operations")
logger.warning("⚠️ Fallback behavior")
logger.error("❌ Critical failures")

# Error Handling
try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    return fallback_result()
```

### Testing Architecture
```python
# Mock Components (CI/CD compatible)
MockTTSGenerator     # Sine waves instead of TTS
MockWhisperValidator # Predefined transcriptions
MockAudioProcessor   # Basic audio operations

# Test Commands
python scripts/test_mock_pipeline.py           # No heavy models
python scripts/run_chunker.py                  # Text processing only
pytest tests/ -v                               # Full test suite
```

> **See TECHNICAL_OVERVIEW.md** for detailed testing strategies and mock component architecture.

## Troubleshooting Patterns

### Common Issues
```python
# Memory Errors
Solution: Reduce num_candidates, use smaller whisper_model

# Validation Failures ("No valid candidates")  
Solution: Lower similarity_threshold, increase num_candidates, check reference_audio quality

# Audio Artifacts
Solution: Tune TTS parameters (lower temperature, higher min_p), generate more candidates

# Token Repetition Warnings (Multilingual Model)
"🚨 Detected Nx repetition of token XXXX"
"⚠️ forcing EOS token, token_repetition=True"
Solution: Increase repetition_penalty (1.8-2.0), lower temperature (0.75-0.8), 
         increase min_p (0.08-0.10), decrease top_p (0.88-0.92)

# Turbo/standard model: EN-only language warning
# "Language 'de' is not supported by model type 'turbo'"
# The speaker switch itself is NOT blocked – only language-specific TTS quality may suffer.
Solution: Use speakers with language: en for standard/turbo models, or switch to multilingual/qwen3.

# Qwen3: no reference text / low cloning quality
Solution: Add a .txt sidecar file with the transcript next to the reference .wav file.
         Without it, x_vector_only mode is used (weaker voice cloning).

# Qwen3: reference audio too long
Warning logged when reference audio > 3s. Trim to best 3-second segment for optimal quality.

# VibeVoice: too many tiny chunks ("Einleitung.", "Ein Text von …" as own chunks)
Cause: force_paragraph_chunks=true.
Solution: Inherit defaults/vibevoice.yaml (force_paragraph_chunks=false).
         With force_paragraph_chunks=false only \n\n\n+ (2+ empty lines) is a hard boundary;
         single blank lines are soft split-points. Speaker-transition splits remain active.

# VibeVoice: unsupported language warning / error
"VibeVoice officially supports only 'en' and 'zh'; got 'de'"
Solution: Keep generation.vibevoice.language_strict: false (default) to proceed with a
         one-off warning. VibeVoice still clones from the reference audio regardless of
         language. Set language_strict: true to abort early on non-en/zh speakers.

# VibeVoice: flat prosody / candidates too similar
Solution: Enable use_sampling: true, raise temperature_max_deviation (0.4–0.6) and/or
         cfg_scale_max_deviation (0.15–0.30); lower base cfg_scale (1.2–1.3) for more
         liveliness. Conservative candidate can keep a stable fallback.

# VibeVoice-ASR: OOM or off-by-one mask error on long clips
Expected on ≤16 GB GPUs; default asr_backend: auto already routes to Whisper. Only switch
back to vibevoice_asr for short clips on a GPU with ≥24 GB VRAM.

# MOS score suspiciously low on long VibeVoice chunks
Check validation.mos.segmentation.max_unsegmented_seconds – if a provider without
score_segmented is active, long audio is still scored as one shot and drifts out of
distribution. CombinedMOSProvider (default) supports windowing.

# Unsupported tts_params for model type
e.g. "Skipping unsupported tts_param 'exaggeration' for model type 'qwen3'"
Solution: This is expected and harmless. params are filtered per model type automatically.

# Cache Miss on Every Run
Expected: Normal behavior, model loads once per process

# Speaker Not Found
Solution: Check speaker IDs in config, verify reference_audio files exist
```

### Debug Commands
```bash
# Component Testing
python scripts/run_chunker.py                  # Test text segmentation
python scripts/test_speaker_pipeline.py        # Test multi-speaker system
python scripts/test_model_cache.py            # Test cache behavior

# Verbose Execution
python src/cbpipe.py --verbose                 # Detailed logging
export PYTHONPATH=src:$PYTHONPATH             # Module path for debugging
```

## Key Data Structures

```python
@dataclass
class TextChunk:
    text: str
    start_pos: int
    end_pos: int
    speaker_id: str
    speaker_transition: bool
    speaker_transition_context: Optional[str]   # "paragraph" | "normal" | "none"
    original_markup: Optional[str]              # raw speaker id from <speaker:…>
    has_paragraph_break: bool
    paragraph_break_type: Optional[str]         # "paragraph" (1 blank line) | "long" (>=2)
    language_id: Optional[str]                  # inherited from speaker config
    is_fallback_split: bool                     # True if sentence was split by secondary delimiter
    estimated_tokens: int
    idx: int

@dataclass  
class AudioCandidate:
    audio_tensor: torch.Tensor
    audio_path: Path
    chunk_idx: int
    candidate_idx: int
    generation_params: Dict                     # includes model-specific keys + seed + type
    # type ∈ {"EXPRESSIVE", "CONSERVATIVE"}; seed = per-candidate torch seed

@dataclass
class ValidationResult:
    transcription: str
    similarity_score: float
    is_valid: bool
    processing_time: float
    # tail_trim metrics are persisted to whisper/whisper_metrics.json

@dataclass
class QualityScore:
    overall_score: float
    similarity_score: float
    length_score: float
    transcription_score: float
    # ProsodyScore subscores: semantic_alignment, flow, liveliness, intelligibility, mos
```

---

**Entry Point**: `src/cbpipe.py`  
**Key Classes**: JobManager, TaskOrchestrator, SpaCyChunker, TTSGenerator, ASRBackend/WhisperValidator, ProsodyScorer, QualityScorer  
**Config Root**: `config/default_config.yaml` (+ `config/defaults/{chatterbox,turbo,qwen3,vibevoice}.yaml`)  
**Model-type registry**: `src/utils/language_registry.py` (SUPPORTED_TTS_PARAMS, MODEL_FEATURES, language gates)  
**Data Flow**: Text → Preprocess → Chunks → Candidates → Tail-Trim → ASR → Prosody/MOS → Selection → Assembly → Output