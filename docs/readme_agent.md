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
# Parameters: target_chunk_limit: 380, max_chunk_limit: 460
# Features: sentence-boundary aware, speaker-transition priority

# Speaker Support
<speaker:id> markup → automatic speaker switching
```

### 2. TTS Generation System
```python
# Core Generation
TTSGenerator.generate_candidates() → List[AudioCandidate]

# Supported model types (generation.model_type):
# - "standard"     ChatterboxTTS           EN-only, full speaker switching
# - "multilingual" ChatterboxMultilingualTTS  23 languages, full speaker switching
# - "turbo"        ChatterboxTurboTTS      EN-only, full speaker switching, norm_loudness, paralinguistic tags
# - "qwen3"        Qwen3TTSModel 1.7B Base 10 languages, full speaker switching, voice clone, requires .txt sidecar
# - "vibevoice"    VibeVoice-Large-Q8      EN/ZH, full speaker switching via pipeline, long-form capable

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
# - Candidate 1: Exact config parameters (baseline, with center-offset for RAMP-DOWN-OVER-CENTER params)
# - Candidates 2-N: Linear interpolation from config to deviation limits
# - Last candidate: Conservative parameters (optional, for stability)
# - conservative_candidate + num_candidates == 2 → Ramping disabled (only 1 expressive slot)

# Parameter behavior (Chatterbox models: standard / multilingual / turbo)
exaggeration: RAMP-DOWN from MAX (config) to MIN (config - max_deviation)
cfg_weight: RAMP-UP from MIN (config) to MAX (config + max_deviation)
temperature: RAMP-UP from MIN (config) to MAX (config + max_deviation)

# Parameter behavior (Qwen3)
temperature:           RAMP-UP from MIN (config) to MAX (config + temperature_max_deviation)
top_k:                 RAMP-UP from MIN (config) to MAX (config + top_k_max_deviation)
subtalker_top_k:       RAMP-UP from MIN (config) to MAX (config + subtalker_top_k_max_deviation)
subtalker_temperature: RAMP-DOWN-OVER-CENTER: config value is CENTER
                       Candidate 1 = config + dev/2 (max), last expressive = config - dev/2 (min)

# Parameter filtering (all models)
# - Unsupported params in tts_params are gracefully skipped with a one-time info log
# - SUPPORTED_TTS_PARAMS per model type defined in src/utils/language_registry.py
# - conservative_candidate unsupported keys also gracefully skipped

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
# - Supported params: cfg_scale, temperature, top_p, diffusion_steps, voice_speed_factor, use_sampling
# - Deterministic by default (use_sampling=false), sampling mode optional per speaker
```

### 3. Quality Validation Pipeline
```python
# Validation Chain
WhisperValidator.validate() → transcription + similarity_score
QualityScorer.score_candidate() → QualityScore (combined metrics)

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
- `defaults/vibevoice.yaml`  — VibeVoice (all three size variants)

### Critical Parameters
```yaml
# Text Processing
chunking:
  target_chunk_limit: 380        # Balance: quality vs context window
  max_chunk_limit: 460
  spacy_model: en_core_web_sm

# Generation Quality
generation:
  num_candidates: 3              # More candidates = better quality, higher compute
  max_retries: 1
  model_type: "standard"         # "standard" | "multilingual" | "turbo" | "qwen3"
  default_language: "en"         # Default language for multilingual model
  
# TTS Parameters (per speaker) – unsupported params for the active model are gracefully skipped
tts_params:
  exaggeration: 0.40             # MAX value for RAMP-DOWN (Chatterbox only)
  exaggeration_max_deviation: 0.20
  cfg_weight: 0.2                # MIN value for RAMP-UP (Chatterbox only)
  cfg_weight_max_deviation: 0.20
  temperature: 0.9               # MIN value for RAMP-UP (all models)
  temperature_max_deviation: 0.3
  min_p: 0.03                    # Stability (Chatterbox only)
  top_p: 0.99                    # Creativity (Chatterbox + Qwen3)
  # turbo extras: top_k: 1000, norm_loudness: false
  # qwen3 extras: top_k, top_k_max_deviation,
  #               repetition_penalty, max_new_tokens, do_sample,
  #               subtalker_top_k, subtalker_top_k_max_deviation,
  #               subtalker_temperature, subtalker_temperature_max_deviation,
  #               subtalker_top_p, subtalker_dosample

# Quality Gates
validation:
  similarity_threshold: 0.8      # Text similarity requirement
  min_quality_score: 0.75       # Combined quality threshold
  whisper_model: small          # base/small/medium/large

# Audio Assembly
audio:
  silence_duration:
    normal: 0.20                 # Inter-sentence pause
    paragraph: 0.80              # Paragraph break pause
  sample_rate: 24000            # ChatterboxTTS native (fixed)
```

## Command Line Interface

### Execution Patterns
```bash
# Basic Usage
python src/cbpipe.py                                    # Run default job
python src/cbpipe.py my_job.yaml                       # Run specific job config
python src/cbpipe.py --job "pattern*"                  # Pattern matching jobs

# Execution Strategies
python src/cbpipe.py --mode new                        # Create new task
python src/cbpipe.py --mode last                       # Resume latest task
python src/cbpipe.py --mode last-new                   # Resume + regenerate final audio
python src/cbpipe.py --mode all                        # Process all tasks
python src/cbpipe.py --mode "job1:last,job2:new"      # Per-job strategies

# Options
python src/cbpipe.py --verbose                         # Debug output
python src/cbpipe.py --device cuda                     # Force GPU
python src/cbpipe.py --force-final-generation          # Regenerate final audio
```

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
    has_paragraph_break: bool

@dataclass  
class AudioCandidate:
    audio: torch.Tensor
    chunk_text: str
    generation_params: Dict
    candidate_id: str
    speaker_id: str

@dataclass
class ValidationResult:
    transcription: str
    similarity_score: float
    is_valid: bool
    processing_time: float

@dataclass
class QualityScore:
    overall_score: float
    similarity_score: float
    length_score: float
    transcription_score: float
```

---

**Entry Point**: `src/cbpipe.py`  
**Key Classes**: JobManager, TaskOrchestrator, SpaCyChunker, TTSGenerator, WhisperValidator  
**Config Root**: `config/default_config.yaml`  
**Data Flow**: Text → Chunks → Candidates → Validation → Selection → Assembly → Output