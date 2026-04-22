# Technical Overview - Enhanced TTS Pipeline

> **Quick Reference for Developers and AI Agents**

This document is an unsorted collection of useful information that goes beyond the readme.md.

## System Architecture at a Glance

**Pipeline**: Text → Preprocessor → Chunks → Candidates → Validation → Selection → Audio Chunks → Assembly → Output  
**Language**: Python 3.8+, PyTorch, Whisper, SpaCy  
**Architecture**: Job/Task-based pipeline with cascading configuration  
**Entry Point**: `src/cbpipe.py`

### Tail-End Processing Overview
- Smart Tail-Trim (WhisperX + VAD/Energy): Before Whisper validation, candidates are trimmed at the tail to remove hallucinations and noise.
  - Smart search for the last N input-words within the final `search_window_words` of aligned ASR words via WhisperX; cut right after the last match.
  - Fallbacks: WhisperX last-word end → WebRTC VAD backward scan → energy heuristic.
  - Post-processing: keep `post_speech_silence_ms` trailing silence, then apply `fade_out_ms` to the kept segment end.
  - Diagnostics persisted in `whisper/whisper_metrics.json` under `tail_trim` per candidate; `*_trimmed.wav` is preferred by assembly.


## Project structure

### Documentation (`docs/`)

| Document | Purpose |
|----------|---------|
| `readme_agent.md` | Primary reference for AI agents: pipeline flow, cascading config, model types (incl. VibeVoice), chunking, validation, prosody/MOS, CLI, troubleshooting. |
| `TECHNICAL_OVERVIEW.md` | This file: architecture notes, component map, configuration, job/task lifecycle. |
| `mermaid_diagram.md` | Visual workflow diagrams (Mermaid) for the pipeline. |
| `SPEAKER_SYSTEM.md` | Multi-speaker setup: `<speaker:id>` markup, YAML speakers, chunking behaviour. |
| `BEST-CANDIDATE-SELECTION.md` | How candidates are validated, scored (quality + prosody), gated, and selected. |
| `PROSODY_VALIDATION.md` | Prosody/MOS layer: tail-trim, WhisperX alignment, weights, `validation.prosody` / `validation.mos`. |
| `PERSISTENT_CACHE.md` | Why TTS model cache shows “miss” on each new process; HuggingFace disk cache. |
| `CONTRIBUTING.md` | Development setup, style, and contribution workflow. |
| `TESTRUN_FINDINGS.md` | Benchmarks and findings from test runs (e.g. ASR backends, VibeVoice). |
| `basic-usage_CLI-arguments.md` | `cbpipe.py` CLI flags and execution modes. |
| `normalization.md` | Audio normalisation (`audio.normalization`) in assembly. |
| `turbo_paralinguistic_tags.md` | Turbo model tags such as `[laugh]` through chunking and TTS. |
| `candidate_quality_scorer_diagram.md` | Diagram / notes on quality scoring structure. |
| `cursor_audio_player_issues_and_troubles.md` | Cursor IDE audio preview quirks and workarounds. |

```
chatterbox-pipeline/
├── config/
│   ├── __init__.py
│   ├── default_config.yaml        # Central default configuration
│   └── example_job_config.yaml    # Example job configuration
├── data/
│   ├── input/
│   │   ├── reference_audio/       # Reference audio files
│   │   │   └── stephan_moebius.wav
│   │   └── texts/                 # Input texts
│   │       └── input-document.txt
│   └── output/                    # Job output directories
├── docs/                          # supplementary documentation (see table above)
├── logs/                           # main.log
├── scripts/                        # Unit Tests
├── src/
│   ├── __init__.py
│   ├── chunking/                  # Text segmentation
│   │   ├── __init__.py
│   │   ├── base_chunker.py
│   │   ├── chunk_validator.py
│   │   └── spacy_chunker.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── github_config.py
│   ├── generation/                # Audio generation
│   │   ├── __init__.py
│   │   ├── audio_processor.py
│   │   ├── batch_processor.py
│   │   ├── candidate_manager.py
│   │   ├── model_cache.py
│   │   ├── selection_strategies.py
│   │   └── tts_generator.py
│   ├── cbpipe.py                  # Main pipeline script
│   ├── pipeline/                  # Pipeline orchestration
│   │   ├── __init__.py
│   │   ├── task_orchestrator.py
│   │   ├── job_manager/
│   │   │   ├── ...
│   │   ├── job_manager_wrapper.py
│   │   ├── task_executor/
│   │   │   ├── __init__.py
│   │   │   ├── retry_logic.py
│   │   │   ├── stage_handlers/
│   │   │   │   ├── ...
│   │   │   └── task_executor.py
│   │   └── task_executor_original.py
│   ├── preprocessor/              # Text preprocessing
│   │   ├── __init__.py
│   │   └── text_preprocessor.py
│   └── utils/                     # Helper functions
│       ├── __init__.py
│       ├── audio_utils.py
│       ├── config_manager.py
│       ├── file_manager/
│       │   ├── __init__.py
│       │   ├── file_manager.py
│       │   ├── io_handlers/
│       │   │   ├── ...
│       │   ├── state_analyzer.py
│       │   └── validation_helpers.py
│       ├── file_manager.py
│       ├── logging_config.py
│       └── progress_tracker.py
├── chatterbox_tester.py        # A tool to quickly render test generations models and the audio input-reference-file to find optimal kwargs 
├── chatterbox_tester_presets.yaml   # presets config inside chatterbox_tester.py
├── CHATTERBOX_TESTER.md        # documentation for chatterbox_tester.py
├── dev-requirements.txt          # Development dependencies
├── mypy.ini
├── pyproject.toml               # Project configuration
├── README.md
└── requirements.txt              # Production dependencies
```

## Core Components

### 1. Job/Task Management (`src/pipeline/`)
```python
from enum import Enum

class ExecutionStrategy(Enum):
    LAST = "last"        # Use latest task
    ALL = "all"         # Use all tasks
    NEW = "new"         # Create new task
    LAST_NEW = "last-new"  # Use latest task + new final audio
    ALL_NEW = "all-new"    # Use all tasks + new final audio

class UserChoice(Enum):
    LATEST = "latest"           # Use latest task
    ALL = "all"                # Use all tasks
    NEW = "new"                # Create new task
    LATEST_NEW = "latest-new"  # Use latest task + new final audio
    ALL_NEW = "all-new"        # Use all tasks + new final audio
    SPECIFIC = "specific"      # Select specific task
    SPECIFIC_NEW = "specific-new"  # Select specific task + new final audio
    CANCEL = "cancel"          # Cancel execution

JobManager:
    - load_job_config() → JobConfig
    - track_job_state() → JobState
    - queue_jobs() → JobQueue
    - resolve_execution_plan() → ExecutionPlan
    - prompt_user_selection() → UserChoice

TaskOrchestrator:
    - execute_task() → TaskResult
    - detect_state() → TaskState
    - analyze_gaps() → GapAnalysis
    - force_final_regeneration() → bool

TaskExecutor:
    - execute_task() → TaskResult  # Individual task execution
    - process_chunks() → ChunkResults
    - validate_candidates() → ValidationResults
```

### 2. Text Pre-Processing (`src/preprocessing/`)
```python
TextPreprocessor.process_text_file()
 - normalize_line_endings: bool
 - normalize_quotes: bool
```

### 3. Text Processing (`src/chunking/`)
```python
SpaCyChunker.chunk_text(text) → List[TextChunk]
# Linguistic sentence segmentation with speaker-transition priority.
# Defaults depend on the parent config:
#   Chatterbox: target=380, max=460, force_paragraph_chunks=true (see default_config.yaml)
#               → every \n\n is a HARD chunk boundary (paragraph-break silence guaranteed)
#   Qwen3:      target=1200, max=2000, force_paragraph_chunks=false (see config/defaults/qwen3.yaml)
#   VibeVoice:  target=2800, max=4800, force_paragraph_chunks=false (see config/defaults/vibevoice.yaml)
#   Longform (Qwen3 / VibeVoice):
#     HARD boundaries: only \n\n\n+ (2+ empty lines); last chunk before such a break gets
#     paragraph_break_type="long" → Assembly inserts long silence.
#     SOFT boundaries: single \n\n is a preferred split-point but greedy packing still applies
#     within a hard section.
#
# Finalization (_finalize_chunks):
#   _merge_micro_chunks: always runs.  Hard-break stops:
#   speaker_transition OR paragraph_break_type=="long".
#   With force_paragraph_chunks=true, plain paragraph breaks also stop the merge.
```

### 4. Audio Generation (`src/generation/`)
```python
TTSGenerator.generate_candidates(text, N) → List[AudioCandidate]
CandidateManager.generate_candidates_for_chunk() → GenerationResult
# N candidates per chunk. Ramped params use one rule: value = base + (dev * ramp_pos).
# Positive dev = ramp up, negative = ramp down. Last candidate may be a
# "CONSERVATIVE" fallback (params from speaker.conservative_candidate) when
# conservative_candidate.enabled=true.

# Supported model types (src/utils/language_registry.py):
#   standard / multilingual / turbo       – Chatterbox family
#   qwen3                                  – Qwen3 TTS 1.7B
#   vibevoice / vibevoice_1_5b / vibevoice_q4 – VibeVoice family (local vendored code)
#
# VibeVoice specifics (src/generation/tts_generator.py):
#   - Supported tts_params: cfg_scale, temperature, top_p, diffusion_steps,
#                           voice_speed_factor, use_sampling
#   - Expressive ramp: same unified formula as other models; cfg_scale, temperature,
#     diffusion_steps when deviation non-zero (candidate 0 = base).
#   - use_sampling=false → greedy LM (top_p/temperature ignored).
#   - voice_speed_factor resamples the *reference* audio, not the output.
#   - Per-candidate seed: base_seed(speaker) + i*1000 + hash(text)%10000
#     (unless generation.seed_fixed=true; see docs/SEEDING.md).
#   - generation.vibevoice.language_strict guards non-en/zh languages.

# Multilingual Support
TTSGenerator(config, model_type="multilingual") → multilingual model loading
generate_single(text, language_id="de") → language-specific generation
# Automatic fallback: multilingual → standard model if unavailable
```

### 5. Quality Validation (`src/validation/`)
```python
ASRBackend.transcribe() → transcription (Whisper or VibeVoice-ASR)
WhisperValidator.validate_candidate() → ValidationResult
ProsodyScorer.score() → ProsodyResult (flow, liveliness, mos, semantic_alignment, …)
QualityScorer.score_candidate() → QualityScore
# Speech-to-Text validation + prosody/MOS + multi-criteria scoring

# ASR backend routing (src/validation/asr/factory.py)
# - validation.asr_backend: auto | whisper | vibevoice_asr
# - "auto" resolves to whisper for ALL model types (VibeVoice-ASR disabled by default
#   due to ~16 GB VRAM footprint and a long-form mask bug on ≥4-min clips).
# - "vibevoice_asr" is opt-in and falls back to Whisper on load failure.

# Robust Similarity
# - quality_calculator.calculate_similarity() applies robust normalization
#   (lowercase, unicode punctuation handling, whitespace/hyphen unification,
#    ß→ss tolerance). Uses RapidFuzz token ratios if available,
#    falls back to Jaccard on tokens.

# Numbers Normalization (non-EN only)
# - validation.numbers_normalization_mode: off|placeholder|digits|words
# - Similarity is computed on normalized texts; length uses normalized original length.

# Dynamic Thresholding
# - WhisperValidator computes an effective similarity threshold based on
#   text length and punctuation density (no new config flags).

# Prosody + MOS (src/validation/prosody_scorer.py)
# - Weights: validation.prosody.weights.{semantic_alignment, flow, liveliness,
#            intelligibility, mos}  (default mos=0.4, flow=0.3, liveliness=0.3)
# - Targets: wpm_min/wpm_max for flow, liveliness_target
# - Select:  alpha_quality + beta_prosody + gamma_mos for final combined score
# - MOS is language-gated (validation.mos.enabled_languages, default [de, en]).
# - Long-audio MOS windowing (validation.mos.segmentation):
#     when duration > max_unsegmented_seconds AND provider implements
#     score_segmented, audio is scored in sliding windows (window_s/hop_s)
#     and aggregated (median|mean|min). Added so short-utterance MOS models
#     (NISQA/UTMOS) stay in-distribution on ~1-minute VibeVoice chunks.

# Tail-End Trimming (Smart + Fallbacks)
# - validation.preprocessing.tail_trim (see default_config.yaml)
#   smart_match.enabled: word-level alignment via WhisperX
#   last_n_words: input tail words to match
#   fuzzy_ratio: acceptance threshold (0..1)
#   search_window_words: limit scan near audio end
#   language_gate: enable Smart-Trim only for specific languages
#   persist_trimmed_candidate: write candidate_YY_trimmed.wav
#   debug_save_removed_tail: write removed tail snippet for inspection
```

Numbers Normalization (non-EN only)
- Config: validation.numbers_normalization_mode: off | placeholder | digits | words
- placeholder: Replaces digits and recognized number words with <NUM>
- digits: Converts number words → digits (e.g., "tausend" → 1000) [uses word2num-de if available]
- words: Converts digits → words (e.g., 1995 → "neunzehnhundertfünfundneunzig") [uses num2words]

Effects
- Similarity: computed on robustly normalized texts (original + transcription)
- Length: uses normalized original length; transcription length remains raw
- Optional Whisper initial prompt: validation.whisper_initial_prompt_enabled + validation.whisper_initial_prompt_text

### 6. Audio Assembly (`src/generation/audio_processor.py`)
```python
AudioProcessor.concatenate_segments() → torch.Tensor
# Intelligente Audio-Verkettung mit Pausenverarbeitung
# Note: Assembly loads candidate_YY_trimmed.wav when present, otherwise falls back to candidate_YY.wav
```


### Core classes 

#### 1. Chunking Layer
```python
class SpaCyChunker:
    def chunk_text(text: str) → List[TextChunk]
    
class TextChunk:
    text: str
    start_pos: int  
    end_pos: int
    has_paragraph_break: bool
```

#### 2. Generation Layer  
```python
class TTSGenerator:
    def generate_candidates(text, num_candidates) → List[AudioCandidate]
    def __init__(config, model_type="standard|multilingual")
    # Properties: model_type, default_language, is_multilingual
    
class CandidateManager:
    def generate_candidates_for_chunk() → GenerationResult
    
class AudioCandidate:
    audio: torch.Tensor
    chunk_text: str
    generation_params: Dict
    candidate_id: str
    speaker_id: str      # speaker reference
    language_id: str     # language used (for multilingual)
```

#### 3. Validation Layer
```python
class WhisperValidator:
    def validate_candidate() → ValidationResult
    
class FuzzyMatcher:
    def match_texts() → MatchResult
    
class QualityScorer:
    def score_candidate() → QualityScore
```

#### 4. Processing Layer
```python  
class AudioProcessor:
    def concatenate_segments() → torch.Tensor
    def save_audio() → bool
```

### Important data structures

#### AudioCandidate
```python
@dataclass
class AudioCandidate:
    audio: torch.Tensor           # Generated audio tensor
    chunk_text: str              # Original text
    generation_params: Dict      # TTS parameters used
    timestamp: datetime          # Creation time
    candidate_id: str           # Unique identifier
```

#### ValidationResult
```python
@dataclass  
class ValidationResult:
    transcription: str           # Whisper transcription
    similarity_score: float      # Text similarity (0-1)
    is_valid: bool              # Passes threshold
    processing_time: float       # Validation duration
```

#### QualityScore
```python
@dataclass
class QualityScore:
    overall_score: float         # Combined score (0-1)
    similarity_score: float      # Text similarity component
    length_score: float          # Audio length component  
    transcription_score: float   # Transcription quality


## Data Flow

### Input/Output Formats
- **Input**: Text file (`data/input/texts/*.txt`)
- **Output**: WAV file (`data/output/{job_name}/*.wav`)
- **Config**: YAML (`config/default_config.yaml`, `config/jobs/*.yaml`)

### Per-Task JSON Outputs

Each task directory contains three JSON artifacts written by `TaskMetricsGenerator`
(`src/utils/file_manager/task_metrics_generator.py`):

| File | Purpose |
|---|---|
| `whisper/whisper_metrics.json` | Raw Whisper/validation data for all candidates, including full transcriptions. |
| `task_metrics.json` | Task overview: selected candidates, summary statistics, per-chunk text. Used by the pipeline and the Audio User Selection Editor. |
| `analysis_metrics.json` | Compact analysis file for parameter-sweep experiments (see below). |

#### `analysis_metrics.json`

Written alongside `task_metrics.json` after every successful task completion.
Designed for efficient offline analysis of multi-speaker, multi-candidate jobs
(e.g. temperature sweeps or RAMP-strategy comparisons).

**What it contains** (per candidate, per chunk):
- `generation_params` – all sampling parameters (temperature, top_k, …), without
  `seed` or `language_id`
- `scores` – `final_selection_score`, `overall_quality_score`, `whisper_similarity`,
  `whisper_quality`, `length_score`, `penalty_score`, and all `prosody_*` subscores
  (`prosody_score`, `prosody_flow`, `prosody_liveliness`, `prosody_intelligibility`,
  `prosody_mos`, `raw_mos`, `wpm`)
- `gates` – `is_valid`, `passes_mos_gate`, `passes_similarity_gate`
- `audio_duration`
- `selected_candidate` (1-based) per chunk

**What it omits:** full text, transcriptions, audio filenames, and any path-based
fields.

**Schema stability rules:**
- Prosody fields are always present; they are `null` when prosody scoring was disabled
  for a run (consistent schema over compact schema).
- All float scores are rounded to 4 decimal places; `wpm` to 1; `audio_duration` to 2.
- `schema_version` (currently `"1.0"`) is a module-level constant
  (`ANALYSIS_METRICS_SCHEMA_VERSION` in `task_metrics_generator.py`) so breaking
  changes can be tracked.
- The file is a pure addition; `task_metrics.json` and `whisper_metrics.json` are
  never modified by `generate_analysis_metrics()`.

### Key Data Structures
```python
JobConfig:      {job_name, input_file, output_dir, task_configs}
TaskConfig:     {task_type, params, dependencies}
TextChunk:      {text, start_pos, end_pos, has_paragraph_break}
AudioCandidate: {audio: torch.Tensor, chunk_text, generation_params, candidate_id}
ValidationResult: {transcription, similarity_score, is_valid, processing_time}
QualityScore:   {overall_score, similarity_score, length_score, transcription_score}
```

## Configuration System

### N-Level Cascading Configuration via `parent:`

Each YAML file (except `default_config.yaml` and runtime task snapshots) may declare an
optional `parent:` field at the top, referencing another config relative to `config/`.
`ConfigManager._resolve_parent_chain()` walks the chain recursively until a file without
a `parent:` is reached, then falls back to `default_config.yaml`.

```
default_config.yaml
    ↑  (parent: default_config.yaml)
config/defaults/qwen3.yaml
    ↑  (parent: defaults/qwen3.yaml)
config/test_qwen3.yaml          ← job config
    ↑  (runtime snapshot, no parent resolution)
data/output/test_qwen3/...task_config.yaml
```

```yaml
# config/defaults/qwen3.yaml  – model-specific defaults
parent: default_config.yaml

generation:
  model_type: qwen3
  ...

# config/my_job.yaml  – minimal job config
parent: defaults/qwen3.yaml

job:
  name: my-qwen3-job
input:
  text_file: my-text.txt
generation:
  default_speaker: alice
  speakers:
    - id: alice
      reference_audio: alice.wav
      # tts_params inherited from defaults/qwen3.yaml if omitted
```

Without a `parent:` field the behavior is identical to the previous 2-level system
(job.yaml → default_config.yaml).

**Pre-built model defaults** (`config/defaults/`):
- `chatterbox.yaml` — standard / multilingual
- `turbo.yaml`      — ChatterboxTurboTTS
- `qwen3.yaml`      — Qwen3 TTS
- `vibevoice.yaml`  — VibeVoice (all three variants; longform chunking, refinement off,
                       language_strict=false, `vibevoice` subsection for backend tuning)

### Critical Parameters
```yaml
generation:
  num_candidates: 3
  model_type: standard          # standard|multilingual|turbo|qwen3|vibevoice|vibevoice_1_5b|vibevoice_q4
  global_seed: 12345            # 0 = random per candidate (see docs/SEEDING.md)
  seed_fixed: false             # true = single torch seed end-to-end (requires seed > 0)

validation:
  similarity_threshold: 0.64    # dynamically adjusted by length/punctuation
  min_quality_score: 0.72
  whisper_model: small|base|medium|large
  numbers_normalization_mode: off|placeholder|digits|words   # non-EN only
  whisper_initial_prompt_enabled: false
  whisper_initial_prompt_text: ""
  asr_backend: auto             # auto|whisper|vibevoice_asr
  mos:
    enabled_languages: [de, en]
    segmentation:               # windowed MOS for long VibeVoice outputs
      max_unsegmented_seconds: 20.0
      window_s: 12.0
      hop_s: 10.0
      aggregator: median

chunking:
  # Chatterbox
  target_chunk_limit: 380
  max_chunk_limit: 460
  force_paragraph_chunks: true  # every \n\n is hard boundary
  # Qwen3 (via defaults/qwen3.yaml)
  # target_chunk_limit: 1200
  # max_chunk_limit: 2000
  # force_paragraph_chunks: false  # longform: only \n\n\n+ is hard boundary
  # VibeVoice (via defaults/vibevoice.yaml)
  # target_chunk_limit: 2800
  # max_chunk_limit: 4800
  # force_paragraph_chunks: false  # only \n\n\n+ is hard boundary (→ long-pause silence)
```

## Job/Task Management

### Job Lifecycle
1. **Job Creation**: Load job config, validate dependencies
2. **Task Generation**: Create task configs based on job requirements
3. **State Tracking**: Monitor job progress, track task states
4. **Recovery**: Automatic gap detection and task resumption

### Task Orchestration
1. **State Detection**: Analyze existing output files
2. **Gap Analysis**: Identify missing or failed tasks
3. **Orchestration**: Run tasks in correct order via TaskOrchestrator
4. **Validation**: Verify task completion and output quality

## Common Issues & Solutions

### Memory Errors
- **Problem**: Out of memory with high candidate count
- **Solution**: Reduce `num_candidates` in job config

### Validation Failures  
- **Problem**: No valid candidates found
- **Solution**: Lower `similarity_threshold` or Increase `num_candidates` or 'finteune config values (lower temperature)' or check reference audio quality

## Testing Strategy

### Mock Components (CI/CD)
```python
scripts/test_mock_pipeline.py     # No heavy models, sine wave generation
tests/test_integration.py         # pytest-based end-to-end tests
```

### Debug Mode
```bash
# Job-specific output directories
data/output/{job_name}/
├── tasks/           # Task-specific outputs
├── candidates/      # All generated candidates  
├── transcriptions/  # Whisper validation results
└── log.txt         # Detailed pipeline log
```

## Development Quick Start


### Important Files
```
src/cbpipe.py                      # Main pipeline orchestration
src/pipeline/job_manager.py      # Job management
src/pipeline/task_orchestrator.py # Task orchestration
src/pipeline/task_executor.py    # Individual task execution
config/default_config.yaml       # Default configuration
```

### Debug Commands
```bash
# Test individual components
python scripts/run_chunker.py
python scripts/test_mock_pipeline.py

# Full pipeline with timing
python src/cbpipe.py --job my_job
```

## API Summary

### Main Pipeline
```python
def main() -> bool:
    # Phase 1: Job Management
    job_manager = JobManager()
    job = job_manager.load_job(job_name)
    
    # Phase 2: Task Orchestration
    task_orchestrator = TaskOrchestrator(job)
    for task in job.tasks:
        result = task_orchestrator.execute_task(task)
        if not result.success:
            task_orchestrator.handle_failure(task, result)
    
    return job.is_complete()
```

### Configuration Access
```python
# Cascading configuration access
config_manager = ConfigManager()
default_config = config_manager.load_default_config()
job_config = config_manager.load_job_config(job_name)
task_config = config_manager.create_task_config(job_config, task_type)
```

### Logging Patterns
```python
logger.info("📝 High-level progress")      # User-facing progress
logger.debug("🔍 Detailed validation")     # Debug information  
log_info_file_only("💾 File operations")   # File-only logs
logger.warning("⚠️ Fallback usage")        # Warnings
logger.error("❌ Critical failures")       # Errors
```

## Status: 

- **Phase 1**: Job/Task Management System
- **Phase 2**: Text chunking and TTS generation
- **Phase 3**: Whisper validation and quality scoring  
- **Phase 4**: Audio post-processing and pipeline orchestration
- **Testing**: Mock pipeline for CI/CD, integration tests
- **Documentation**: Comprehensive technical documentation

**Ready for**: feature extensions 

### Execution Strategies
The pipeline supports various execution strategies:

1. **Global Strategies** (`--mode`):
   - `last`: Use latest task
   - `all`: Execute all tasks
   - `new`: Create new task
   - `last-new`: Latest task + new final audio
   - `all-new`: All tasks + new final audio

2. **Job-specific Strategies** (`--job-mode`):
   ```bash
   --job-mode "job1:last-new,job2:all-new,job3:last"
   ```

3. **Interactive Selection**:
   - Used when no strategy is specified
   - Offers all options from global strategies
   - Additional option for specific task selection

### Priorities
1. Job-specific strategies (`--job-mode`)
2. Global strategy (`--mode`)
3. Interactive selection (fallback)



### Testing Architecture

#### Mock Components
```python
# CI/CD compatible testing without heavy models
MockTTSGenerator     # Generates sine waves instead of speech
MockWhisperValidator # Returns predefined transcriptions
MockAudioProcessor   # Basic audio operations
```

#### Test Coverage
- **Unit Tests**: Individual component testing
