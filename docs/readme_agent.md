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

# Multilingual Support
# - model_type: "standard" (ChatterboxTTS) or "multilingual" (ChatterboxMultilingualTTS)
# - default_language: Default language ID for multilingual model (e.g., "en", "de")
# - language_id: Optional per-generation language override

# RAMP Strategy (N candidates per chunk)
# - Candidate 1: Exact config parameters (baseline)
# - Candidates 2-N: Linear interpolation from config to deviation limits
# - Last candidate: Conservative parameters (optional, for stability)

# Parameter Behavior
exaggeration: RAMP-DOWN from MAX (config) to MIN (config - max_deviation)
cfg_weight: RAMP-UP from MIN (config) to MAX (config + max_deviation)  
temperature: RAMP-UP from MIN (config) to MAX (config + max_deviation)
```

### 3. Quality Validation Pipeline
```python
# Validation Chain
WhisperValidator.validate() → transcription + similarity_score
FuzzyMatcher.match_texts() → MatchResult (token/partial/ratio methods)
QualityScorer.score_candidate() → QualityScore (combined metrics)

# Selection Logic (3-stage fallback)
# Stage 1: Best valid candidate (similarity > threshold)
# Stage 2: Best invalid candidate (highest quality score)
# Stage 3: Emergency fallback (first candidate)
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

### Cascading Configuration (3-Level Inheritance)
```yaml
default_config.yaml          # Base template
    ↓
job_config.yaml             # Job-specific overrides  
    ↓
task_config.yaml            # Runtime task instance
```

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
  model_type: "standard"         # "standard" (ChatterboxTTS) or "multilingual" (ChatterboxMultilingualTTS)
  default_language: "en"         # Default language for multilingual model (e.g., "en", "de", "fr")
  
# TTS Parameters (per speaker)
tts_params:
  exaggeration: 0.40             # MAX value for RAMP-DOWN
  exaggeration_max_deviation: 0.20
  cfg_weight: 0.2                # MIN value for RAMP-UP  
  cfg_weight_max_deviation: 0.20
  temperature: 0.9               # MIN value for RAMP-UP
  temperature_max_deviation: 0.3
  min_p: 0.03                    # Stability: higher = more conservative
  top_p: 0.99                    # Creativity: lower = more conservative

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