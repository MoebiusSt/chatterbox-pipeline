# Enhanced TTS Pipeline Development Plan

## Table of Contents
1. [Project Overview](#project-overview)
2. [Current State Analysis](#current-state-analysis)
3. [Technical Architecture](#technical-architecture)
4. [Product Requirements](#product-requirements)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Development Stack](#development-stack)
7. [Project Structure](#project-structure)
8. [Quality Assurance](#quality-assurance)
9. [Deployment Strategy](#deployment-strategy)

---

## Project Overview

### Vision Statement
Transform the existing Chatterbox TTS pipeline into a production-ready, task bases artifact-free long-form audio generation CLI capable of batch processing documents up to 50,000 characters with professional-grade quality.

### Core Problem
Current TTS pipeline limitations:
- **Character-based chunking** creates unnatural breaks mid-sentence
- **Audio artifacts** from poor concatenation and low-volume noise
- **Context loss** affecting pronunciation and emphasis
- **No quality validation** leading to inconsistent output
- **Manual post-processing** required for professional results

### Solution Strategy
Three-phase enhancement approach:
1. **Intelligent Chunking** - SpaCy-powered linguistic segmentation
2. **Quality Validation** - Whisper AI-based candidate selection
3. **Automated Post-Processing** - Auto-editor artifact removal

# TTS Pipeline Enhancement - Software Architecture Plan

## Functional Requirements

### 1. Enhanced Text Chunking with SpaCy
- **Sentence Boundary Detection**: Use SpaCy's sentence segmentation for accurate splitting
- **Token-based Length Estimation**: Calculate chunk lengths based on actual tokens rather than characters
- **Context Window Management**: Predict model context usage more accurately to minimize exceedances
- **Balanced Chunking**: Ensure more evenly distributed chunk sizes while respecting sentence boundaries

### 2. Audio Post-Processing with Auto-Editor
- **Artifact Removal**: Remove low-volume artifacts below speaking threshold
- **Dynamic Threshold Calculation**: Set volume threshold relative to reference audio noise floor
- **Natural Sound Preservation**: Preserve inhalations, exhalations, and lip sounds using margin settings
- **Attack/Decay Margins**: Configure cushion periods before/after cuts to maintain natural flow

### 3. Quality Validation with Whisper
- **Multi-Candidate Generation**: Generate 3 candidates per chunk by default (configurable)
- **Speech-to-Text Validation**: Transcribe each candidate using Whisper locally
- **Fuzzy Text Matching**: Match transcriptions against original text with 95% threshold
- **Retry Mechanism**: Generate 2 additional attempts for failed candidates
- **Selection Strategy**: Choose shortest validated candidate (artifacts typically increase length)
- **Fallback Selection**: If all candidates fail validation, select highest-scoring candidate

---

## Technical Architecture

### System Design Principles
- **Modularity**: Independent, reusable, testable components
- **Configurability**: YAML file-based settings management
- **Extensibility**: Plugin architecture for new features (perspectivly, switching of reference input audio to be able to switch reading roles, characters or voices during the generation of texts with roles in between tasks; managment of said input reference audio files/characters/voices)
- **Reliability**: error handling and fallbacks

### Core Components Architecture

```mermaid
  currently undergoing refactoring
```

### Data Flow Architecture

1. **Input Processing**: Text → Preprocessing → SpaCy Analysis → Linguistic Chunks
2. **Generation Phase**: Chunks → Multiple Candidates → Audio Tensors
3. **Validation Phase**: Audio → Whisper Transcription → Fuzzy Matching with scoring
4. **Selection Phase**: Validate Audio Candidates → Quality Scoring → Best Selection
5. **Post-Processing**: Selected Audio → Artifact Removal → Final Output Assembly

---

## Product Requirements

### Functional Requirements

#### FR-001: Intelligent Text Chunking
- **SpaCy Integration**: Use linguistic models for sentence boundary detection
- **Token-Based Estimation**: Accurate context window management
- **Configurable Limits**: Target ~500 chars, max ~600 chars, safety margins (max of Chatterbox is 1000)
- **Multi-Language Support**: English initially

#### FR-002: Quality Validation System
- **Candidate Generation**: 3 candidates per chunk (configurable)
- **Whisper Validation**: Local speech-to-text transcription
- **Fuzzy Matching**: similarity threshold with retry mechanism
- **Selection Logic**: best validated candidate wins

#### FR-003: Automated Post-Processing
- **Auto-Editor Integration**: Dynamic threshold-based artifact removal
- **Natural Sound Preservation**: Breathing, lip sounds with margins and attack decay values
- **Reference Audio Analysis**: Noise floor calculation for thresholds

#### Scalability Requirements  
- **Document Size**: Support up to 50,000 characters
- **Concurrent Processing**: not required, as it is oo memory expansive - instead task based batch processing

### User Stories

#### Primary User Stories
**ST-101**: As a content creator, I want to convert long texts to high-quality audio without manual editing
- Generate professional audiobooks from manuscripts
- Maintain consistent voice quality across chapters
- Eliminate need for manual post-processing

**ST-102**: As a developer, I want reliable API integration for TTS services
- Consistent, predictable output quality
- Comprehensive error handling and logging
- Configurable parameters for different use cases

#### Edge Case Handling
**ST-301**: Handle validation failures gracefully with fallback mechanisms
**ST-302**: Recover from individual functional failures without stopping entire process. Capability to resume failed, incomplete tasks runs

---

## Implementation Roadmap

### Phase 1: Foundation 
**Objective**: Establish core architecture and SpaCy integration

#### Project Setup
- [x] Initialize project structure with modular architecture
- [x] Set up development environment and dependencies
- [x] Create configuration management system
- [x] Implement basic logging and error handling

#### Job/Task Management
- [x] Develop `JobManager` for job configuration and state tracking
- [x] Implement `TaskExecutor` for task execution and state management
- [x] Create flexible execution strategies:
  - Global strategies (`--mode`)
  - Job-specific strategies (`--job-mode`)
  - Interactive selection
- [x] Add final audio regeneration options:
  - Force new final audio for specific tasks
  - Batch regeneration for multiple tasks
  - Interactive regeneration choice

#### SpaCy Integration
- [x] Develop `SpaCyChunker` class with sentence boundary detection
- [x] Implement token-based length estimation
- [x] Create `ChunkValidator` for quality checks

### Phase 2: Generation & Validation
**Objective**: Implement candidate generation and Whisper validation

#### Candidate Generation
- [x] Develop `CandidateManager` for multiple candidate generation
- [x] Implement retry mechanisms for invalid initial generations
- [x] Create `AudioProcessor` for segment management
- [x] Add progress tracking and monitoring

#### Whisper Validation
- [x] Integrate Whisper AI for local speech-to-text
- [x] Develop `FuzzyMatcher` with configurable thresholds
- [x] Implement `QualityScorer` for candidate selection
- [x] Create validation result reporting system

### Phase 3: Post-Processing
**Objective**: Auto-editor integration and pipeline orchestration

#### Auto-Editor Integration
- [ ] Develop `AutoEditorWrapper` for artifact removal
- [ ] Implement `NoiseAnalyzer` for threshold calculation
- [ ] Create natural sound preservation logic
- [ ] Add configurable margin settings

#### Pipeline Orchestration
- [ ] Build `job_orchestrator` for end-to-end processing pf many new or incomplete jobs
- [ ] Implement `TaskManager` for workflow control
- [ ] Add comprehensive error handling and recovery
- [x] Create final output generation and reporting

### Phase 4: Testing & Optimization
**Objective**: Comprehensive testing and performance optimization

#### Testing Suite
- [x] Unit tests for all core components
- [x] Integration tests for full pipeline using mock components
- [x] Mock TTS pipeline for CI/CD environments
- [x] Separation of production and test code

#### Code Quality & Architecture
- [x] Clean separation of production and test code
- [x] Mock pipeline for CI/CD without heavy model dependencies
- [x] Modular test architecture with pytest integration
- [x] Dual logging system with reduced console output and complete file logging per task
- [ ] Complete API documentation
- [ ] User guide and deployment instructions

---

## Development Stack

### Core Dependencies
```python
# requirements.txt
torch>=2.0.0                    # PyTorch for TTS model
torchaudio>=2.0.0              # Audio processing
spacy>=3.6.0                   # NLP and sentence segmentation
openai-whisper>=20231117       # Speech-to-text validation
auto-editor>=24.0.0            # Audio post-processing
fuzzywuzzy>=0.18.0             # Fuzzy text matching
python-Levenshtein>=0.21.0    # String similarity calculations
pytest>=7.0.0                  # Testing framework
pytest-cov>=4.0.0             # Coverage reporting
```


### Language Models Setup
```bash
# SpaCy models
python -m spacy download en_core_web_sm

# Whisper models (downloaded automatically)
# Base model: ~140MB, good balance of speed/accuracy
```

### Development Tools
```python
# dev-requirements.txt
black>=23.0.0                  # Code formatting
isort>=5.12.0                  # Import sorting
flake8>=6.0.0                  # Linting
mypy>=1.5.0                    # Type checking
pre-commit>=3.3.0              # Git hooks
sphinx>=7.0.0                  # Documentation
```

## Configuration Management

### Pipeline Configuration (`config/default_config.yaml`)

```yaml
job:
  name: "default"
  run-label: ""

input:
  reference_audio: "fry.wav"        # Dateiname im reference_audio Ordner
  text_file: "input-document.txt"   # Text-Datei im texts Ordner

preprocessing:
  enabled: true
  # Text normalization options
  normalize_line_endings: true    # Convert \r\n and \r to \n
  # Future preprocessing options can be added here
  # normalize_quotes: false
  # remove_extra_whitespace: false
  # fix_encoding_issues: false

chunking:
  target_chunk_limit: 380
  max_chunk_limit: 460
  min_chunk_length: 80
  spacy_model: "en_core_web_sm"

generation:
  num_candidates: 2
  max_retries: 2
  tts_params:
    exaggeration: 0.40
    cfg_weight: 0.30
    temperature: 0.9
  # Conservative candidate parameters for guaranteed correctness
  conservative_candidate:
    enabled: true
    exaggeration: 0.45
    cfg_weight: 0.4
    temperature: 0.7
  
validation:
  whisper_model: "small" # "base" "small" "medium" "large"
  similarity_threshold: 0.8
  min_quality_score: 0.75

postprocessing:
  audio_cleaning:
    enabled: false  # Disable AudioCleaner completely to avoid chunky audio
  auto_editor:
    enabled: false  # Disable Auto-Editor completely
    margin_before: 0.4    # longer puffer before speech segments
    margin_after: 0.3     # longer puffer after speech segments
    preserve_natural_sounds: true
  noise_threshold_factor: 0.4  # multiplication factor for recommended threshold (0.8 = 80% of the detected voice threshhold)


audio:
  silence_duration:
    normal: 0.20
    paragraph: 0.80
  # ChatterboxTTS native sample rate - NOT a user setting!
  # This must match the actual output sample rate of ChatterboxTTS (24kHz)
  # Only change this if ChatterboxTTS itself changes its output sample rate
  sample_rate: 24000 
```

## Data Models

```python
# Text Processing Models
@dataclass
class TextChunk:
    text: str
    start_pos: int
    end_pos: int
    has_paragraph_break: bool
    estimated_tokens: int

# Audio Generation Models
@dataclass
class AudioCandidate:
    audio: torch.Tensor
    chunk: TextChunk
    generation_params: Dict[str, Any]
    timestamp: datetime
    candidate_id: str

# Validation Models
@dataclass
class ValidationResult:
    is_valid: bool
    transcription: str
    similarity_score: float
    quality_score: float
    validation_time: float

@dataclass
class MatchResult:
    similarity: float
    is_match: bool
    original_text: str
    compared_text: str
```

## Logging System

### Dual Logging Configuration
The pipeline implements a dual logging system:
- **Console Output**: Reduced verbosity showing only essential information
- **Log File**: Complete detailed logging saved to `\data\output\{timestamp}\log.txt`

#### Log File Features
- Full timestamp and logger name information
- Complete validation metrics and debugging details
- Candidate generation and selection details
- Fuzzy matching and word count validation results
- File I/O operations and transcription saves

#### Console Features
- Clean, minimal output focusing on progression and results
- Essential phase information and error messages
- Candidate selection results without detailed scoring
- Best candidate selection result with config-values of candidate
- Progress tracking and completion summaries

---

## Project Structure

```
tts_pipeline_enhanced/
├── README.md
├── requirements.txt
├── dev-requirements.txt
├── pyproject.toml
├── .pre-commit-config.yaml
├── .github/
│   └── workflows/
│       ├── tests.yml
│       └── release.yml
├── config/
│   ├── __init__.py
│   ├── settings.py
│   ├── default_config.yaml
│   └── logging_config.yaml
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── base_chunker.py
│   │   ├── spacy_chunker.py
│   │   └── chunk_validator.py
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── tts_generator.py
│   │   ├── candidate_manager.py
│   │   └── audio_processor.py
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── whisper_validator.py
│   │   ├── fuzzy_matcher.py
│   │   └── quality_scorer.py
│   ├── postprocessing/
│   │   ├── __init__.py
│   │   ├── auto_editor_wrapper.py
│   │   ├── noise_analyzer.py
│   │   └── audio_cleaner.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── task_executor.py
│   │   ├── job_manager.py
│   │   ├── batch_executor.py
│   │   └── error_handler.py
│   └── utils/
│       └── __init__.py
│       ├── audio_utils.py
│       ├── file_manager.py
│       └── logging_config.py
├── data/
│   ├── input/
│   │   ├── texts/
│   │   └── reference_audio/
│   ├── output/
│   │   └── job/
│   │       ├── {textfile_date_time}/ # task config
│   │       ├── texts/     # Combined chunks and transcriptions
│   │       ├── candidates/ # audio renderings
│   │       └── final/     # Final enhanced audio output
│   └── temp/
│      ├── transcriptions/
│      └── validation_logs/
├── scripts/
│   ├── setup_dependencies.py
│   ├── download_models.py
│   └── run_pipeline.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_chunking/
│   ├── test_generation/
│   ├── test_validation/
│   ├── test_postprocessing/
│   └── test_pipeline/
└── docs/
    ├── api/
    ├── user_guide/
    └── examples/
```

## Core Classes and Modules

### 1. Text Chunking Module (`src/chunking/`)

```python
# base_chunker.py
class BaseChunker(ABC):
    @abstractmethod
    def chunk_text(self, text: str) -> List[TextChunk]

# spacy_chunker.py
class SpaCyChunker(BaseChunker):
    def __init__(self, model_name: str = "en_core_web_sm")
    def chunk_text(self, text: str) -> List[TextChunk]
    def _estimate_token_length(self, text: str) -> int
    def _find_optimal_split_point(self, sentences: List[Span]) -> int

# chunk_validator.py
class ChunkValidator:
    def validate_chunk_length(self, chunk: TextChunk) -> bool
    def validate_sentence_boundaries(self, chunk: TextChunk) -> bool
```

### 2. Generation Module (`src/generation/`)

```python
# tts_generator.py
class TTSGenerator:
    def __init__(self, model: ChatterboxTTS, device: str)
    def generate_single(self, text: str, **kwargs) -> torch.Tensor
    def generate_candidates(self, text: str, num_candidates: int) -> List[AudioCandidate]

# candidate_manager.py
class CandidateManager:
    def __init__(self, max_candidates: int = 3, max_retries: int = 2)
    def generate_candidates_for_chunk(self, chunk: TextChunk) -> List[AudioCandidate]
    def select_best_candidate(self, candidates: List[AudioCandidate]) -> AudioCandidate

# audio_processor.py
class AudioProcessor:
    def concatenate_segments(self, segments: List[torch.Tensor]) -> torch.Tensor
    def add_silence(self, audio: torch.Tensor, duration: float) -> torch.Tensor
```

### 3. Validation Module (`src/validation/`)

```python
# whisper_validator.py
class WhisperValidator:
    def __init__(self, model_size: str = "base")
    def transcribe_audio(self, audio: torch.Tensor) -> str
    def validate_candidate(self, candidate: AudioCandidate, original_text: str) -> ValidationResult

# fuzzy_matcher.py
class FuzzyMatcher:
    def __init__(self, threshold: float = 0.95)
    def match_texts(self, text1: str, text2: str) -> MatchResult
    def calculate_similarity(self, text1: str, text2: str) -> float

# quality_scorer.py
class QualityScorer:
    def score_candidate(self, candidate: AudioCandidate, validation: ValidationResult) -> float
    def _calculate_length_score(self, audio: torch.Tensor) -> float
    def _calculate_validation_score(self, validation: ValidationResult) -> float
```

### 4. Post-Processing Module (`src/postprocessing/`)

```python
# auto_editor_wrapper.py
class AutoEditorWrapper:
    def __init__(self, margin_before: float = 0.1, margin_after: float = 0.1)
    def clean_audio(self, audio_path: str, threshold: float) -> str
    def calculate_threshold_from_reference(self, reference_path: str) -> float

# noise_analyzer.py
class NoiseAnalyzer:
    def analyze_noise_floor(self, audio: torch.Tensor) -> float
    def detect_speech_segments(self, audio: torch.Tensor) -> List[Tuple[int, int]]

# audio_cleaner.py
class AudioCleaner:
    def remove_artifacts(self, audio: torch.Tensor, threshold: float) -> torch.Tensor
    def preserve_natural_sounds(self, audio: torch.Tensor, margins: Tuple[float, float]) -> torch.Tensor
```

### 5. Pipeline Orchestration (`src/pipeline/`)

```python
# task_executor.py
class TaskExecutor:
    def execute_task(self, task: Task) -> Any
    def handle_task_failure(self, task: Task, error: Exception) -> None

# job_manager.py
class JobManager:
    def __init__(self, config: TaskConfig)
    def run_job(self, job: Job) -> Any

# batch_executor.py
class BatchExecutor:
    def execute_batch(self, tasks: List[Task]) -> List[Any]
    def handle_batch_failure(self, batch: List[Task], errors: List[Exception]) -> List[Any]

# error_handler.py
class ErrorHandler:
    def handle_task_error(self, error: Exception, task: Task) -> Any
    def handle_job_error(self, error: Exception, job: Job) -> Any
    def handle_batch_error(self, error: Exception, batch: List[Task]) -> List[Any]
```

---

## Quality Assurance

### Testing Strategy

#### Unit Testing
```python
# Example test structure
class TestSpaCyChunker:
    def test_sentence_boundary_detection(self):
        # Test accurate sentence splitting
        
    def test_token_based_length_estimation(self):
        # Test accurate token counting
        
```

#### Integration Testing
- Full pipeline end-to-end testing


### Continuous Integration
```yaml
# .github/workflows/ci.yml
name: CI Pipeline
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          python -m spacy download en_core_web_sm
      - name: Run tests
        run: |
          # Test chunking module (lightweight)
          python scripts/run_chunker.py
          
          # Test mock pipeline (without heavy TTS models)
          python scripts/test_mock_pipeline.py
          
          # Test basic pipeline components
          python scripts/test_basic_pipeline.py
          
          # Run integration tests
          python -m pytest tests/test_integration.py -v
      - name: Lint with flake8
        run: |
          pip install flake8
          flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
```

---

## Deployment Strategy

### Environment Setup
```bash
# Production environment setup
git clone <repository>
cd tts_pipeline_enhanced
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
python scripts/download_models.py
```

### Usage Examples
```bash
# Basic usage
to be updated

# Advanced configuration
to be updated
```

*This development plan serves as the master document for implementing the Enhanced TTS Pipeline project. All team members should reference this document for architecture decisions, implementation details, and project milestones.* 