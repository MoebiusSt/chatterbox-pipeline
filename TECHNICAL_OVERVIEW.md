# Technical Overview - Enhanced TTS Pipeline

> **Quick Reference for Developers and AI Agents**

## System Architecture at a Glance

**Pipeline**: Text → Preprocessor → Chunks → Candidates → Validation → Selection → Audio Chunks → Assembly → Output  
**Language**: Python 3.8+, PyTorch, Whisper, SpaCy  
**Architecture**: Job/Task-based pipeline with cascading configuration  
**Entry Point**: `src/main.py`

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
tbd;
# Use small ai LLM to detect non-english-words or passages, detect the the language of those passages, and convert them to a pseudo english string that would help an english speaker pronounce the phonemes right. BUT DON'T(!) use IPA (International Phonetic Alphabet).
```

### 3. Text Processing (`src/chunking/`)
```python
SpaCyChunker.chunk_text(text) → List[TextChunk]
# Linguistische Segmentierung, 520-650 Zeichen pro Chunk
```

### 4. Audio Generation (`src/generation/`)
```python
TTSGenerator.generate_candidates(text, N) → List[AudioCandidate]
CandidateManager.generate_candidates_for_chunk() → GenerationResult
# N candidates per chunk with inverse parameter correlation:
# exaggeration ↑ → cfg_weight ↓, temperature ↓ (prevents "wild" parameter combinations)
```

### 5. Quality Validation (`src/validation/`)
```python
WhisperValidator.validate_candidate() → ValidationResult
FuzzyMatcher.match_texts() → MatchResult  
QualityScorer.score_candidate() → QualityScore
# Speech-to-Text Validierung + Multi-Kriterien Scoring
```

### 6. Audio Assembly (`src/generation/audio_processor.py`)
```python
AudioProcessor.concatenate_segments() → torch.Tensor
# Intelligente Audio-Verkettung mit Pausenverarbeitung
```

## Data Flow

### Input/Output Formats
- **Input**: Text file (`data/input/texts/*.txt`)
- **Output**: WAV file (`data/output/{job_name}/*.wav`)
- **Config**: YAML (`config/default_config.yaml`, `config/jobs/*.yaml`)

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

### Cascading Configuration
```yaml
# 1. Default Configuration (config/default_config.yaml)
default_config:
  generation:
    num_candidates: 3
    temperature: 0.7

# 2. Job Configuration (config/jobs/my_job.yaml)
job_config:
  generation:
    num_candidates: 5  # Override default
    temperature: 0.8   # Override default

# 3. Task Configuration (generated at runtime)
task_config:
  generation:
    temperature: 0.9  # Override job config
```

### Critical Parameters
```yaml
generation:
  num_candidates: 3              
  
validation:
  similarity_threshold: 0.80         # Quality gate: higher = stricter
  min_quality_score: 0.75        # Minimum quality score for validation
  
chunking:
  target_chunk_limit: 520       # Balance: quality vs risk of context window overload
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
src/main.py                      # Main pipeline orchestration
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
python src/main.py --job my_job
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