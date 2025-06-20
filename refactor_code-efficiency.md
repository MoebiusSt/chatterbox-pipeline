# 🎯 **TTS-Pipeline Code-Effizienz Refactoring Plan**

## **Executive Summary**

**Ziel**: Verschlankung der Code-Basis durch Kürzung redundanter, oder selbsterklärender Kommentare
**Context Window**: Auf deutlich weniger Zeilen pro Hauptmodul reduzieren  indem in Submodule aufgeteilt wird.

---

## **📊 Aktueller Zustand**

### **Problematische Dateien (> 800 Zeilen)**
- `task_executor.py`: **1434 Zeilen** 
- `file_manager.py`: **1072 Zeilen**  
- `job_manager.py`: **804 Zeilen** 
- `candidate_manager.py`: **640 Zeilen** 
- `whisper_validator.py`: **458 Zeilen** 
- `auto_editor_wrapper.py`: **416 Zeilen**

### **Identifizierte Probleme**
1. **sys.path.append() Hacks**: 16 Dateien betroffen
2. **Recovery System Kommentare**: ~200 Zeilen Legacy-Dokumentation
3. **Import-Inkonsistenzen**: Mixed `src.` prefixes in Scripts
4. **Redundante Kommentare**: Selbsterklärende Docstrings
5. **Monolithische Klassen**: Mehrere Verantwortlichkeiten pro Klasse

---

## **🏗️ PHASE 1: Verschlankung**

### **1.1 Recovery System Kommentare entfernen **

**Betroffene Dateien:**
- `src/validation/whisper_validator.py` (Lines 332-370: 38 Zeilen)
- `src/chunking/spacy_chunker.py` (Recovery warnings)

### **1.2 sys.path.append() Hacks entfernen (45 Min)**

**Betroffene Dateien (16 Files):**
```python
# Scripts (9 Files):
scripts/test_validation_pipeline.py
scripts/run_chunker.py  
scripts/test_mock_pipeline.py
scripts/test_basic_pipeline.py
scripts/check_ci.py
scripts/test_integration.py
scripts/test_task_system.py
scripts/test_regenerate_final.py
scripts/test_postprocessing_pipeline.py

# Source Files (4 Files):
src/postprocessing/audio_cleaner.py
src/postprocessing/auto_editor_wrapper.py
src/validation/quality_scorer.py
src/validation/whisper_validator.py
```

**Ersetzungspattern:**
```python
# VORHER (schlecht):
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.file_manager import AudioCandidate

# NACHHER (richtig):
from ..utils.file_manager import AudioCandidate
```

### **1.3 Import-Inkonsistenzen beheben**

**Inkonsistente Patterns gefunden:**
```python
# scripts/test_task_system.py - INKONSISTENT:
from src.pipeline.job_manager import JobManager      # ← mit src. prefix
from pipeline.task_executor import TaskExecutor     # ← ohne src. prefix
```

**Vereinheitlichen zu:**
```python
# Für Scripts: IMMER src. prefix verwenden
from src.pipeline.job_manager import JobManager
from src.pipeline.task_executor import TaskExecutor

# Für Source Files: IMMER relative imports
from ..pipeline.job_manager import JobManager
from .task_executor import TaskExecutor
```

### **1.4 Redundante Kommentare entfernen**

**Beispiele zum Entfernen:**
```python
# REDUNDANT (löschen):
def get_input_text(self) -> str:
    """Load input text file."""  # ← offensichtlich!
    
def save_chunks(self, chunks: List[TextChunk]) -> bool:
    """Save text chunks to files."""  # ← offensichtlich!

# BEHALTEN (wertvoll):
def _fallback_split_long_sentence(self, sentence: Span, max_limit: int) -> List[str]:
    """Attempts to split a very long sentence ONCE at a good delimiter..."""  # ← erklärt Algorithmus
```

---

## **PHASE 2: Strukturelles Refactoring**

### **2.1 TaskExecutor aufteilen**

**Neue Struktur:**
```
src/pipeline/task_executor/
├── task_executor.py              # Hauptlogik: execute_task(), stage coordination (300 Zeilen)
├── stage_handlers/
│   ├── __init__.py
│   ├── preprocessing_handler.py  # execute_preprocessing() (80 Zeilen)
│   ├── generation_handler.py     # execute_generation() (200 Zeilen)
│   ├── validation_handler.py     # execute_validation() (300 Zeilen)
│   └── assembly_handler.py       # execute_assembly() (150 Zeilen)
├── retry_logic.py                # _generate_retry_candidates() (200 Zeilen)
└── component_factory.py          # Lazy loading properties (100 Zeilen)
```

**Aufteilung Details:**

**task_executor.py (300 Zeilen):**
```python
class TaskExecutor:
    def __init__(self, file_manager, task_config)
    def execute_task(self) -> TaskResult
    def _execute_stages_from_state(self, task_state) -> bool
    def _detect_device(self) -> str
    
    # Properties delegieren an component_factory
    @property
    def chunker(self) -> SpaCyChunker
    @property  
    def tts_generator(self) -> TTSGenerator
    # etc.
```

**stage_handlers/generation_handler.py (200 Zeilen):**
```python
class GenerationHandler:
    def __init__(self, file_manager, config, tts_generator)
    def execute_generation(self) -> bool
    def _generate_candidates_for_chunk(self, chunk) -> List[AudioCandidate]
    def _generate_missing_candidates(self, chunk, missing_indices) -> List[AudioCandidate]
```

### **2.2 FileManager aufteilen**

**Neue Struktur:**
```
src/utils/file_manager/
├── file_manager.py               # Core API + delegation (200 Zeilen)
├── io_handlers/
│   ├── __init__.py
│   ├── chunk_io.py              # save/get_chunks() (120 Zeilen)
│   ├── candidate_io.py          # save/get_candidates() + disk_saver logic (200 Zeilen)
│   ├── whisper_io.py            # save/get_whisper() + migration (150 Zeilen)
│   ├── metrics_io.py            # save/get_metrics() (80 Zeilen)
│   └── final_audio_io.py        # save/get_final_audio() (100 Zeilen)
├── state_analyzer.py            # analyze_task_state() (200 Zeilen)
└── validation_helpers.py        # _audio_file_exists(), _remove_stale_* (120 Zeilen)
```

**file_manager.py:**
```python
class FileManager:
    def __init__(self, task_config, preloaded_config=None)
    def _find_project_root(self) -> Path
    
    # Input Operations (delegiert)
    def get_input_text(self) -> str
    def get_reference_audio(self) -> Path
    
    # Delegation to handlers
    def save_chunks(self, chunks): return self._chunk_handler.save_chunks(chunks)
    def get_chunks(self): return self._chunk_handler.get_chunks()
    # etc.
```

**io_handlers/candidate_io.py:**  
*Hier kommt die disk_saver.py Logik aus candidate_manager rein!*
```python
class CandidateIOHandler:
    def save_candidates(self, chunk_idx, candidates, overwrite_existing=False) -> bool
    def get_candidates(self, chunk_idx=None) -> Dict[int, List[AudioCandidate]]
    def _save_candidates_to_disk(self, candidates, chunk_index, sample_rate) -> List[str]  # von candidate_manager
    def _remove_corrupt_candidate(self, chunk_idx, candidate_idx) -> bool
```


### **2.3 JobManager aufteilen**

**Neue Struktur:**
```
src/pipeline/job_manager/
├── job_manager.py               # Core job discovery (200 Zeilen)
├── user_interaction.py          # prompt_user_selection() (150 Zeilen)
├── execution_planner.py         # resolve_execution_plan() (300 Zeilen)
└── config_validator.py          # _validate_mixed_configurations() (100 Zeilen)
```

**job_manager.py:**
```python
class JobManager:
    def __init__(self, config_manager)
    def is_task_config(self, config_path) -> bool
    def get_jobs(self, job_name=None) -> List[TaskConfig]
    def find_jobs_by_name(self, job_name) -> List[TaskConfig]
    def find_all_jobs(self) -> List[TaskConfig]
    def create_new_task(self, job_config) -> TaskConfig
    def parse_mode_argument(self, mode_arg) -> tuple
```

**execution_planner.py:**
```python
class ExecutionPlanner:
    def resolve_execution_plan(self, args, config_files=None) -> ExecutionPlan
    def validate_execution_plan(self, plan) -> bool
    def print_execution_summary(self, plan) -> None
```

---

## **PHASE 3: Spezialisierte Module **

### **3.1 Validation Components aufteilen**

**whisper_validator.py:**

**Neue Struktur:**
```
src/validation/
├── whisper_validator.py         # Core Whisper logic (200 Zeilen)
├── transcription_io.py          # save_transcriptions_to_disk() (150 Zeilen)
├── similarity_calculator.py     # _calculate_similarity() (50 Zeilen)
└── quality_metrics.py           # _calculate_quality_score() (80 Zeilen)
```

**whisper_validator.py:**
```python
class WhisperValidator:
    def __init__(self, model_size, device, similarity_threshold, min_quality_score)
    def _load_model(self)
    def transcribe_audio(self, audio, sample_rate, language) -> str
    def validate_candidate(self, candidate, original_text, sample_rate) -> ValidationResult
    def batch_validate(self, candidates, original_texts, sample_rate) -> list[ValidationResult]
```

### **3.2 Generation Components aufteilen**

**candidate_manager.py:**

**Neue Struktur:**
```
src/generation/
├── candidate_manager.py         # Core logic (300 Zeilen)
├── batch_processor.py          # process_chunks() (120 Zeilen)
└── selection_strategies.py      # select_best_candidate*() (150 Zeilen)

save_candidates_to_disk() MOVE TO:
src/utils/file_manager/io_handlers/candidate_io.py  # ← Richtige Stelle!
```

### **3.3 Postprocessing – NICHT STRUKTURELL ÄNDERN - WIRD ZU EINEM ANDEREN ZEITPUNKT WEITGEHEND ENTFERNT **


---

## **PHASE 4: Utility-Extraktion**

### **4.1 Common Utilities erstellen**

**Neue Module:**
```
src/utils/
├── audio_utils.py               # Audio tensor operations 
├── path_utils.py                # Path resolution, temp files 
├── validation_utils.py          # Common validation patterns 
├── logging_utils.py             # Enhanced logging helpers 
└── device_utils.py              # Device detection logic
```

**audio_utils.py:**
```python
def ensure_audio_dimensions(audio: torch.Tensor) -> torch.Tensor
def validate_audio_tensor(audio: torch.Tensor) -> bool
def resample_audio(audio: torch.Tensor, old_rate: int, new_rate: int) -> torch.Tensor
def calculate_audio_duration(audio: torch.Tensor, sample_rate: int) -> float
```

---

## **PHASE 5: Integration & Testing **

### **5.1 Import-Updates**

**Systematisch alle Import-Statements anpassen:**
```bash
# 1. Hauptmodule aktualisieren
find src/ -name "*.py" -exec sed -i 's/from task_executor import/from pipeline.task_executor import/g' {} \;

# 2. Scripts aktualisieren  
find scripts/ -name "*.py" -exec sed -i 's/from src\./from src\./g' {} \;

# 3. __init__.py Files erstellen
touch src/pipeline/task_executor/__init__.py
touch src/utils/file_manager/__init__.py
# etc.
```

### **5.2 Funktionstest**

**Test-Reihenfolge:**
1. **Import-Test**: `python -c "from src.main import main"`
2. **Pipeline-Test**: `python src/main.py --mode new` (default-job)
3. **Recovery-Test**: Vorhandene Task ausführen `python src/main.py --mode last`

---

### **Mittlere Risiken (🟡):**
1. **Import-Pfade**: Alle müssen korrekt sein
2. **Circular Imports**: Bei neuen Strukturen möglich
   - **Mitigation**: Dependency-Graph prüfen

---

## **✅ Definition of Done**

- [ ] Alle Module Zeilenumfang vermindern
- [ ] Keine sys.path.append() Hacks
- [ ] Konsistente Import-Patterns  
- [ ] Task-System funktioniert noch?