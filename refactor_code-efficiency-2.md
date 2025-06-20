# 🎯 **TTS-Pipeline Code-Effizienz Refactoring Plan 2**

## **Executive Summary**

**Ziel**: Aufteilung Monolithische Mauptmodule/Klassen die mehrere Verantwortlichkeiten haben, in kleinere Submodule mit kleinerer Verantwortlichkeit. Dabei auch leichte Verschlankung durch Kommentar-Reduzierung.

---

## **📊 Aktueller Zustand**

### **Problematische Datei**
- `task_executor.py`: **1434 Zeilen** 

---

## ** PHASE 1: Verschlankung**

### **1.0 venv **
source venv/bin/activate

### **1.1 Redundante Kommentare entfernen**

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

Prüfe nach jedem Schritt:
1. **Import-Pfade**: Alle müssen korrekt sein 
2. **Circular Imports**: Bei neuen Strukturen möglich


### **2.1 TaskExecutor aufteilen**

'src/pipeline/task_executor.py' aufteilen in neue Struktur

```
src/pipeline/task_executor/
├── task_executor.py              
├── stage_handlers/
│   ├── __init__.py
│   ├── preprocessing_handler.py  
│   ├── generation_handler.py      
│   ├── validation_handler.py      
│   └── assembly_handler.py        
└── retry_logic.py                 
```

**Aufteilung Details:**

**src/pipeline/task_executor/task_executor.py (~300 Zeilen):**
Hauptlogik: execute_task(), stage coordination (300 Zeilen)
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

**src/pipeline/task_executor/stage_handlers/generation_handler.py (~200 Zeilen):**
execute_generation() (200 Zeilen)
```python
class GenerationHandler:
    def __init__(self, file_manager, config, tts_generator)
    def execute_generation(self) -> bool
    def _generate_candidates_for_chunk(self, chunk) -> List[AudioCandidate]
    def _generate_missing_candidates(self, chunk, missing_indices) -> List[AudioCandidate]
```
**src/pipeline/task_executor/stage_handlers/preprocessing_handler.py**
execute_preprocessing() (80 Zeilen)
...
**src/pipeline/task_executor/stage_handlers/validation_handler.py**
execute_validation() (300 Zeilen)
...
**src/pipeline/task_executor/stage_handlers/assembly_handler.py**
execute_assembly() (150 Zeilen)
...
**src/pipeline/task_executor/stage_handlers/retry_logic.py**
_generate_retry_candidates() (200 Zeilen)
...


### **2.2 Imports.Inkonsistente Patterns prüfen:**

Hat der neue Code nach der Aufteilung des Moduls inkosistente Import Patterns? 
Als Beispiel:

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


### **2.3 Funktionstest**

**Test-Reihenfolge:**
1. **Neue Abhängigkeiten**: Prüfe: Andere Stellen in der Code-Basis welche Funktionen des aufgeteilten Moduls verwenden - verwenden diese nun die neue Submodule?
2. **Import-Test**: Beispielsweise `python -c "from src.main import main"`
3. **Pipeline-Test**: `python src/main.py --mode new` (default-job ausführen)
4. **Recovery-Test**: Vorhandene Task ausführen `python src/main.py --mode last` (letzten default-job nochmal prüfen lassen)
5. **Fehler aufwicklung** Wenn Error oder Laufzeitabbrücke auftreten nach der Aufteilung des Moduls, verwende zum Debuggen als Referenz die Backupup-Datei 'src/pipeline/task_executor_backup-before-refactor.py', um zu sehen wie es vor der Aufteilung funktionierte.
