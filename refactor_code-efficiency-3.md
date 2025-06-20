# 🎯 **TTS-Pipeline Code-Effizienz Refactoring Plan 3**

## **Executive Summary**

**Ziel**: Aufteilung Monolithische Mauptmodule/Klassen die mehrere Verantwortlichkeiten haben, in kleinere Submodule mit kleinerer Verantwortlichkeit. Dabei auch leichte Verschlankung durch Kommentar-Reduzierung.

---

## **📊 Aktueller Zustand**

### **Problematische Datei **
- `file_manager.py`: **1072 Zeilen**  

### **Identifizierte Probleme**
4. **Redundante Kommentare**: Selbsterklärende oder allzu verbose Docstrings oder Kommentare enthalten? Dann reduzieren
5. **Monolithische Klassen**: Mehrere Verantwortlichkeiten pro Klasse

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


### **2.1 FileManager aufteilen**

'src/utils/file_manager.py' aufteilen in neue Struktur:

**Neue Struktur:**
```
src/utils/file_manager/
├── file_manager.py               
├── io_handlers/
│   ├── __init__.py
│   ├── chunk_io.py              
│   ├── candidate_io.py          
│   ├── whisper_io.py            
│   ├── metrics_io.py            
│   └── final_audio_io.py         
├── state_analyzer.py             
└── validation_helpers.py         
```

**src/utils/file_manager/file_manager.py:**
Core API + delegation (200 Zeilen)
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

**src/utils/file_manager/io_handlers/chunk_io.py:** 
save/get_chunks() (120 Zeilen)
...

**src/utils/file_manager/io_handlers/candidate_io.py:** 
save/get_candidates()
Also, move input-output operations from "/src/genertation/candidatemanager.py" to here into file_manager
```python
class CandidateIOHandler:
    def save_candidates(self, chunk_idx, candidates, overwrite_existing=False) -> bool
    def get_candidates(self, chunk_idx=None) -> Dict[int, List[AudioCandidate]]
    def _save_candidates_to_disk(self, candidates, chunk_index, sample_rate) -> List[str]  # von candidate_manager! Nicht von file_manager
    def _remove_corrupt_candidate(self, chunk_idx, candidate_idx) -> bool
```

**src/utils/file_manager/io_handlers/whisper_io.py:** 
save/get_whisper() + migration (150 Zeilen)
...
**src/utils/file_manager/io_handlers/metrics_io.py:** 
save/get_metrics() (80 Zeilen)
...
**src/utils/file_manager/io_handlers/final_audio_io.py:** 
save/get_final_audio() (100 Zeilen)
...
**src/utils/file_manager/state_analyzer.py:** 
analyze_task_state() (200 Zeilen)
...
**src/utils/file_manager/validation_helpers.py:** 
_audio_file_exists(), _remove_stale_* (120 Zeilen)
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
