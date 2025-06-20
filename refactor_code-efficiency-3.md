# 🎯 **TTS-Pipeline Code-Effizienz Refactoring Plan 3**

## **Executive Summary**

**Ziel**: Aufteilung Monolithische Mauptmodule/Klassen die mehrere Verantwortlichkeiten haben, in kleinere Submodule mit kleinerer Verantwortlichkeit. Dabei auch leichte Verschlankung durch Kommentar-Reduzierung.

---

## **🚀 UNIVERSELLE IMPORT-STRATEGIE**

**Problemstellung**: Inkonsistente Import-Patterns führen zu `ModuleNotFoundError` beim Refactoring.

### **Lösung: Einheitliche Import-Regeln**

**1. Für Scripts (`scripts/`):**
```bash
# Terminal-Ausführung IMMER mit PYTHONPATH:
PYTHONPATH=/path/to/project/src python3 script_name.py

# Oder in Script mit aktiviertem venv:
source venv/bin/activate && PYTHONPATH=$(pwd)/src python3 scripts/script_name.py
```

**2. Für Source-Module (`src/`):**
```python
# NEUE SUBMODULE: Verwende relative Imports zu src/ Ebene
# Beispiel in src/utils/file_manager/io_handlers/chunk_io.py:
from ...chunking.spacy_chunker import SpaCyChunker    # 3 Ebenen hoch zu src/
from ...validation.whisper_validator import WhisperValidator

# BESTEHENDE MODULE: Verwende bestehende Pattern
from utils.file_manager import FileManager           # Funktioniert bereits
from validation.whisper_validator import WhisperValidator
```

**3. Test-Strategie:**
```bash
# Import-Test IMMER mit PYTHONPATH:
PYTHONPATH=/path/to/project/src python3 -c "from module import Class"

# Pipeline-Test:
cd /path/to/project && PYTHONPATH=./src python3 src/main.py --mode new
```

**4. __init__.py Export-Regel:**
```python
# Neue Submodule MÜSSEN Hauptklassen exportieren:
# src/utils/file_manager/__init__.py
from .file_manager import FileManager
from .state_analyzer import StateAnalyzer
# Legacy-Kompatibilität:
from ...utils.file_manager import OriginalClass  # Falls nötig
```

---

## **📊 Aktueller Zustand**

### **Problematische Datei **
- `file_manager.py`: **1072 Zeilen**  


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


### **2.2 Imports: Anwendung der universellen Strategie**

**Nach der Aufteilung SOFORT prüfen:**

```python
# ✅ RICHTIG - Neue IO Handler (3 Ebenen hoch zu src/):
# src/utils/file_manager/io_handlers/chunk_io.py:
from ...chunking.spacy_chunker import SpaCyChunker
from ...validation.whisper_validator import WhisperValidator

# ✅ RICHTIG - Neue FileManager (2 Ebenen hoch zu src/):
# src/utils/file_manager/file_manager.py:
from ..config_manager import ConfigManager
from .io_handlers.chunk_io import ChunkIOHandler

# ✅ RICHTIG - Export in __init__.py:
# src/utils/file_manager/__init__.py:
from .file_manager import FileManager
from .state_analyzer import StateAnalyzer
```


### **2.3 Funktionstest mit universeller Import-Strategie**

**Test-Reihenfolge:**
1. **Import-Test**: `PYTHONPATH=$(pwd)/src python3 -c "from utils.file_manager import FileManager"`
2. **Pipeline-Test**: `cd $(pwd) && PYTHONPATH=./src python3 src/main.py --mode new`
3. **Recovery-Test**: `cd $(pwd) && PYTHONPATH=./src python3 src/main.py --mode last`
4. **Script-Test**: `source venv/bin/activate && PYTHONPATH=$(pwd)/src python3 scripts/test_basic_pipeline.py`
5. **Fehler aufwicklung** Wenn Error oder Laufzeitabbrücke auftreten nach der Aufteilung des Moduls, verwende zum Debuggen als Referenz die Backupup-Datei '/src/utils/file_manager_backup-before-refactor.py', um zu sehen wie es vor der Aufteilung funktionierte.
