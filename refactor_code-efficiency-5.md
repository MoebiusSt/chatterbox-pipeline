# 🎯 **TTS-Pipeline Code-Effizienz Refactoring Plan 5**

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
# Beispiel in src/generation/batch_processor.py:
from ..utils.file_manager import FileManager           # 2 Ebenen hoch zu src/
from ..validation.whisper_validator import WhisperValidator

# BESTEHENDE MODULE: Verwende bestehende Pattern  
from utils.file_manager import FileManager             # Funktioniert bereits
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
# src/generation/__init__.py
from .candidate_manager import CandidateManager
from .batch_processor import BatchProcessor
# Legacy-Kompatibilität für bestehende Imports
```

---

## **📊 Aktueller Zustand**

### **Problematische Datei**
- `candidate_manager.py`: **640 Zeilen** 

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


### **2.1 Generation Components aufteilen**

'/src/generation/candidate_manager.py' aufteilen in neue Struktur

**Neue Struktur:**
```
src/generation/
├── candidate_manager.py         
├── batch_processor.py         
└── selection_strategies.py      
```

**/src/generation/candidate_manager.py**
nur Core logic (300 Zeilen) 

**/src/generation/batch_processor.py**
process_chunks() (120 Zeilen)

**/src/generation/selection_strategies.py**
select_best_candidate*() (150 Zeilen)

**src/utils/file_manager/io_handlers/candidate_io.py**
save_candidates_to_disk() wird bewegt zu "src/utils/file_manager/io_handlers/candidate_io.py" – Richtige Stelle für input-output! Unterhalb von save_candidates() und get_candidates() von urpsrünglich file_manager.py einordnen:
...
```python
class CandidateIOHandler:
    def save_candidates(self, chunk_idx, candidates, overwrite_existing=False) -> bool
    def get_candidates(self, chunk_idx=None) -> Dict[int, List[AudioCandidate]]
    def _save_candidates_to_disk(self, candidates, chunk_index, sample_rate) -> List[str]  # von candidate_manager! Nicht von file_manager
    def _remove_corrupt_candidate(self, chunk_idx, candidate_idx) -> bool
```

### **2.2 Imports: Anwendung der universellen Strategie**

**Nach der Aufteilung SOFORT prüfen:**

```python
# ✅ RICHTIG - Neue Generation Module (2 Ebenen hoch zu src/):
# src/generation/batch_processor.py:
from ..utils.file_manager import FileManager
from ..validation.whisper_validator import WhisperValidator

# ✅ RICHTIG - CandidateManager (bestehende Pattern beibehalten):
# src/generation/candidate_manager.py:
from utils.file_manager import AudioCandidate
from .batch_processor import BatchProcessor

# ✅ RICHTIG - Export in __init__.py:
# src/generation/__init__.py:
from .candidate_manager import CandidateManager
from .batch_processor import BatchProcessor
```


### **2.3 Funktionstest mit universeller Import-Strategie**

**Test-Reihenfolge:**
1. **Import-Test**: `PYTHONPATH=$(pwd)/src python3 -c "from generation.candidate_manager import CandidateManager"`
2. **Pipeline-Test**: `cd $(pwd) && PYTHONPATH=./src python3 src/main.py --mode new`
3. **Recovery-Test**: `cd $(pwd) && PYTHONPATH=./src python3 src/main.py --mode last`
4. **Script-Test**: `source venv/bin/activate && PYTHONPATH=$(pwd)/src python3 scripts/test_basic_pipeline.py`
5. **Fehler aufwicklung** Wenn Error oder Laufzeitabbrücke auftreten nach der Aufteilung des Moduls, verwende zum Debuggen als Referenz die Backupup-Datei 'src/generation/candidate_manager_backup-before-refactor.py', um zu sehen wie es vor der Aufteilung funktionierte.
