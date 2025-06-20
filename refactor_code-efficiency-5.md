# 🎯 **TTS-Pipeline Code-Effizienz Refactoring Plan 5**

## **Executive Summary**

**Ziel**: Aufteilung Monolithische Mauptmodule/Klassen die mehrere Verantwortlichkeiten haben, in kleinere Submodule mit kleinerer Verantwortlichkeit. Dabei auch leichte Verschlankung durch Kommentar-Reduzierung.

---

## **📊 Aktueller Zustand**

### **Problematische Datei**
- `candidate_manager.py`: **640 Zeilen** 


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
