# 🎯 **TTS-Pipeline Code-Effizienz Refactoring Plan 4**

## **Executive Summary**

**Ziel**: Aufteilung Monolithische Mauptmodule/Klassen die mehrere Verantwortlichkeiten haben, in kleinere Submodule mit kleinerer Verantwortlichkeit. Dabei auch leichte Verschlankung durch Kommentar-Reduzierung.

---

## **📊 Aktueller Zustand**

### **Problematische Datei**
- `job_manager.py`: **804 Zeilen** 

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


### **2.1 JobManager aufteilen**

'src/pipeline/job_manager.py' aufteilen in neue Struktur:

**Neue Struktur:**
```
src/pipeline/job_manager/
├── job_manager.py               
├── user_interaction.py         
├── execution_planner.py         
└── config_validator.py          
```

**src/pipeline/job_manager/job_manager.py:**
Core job discovery (200 Zeilen)
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

**src/pipeline/job_manager/execution_planner.py:**
resolve_execution_plan() (300 Zeilen)
```python
class ExecutionPlanner:
    def resolve_execution_plan(self, args, config_files=None) -> ExecutionPlan
    def validate_execution_plan(self, plan) -> bool
    def print_execution_summary(self, plan) -> None
```
**src/pipeline/job_manager/user_interaction.py:**
prompt_user_selection() (150 Zeilen)
...
**src/pipeline/job_manager/config_validator.py:**
_validate_mixed_configurations() (100 Zeilen)
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
