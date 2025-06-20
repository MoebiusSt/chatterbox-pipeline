# 🎯 **TTS-Pipeline Code-Effizienz Refactoring Plan 4**

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
# Beispiel in src/pipeline/job_manager/execution_planner.py:
from ...utils.config_manager import ConfigManager      # 3 Ebenen hoch zu src/
from ...utils.file_manager import FileManager

# BESTEHENDE MODULE: Verwende bestehende Pattern
from utils.file_manager import FileManager             # Funktioniert bereits
from pipeline.task_executor import TaskExecutor
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
# src/pipeline/job_manager/__init__.py
from .job_manager import JobManager
from .execution_planner import ExecutionPlanner
# Legacy-Kompatibilität für bestehende Imports
```

---

## **📊 Aktueller Zustand**

### **Problematische Datei**
- `job_manager.py`: **804 Zeilen** 


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


### **2.2 Imports: Anwendung der universellen Strategie**

**Nach der Aufteilung SOFORT prüfen:**

```python
# ✅ RICHTIG - Neue JobManager Submodule (3 Ebenen hoch zu src/):
# src/pipeline/job_manager/execution_planner.py:
from ...utils.config_manager import ConfigManager
from ...utils.file_manager import FileManager, TaskConfig

# ✅ RICHTIG - Neue JobManager (2 Ebenen hoch zu src/):
# src/pipeline/job_manager/job_manager.py:
from ..task_executor import TaskExecutor
from .execution_planner import ExecutionPlanner

# ✅ RICHTIG - Export in __init__.py:
# src/pipeline/job_manager/__init__.py:
from .job_manager import JobManager
from .execution_planner import ExecutionPlanner
```


### **2.3 Funktionstest mit universeller Import-Strategie**

**Test-Reihenfolge:**
1. **Import-Test**: `PYTHONPATH=$(pwd)/src python3 -c "from pipeline.job_manager import JobManager"`
2. **Pipeline-Test**: `cd $(pwd) && PYTHONPATH=./src python3 src/main.py --mode new`
3. **Recovery-Test**: `cd $(pwd) && PYTHONPATH=./src python3 src/main.py --mode last`
4. **Script-Test**: `source venv/bin/activate && PYTHONPATH=$(pwd)/src python3 scripts/test_task_system.py`
5. **Fehler aufwicklung** Wenn Error oder Laufzeitabbrücke auftreten nach der Aufteilung des Moduls, verwende zum Debuggen als Referenz die Backupup-Datei 'src/pipeline/job_manager_backup-before-refactor.py', um zu sehen wie es vor der Aufteilung funktionierte.
