# 🎯 **TTS-Pipeline Code-Effizienz Refactoring Plan 7**

## **Executive Summary**

**Ziel**: **sys.path.append() Hacks** überprüfen und ggf. fixen

---

## **🚀 UNIVERSELLE IMPORT-STRATEGIE**

**Problemstellung**: `sys.path.append()` Hacks und inkonsistente Import-Patterns führen zu fragilen Scripts.

### **Lösung: Einheitliche Import-Regeln**

**1. Für Scripts (`scripts/`):**
```bash
# Terminal-Ausführung IMMER mit PYTHONPATH:
PYTHONPATH=/path/to/project/src python3 script_name.py

# Oder in Script mit aktiviertem venv:
source venv/bin/activate && PYTHONPATH=$(pwd)/src python3 scripts/script_name.py
```

**2. Sys.path.append() entfernen:**
```python
# ❌ VORHER (schlecht):
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.file_manager import AudioCandidate

# ✅ NACHHER (richtig):
# Im Script: KEINE sys.path.append()!
# Terminal: PYTHONPATH=$(pwd)/src python3 scripts/script_name.py
from utils.file_manager import AudioCandidate
```

**3. Source Files bereinigen:**
```python
# ❌ VORHER (in Source-Modulen):
sys.path.append(str(Path(__file__).resolve().parents[2]))
from utils.audio_utils import AudioUtils

# ✅ NACHHER (relative imports):
from ..utils.audio_utils import AudioUtils
```

---

## **📊 Aktueller Zustand**

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

---

## **PHASE 1: Bereinigung Scripts**

### **1.1 Entferne sys.path.append() aus allen Scripts**

**Systematische Bereinigung:**
```bash
# Finde alle sys.path.append() Vorkommen:
grep -r "sys.path.append" scripts/

# Für jede gefundene Datei:
# 1. Entferne sys.path.append() Zeilen
# 2. Teste mit PYTHONPATH: 
source venv/bin/activate && PYTHONPATH=$(pwd)/src python3 scripts/SCRIPT_NAME.py
```

### **1.2 Scripts-Template erstellen**

**Standard-Script-Header:**
```python
#!/usr/bin/env python3
"""
Script description.

Usage:
    source venv/bin/activate && PYTHONPATH=$(pwd)/src python3 scripts/script_name.py
"""

# KEINE sys.path.append() mehr!
from utils.file_manager import FileManager
from pipeline.task_executor import TaskExecutor
# etc.
```

---

## **PHASE 2: Bereinigung Source Files**

### **2.1 Entferne sys.path.append() aus Source-Modulen**

**Betroffen:**
- `src/postprocessing/audio_cleaner.py`
- `src/postprocessing/auto_editor_wrapper.py` 
- `src/validation/quality_scorer.py`
- `src/validation/whisper_validator.py`

**Ersetzungspattern:**
```python
# ❌ ENTFERNEN:
sys.path.append(str(Path(__file__).resolve().parents[2]))

# ✅ ERSETZEN mit relativen Imports:
from ..utils.file_manager import AudioCandidate
from ..utils.audio_utils import AudioUtils
```

---

## **PHASE 3: Funktionstest mit universeller Import-Strategie**

### **3.1 Test-Reihenfolge:**

**Scripts testen:**
```bash
# Für jedes bereinigte Script:
source venv/bin/activate && PYTHONPATH=$(pwd)/src python3 scripts/SCRIPT_NAME.py

# Batch-Test aller Scripts:
for script in scripts/*.py; do
    echo "Testing $script..."
    PYTHONPATH=$(pwd)/src python3 "$script" || echo "FAILED: $script"
done
```

**Source-Module testen:**
```bash
# Import-Tests:
PYTHONPATH=$(pwd)/src python3 -c "from postprocessing.audio_cleaner import AudioCleaner"
PYTHONPATH=$(pwd)/src python3 -c "from validation.quality_scorer import QualityScorer"

# Pipeline-Test:
cd $(pwd) && PYTHONPATH=./src python3 src/main.py --mode new
```

### **3.2 Dokumentation aktualisieren**

**README.md ergänzen:**
```markdown
## Entwicklung

### Scripts ausführen:
```bash
source venv/bin/activate && PYTHONPATH=$(pwd)/src python3 scripts/script_name.py
```

### Tests ausführen:
```bash
PYTHONPATH=$(pwd)/src python3 -m pytest tests/
```
```

---

## **PHASE 4: Validierung**

### **4.1 Finale Überprüfung:**

```bash
# 1. Keine sys.path.append() mehr vorhanden:
grep -r "sys.path.append" src/ scripts/ || echo "✅ Alle bereinigt"

# 2. Alle Scripts funktionsfähig:
for script in scripts/*.py; do PYTHONPATH=$(pwd)/src python3 "$script" --help 2>/dev/null || echo "CHECK: $script"; done

# 3. Pipeline funktionsfähig:
PYTHONPATH=$(pwd)/src python3 src/main.py --mode new --job default

# 4. Import-Tests bestehen:
PYTHONPATH=$(pwd)/src python3 -c "import sys; [print(f'✅ {m}') for m in ['utils.file_manager', 'pipeline.task_executor', 'validation.whisper_validator'] if __import__(m)]"
```
