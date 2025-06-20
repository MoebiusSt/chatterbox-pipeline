# 🎯 **TTS-Pipeline Code-Effizienz Refactoring Plan 6**

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
# Beispiel in src/validation/transcription_io.py:
from ..utils.file_manager import FileManager           # 2 Ebenen hoch zu src/
from ..utils.audio_utils import AudioUtils

# BESTEHENDE MODULE: Verwende bestehende Pattern
from utils.file_manager import AudioCandidate          # Funktioniert bereits
from utils.audio_utils import AudioUtils
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
# src/validation/__init__.py
from .whisper_validator import WhisperValidator
from .transcription_io import TranscriptionIO
# Legacy-Kompatibilität für bestehende Imports
```

---

## **📊 Aktueller Zustand**

### **Problematische Datei**
- `whisper_validator.py`: **458 Zeilen** 

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

### **2.1 whisper_validator.py:**

Aufteilen von "src/validation/whisper_validator.py" in neue Struktur.

**Neue Struktur:**
```
src/validation/
├── whisper_validator.py         # 
├── transcription_io.py          # 
├── quality_calculator.py     # 
```

**src/validation/whisper_validator.py:**
Core Whisper logic (200 Zeilen)
```python
class WhisperValidator:
    def __init__(self, model_size, device, similarity_threshold, min_quality_score)
    def _load_model(self)
    def transcribe_audio(self, audio, sample_rate, language) -> str
    def validate_candidate(self, candidate, original_text, sample_rate) -> ValidationResult
    def batch_validate(self, candidates, original_texts, sample_rate) -> list[ValidationResult]
```

**src/validation/transcription_io.py:**
save_transcriptions_to_disk() (150 Zeilen)

**src/validation/quality_calculator.py:**
_calculate_similarity() (50 Zeilen)
_calculate_quality_score() (80 Zeilen)


### **2.2 Imports: Anwendung der universellen Strategie**

**Nach der Aufteilung SOFORT prüfen:**

```python
# ✅ RICHTIG - Neue Validation Module (2 Ebenen hoch zu src/):
# src/validation/transcription_io.py:
from ..utils.file_manager import FileManager
from ..utils.audio_utils import AudioUtils

# ✅ RICHTIG - WhisperValidator (bestehende Pattern beibehalten):
# src/validation/whisper_validator.py:
from utils.file_manager import AudioCandidate
from .transcription_io import TranscriptionIO

# ✅ RICHTIG - Export in __init__.py:
# src/validation/__init__.py:
from .whisper_validator import WhisperValidator
from .transcription_io import TranscriptionIO
```


### **2.3 Funktionstest mit universeller Import-Strategie**

**Test-Reihenfolge:**
1. **Import-Test**: `PYTHONPATH=$(pwd)/src python3 -c "from validation.whisper_validator import WhisperValidator"`
2. **Pipeline-Test**: `cd $(pwd) && PYTHONPATH=./src python3 src/main.py --mode new`
3. **Recovery-Test**: `cd $(pwd) && PYTHONPATH=./src python3 src/main.py --mode last`
4. **Script-Test**: `source venv/bin/activate && PYTHONPATH=$(pwd)/src python3 scripts/test_validation_pipeline.py`
5. **Fehler aufwicklung** Wenn Error oder Laufzeitabbrücke auftreten nach der Aufteilung des Moduls, verwende zum Debuggen als Referenz die Backupup-Datei 'src/validation/whisper_validator_backup-before-refactor.py', um zu sehen wie es vor der Aufteilung funktionierte.
