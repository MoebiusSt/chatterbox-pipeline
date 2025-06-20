# 🎯 **TTS-Pipeline Code-Effizienz Refactoring Plan 7**

## **Executive Summary**

**Ziel**: **sys.path.append() Hacks** überprüfen und ggf. fixen
---

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
