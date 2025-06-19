# Contributing to Enhanced TTS Pipeline

Vielen Dank für dein Interesse, zu diesem Projekt beizutragen! Diese Anleitung hilft dir, effektiv mitzumachen.

## Entwicklungsumgebung einrichten

### 1. Repository forken und klonen
```bash
# Fork das Repository auf GitHub, dann:
git clone https://github.com/YOUR_USERNAME/tts_pipeline_enhanced.git
cd tts_pipeline_enhanced
```

### Environment Setup
```bash
# Production environment setup
git clone <repository>
cd tts_pipeline_enhanced
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
python scripts/download_models.py
```

### 2. Entwicklungsumgebung vorbereiten
```bash
# Virtuelle Umgebung erstellen (ERFORDERLICH)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows


# Entwicklungsabhängigkeiten installieren
pip install -r requirements.txt
pip install -r dev-requirements.txt  # falls vorhanden
python -m spacy download en_core_web_sm

# Zusätzliche Tools für Entwicklung
pip install black isort flake8 mypy pytest pytest-cov
```

### 3. Pre-commit Hooks (empfohlen)
```bash
pip install pre-commit
pre-commit install
```

## Code-Standards

### Code-Formatierung
Wir verwenden **Black** für einheitliche Code-Formatierung:
```bash
# Code automatisch formatieren
black src/ scripts/

# Nur prüfen (ohne Änderungen)
black --check src/ scripts/
```

### Import-Sortierung
Verwende **isort** für konsistente Import-Reihenfolge:
```bash
# Imports sortieren
isort src/ scripts/

# Nur prüfen
isort --check-only src/ scripts/
```

### Linting
Verwende **flake8** für Code-Qualitätsprüfung:
```bash
flake8 src/ scripts/
```

### Type Hints
Verwende Type Hints, wo möglich:
```python
def process_chunk(chunk: TextChunk, params: Dict[str, Any]) -> AudioCandidate:
    ...
```

## Testing

### Tests ausführen
```bash
# Einfache Tests (ohne TTS)
python scripts/run_chunker.py
python scripts/test_basic_pipeline.py

# Mit pytest (falls verfügbar)
pytest tests/
```

### Neue Tests schreiben
- Füge Tests in das `tests/` Verzeichnis hinzu
- Verwende pytest-Konventionen

## Projektstruktur verstehen

```
src/
├── chunking/       # Text-Segmentierung
├── generation/     # Audio-Generierung  
├── validation/     # Qualitäts-Validierung
├── postprocessing/ # Audio-Verarbeitung
├── preproccessor/  # Text-Vorarbeiten
├── pipeline/       # Pipeline-Orchestrierung
└── utils/          # Hilfsfunktionen
```

## Contribution-Workflow

### 1. Tests und Qualitätsprüfung
```bash
# Code formatieren
black src/ scripts/
isort src/ scripts/

# Linting
flake8 src/ scripts/
```

### 5. Pull Request erstellen
- Beschreibe die Änderungen ausführlich
- Verweise auf relevante Issues
- Füge Screenshots/Logs hinzu, falls relevant


## Spezifische Bereiche

### TTS-Integration
- Verwende einheitliche Parameter-Reihenfolge: `exaggeration, cfg_weight, temperature`

### Debugging-Tipps
```bash
# Verbose Logging aktivieren
export PYTHONPATH=src:$PYTHONPATH
python -m logging.basicConfig level=DEBUG src/main.py