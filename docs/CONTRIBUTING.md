# Contributing to Enhanced TTS Pipeline

Thank you for your interest in contributing to this project! These instructions will help you to participate effectively.

## Set up development environment

### 1. Fork and clone repository
```bash
git clone https://github.com/YOUR_USERNAME/chatterbox-pipeline.git
cd chatterbox-pipeline
```

### Environment Setup
```bash
# Production environment setup
git clone <repository>
cd chatterbox-pipeline
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Prepare the development environment
```bash
# Create virtual environment (REQUIRED)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate   # Windows


# Install development dependencies
pip install -r requirements.txt
pip install -r dev-requirements.txt  # if available
python -m spacy download en_core_web_sm

# Additional Tool for development
pip install black isort flake8 mypy pytest pytest-cov
```

### 3. Pre-commit Hooks (recommended)
```bash
pip install pre-commit
pre-commit install
```

## Code Standards

### Code Formatting
We use **Black** for consistent code formatting:
```bash
# Format code automatically
black src/ scripts/

# Check only (without changes)
black --check src/ scripts/
```

### Import Sorting
Use **isort** for consistent import ordering:
```bash
# Sort imports
isort src/ scripts/

# Check only
isort --check-only src/ scripts/
```

### Linting
Use **flake8** for code quality checking:
```bash
flake8 src/ scripts/
flake8 src/ --select=E999,F821,F401,F541,F841 --statistics --count
flake8 src/ --statistics --count --ignore=E501,W503,E226,E302
```

### Type Hints
Use Type Hints where possible:
```python
def process_chunk(chunk: TextChunk, params: Dict[str, Any]) -> AudioCandidate:
    ...
```

## Testing

### Run Tests
```bash
# Simple tests (without TTS)
python scripts/run_chunker.py
python scripts/test_basic_pipeline.py

# With pytest (if available)
pytest tests/
```

### Writing New Tests
- Add tests to the `tests/` directory
- Use pytest conventions

## Understanding Project Structure

```
src/
├── chunking/       # Text segmentation
├── generation/     # Audio generation  
├── validation/     # Quality validation
├── postprocessing/ # Audio processing
├── preproccessor/  # Text preprocessing
├── pipeline/       # Pipeline orchestration
└── utils/          # Utility functions
```

## Contribution Workflow

### 1. Tests and Quality Checks
```bash
# Format code
black src/ scripts/
isort src/ scripts/

# Linting
flake8 src/ scripts/
```

### 5. Create Pull Request
- Describe changes in detail
- Reference relevant issues
- Add screenshots/logs if relevant


## Specific Areas

### TTS Integration
- Use consistent parameter order: `exaggeration, cfg_weight, temperature`

### Debugging Tips
```bash
# Enable verbose logging
export PYTHONPATH=src:$PYTHONPATH
python -m logging.basicConfig level=DEBUG src/cbpipe.py