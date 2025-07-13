# Model Cache System - Explanation

## The Problem

Every time `cbpipe.py` is called, the message "cache miss" is displayed because ChatterboxTTS must be reloaded on each new run. This is **normal behavior**.

## Why Cache Miss on Every Run?

### 1. **In-Memory Cache**
The cache only works within a Python process. Each new call to `cbpipe.py` starts a new process with an empty cache.

### 2. **Complex Models**
ChatterboxTTS is a composite model (TTS + Voice Encoder + Text Processor) that cannot be simply serialized to disk.

### 3. **HuggingFace Cache Still Works**
The model is **not** downloaded again. The model files are already stored in `~/.cache/huggingface/`. Only the initialization takes time.

## Cache Behavior

### ✅ Cache Hit (within a run)
```
# First call in a run
🔄 Loading ChatterboxTTS model for device: cuda (cache miss)
✅ Model loaded in 8.2s and cached for future use in this session

# Second call in the same run
♻️ Using cached ChatterboxTTS model for device: cuda (cache hit, originally loaded in 8.2s)
```

### ❌ Cache Miss (new run)
```
# New program start
🔄 Loading ChatterboxTTS model for device: cuda (cache miss)
✅ Model loaded in 8.2s and cached for future use in this session
```

## Optimization Strategies

### 1. **Multiple Tasks in One Run**
```bash
# Bad: Multiple separate runs
python src/cbpipe.py task1.yaml
python src/cbpipe.py task2.yaml
python src/cbpipe.py task3.yaml

# Better: All tasks in one run
python src/cbpipe.py task1.yaml task2.yaml task3.yaml
```

### 2. **Use Job System**
```bash
# All tasks of a job in one run
python src/cbpipe.py --mode all

# Process all existing tasks
python src/cbpipe.py --job "myjob*" --mode all
```

### 3. **Accept Loading Time**
8-12 seconds loading time is normal for a 797M parameter model.

## Commands

### Show Cache Explanation
```bash
python src/cbpipe.py --explain-cache
```

### Run Cache Test
```bash
python scripts/test_model_cache.py
```

### Cache Info in Code
```python
from generation.model_cache import ChatterboxModelCache

# Cache information
info = ChatterboxModelCache.get_cache_info()
print(info)

# Cache explanation
ChatterboxModelCache.explain_cache_behavior()
```

## Technical Details

### Cache Implementation
```python
class ChatterboxModelCache:
    _model_cache: Dict[str, Any] = {}          # In-Memory cache
    _load_times: Dict[str, float] = {}         # Loading time tracking
    
    # Cache only works within a process
    # Persistent caching is not possible due to model complexity
```

### Why No Persistent Cache?
1. **Threading Components**: Models contain non-serializable objects
2. **Complex Dependencies**: Various submodels with different states
3. **Version Issues**: Cache files would not be compatible between Python versions
4. **Storage Space**: Persistent cache would require ~5GB per model

## Conclusion

**Cache miss on every run is normal behavior.** The 8-12 seconds loading time is the normal price for a state-of-the-art TTS system. The HuggingFace cache system already prevents re-downloading the models.

**Optimization:** Use the job system to process multiple tasks in one run. 