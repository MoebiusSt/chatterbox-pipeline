# 🎯 **TTS-Pipeline Code-Effizienz Refactoring Plan 1**

## **Executive Summary**

**Ziel**: Leichte Verschlankung durch Kommentar-Reduzierung

---

### **Identifizierte Probleme**

1. **Recovery System Kommentare**: ~200 Zeilen Legacy-Dokumentation
2. **Redundante Kommentare**: Selbsterklärende Docstrings

---

## ** PHASE 1:**

### **1.1 Recovery System Kommentare entfernen **

**Betroffene Dateien:**
- `src/validation/whisper_validator.py` (Lines 332-370: 38 Zeilen)
- `src/chunking/spacy_chunker.py` (Recovery warnings)

### **1.2 Redundante Kommentare entfernen**

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
