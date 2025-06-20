# 🎯 **TTS-Pipeline Code-Effizienz Refactoring Plan 1**

## **Executive Summary**

**Ziel**: Leichte Verschlankung durch Kommentar-Reduzierung in der gesammten Code-Basis

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

Suche in allen .py-Dateien in den Ordnern:
/src/chunking
/src/generation
/src/pipeline
/src/preprocessing
/src/utils
/src/Validation

Finde allzu selbsterklärende Code Kommentare oder docstrings die durch den sprechenden umgebenden Code sowieso klar sein sollten. Behalte gute, erklärende Kommentar, aber wenn möglich kürze diese.

Finde auch Doc-Strings wenn diese eine Liste der zu übergebenden Argumente enthalten. Meist sind diese durch sprechende Property-Namen schon in der Definition klar genug. Die "Args:" sind also meist entfernbar. Die "Returns:" aber behalten!

```python
def rank_candidates(
        self,
        candidates: List[AudioCandidate],
        validation_results: List[ValidationResult],
        match_results: Optional[List[MatchResult]] = None,
        expected_durations: Optional[List[float]] = None,
    ) -> List[Tuple[AudioCandidate, QualityScore]]:
        """
        Rank multiple candidates by quality score.
        
        Candidates are sorted by overall_score (descending). 
        In case of tied scores, shorter audio duration is preferred as tie-breaker.

        Args:
            candidates: List of audio candidates
            validation_results: List of validation results
            match_results: Optional list of fuzzy match results
            expected_durations: Optional list of expected durations

        Returns:
            List of (candidate, score) tuples, sorted by score (best first)
        """
```

**Beispiele zum Entfernen redundanter Code-Kommentare:**
```python
# REDUNDANTE BEISPIELE (löschen):
def get_input_text(self) -> str:
    """Load input text file."""  # ← offensichtlich!
    
def save_chunks(self, chunks: List[TextChunk]) -> bool:
    """Save text chunks to files."""  # ← offensichtlich!

class ScoringStrategy(Enum):
    """Scoring strategy for combining scores."""

class QualityScore:
    """Comprehensive quality score for an audio candidate."""

# BEHALTEN BEISPIELE (wertvoll):
def _fallback_split_long_sentence(self, sentence: Span, max_limit: int) -> List[str]:
    """Attempts to split a very long sentence ONCE at a good delimiter..."""  # ← erklärt Algorithmus

class QualityScorer:
    """
    Evaluates and scores audio candidates based on multiple quality metrics.
    Used for selecting the best candidate from multiple generations.
    """ # ← erklärt Algorithmus

```
