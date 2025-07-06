# Plan: pygoruut Language Tag Integration

## 📚 **Repository & Project Context**

### **pygoruut Repository**
- **GitHub**: https://github.com/neurlang/pygoruut
- **Lizenz**: MIT License (kommerziell nutzbar)
- **Beschreibung**: Python wrapper für Grapheme-to-Phoneme (G2P) und Phoneme-to-Grapheme (P2G) Transformationen
- **Hugging Face Demo**: https://huggingface.co/spaces/neurlang/pygoruut
- **Basiert auf**: gruut (https://github.com/rhasspy/gruut) - etabliertes G2P Framework

### **Unser TTS Pipeline Projekt**
- **Workspace**: `~/projekte/tts_pipeline_enhanced`
- **Aktuelle Pipeline**: Text → Preprocessor → Chunker → TTS
- **Text Preprocessor**: `src/text_preprocessing/text_preprocessor.py`
- **Config System**: 3-stufige Kaskade (default_config.yaml → job.yaml → task.yaml)
- **Virtual Environment**: `venv/` (bereits vorhanden)

### **pygoruut Technische Details**

#### **🔄 Zweistufige Transformation: OOC → IPA → Pseudo-British**

Das Kernkonzept ist eine **bidirektionale Phonem-Konversion** in 2 Schritten:

1. **Schritt 1 - G2P (Grapheme-to-Phoneme)**: Out-of-Context Text → IPA
2. **Schritt 2 - P2G (Phoneme-to-Grapheme)**: IPA → Pseudo-British Interpretation

```python
from pygoruut.pygoruut import Pygoruut, PygoruutLanguages

pygoruut = Pygoruut()

# SCHRITT 1: Deutsche Aussprache → IPA Phoneme
response = pygoruut.phonemize(language="German", sentence="Sindelfingen")
# Output: ˈzɪndl̩fɪŋ

# SCHRITT 2: IPA → Pseudo-Englische Schreibweise (reverse=True)
response = pygoruut.phonemize(language="English", sentence="ˈzɪndl̩fɪŋ", is_reverse=True)
# Output: Zindell'fyngn
```

#### **🎯 Ziel der Transformation:**
- **Problem**: `[lang="de"]Sindelfingen[/lang]` würde von TTS falsch ausgesprochen
- **Lösung**: `Zindell'fyngn` wird von englischer TTS korrekt ausgesprochen
- **Resultat**: Deutsche Ortsnamen klingen in englischer TTS natürlich

#### **🔗 Kompletter Workflow-Beispiel:**
```
INPUT:  "Visit [lang="de"]München[/lang] and [lang="fr"]Paris[/lang] today!"

STEP 1: Tag Parsing
- Erkannt: "München" (Deutsch), "Paris" (Französisch)

STEP 2: G2P Transformation (OOC → IPA)
- "München" (de) → ˈmʏnçən
- "Paris" (fr) → paˈʁi

STEP 3: P2G Transformation (IPA → Pseudo-English)
- ˈmʏnçən → "Mynchn"
- paˈʁi → "Parree"

STEP 4: Text Replacement
OUTPUT: "Visit Mynchn and Parree today!"

→ An Chunker: Tag-freier Text mit aussprechbaren Pseudo-Wörtern
```

### **Sprach-Support**
- **ISO 639 Codes**: 80+ Sprachen (de, en, fr, es, ja, zh, etc.)
- **Non-ISO Dialekte**: EnglishBritish, EnglishAmerican, VietnameseCentral, etc.
- **Vollständige Liste**: Via `PygoruutLanguages().get_all_supported_languages()`

### **Installation & Dependencies**
```bash
# Virtual Environment aktivieren (IMMER ZUERST!)
source venv/bin/activate

# pygoruut installieren
pip install pygoruut

# requirements.txt erweitern
echo "pygoruut" >> requirements.txt
```

### **Aktuelle Projektstruktur**
```
tts_pipeline_enhanced/
├── config/
│   ├── default_config.yaml          # Basis-Konfiguration
│   ├── job_configs/                 # Job-spezifische Configs
│   └── task_configs/                # Task-spezifische Configs
├── src/
│   ├── text_preprocessing/
│   │   ├── text_preprocessor.py     # Hauptklasse für Text-Preprocessing
│   │   └── [NEU] language_tag_phonemizer.py  # Geplante neue Klasse
│   ├── chunking/                    # Text Chunking
│   └── tts/                         # TTS Engine Integration
├── venv/                            # Python Virtual Environment
└── requirements.txt                 # Python Dependencies
```

### **Wichtige Projekt-Regeln**
- **Virtual Environment**: Immer `source venv/bin/activate` vor ersten Commands
- **Code-Verfolgung**: Nach Änderungen alle Referenzen proaktiv überprüfen
- **myPy Type Checking**: Präventive Validierung vor Tests


## 🎯 **Ziel**
Integration von pygoruut für G2P→P2G Transformation von Language-Tagged Text-Bereichen.

## 🔬 **Phase 1: Proof-of-Concept & Testing**

### **1.1 Grundlegende pygoruut Tests**
- **Ziel**: Verhalten und Ergebnisse von pygoruut evaluieren
- **Deliverable**: `test_pygoruut_poc.py` 
- **Tests**:
  - Einzelwort-Transformationen verschiedener Sprachen
  - Fehlerbehandlung bei unbekannten Sprachen
  - Qualität der P2G-Rückkonvertierung
  - Verhalten bei längeren Texten? (Speicher?)

### **1.2 Language Mapping Tests**
- **Ziel**: Testen ob pygoruuts eingebaute ISO-639 Unterstützung ausreicht
- **Tests**:
  - ISO-Code → Language-Name Mapping
  - Nicht-ISO Codes (EnglishBritish, etc.)
  - Edge Cases und Fallback-Strategien

### **1.3 Batch Processing Evaluation**
- **Ziel**: Optimaler Workflow für mehrere Tags in einem Text
- **Szenarien**:
  - Batch-Processing: Mehrere Sprachen in einem Call? Text mit 3-5 verschiedenen Sprach-Tags?
  - Performance: Einzeln vs. Batch vs. Grouped by Language
  - **Entscheidung**: Basierend auf Testergebnissen

## 🏗️ **Phase 2: Implementierung**

### **2.1 Configuration erweitern**
```yaml
# config/default_config.yaml
text_preprocessing:
  phonemize_language_tagged_words: true  # NEW
```

### **2.2 Language Tag Phonemizer Klasse**
- **Neue Datei**: `src/text_preprocessing/language_tag_phonemizer.py`
- **Klasse**: `LanguageTagPhonemizer`
- **Nicht** in `text_preprocessor.py` integriert → Separation of Concerns

### **2.3 Tag Parser & Validator**
- **Regex Pattern**: `r'\[lang="([^"]+)"\](.*?)\[/lang\]'`
- **Validierung**:
  - Unterstützte Sprachen via `pygoruut.get_supported_languages()`
  - Nested Tags Detection & Warnung
  - Malformed Tags Detection (and removal)

### **2.4 Core Transformation Pipeline**
```python
def process_language_tagged_text(text: str, target_language: str = "en") -> str:
    """
    Input:  "Hello! [lang="de"]Sindelfingen[/lang] is beautiful."
    Output: "Hello! Zindell'fyngn is beautiful."
    """
    # 1. Parse & Validate Tags
    # 2. Extract Tagged Regions
    # 3. Transform: G2P(source_lang) → P2G(target_lang)
    # 4. Replace in Original Text
    # 5. Return Clean Text
```

### **2.5 Integration in Text Preprocessor**
- **Minimale Änderung** in `text_preprocessor.py`
- **Delegation**: `if config.phonemize_language_tagged_words: self.phonemizer.process(...)`
- **Instanziierung**: `self.phonemizer = LanguageTagPhonemizer()`

## 🧪 **Phase 3: Testing & Validation**

### **3.1 Unit Tests**
- Tag Parsing Edge Cases
- Transformation Accuracy
- Error Handling
- Performance Benchmarks

### **3.2 Integration Tests**
- End-to-End Pipeline
- Config-basierte Aktivierung/Deaktivierung
- Chunker-Compatibility

### **3.3 User Feedback Features**
- **Detailliertes Logging**:
  - Erkannte Tags
  - Transformationen
  - Fehlgeschlagene Konvertierungen
- **Validation Messages**:
  - Unsupported Languages
  - Malformed Tags
  - Nested Tags Warnings

## 📋 **Technische Spezifikationen**

### **Supported Tag Format**
```
[lang="LANGUAGE_CODE"]Text[/lang]
```

### **Language Codes**
- **ISO 639**: `de`, `en`, `fr`, `es`, etc.
- **Non-ISO**: `EnglishBritish`, `EnglishAmerican`, etc.
- **Validation**: Via `PygoruutLanguages().get_all_supported_languages()`

### **Error Handling**
- **Unbekannte Sprachen**: Warnung + Original Text beibehalten
- **Malformed Tags**: Warnung + Tag entfernen
- **Nested Tags**: Warnung + Outer Tag verwenden

### **Performance Considerations**
- **Batch Strategy**: Basierend auf Phase 1 Ergebnissen
- **Caching**: KISS! Verlass auf pygoruut's eingebautes Caching
- **Lazy Loading**: pygoruut erst bei Bedarf initialisieren

## 🔄 **Workflow**

### **Config: `phonemize_language_tagged_words: true`**
```
Input Text → Tag Parser → Validator → Phonemizer → Clean Text → Chunker
```

### **Config: `phonemize_language_tagged_words: false`**
```
Input Text → Tag Remover → Clean Text → Chunker
```

## 📚 **Phase 4: Documentation**

### **4.1 User Guide**
- **Tag-Syntax**: Beispiele und Best Practices
- **Unterstützte Sprachen**: Liste aller ISO/Non-ISO Codes (Kurz erwähen)
- **Konfiguration**: Setup und Troubleshooting (kurz)

### **4.2 Developer Documentation**
- in "TECHNICAL_OVERVIEW.md"
- **API Reference**: `LanguageTagPhonemizer` Klasse

## 🚀 **Nächste Schritte**

1. **Phase 1**: Proof-of-Concept erstellen und evaluieren
2. **Entscheidung**: Batch-Processing-Strategie basierend auf Tests
3. **Phase 2**: Implementierung der `LanguageTagPhonemizer` Klasse
4. **Phase 3**: Testing & Integration
5. **Phase 4**: Documentation 