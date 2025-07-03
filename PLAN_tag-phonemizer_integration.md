# Plan: pygoruut & deepphoneme Language Tag Integration

## 📚 **Repository & Project Context**

### **pygoruut Repository**
- **GitHub**: https://github.com/neurlang/pygoruut
- **Lizenz**: MIT License (kommerziell nutzbar)
- **Beschreibung**: Python wrapper für Grapheme-to-Phoneme (G2P)
- **Hugging Face Demo**: https://huggingface.co/spaces/neurlang/pygoruut

### **DeepPhonemizer Repository**
- **DeepPhonemizer**: Phoneme-to-Grapheme (P2G) Transformationen: IPA → English Approximations
https://huggingface.co/spaces/roedoejet/English-Phonemes-to-Graphemes-p2g/tree/main


### **Unser TTS Pipeline Projekt**
- **Workspace**: `~/projekte/tts_pipeline_enhanced`
- **Aktuelle Pipeline**: Text → Preprocessor → Chunker → TTS
- **Text Preprocessor**: `src/text_preprocessing/text_preprocessor.py`
- **Config System**: 3-stufige Kaskade (default_config.yaml → job.yaml → task.yaml)
- **Virtual Environment**: `venv/` (bereits vorhanden)

### **pygoruut Technische Details**

## 🎯 **Ziel**
 G2P→P2G Transformation von Language-Tagged Text-Bereichen.

### **🎯 Warum diese Transformation:**
- **Problem**: `[lang="de"]Sindelfingen[/lang]` würde von TTS falsch ausgesprochen
- **Lösung**: `Zindell'fyngn` wird von englischer TTS korrekter ausgesprochen
- **Resultat**: Deutsche Ortsnamen klingen in englischer TTS natürlich

#### **🔄 Zweistufige Transformation: OOC → IPA → Pseudo-British**
Das Kernkonzept ist eine **bidirektionale Phonem-Konversion** in 2 Schritten:
1. **Schritt 1 - G2P (Grapheme-to-Phoneme)**: Out-of-Context Text → IPA
2. **Schritt 2 - P2G (Phoneme-to-Grapheme)**: IPA → DeepPhonemizer(Pseudo-British Interpretation)

#### **Transformation Flow**
```
1. Parse Tags: [lang="de"]München[/lang] → ("de", "München")
2. G2P: "München" → "mˈyːnçən" (pygoruut)
3. IPA Fix: "mˈyːnçən" → "mɹoncen" (custom phoneme-mapping.yaml for finetuning)
4. P2G: "mɹoncen" → "moncen" (DeepPhonemizer)
5. Replace: "Visit moncen today!"
```

### **Sprach-Support**
- **ISO 639 Codes**: 80+ Sprachen per pygoruut (de, en, fr, es, ja, zh, etc.)
- **Non-ISO Dialekte**: EnglishBritish, EnglishAmerican, VietnameseCentral, etc.
- **Vollständige Liste**: Via `PygoruutLanguages().get_all_supported_languages()`

## **Installation & Dependencies**
```bash
# Virtual Environment aktivieren (IMMER ZUERST!)
source venv/bin/activate

# pygoruut installieren
pip install pygoruut

# DeepPhonemizer installieren (für P2G)
pip install panphon ipatok nltk
pip install git+https://github.com/roedoejet/DeepPhonemizer.git

# DeepPhonemizer Model wird automatisch in ~/.deepphoneme/ heruntergeladen

# requirements.txt erweitern
echo "pygoruut" >> requirements.txt
echo "panphon" >> requirements.txt
echo "ipatok" >> requirements.txt
echo "nltk" >> requirements.txt
echo "git+https://github.com/roedoejet/DeepPhonemizer.git" >> requirements.txt
```

### **Projektstruktur**
```
tts_pipeline_enhanced/
├── config/
│   ├── default_config.yaml              # Erweiterte Config
│   └── [NEU] phoneme_mappings.yaml      # Sprachspezifische Mapping-Fixes
└── src/
    ├── text_preprocessing/
    │   ├── text_preprocessor.py         # Hauptklasse (erweitert)
    │   └── [NEU] language_tag_processor.py  # Neue Klasse
    
# DeepPhonemizer Model wird automatisch in ~/.deepphoneme/ verwaltet
```

## 🏗️ **Implementierung**

### **Deliverable 1: Language Tag Processor**
- **Datei**: `src/text_preprocessing/language_tag_processor.py`
- **Klasse**: `LanguageTagProcessor`
- **Aufgabe**: Language Tags finden und transformieren

### **Deliverable 2: Text Preprocessor Integration**
- **Erweitere**: `text_preprocessor.py` 
- **Config-Option**: `process_language_tags: true`
- **Delegation**: Ruft `LanguageTagProcessor` auf

### **Wichtige Projekt-Regeln**
- **Virtual Environment**: Immer `source venv/bin/activate` vor ersten Commands
- **Code-Verfolgung**: Nach Änderungen alle Referenzen proaktiv überprüfen
- **myPy Type Checking**: Präventive Validierung vor Tests

## 💻 **Core **

### **LanguageTagProcessor Class**
```python
import re
import unicodedata
import ipatok
import panphon.distance
import torch
from typing import List, Dict, NamedTuple
from pygoruut.pygoruut import Pygoruut
from dp.phonemizer import Phonemizer

class LanguageTag(NamedTuple):
    original: str
    language: str
    text: str

class LanguageTagProcessor:
    def __init__(self, config):
        self.pygoruut = Pygoruut(writeable_bin_dir='')
        
        # Load DeepPhonemizer model (automatically managed in ~/.deepphoneme/)
        original_load = torch.load
        torch.load = lambda *args, **kwargs: original_load(*args, **kwargs, weights_only=False)
        self.deepphoneme = Phonemizer.from_checkpoint("model_step_140k.pt")
        torch.load = original_load
        
        self.load_phoneme_mappings(config.phoneme_mappings_path)
        
        # DeepPhonemizer phone inventory (required for IPA → model phone mapping)
        # Diese Phones sind vom DeepPhonemizer-Model vordefiniert und müssen so bleiben
        self.MODEL_PHONES = ['a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 's', 't', 'u', 'v', 'w', 'z', 'æ', 'ð', 'ŋ', 'ɑ', 'ɔ', 'ə', 'ɛ', 'ɝ', 'ɡ', 'ɪ', 'ɫ', 'ɹ', 'ʃ', 'ʊ', 'ʒ', 'θ']
        self.DST = panphon.distance.Distance()
    
    def find_language_tags(self, text: str) -> List[LanguageTag]:
        """Find all language tags in text"""
        pattern = r'\[lang="([^"]+)"\](.*?)\[/lang\]'
        matches = re.findall(pattern, text, re.DOTALL)
        
        tags = []
        for match in re.finditer(pattern, text, re.DOTALL):
            language, content = match.groups()
            tags.append(LanguageTag(
                original=match.group(0),
                language=language,
                text=content
            ))
        return tags
    
    def process_text(self, text: str) -> str:
        """Main processing function using optimized batch-by-language strategy"""
        # 1. Find language tags
        tags = self.find_language_tags(text)
        
        if not tags:
            return text
        
        # 2. Group tags by language (optimization from evaluation)
        language_groups = {}
        for tag in tags:
            lang = tag.language.capitalize()
            if lang not in language_groups:
                language_groups[lang] = []
            language_groups[lang].append(tag)
        
        # 3. Process each language group using batch processing
        transformations = {}
        for language, tag_group in language_groups.items():
            try:
                # Batch process all texts for this language
                texts = [tag.text for tag in tag_group]
                batch_results = self.pygoruut.phonemize_list(
                    language=language,
                    sentence_list=texts
                )
                
                # Apply custom mappings and P2G transformation
                for tag, ipa_result in zip(tag_group, batch_results):
                    ipa_text = str(ipa_result)
                    normalized_ipa = self.apply_phoneme_mappings(ipa_text, tag.language)
                    mapped_phones = self.map_phones(normalized_ipa)
                    
                    # P2G transformation
                    result = self.deepphoneme.phonemise_list([mapped_phones], lang="eng")
                    transformations[tag.original] = result.phonemes[0]
                    
            except Exception as e:
                # Fallback to individual processing for this language group
                for tag in tag_group:
                    try:
                        transformations[tag.original] = self.transform_language_tag(tag)
                    except Exception as e2:
                        print(f"Warning: Failed to process [{tag.language}]{tag.text}[/lang]: {e2}")
                        transformations[tag.original] = tag.text
        
        # 4. Replace tags in original text (maintain order)
        for tag in tags:
            if tag.original in transformations:
                text = text.replace(tag.original, transformations[tag.original])
        
        return text
    
    def transform_language_tag(self, tag: LanguageTag) -> str:
        """Transform: Text → IPA → English"""
        try:
            # Step 1: pygoruut G2P
            ipa_result = self.pygoruut.phonemize(
                language=self.language_mappings.get(tag.language, tag.language.capitalize()), 
                sentence=tag.text
            )
            ipa_text = str(ipa_result)
            
            # Step 2: Apply custom mappings
            normalized_ipa = self.apply_phoneme_mappings(ipa_text, tag.language)
            
            # Step 3: Map to DeepPhonemizer phone inventory
            mapped_phones = self.map_phones(normalized_ipa)
            
            # Step 4: DeepPhonemizer P2G
            result = self.deepphoneme.phonemise_list([mapped_phones], lang="eng")
            return result.phonemes[0]
            
        except Exception as e:
            # Fallback: return original text on error
            print(f"Warning: Failed to process [{tag.language}]{tag.text}[/lang]: {e}")
            return tag.text
    
    def apply_phoneme_mappings(self, ipa_text: str, language: str) -> str:
        """Apply custom phoneme mappings from YAML configuration"""
        # Unicode normalization
        normalized = unicodedata.normalize("NFC", ipa_text)
        
        # Apply global mappings (including R-Sound fixes)
        if hasattr(self, 'global_mappings'):
            for old_phone, new_phone in self.global_mappings.items():
                normalized = normalized.replace(old_phone, new_phone)
        
        # Apply language-specific mappings from YAML
        if hasattr(self, 'custom_mappings') and language in self.custom_mappings:
            for old_phone, new_phone in self.custom_mappings[language].items():
                normalized = normalized.replace(old_phone, new_phone)
        
        return normalized
    
    def map_phones(self, ipa_text: str) -> str:
        """Map IPA phones to DeepPhonemizer phone inventory"""
        phones = ipatok.tokenise(ipa_text)
        mapped = []
        
        for phone in phones:
            # Force r-sounds to English r
            if phone in ['r', 'ɾ', 'ʁ', 'ʀ', 'ɹ']:
                mapped.append('ɹ')
            else:
                distances = [self.DST.hamming_feature_edit_distance(phone, p) for p in self.MODEL_PHONES]
                best_phone = self.MODEL_PHONES[distances.index(min(distances))]
                mapped.append(best_phone)
        
        return "".join(mapped)
    
    def load_phoneme_mappings(self, mappings_path: str):
        """Load phoneme mappings and language mappings from YAML file"""
        try:
            import yaml
            with open(mappings_path, 'r') as f:
                data = yaml.safe_load(f)
                self.language_mappings = data.get('language_mappings', {})
                self.global_mappings = data.get('global_mappings', {})
                self.custom_mappings = data.get('custom_mappings', {})
        except Exception as e:
            print(f"Warning: Could not load phoneme mappings from {mappings_path}: {e}")
            # Fallback mappings
            self.language_mappings = {'de': 'German', 'en': 'English', 'fr': 'French', 'es': 'Spanish'}
            self.global_mappings = {'r': 'ɹ', 'ɾ': 'ɹ', 'ʁ': 'ɹ', 'ʀ': 'ɹ', 'l': 'ɫ', 'g': 'ɡ', 'ʌ': 'ə'}
            self.custom_mappings = {}
```

### **Text Preprocessor Integration**
```python
class TextPreprocessor:
    def __init__(self, config):
        self.config = config
        if config.process_language_tags:
            self.language_processor = LanguageTagProcessor(config.language_tag_processor)
    
    def preprocess_text(self, text: str) -> str:
        """Main preprocessing with language tag support"""
        # Process language tags if enabled
        if self.config.process_language_tags:
            text = self.language_processor.process_text(text)
        
        # Other preprocessing...
        return text
```

# Arbeitsablauf

## 🔬 **Phase 1: Proof-of-Concept & Testing**

### **1.1 Grundlegende pygoruut Tests** ✅ ERLEDIGT
- **Ziel**: Verhalten und Ergebnisse von pygoruut evaluieren

### **1.2 Language Mapping Tests** ✅ ERLEDIGT
- **Ziel**: Testen ob pygoruuts eingebaute ISO-639 Unterstützung ausreicht

### **1.3 Batch Processing Evaluation** ✅ ERLEDIGT
- **Entscheidung**: **"Batch by Language"** - 64x schneller als Individual Processing bei 100% Erfolgsrate. (Siehe: test_batch_processing.py)
**Wichtige Erkenntnisse**
- pygoruut's `phonemize_list()` ist extrem effizient für Batch Processing
- Sprach-Gruppierung ist der Schlüssel für optimale Performance
- Fallback auf Individual Processing ist robuste Fehlerbehandlung


## 🏗️ **Phase 2: Implementierung**

### **2.1 Configuration erweitern**
```yaml
# config/default_config.yaml
text_preprocessing:
  process_language_tags: true
  language_tag_processor:
    phoneme_mappings_path: "config/phoneme_mappings.yaml"
```

### **2.2 Language Tag Processor Klasse**

- **Neue Datei**: `src/text_preprocessing/language_tag_processor.py`
- **Klasse**: `LanguageTagProcessor`
- **Nicht** in `text_preprocessor.py` integriert → Separation of Concerns

### **2.3 Tag Parser & Validator**
- **Regex Pattern**: `r'\[lang="([^"]+)"\](.*?)\[/lang\]'`
- **Validierung**:
  - Unterstützte Sprachen via `pygoruut.get_supported_languages()`
  - Nested Tags Detection & Warnung
  - Malformed Tags Detection (and removal)

### **2.4 Core Transformation Pipeline** (✅ **Optimiert mit Batch-by-Language**)
```python
def process_language_tagged_text(text: str, target_language: str = "en") -> str:
    """
    Input:  "Hello! [lang="de"]Sindelfingen[/lang] is beautiful."
    Output: "Hello! Zindell'fyngn is beautiful."
    """
    # 1. Parse & Validate Tags
    # 2. Group Tags by Language (OPTIMIZATION)
    # 3. Batch Transform per Language: G2P(source_lang) → P2G(target_lang)
    # 4. Replace in Original Text (maintain order)
    # 5. Return Clean Text
```

### **2.5 Integration in Text Preprocessor**
- **Minimale Änderung** in `text_preprocessor.py`
- **Delegation**: `if config.process_language_tags: self.language_processor.process_text(...)`
- **Instanziierung**: `self.language_processor = LanguageTagProcessor()`

### **2.6 R-Sound-Fix implementieren

- **YAML-basierte Konfiguration**: Custom Mappings to finetune often occuring mapping errors via "phoneme_mappings.yaml"-file

ɾ (deutscher Tap) → ɫ (L-Laut) → englisches "l"
Nur r → ɹ funktioniert korrekt

Verbesserte Mapping-Strategie: 100% R-Erhaltung (5/5 Wörter)
Brot → bɹot → brot (statt blt)
größer → gɹesə → gresa (statt glesa)

```yaml
# config/phoneme_mappings.yaml

# Language code mappings (ISO → pygoruut names)
language_mappings:
  de: German
  en: English
  fr: French
  es: Spanish
  it: Italian
  pt: Portuguese
  nl: Dutch
  ru: Russian
  ja: Japanese
  zh: Chinese
  ga: Irish
  da: Danish

# Global phoneme mappings (applied to all languages)
global_mappings:
  # Standard IPA normalizations
  l: ɫ      # l → velarized l
  ʌ: ə      # ʌ → schwa
  g: ɡ      # g → voiced velar stop
  
  # R-Sound preservation (bewährt - applies to all languages)
  ʁ: ɹ      # uvular r → English r
  ɾ: ɹ      # tap r → English r
  ʀ: ɹ      # trill r → English r
  r: ɹ      # any remaining r → English r

# Language-specific custom mappings
custom_mappings:
  de:  # German
    ʏ: ɪ    # ü-Laut → i-Laut
    ø: e    # ö-Laut → e-Laut
    ç: k    # ch-Laut → k-Laut (optional)
  fr:  # French  
    # French-specific mappings (R-fix already global)
    œ: e    # œ → e
  es:  # Spanish
    # Spanish-specific mappings (R-fix already global)
    β: b    # β → b
```

### **2.6.1 Mapping Validation System**
- **Schema-Validation**: Automatische Überprüfung der YAML-Dateien
- **Konflikt-Erkennung**: Widersprüchliche Mappings zwischen Dateien



## 🧪 **Phase 3: Testing & Validation**

### **3.1 Unit Tests**
- Tag Parsing Edge Cases
- Error Handling

### **3.2 Integration Tests**
- End-to-End Pipeline
- Chunker-Compatibility

### **3.3 User Feedback Features**
- **Detailliertes Logging in verbose-mode**:
  - Erkannte Tags
  - Transformationen
  - Fehlgeschlagene Konvertierungen
- **Validation Messages in verbose-mode**:
- **Unbekannte Sprachen**: Warnung + Original Text beibehalten
- **Malformed Tags**: Warnung + Tag entfernen
- **Nested Tags**: Warnung + Outer Tag verwenden

## 📋 **Technische Spezifikationen**

### **Supported Tag Format**
```
[lang="LANGUAGE_CODE"]Text[/lang]
```

### **Language Codes**
- **ISO 639**: `de`, `en`, `fr`, `es`, etc.
- **Non-ISO**: `EnglishBritish`, `EnglishAmerican`, etc.
- **Validation**: Via `PygoruutLanguages().get_all_supported_languages()`

### **Performance Considerations** 
- **Batch Strategy**: ✅ **"Batch by Language"** - Evaluationsergebnis: 64x schneller als Individual Processing
- **Optimierungsstrategie**: 
  - Tags nach Sprache gruppieren
  - `pygoruut.phonemize_list()` für alle Texte derselben Sprache
  - Fallback auf Individual Processing bei Batch-Fehlern
- **Caching**: KISS! Verlass auf pygoruut's und deepphoneme's eingebautes Caching
- **Lazy Loading**: pygoruut erst bei Bedarf initialisieren

## 🔄 **Workflow**

### **Config: `process_language_tags: true`**
```
Input Text → Tag Parser → Group by Language → Batch pygoruut (G2P) → Custom Mapping → DeepPhonemizer (P2G) → Clean Text → Chunker
                     ↓
         [lang="de"]München[/lang] + [lang="de"]Berlin[/lang] 
                     ↓
              Batch: ["München", "Berlin"] → ["mˈyːnçən", "bɛɾlˈiːn"]
```

### **Config: `process_language_tags: false`**
```
Input Text → [Tags ignored] → Clean Text → Chunker
```

## 📚 **Phase 4: Documentation**

### **4.1 User Guide**
- **Tag-Syntax**: Beispiele und Best Practices
- **Unterstützte Sprachen**: Liste aller ISO/Non-ISO Codes (Kurz erwähen)
- **Konfiguration**: Setup und Troubleshooting (kurz)

### **4.2 Developer Documentation**
- in "TECHNICAL_OVERVIEW.md"
- **API Reference**: `LanguageTagPhonemizer` Klasse