# Plan: Speaker-System Refactoring

## 📋 Überblick

**Ziel**: Implementierung eines Multi-Speaker-Systems für wechselnde Sprecherrollen im TTS-Pipeline mit dynamischem reference_audio und TTS-Parameter Switching.

**Kernprinzipien**:
- Speaker 0 = Default-Speaker (kein Markup erforderlich)
- Nur Start-Tags `<speaker:id>` erforderlich, keine End-Tags
- Sprecherwechsel hat höchste Chunking-Priorität
- Vollständig kaskadierendes Konfigurationssystem
- Serielle Verarbeitung (Thread-Safe)

---

## 🏗️ Architektur-Änderungen

### Alte Struktur
```yaml
input:
  reference_audio: single_voice.wav
generation:
  tts_params: {...}
  conservative_candidate: {...}
```

### Neue Struktur
```yaml
input:
  text_file: document.txt
generation:
  num_candidates: 3
  max_retries: 2
  speakers:
    - id: speaker_0        # Default speaker
      reference_audio: voice1.wav
      tts_params: {...}
      conservative_candidate: {...}
    - id: narrator
      reference_audio: voice2.wav
      tts_params: {...}
```

---

## 📝 Text-Markup System

### Markup-Syntax
```text
Default speaker text ohne Markup.

<speaker:narrator>
Ab hier spricht der Narrator.
Auch dieser Text ist vom Narrator.

<speaker:character>
"Hallo!", sagte der Charakter.

<speaker:0>
Zurück zum Default-Speaker.
```

### Markup-Regeln
- `<speaker:id>` wechselt zu Speaker mit entsprechender ID
- `<speaker:0>` oder `<speaker:speaker_0>` zurück zum Default
- `<speaker:reset>` als Alternative für Default
- Unbekannte IDs → Warnung + Fallback auf Speaker 0
- Syntax-Fehler → Ignorieren + Warnung

---

## 🔧 Chunker-Änderungen

### Neue Prioritäten (in Reihenfolge)
1. **Sprecherwechsel** (höchste Priorität)
2. Paragraph-Grenzen
3. Satzgrenzen  
4. Längen-Limits (niedrigste Priorität)

### Erweiterte TextChunk Klasse
```python
@dataclass
class TextChunk:
    text: str
    start_pos: int
    end_pos: int
    has_paragraph_break: bool
    # NEU:
    speaker_id: str = "0"               # Aktueller Speaker
    speaker_transition: bool = False    # Chunk beginnt mit Sprecherwechsel
    original_markup: Optional[str] = None  # Original-Markup für Debugging
    idx: int = 0
```

### Chunker-Algorithmus
```python
def chunk_text_with_speakers(text: str) -> List[TextChunk]:
    1. Parse Markup und extrahiere Speaker-Wechsel
    2. Erstelle primäre Splits bei jedem <speaker:xxx>
    3. Für jeden Speaker-Bereich:
       - Anwende normale Chunking-Regeln
       - Behalte Speaker-ID für alle Sub-Chunks
    4. Markiere Chunks mit speaker_transition=True
    5. Validiere Speaker-IDs gegen verfügbare Konfiguration
```

---

## 🛠️ Code-Änderungen (Systematisch)

### 1. Konfiguration (`src/utils/config_manager.py`)

#### Neue Validierungsmethoden
```python
def validate_speakers_config(self, config: Dict) -> bool:
    """Validiere speakers[] Array"""
    speakers = config.get("generation", {}).get("speakers", [])
    if not speakers:
        logger.error("No speakers defined in generation.speakers")
        return False
    
    # Validiere Speaker 0
    if speakers[0].get("id") != "0" and not speakers[0].get("id"):
        logger.warning("Speaker 0 should have id='0' or be first speaker")
    
    # Validiere eindeutige IDs
    speaker_ids = [s.get("id") for s in speakers]
    if len(speaker_ids) != len(set(speaker_ids)):
        logger.error("Duplicate speaker IDs found")
        return False
    
    return True

def get_speaker_config(self, config: Dict, speaker_id: str) -> Dict:
    """Hole Speaker-spezifische Konfiguration"""
    speakers = config.get("generation", {}).get("speakers", [])
    
    # Suche Speaker nach ID
    for speaker in speakers:
        if speaker.get("id") == speaker_id:
            return speaker
    
    # Fallback auf Speaker 0
    if speakers:
        logger.warning(f"Speaker '{speaker_id}' not found, using speaker 0")
        return speakers[0]
    
    raise RuntimeError("No speakers configured")

def merge_speaker_params(self, base_config: Dict, speaker_config: Dict) -> Dict:
    """Merge Speaker-Config mit Basis-Config (kaskadierende Vererbung)"""
    # Implementiere intelligentes Merging für tts_params und conservative_candidate
    pass
```

#### Migration bestehender Configs
```python
def migrate_legacy_config(self, config: Dict) -> Dict:
    """Migriere alte Config-Struktur zu Speaker-System"""
    if "speakers" not in config.get("generation", {}):
        # Extrahiere alte Struktur
        old_reference_audio = config.get("input", {}).get("reference_audio")
        old_tts_params = config.get("generation", {}).get("tts_params", {})
        old_conservative = config.get("generation", {}).get("conservative_candidate", {})
        
        # Erstelle Speaker 0
        speaker_0 = {
            "id": "0",
            "reference_audio": old_reference_audio,
            "tts_params": old_tts_params,
            "conservative_candidate": old_conservative
        }
        
        # Update config
        config["generation"]["speakers"] = [speaker_0]
        config["input"].pop("reference_audio", None)
        config["generation"].pop("tts_params", None)
        config["generation"].pop("conservative_candidate", None)
        
        logger.info("Migrated legacy config to speaker system")
    
    return config
```

### 2. Chunker (`src/chunking/spacy_chunker.py`)

#### Speaker-Markup Parser
```python
import re
from typing import List, Tuple

class SpeakerMarkupParser:
    SPEAKER_PATTERN = r'<speaker:([^>]+)>'
    
    def parse_speaker_transitions(self, text: str) -> List[Tuple[int, str]]:
        """Parse Speaker-Wechsel und gebe (position, speaker_id) zurück"""
        transitions = []
        for match in re.finditer(self.SPEAKER_PATTERN, text):
            position = match.start()
            speaker_id = match.group(1).strip()
            transitions.append((position, speaker_id))
        return transitions
    
    def remove_markup(self, text: str) -> str:
        """Entferne Markup-Tags aus Text"""
        return re.sub(self.SPEAKER_PATTERN, '', text)
    
    def validate_speaker_id(self, speaker_id: str, available_speakers: List[str]) -> str:
        """Validiere und normalisiere Speaker-ID"""
        if speaker_id in ["0", "reset"]:
            return "0"
        
        if speaker_id in available_speakers:
            return speaker_id
            
        logger.warning(f"Unknown speaker '{speaker_id}', falling back to speaker 0")
        return "0"
```

#### Erweiterte SpaCyChunker
```python
class SpaCyChunker(BaseChunker):
    def __init__(self, ...):
        super().__init__(...)
        self.speaker_parser = SpeakerMarkupParser()
        self.available_speakers = []  # Wird von außen gesetzt
    
    def set_available_speakers(self, speakers: List[str]):
        """Setze verfügbare Speaker-IDs für Validierung"""
        self.available_speakers = speakers
    
    def chunk_text(self, text: str) -> List[TextChunk]:
        """Erweiterte Chunking mit Speaker-Support"""
        
        # 1. Parse Speaker-Transitionen
        transitions = self.speaker_parser.parse_speaker_transitions(text)
        clean_text = self.speaker_parser.remove_markup(text)
        
        # 2. Erstelle primäre Splits bei Speaker-Wechseln
        primary_splits = self._create_speaker_splits(clean_text, transitions)
        
        # 3. Anwende normale Chunking-Logik auf jeden Speaker-Bereich
        all_chunks = []
        for speaker_section in primary_splits:
            section_chunks = self._chunk_speaker_section(speaker_section)
            all_chunks.extend(section_chunks)
        
        # 4. Post-processing und Validierung
        return self._finalize_chunks(all_chunks)
    
    def _create_speaker_splits(self, text: str, transitions: List[Tuple[int, str]]) -> List[Dict]:
        """Erstelle primäre Aufteilung nach Speaker-Bereichen"""
        sections = []
        current_speaker = "0"  # Default speaker
        start_pos = 0
        
        for position, new_speaker in transitions:
            # Validiere und normalisiere Speaker-ID
            validated_speaker = self.speaker_parser.validate_speaker_id(
                new_speaker, self.available_speakers
            )
            
            # Erstelle Sektion für aktuellen Speaker
            if position > start_pos:
                section_text = text[start_pos:position].strip()
                if section_text:
                    sections.append({
                        "text": section_text,
                        "speaker_id": current_speaker,
                        "start_pos": start_pos,
                        "speaker_transition": len(sections) > 0  # Nicht für ersten Bereich
                    })
            
            # Wechsle zu neuem Speaker
            current_speaker = validated_speaker
            start_pos = position
        
        # Letzter Bereich
        if start_pos < len(text):
            final_text = text[start_pos:].strip()
            if final_text:
                sections.append({
                    "text": final_text,
                    "speaker_id": current_speaker,
                    "start_pos": start_pos,
                    "speaker_transition": len(sections) > 0
                })
        
        return sections
    
    def _chunk_speaker_section(self, section: Dict) -> List[TextChunk]:
        """Normale Chunking-Logik für einzelnen Speaker-Bereich"""
        # Verwende bestehende Chunking-Logik
        base_chunks = super().chunk_text(section["text"])
        
        # Erweitere um Speaker-Informationen
        enhanced_chunks = []
        for i, chunk in enumerate(base_chunks):
            enhanced_chunk = TextChunk(
                text=chunk.text,
                start_pos=chunk.start_pos + section["start_pos"],
                end_pos=chunk.end_pos + section["start_pos"],
                has_paragraph_break=chunk.has_paragraph_break,
                speaker_id=section["speaker_id"],
                speaker_transition=(i == 0 and section["speaker_transition"]),
                idx=chunk.idx
            )
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks
```

### 3. FileManager (`src/utils/file_manager/file_manager.py`)

#### Speaker-Support
```python
class FileManager:
    def get_reference_audio_for_speaker(self, speaker_id: str) -> Path:
        """Hole reference_audio für spezifischen Speaker"""
        speaker_config = self.config_manager.get_speaker_config(self.config, speaker_id)
        reference_audio = speaker_config.get("reference_audio")
        
        if not reference_audio:
            raise ValueError(f"No reference_audio defined for speaker '{speaker_id}'")
        
        audio_path = self.reference_audio_dir / reference_audio
        
        if not audio_path.exists():
            available_files = [f.name for f in self.reference_audio_dir.glob("*.wav")]
            raise FileNotFoundError(
                f"Reference audio not found: {audio_path}\n"
                f"Available files: {available_files}"
            )
        
        return audio_path
    
    def get_all_speaker_ids(self) -> List[str]:
        """Hole alle verfügbaren Speaker-IDs"""
        speakers = self.config.get("generation", {}).get("speakers", [])
        return [speaker.get("id", "0") for speaker in speakers]
    
    def validate_speakers_reference_audio(self) -> Dict[str, bool]:
        """Validiere reference_audio für alle Speaker"""
        validation_results = {}
        speakers = self.config.get("generation", {}).get("speakers", [])
        
        for speaker in speakers:
            speaker_id = speaker.get("id", "unknown")
            try:
                self.get_reference_audio_for_speaker(speaker_id)
                validation_results[speaker_id] = True
            except (FileNotFoundError, ValueError) as e:
                logger.error(f"Speaker '{speaker_id}' validation failed: {e}")
                validation_results[speaker_id] = False
        
        return validation_results
```

### 4. TTSGenerator (`src/generation/tts_generator.py`)

#### Speaker-dynamische Generation
```python
class TTSGenerator:
    def __init__(self, config: Dict[str, Any], device: str = "auto", seed: int = 12345):
        # ... existing init ...
        self.current_speaker_id = "0"
        self.speakers_config = config.get("speakers", [])
    
    def switch_speaker(self, speaker_id: str, config_manager=None):
        """Wechsle zu anderem Speaker mit neuer reference_audio"""
        if self.current_speaker_id == speaker_id:
            logger.debug(f"Speaker '{speaker_id}' already active, skipping switch")
            return
        
        # Hole Speaker-Konfiguration
        speaker_config = None
        for speaker in self.speakers_config:
            if speaker.get("id") == speaker_id:
                speaker_config = speaker
                break
        
        if not speaker_config:
            logger.warning(f"Speaker '{speaker_id}' not found, using speaker 0")
            speaker_config = self.speakers_config[0] if self.speakers_config else {}
            speaker_id = "0"
        
        # Lade neue reference_audio
        reference_audio = speaker_config.get("reference_audio")
        if reference_audio and config_manager:
            audio_path = config_manager.get_reference_audio_for_speaker(speaker_id)
            logger.info(f"🎭 Switching to speaker '{speaker_id}' with voice: {audio_path.name}")
            self.prepare_conditionals(str(audio_path))
            self.current_speaker_id = speaker_id
        else:
            logger.warning(f"No reference_audio for speaker '{speaker_id}'")
    
    def generate_candidates_with_speaker(
        self,
        text: str,
        speaker_id: str = "0",
        num_candidates: int = 3,
        config_manager=None,
        **kwargs
    ) -> List[AudioCandidate]:
        """Generiere Kandidaten mit spezifischem Speaker"""
        
        # 1. Wechsle zu Speaker
        self.switch_speaker(speaker_id, config_manager)
        
        # 2. Hole Speaker-spezifische Parameter
        speaker_config = self._get_speaker_config(speaker_id)
        tts_params = speaker_config.get("tts_params", {})
        conservative_config = speaker_config.get("conservative_candidate", {})
        
        # 3. Generiere mit Speaker-Parametern
        return self.generate_candidates(
            text=text,
            num_candidates=num_candidates,
            tts_params=tts_params,
            conservative_config=conservative_config,
            reference_audio_path=None,  # Bereits durch switch_speaker geladen
            **kwargs
        )
    
    def _get_speaker_config(self, speaker_id: str) -> Dict[str, Any]:
        """Hole Konfiguration für spezifischen Speaker"""
        for speaker in self.speakers_config:
            if speaker.get("id") == speaker_id:
                return speaker
        
        # Fallback auf Speaker 0
        if self.speakers_config:
            return self.speakers_config[0]
        
        return {}
```

### 5. GenerationHandler (`src/pipeline/task_executor/stage_handlers/generation_handler.py`)

#### Multi-Speaker Generation
```python
class GenerationHandler:
    def execute_generation(self) -> bool:
        """Erweiterte Generation mit Speaker-Support"""
        
        # 1. Validiere Speaker-Konfiguration
        if not self._validate_speakers():
            return False
        
        # 2. Setze verfügbare Speaker im Chunker
        available_speakers = self.file_manager.get_all_speaker_ids()
        if hasattr(self.chunker, 'set_available_speakers'):
            self.chunker.set_available_speakers(available_speakers)
        
        # 3. Load chunks (mit Speaker-Informationen)
        chunks = self._load_or_create_chunks()
        
        # 4. Generiere Kandidaten pro Chunk mit Speaker-Wechsel
        all_candidates = {}
        current_speaker = "0"
        
        for chunk in chunks:
            # Wechsle Speaker wenn nötig
            if chunk.speaker_id != current_speaker or chunk.speaker_transition:
                logger.info(f"🎭 Speaker change: '{current_speaker}' → '{chunk.speaker_id}'")
                current_speaker = chunk.speaker_id
            
            # Generiere Kandidaten für aktuellen Speaker
            candidates = self._generate_candidates_for_chunk_with_speaker(
                chunk, current_speaker
            )
            all_candidates[chunk.idx] = candidates
        
        # 5. Speichere Ergebnisse
        self._save_generation_results(all_candidates)
        return True
    
    def _generate_candidates_for_chunk_with_speaker(
        self, 
        chunk: TextChunk, 
        speaker_id: str
    ) -> List[AudioCandidate]:
        """Generiere Kandidaten für Chunk mit spezifischem Speaker"""
        
        logger.debug(
            f"Generating candidates for chunk {chunk.idx+1} "
            f"(speaker: {speaker_id}, text: '{chunk.text[:50]}...')"
        )
        
        # Verwende Speaker-spezifische Generation
        candidates = self.tts_generator.generate_candidates_with_speaker(
            text=chunk.text,
            speaker_id=speaker_id,
            num_candidates=self.config["generation"]["num_candidates"],
            config_manager=self.file_manager
        )
        
        # Setze Chunk-spezifische Metadaten
        for candidate in candidates:
            candidate.chunk_idx = chunk.idx
            candidate.chunk_text = chunk.text
            candidate.speaker_id = speaker_id  # NEU: Speaker-ID in Candidate
        
        return candidates
    
    def _validate_speakers(self) -> bool:
        """Validiere Speaker-Konfiguration und reference_audio"""
        validation_results = self.file_manager.validate_speakers_reference_audio()
        
        failed_speakers = [sid for sid, valid in validation_results.items() if not valid]
        if failed_speakers:
            logger.error(f"Speaker validation failed for: {failed_speakers}")
            return False
        
        logger.info(f"✅ All speakers validated: {list(validation_results.keys())}")
        return True
```

---

## 📋 Implementierungsreihenfolge

### Phase 1: Grundlagen (1-2 Tage)
1. ✅ **Konfigurationsstruktur**: Neue default_config.yaml
2. **ConfigManager erweitern**: Speaker-Validierung und -Auflösung
3. **Basis-Tests**: Config-Loading und Speaker-Zugriff

### Phase 2: Chunker (2-3 Tage)  
4. **SpeakerMarkupParser**: Markup-Parsing und Validierung
5. **SpaCyChunker erweitern**: Speaker-Splits mit höchster Priorität
6. **TextChunk erweitern**: Speaker-Metadaten
7. **Chunker-Tests**: Markup-Verarbeitung und Speaker-Wechsel

### Phase 3: TTS-Integration (2-3 Tage)
8. **FileManager erweitern**: Speaker-spezifische reference_audio 
9. **TTSGenerator erweitern**: Dynamischer Speaker-Wechsel
10. **GenerationHandler erweitern**: Multi-Speaker-Generation
11. **Generation-Tests**: Ende-zu-Ende Speaker-Wechsel

### Phase 4: Migration & Testing (1-2 Tage)
12. **Config-Migration**: Alle bestehenden *.yaml Dateien
13. **Dokumentation**: README und Beispiele

---

## 🧪 Testing-Strategie

### Unit Tests
```python
# test_speaker_markup_parser.py
def test_parse_simple_speaker_change():
    parser = SpeakerMarkupParser()
    text = "Hello <speaker:narrator>world"
    transitions = parser.parse_speaker_transitions(text)
    assert transitions == [(6, "narrator")]

# test_config_manager_speakers.py  
def test_get_speaker_config():
    config = load_test_config_with_speakers()
    speaker_config = config_manager.get_speaker_config(config, "narrator")
    assert speaker_config["reference_audio"] == "narrator_voice.wav"

# test_chunker_speakers.py
def test_chunker_speaker_priority():
    chunker = SpaCyChunker()
    text = "Short text. <speaker:narrator>Different speaker here."
    chunks = chunker.chunk_text(text)
    
    assert len(chunks) == 2  # Should split at speaker change
    assert chunks[0].speaker_id == "0"
    assert chunks[1].speaker_id == "narrator"
    assert chunks[1].speaker_transition == True
```

### Integration Tests
```python
# test_speaker_pipeline.py
def test_end_to_end_speaker_generation():
    """Test komplette Pipeline mit mehreren Sprechern"""
    
    # Setup
    config = create_multi_speaker_config()
    test_text = """
    Default speaker introduction.
    <speaker:narrator>The narrator takes over here.
    <speaker:character>"Hello there!", said the character.
    <speaker:0>Back to default speaker.
    """
    
    # Execute
    task_executor = TaskExecutor(config)
    result = task_executor.execute_task()
    
    # Verify
    assert result.success == True
    chunks = task_executor.file_manager.get_chunks()
    assert len([c for c in chunks if c.speaker_id == "0"]) >= 2
    assert len([c for c in chunks if c.speaker_id == "narrator"]) >= 1
    assert len([c for c in chunks if c.speaker_id == "character"]) >= 1
```

### Test-Dokumente
read data/input/texts/input-document.txt via default_config.yaml

---

## 🎯 Qualitätssicherung

### Code-Reviews
- [ ] Architektur-Review nach Phase 1
- [ ] Chunker-Logic Review nach Phase 2  
- [ ] TTS-Integration Review nach Phase 3
- [ ] Performance-Review nach Phase 4

### Dokumentation
- [ ] README mit Speaker-System Anleitung
- [ ] Markup-Syntax Dokumentation

### Backwards-Compatibility  
- [ ] Bestehende Job-Pipelines unverändert
- [ ] Fallback-Mechanismen bei fehlenden Speakern