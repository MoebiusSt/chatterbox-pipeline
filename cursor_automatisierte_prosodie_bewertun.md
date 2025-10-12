---

**User**

**Feature Request: Timestamp-basierter Tail Trimmer mit Whisper-Integration**

**Kernfrage:** Kann Whisper/WhisperX word-level timestamps ausgeben?

**Problemstellung:**
TTS-Halluzinationen am Audio-Ende werden vom aktuellen tail_trimmer nicht zuverlässig erkannt, wenn sie wie gesprochene Wörter klingen (Beispiel: `candidate_02.wav` - Input: "Einleitung Ende" → Transkription: "Einleitung Ende. Uns, Ibia, dem und ......").

**Lösungsansatz:**
1. **Timestamp-basierte Validierung:** 
   - Letzten 2 Wörter des Input-Texts in der Transkription lokalisieren
   - Bei Match: Audio ab dem entsprechenden Timestamp trimmen

2. **Hybrid-Ansatz:**
   - Bestehenden tail_trimmer für nicht-sprachliche Artefakte beibehalten
   - Timestamp-basiertes Trimming für sprachähnliche Halluzinationen hinzufügen

3. **Error-Code Integration:**
   - `long_tail=tensor(True)` Flag von Chatterbox tracken
   - Nur betroffene Chunks mit erweiterter Validierung verarbeiten
   - Nach Trimming: Re-Validierung für Score-Verbesserung

**Frage:** Ist dieser Ansatz mit Whisper/WhisperX implementierbar?


---

**Cursor**

Kurzantwort: Ja. WhisperX liefert Wort-Zeitmarken (Start/Ende je Wort) über das Alignment. Damit können wir exakt nach dem letzten gewünschten Wort schneiden und anschließend weiter unser VAD/Energie-Tail-Trim anwenden. Re-Validierung des getrimmten Kandidaten ist ebenfalls machbar.

Vorschlag für „Smart Tail-Trim” (ergänzend, nicht ersetzend)
- Idee: Vor dem bisherigen Tail-Trim suche die letzten N Wörter des Input-Texts in der WhisperX-Transkription (word-level). Wenn gefunden, schneide unmittelbar nach dem Ende des letzten Treffers (+200 ms Stille, +Fade-out). Fallback bleibt: WhisperX-Ende-Suche → VAD → Energie.
- Optional: Nutze das während der Generierung protokollierte Flag `long_tail=True`, um den Smart-Trim gezielt zu triggern und danach die Validierung neu zu rechnen.

Config-Erweiterung (YAML)
```yaml
validation:
  preprocessing:
    tail_trim:
      smart_match:
        enabled: true
        last_n_words: 2
        fuzzy_ratio: 0.85     # 0..1, ab wann Match akzeptiert wird
        prefer_whisperx: true # nutzt vorhandenes WhisperX-Alignment
```

TailTrimmer: neue Methode (Grober Vorschlag)
```python
# src/validation/preprocessors/tail_trimmer.py
import re
from difflib import SequenceMatcher
import torchaudio

def _normalize_words(self, text: str) -> list[str]:
    # lower, ASCII-ähnlich, Satzzeichen raus
    t = text.lower()
    t = re.sub(r"[^\w\säöüß]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t.split() if t else []

def _trim_after_last_words_with_alignment(
    self,
    audio: torch.Tensor,
    language: str,
    original_text: str,
    last_n_words: int,
    fuzzy_ratio: float,
) -> Optional[int]:
    if whisperx is None:
        return None
    # 1) letzten N Wörter aus dem Input
    tgt_words = self._normalize_words(original_text)[-last_n_words:]
    if not tgt_words:
        return None

    # 2) Alignment holen
    device = "cpu"
    model = whisperx.load_model("small", device=device, compute_type="int8")
    audio16 = torchaudio.transforms.Resample(self.sample_rate, 16000)(audio.unsqueeze(0)).squeeze(0)
    result = model.transcribe(audio16.cpu().numpy(), language=language)

    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    aligned = whisperx.align(
        result.get("segments", []),
        align_model, metadata,
        audio16.cpu().numpy(),
        device=device,
        return_char_alignments=False
    )
    words = aligned.get("word_segments") or []
    if not words:
        return None

    # 3) Normalisierte Wortliste der Erkennung + Indexmapping
    rec_words = []
    for i, w in enumerate(words):
        wtxt = (w.get("word") or "").strip()
        nws = self._normalize_words(wtxt)
        if not nws:
            continue
        rec_words.append((i, nws[0]))  # ein Token pro Align-Wort

    if not rec_words:
        return None

    # 4) Sliding-Window Fuzzy-Match über das Ende
    tgt = " ".join(tgt_words)
    best = (0.0, None, None)  # (ratio, start_idx, end_idx)
    window = len(tgt_words)
    for j in range(max(1, len(rec_words) - 12), len(rec_words) - window + 1):  # fokussiere auf Schluss
        span = " ".join([rec_words[k][1] for k in range(j, j + window)])
        ratio = SequenceMatcher(None, tgt, span).ratio()
        if ratio > best[0]:
            best = (ratio, rec_words[j][0], rec_words[j + window - 1][0])

    if best[0] < fuzzy_ratio or best[1] is None:
        return None

    # 5) Schnittprobe: Ende des letzten gematchten Wortes
    end_word = words[best[2]]
    if end_word.get("end") is None:
        return None
    cut_s = float(end_word["end"])
    cut_idx = int(cut_s * self.sample_rate)
    return cut_idx
```

Integration im bestehenden Trim (vor VAD/Energie)
```python
# innerhalb TailTrimmer.trim(...) – vor VAD/Energie:
sm_cfg = self.config.get("smart_match", {}) if hasattr(self, "config") else {}
if sm_cfg.get("enabled", False):
    sm_cut = self._trim_after_last_words_with_alignment(
        audio,
        language=language,
        original_text=original_text,
        last_n_words=int(sm_cfg.get("last_n_words", 2)),
        fuzzy_ratio=float(sm_cfg.get("fuzzy_ratio", 0.85)),
    )
    if sm_cut is not None:
        cut_idx = sm_cut
        method = "smart_words"
```

Re-Validierung nach Smart-Trim (Vorschlag)
```python
# src/pipeline/task_executor/stage_handlers/validation_handler.py
# nach erfolgreichem Tail-Trim und wenn removed_tail vorhanden:
if removed_tail is not None:
    # Revalidate trimmed candidate for better score
    try:
        # ggf. bestehende Pfade/Objekte verwenden:
        asr_res = self.whisper_validator.transcribe_tensor(
            candidate.audio_tensor,
            language=getattr(chunk, "language_id", "en")
        )
        # Ähnlichkeit/Qualität/Prosodie/MOS neu berechnen und in metrics aktualisieren
        updated = self._score_candidate_with_all_metrics(candidate, chunk, asr_res)
        combined_result.update(updated)
    except Exception as e:
        logger.debug(f"Revalidate trimmed candidate failed: {e}")
```

„long_tail“-Signal nutzen (optional)
- Während der Generierung das Flag aus dem Log in die Kandidaten-Metadaten übernehmen, z. B. `candidate.flags.long_tail = True`.
- In der Validation: Wenn `long_tail==True`, Smart-Trim erzwingen (und danach revalidieren), selbst wenn der VAD-/Energie-Trim keinen Cut gefunden hätte.

Antwort auf Deine Kernfrage
- WhisperX: liefert robuste Wort-Zeitmarken via Alignment (Start/Ende pro Wort). Genau darauf setzen wir bereits auf CPU. Damit ist der „Smart Tail-Trim nach den letzten zwei Input-Wörtern“ zuverlässig umsetzbar.

Wenn Du möchtest, implementiere ich diese „Smart Tail-Trim“ Erweiterung direkt und aktiviere sie testweise für Deinen genannten Chunk.

```yaml
validation:
  preprocessing:
    tail_trim:
      smart_match:
        enabled: true
        last_n_words: 2
        fuzzy_ratio: 0.85     # 0..1, ab wann Match akzeptiert wird
        prefer_whisperx: true # nutzt vorhandenes WhisperX-Alignment
```

```python
# src/validation/preprocessors/tail_trimmer.py
import re
from difflib import SequenceMatcher
import torchaudio

def _normalize_words(self, text: str) -> list[str]:
    # lower, ASCII-ähnlich, Satzzeichen raus
    t = text.lower()
    t = re.sub(r"[^\w\säöüß]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t.split() if t else []

def _trim_after_last_words_with_alignment(
    self,
    audio: torch.Tensor,
    language: str,
    original_text: str,
    last_n_words: int,
    fuzzy_ratio: float,
) -> Optional[int]:
    if whisperx is None:
        return None
    # 1) letzten N Wörter aus dem Input
    tgt_words = self._normalize_words(original_text)[-last_n_words:]
    if not tgt_words:
        return None

    # 2) Alignment holen
    device = "cpu"
    model = whisperx.load_model("small", device=device, compute_type="int8")
    audio16 = torchaudio.transforms.Resample(self.sample_rate, 16000)(audio.unsqueeze(0)).squeeze(0)
    result = model.transcribe(audio16.cpu().numpy(), language=language)

    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)
    aligned = whisperx.align(
        result.get("segments", []),
        align_model, metadata,
        audio16.cpu().numpy(),
        device=device,
        return_char_alignments=False
    )
    words = aligned.get("word_segments") or []
    if not words:
        return None

    # 3) Normalisierte Wortliste der Erkennung + Indexmapping
    rec_words = []
    for i, w in enumerate(words):
        wtxt = (w.get("word") or "").strip()
        nws = self._normalize_words(wtxt)
        if not nws:
            continue
        rec_words.append((i, nws[0]))  # ein Token pro Align-Wort

    if not rec_words:
        return None

    # 4) Sliding-Window Fuzzy-Match über das Ende
    tgt = " ".join(tgt_words)
    best = (0.0, None, None)  # (ratio, start_idx, end_idx)
    window = len(tgt_words)
    for j in range(max(1, len(rec_words) - 12), len(rec_words) - window + 1):  # fokussiere auf Schluss
        span = " ".join([rec_words[k][1] for k in range(j, j + window)])
        ratio = SequenceMatcher(None, tgt, span).ratio()
        if ratio > best[0]:
            best = (ratio, rec_words[j][0], rec_words[j + window - 1][0])

    if best[0] < fuzzy_ratio or best[1] is None:
        return None

    # 5) Schnittprobe: Ende des letzten gematchten Wortes
    end_word = words[best[2]]
    if end_word.get("end") is None:
        return None
    cut_s = float(end_word["end"])
    cut_idx = int(cut_s * self.sample_rate)
    return cut_idx
```

```python
# innerhalb TailTrimmer.trim(...) – vor VAD/Energie:
sm_cfg = self.config.get("smart_match", {}) if hasattr(self, "config") else {}
if sm_cfg.get("enabled", False):
    sm_cut = self._trim_after_last_words_with_alignment(
        audio,
        language=language,
        original_text=original_text,
        last_n_words=int(sm_cfg.get("last_n_words", 2)),
        fuzzy_ratio=float(sm_cfg.get("fuzzy_ratio", 0.85)),
    )
    if sm_cut is not None:
        cut_idx = sm_cut
        method = "smart_words"
```

```python
# src/pipeline/task_executor/stage_handlers/validation_handler.py
# nach erfolgreichem Tail-Trim und wenn removed_tail vorhanden:
if removed_tail is not None:
    # Revalidate trimmed candidate for better score
    try:
        # ggf. bestehende Pfade/Objekte verwenden:
        asr_res = self.whisper_validator.transcribe_tensor(
            candidate.audio_tensor,
            language=getattr(chunk, "language_id", "en")
        )
        # Ähnlichkeit/Qualität/Prosodie/MOS neu berechnen und in metrics aktualisieren
        updated = self._score_candidate_with_all_metrics(candidate, chunk, asr_res)
        combined_result.update(updated)
    except Exception as e:
        logger.debug(f"Revalidate trimmed candidate failed: {e}")
```

---

**User**

Super. bitte plane dieses Feature genauer. Bedenke alle pipeline-Programm-Pfade (gap-filling run muss ebenso selektiv neue Kandidaten tailtrimmen etc.). Bitte lasse auch hier zu Diagnosezwecken das abgeschnittene Audio-Teilstück als Datei speichern. 

'''
# Smart Tail-Trim mit Wort-Zeitmarken (WhisperX) und Revalidierung – mit Sprach-Gating

## Ziel

- Inhaltlich geführtes Tail-Trimming: Nach den letzten N Wörtern des Input-Texts in alignierten WhisperX-Word-Zeitmarken schneiden.
- Falls danach noch „Wortklänge/Halluzinationen“ existieren: Nach dem Cut zusätzlich VAD/Energie-Trim anwenden.
- Immer nicht-sprachliche Geräusche hinter dem letzten Wort via VAD/Energie entfernen.
- Revalidierung getrimmter Kandidaten, damit Scores (Similarity/Quality/Prosody/MOS) auf dem finalen Audio basieren.
- Funktioniert in allen Pfaden: volle Runs, selektive Validierung, und gap-filling.

## Konfiguration (default_config.yaml)

- validation.preprocessing.tail_trim.smart_match
- enabled: true
- last_n_words: 2
- fuzzy_ratio: 0.85
- prefer_whisperx: true
- apply_post_speech_silence_ms: 200
- apply_fade_out_ms: 120
- persist_trimmed_candidate: true   # speichert *_trimmed.wav und nutzt sie downstream
- debug_save_removed_tail: true     # bereits vorhanden – speichert *_removed_tail.wav

## Dateiausgaben

- Kept: candidate_XX_trimmed.wav (nur wenn tatsächlich geschnitten)
- Removed tail: chunk_XXX_cand_YY_removed_tail.wav (wie heute, pro Task in debug_tail_trimmer/)
- Optional Backup: candidate_XX_orig.wav (wenn persist_trimmed_candidate=true und wir im selben Ordner überschreiben würden, ansonsten unnötig)
- Metriken: whisper_metrics.json ergänzt um tail_trim.{method, match_type, cut_sample, matched_words}

## Algorithmus

1) Normalize Input & ASR-Wörter: Lowercase, Satzzeichen entfernen, Unicode-Wortchars erlauben.
2) WhisperX Alignment (CPU, compute_type=int8) → word_segments mit start/end (Sekunden).
3) Suche der letzten N Input-Wörter in den letzten K Align-Wörtern (z. B. 30) mit fuzzy Ratio:

- Finde den rechtsliegenden (letzten) Match; wenn Match die letzten Align-Wörter exakt sind, kein Cut durch "content"-Methode (match_type=exact_final).
- Wenn Match vor dem letzten Align-Wort endet (es folgen noch Wörter/Laute), setze Cut = end(last_matched_word) (match_type=content_cut).
4) Post-Cut Processing:
- addiere apply_post_speech_silence_ms, begrenze auf Audioende.
- Fade-out (apply_fade_out_ms) auf den restlichen gehaltenen Samples.
5) VAD/Energie-Trim danach:
- Suche letzte sprachhaltige Stelle; schneide alles dahinter.
- Falls vorher kein Cut (kein Match), wirke wie heute: WhisperX-Ende → VAD → Energie.
6) Persistenz (optional, per Flag): Speichere kept als *_trimmed.wav; removed tail separat.
7) Update tail_trim-Metadaten: method, match_type (none|content_cut|exact_final|whisperx|vad|energy), cut_sample, kept_ms, removed_ms.

## Revalidierung

- Wenn Trim angewendet wurde (cut_sample != None):
- Transkription erneut (Whisper), Scoring erneut (Similarity/Quality), Prosody erneut (WPM, Pausen, F0), MOS erneut.
- Metrics im Speicher aktualisieren und whisper_metrics.json synchronisieren.
- Candidate-Auswahl für Chunk neu bewerten; Assembly nutzt ggf. *_trimmed.wav.

## Pipeline-Integration

- tail_trimmer.py
- Neue Methode: _trim_after_last_words_with_alignment(audio, language, original_text, last_n_words, fuzzy_ratio) -> Optional[int]
- trim(): Reihenfolge: Smart-Match (wenn enabled) → WhisperX-Ende → VAD → Energie.
- Rückgabe: (kept_audio, removed_tail, meta). Persistenz abhängig von persist_trimmed_candidate.
- validation_handler.py
- execute_validation() und execute_selective_validation():
- Tail-Trim auf Candidate.audio_tensor.
- Wenn geschnitten, optional speichern (trimmed + removed_tail) und Revalidierung starten.
- final_selection_score auf Basis neuer Metriken berechnen.
- Gap-Filling: identische Behandlung, da Path shared ist.
- Generierung (optional):
- long_tail-Flag aus Generation in candidates_metadata.json übernehmen (flags.long_tail=true).
- In Validation: Wenn flags.long_tail, Smart-Match erzwingen (oder last_n_words erhöhen), danach Revalidierung.

## Multilingualität

- WhisperX mit language_code des Chunks.
- Normalisierung tolerant gegenüber nicht-ASCII (äöüß etc.).

## Tests (manuell + Skript)

- Kurzer Chunk „Einleitung Ende“ mit gezielter Halluzination dahinter:
- Erwartet: content_cut nach "Ende" → VAD-Trim residualer Geräusche → Revalidierung → bessere Similarity/Quality.
- Fälle ohne Match (keine letzten N Worte am Ende):
- Erwartet: WhisperX-Ende/VAD/Energie greifen, removed_tail wird gespeichert.
- Exakter Abschluss (Transkription endet exakt mit letzten N Input-Wörtern):
- Erwartet: kein content_cut, aber VAD/Energie-Trim nach letztem Wort.

## Risiken & Mitigation

- Alignment-Kosten: Cache WhisperX-Modelle im TailTrimmer (einmal laden pro Prozess).
- Fuzzy-Falschpositiv: Begrenze Suche auf Schlussbereich, erhöhe fuzzy_ratio; logs + debug WAVs zur Prüfung.
- Persistenzpfade: Kollisionen vermeiden via suffix (_trimmed/_removed_tail) pro Candidate.

## Aktivierung/Flags

- YAML: smart_match.enabled true/false
- CLI (optional): --tail-trim-smart, --tail-trim-last-words 2, --tail-trim-fuzzy 0.85
- Debug: debug_save_removed_tail true
'''

Achtung! Da WhisperX glaube ich in seinen Basismodellen nur ein Subset der Sprachen von Chatterbox beherrscht, nämlich nur {en, fr, de, es, it}, müssen wir das Smarte Tail-Trimming skippen und nur das VAD/Energie Trim ausführen, sollte die Language des Chunks abweichen von diesen verfügbaren Sprachen.
