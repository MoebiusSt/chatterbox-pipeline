# Testrun Findings – Chatterbox Multilingual vs. VibeVoice (Longform)

Stand: 2026-04-16. Eingangsdokument: `data/input/texts/input-document.txt` (≈ 4.4 kchars, deutsch, Redaktionstext mit Fließabschnitten).

## Zusammenfassung

| Metric                         | Chatterbox Multi                   | VibeVoice                            |
| ------------------------------ | ---------------------------------- | ------------------------------------ |
| Task                           | `input-document_20260416_202911`   | `input-document_20260416_235239`     |
| Gesamtlaufzeit                 | **11:38** (698 s)                  | **54:45** (3 285 s)                  |
| Final-Audio-Dauer              | 260.93 s                           | 257.42 s                             |
| Chunks                         | **20** (satz-/absatzorientiert)    | **2** (long-context merged)          |
| Kandidaten gesamt (valid)      | 60 (59)                            | 6 (6)                                |
| Median Similarity              | 0.920                              | **0.954**                            |
| Median Overall-Quality         | 0.932                              | **0.959**                            |
| Median Raw-MOS (NISQA-TTS)     | 3.62                               | **3.72**                             |
| Median Prosody-Score           | 0.46                               | **0.59**                             |
| Median Flow-Subscore           | **0.00**                           | 0.48                                 |
| Median Liveliness              | 0.974                              | 0.994                                |
| Median WPM (selected)          | 140.6                              | 139.5                                |
| ASR-Backend (dieser Testrun)   | Whisper `small`                    | Whisper `small` (VV-ASR forced off)  |

*Hinweis:* Beim VibeVoice-Lauf wurde `validation.asr_backend: whisper` für die
Dauer des Testruns forciert (siehe Bug V1), um die Pipeline in tolerabler Zeit
durchzubringen. Ein isolierter VV-ASR-Test auf einem 167-s-Kandidaten (Chunk
001/Cand 01) lief nach dem Hotfix mit gutem Transkript erfolgreich durch
(~715 s Wall-Clock, 4.3× realtime auf dieser GPU).

## Ergebnisbewertung

### Chatterbox Multilingual

* Alle 60 Kandidaten bis auf eine Ausnahme valid. Einziger Fail: Chunk 2
  (`"Einleitung ende."`, 3 Worte) → ein Kandidat scheitert an der
  Similarity-Gate.
* Scoring plausibel: Similarity und Länge hoch, MOS durchschnittlich 3.6.
* **Zero-Flow-Problem**: `flow` ist in 10 von 20 Chunks für alle drei
  Kandidaten exakt 0.0 (Median 0.0, Mean 0.28). Betrifft sowohl kurze
  Ein-Satz-Chunks als auch längere Absätze. Die Prosody-Komponente ist dadurch
  systematisch gedämpft (Median 0.46). Bei Chatterbox mit kurzen Chunks ist
  das akzeptabel, aber die häufigen harten 0-Werte wirken wie ein Floor, nicht
  wie ein Messwert (siehe Fund P1).
* `intelligibility` ist in 60/60 Kandidaten exakt 1.0 — Subscore ohne
  Diskriminierungsvermögen (Fund P2).
* Task läuft sauber, Assembly und Final-Audio (4:21 min) passen zur
  Chunksumme.

### VibeVoice (Longform)

* Alle 6 Kandidaten valid.
* 2 Chunks à 2 866 bzw. 1 550 chars (nach Fix der Chunk-Verschmelzung, siehe
  V2). Kandidaten-Audio ist 155–164 s bzw. 91–95 s lang und wird
  erwartungsgemäß länger durch-inferiert.
* Deutlich bessere Prosody- (0.59 vs. 0.46) und Flow-Werte (0.48 vs. 0.00),
  weil innerhalb eines langen Chunks ausreichend Wort-Pausen-Signal für den
  Flow-Scorer existiert. Das stützt den Mehrwert der Long-Chunk-Strategie.
* Similarity tighter (0.939 – 0.958). VibeVoice spricht in Englisch
  (Sprachumschaltung auf DE nicht möglich, siehe Fund V3) — die deutsche
  Transkription gelingt trotzdem mit hoher Textähnlichkeit, dank tolerantem
  Similarity-Scorer und Zahlen-/Punktnormalisierung.
* Tail-Trim cut nur 0.01 – 0.06 s → sauberes Ende. Keine Auffälligkeiten.
* Laufzeit-Aufteilung (grob): Generation ≈ 30 min, Whisper-Validation
  ≈ 17 min, Rest (Assembly, IO, Modell-Loads) ≈ 8 min.

## Gefixt während des Testruns (bereits im Code)

| ID  | Finding                                                              | Root Cause                                                                                                 | Fix                                                                                                                                      |
| --- | -------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| V1  | `VibeVoiceAcousticTokenizerModel.encode() got an unexpected keyword argument 'is_final_chunk'` → ASR fiel in jedem Kandidaten in Leerstring zurück | Vendored Tokenizer älter als Upstream-VV-ASR-API.                                                           | Die beiden `encode()`-Methoden in `src/third_party/vibevoice/modular/modular_vibevoice_tokenizer.py` akzeptieren `is_final_chunk` (ignoriert im Single-Pass-Pfad). |
| V2  | VibeVoice zerlegte einen 4.4-kchar-Text in **15** statt 2–3 Chunks.   | `force_paragraph_chunks: true` aus `config/default_config.yaml` überschrieb die VV-Long-Limits.             | `force_paragraph_chunks: false` in `config/defaults/vibevoice.yaml`.                                                                     |
| V5  | `ModuleNotFoundError: No module named 'vibevoice'` beim ASR-Load.    | Vendored Pfad ohne `sys.path`-Shim; `trust_remote_code=True` forderte echtes Package.                       | `sys.path.insert` + `sys.modules`-Shim in `src/validation/asr/vibevoice_asr_backend.py`, `trust_remote_code` entfernt, Tokenizer-Class nachvendort. |
| Q1  | `NISQA scoring failed: float() argument must be a string or a real number, not 'dict'` (16× pro Chunk-Candidate-Kombi) | VERSA `nisqa_metric` gibt neuerdings ein Dict zurück (`nisqa_mos_pred`, `nisqa_noi_pred`, …), der Code rief `float()` direkt darauf auf. | `src/validation/mos_providers/nisqa.py`: Dict-Rückgabe erkannt, `nisqa_mos_pred` extrahiert, Sub-Scores in `_last_details["sub_scores"]` gespeichert. |
| M1  | `asr_backend`-Feld fehlte in `whisper_metrics.json` (nur in den Einzeldateien vorhanden). | `_sync_whisper_to_enhanced_metrics` hat das Feld nicht übernommen; `ValidationResult` trug es gar nicht. | `ValidationResult.asr_backend` ergänzt, in `ASRBackend.score_transcription` + Legacy `WhisperValidator` gesetzt, `quality_scorer.validation_metrics` führt es auf, `whisper_io` synct es. |
| M2  | `candidate_metrics.audio_duration = 0.0` in `task_metrics.json` für jeden ausgewählten Kandidaten. | `TaskMetricsGenerator` las `audio_duration` nur top-level, das Feld existiert aber nur in `quality_details`. | Fallback auf `quality_details.audio_duration` in `src/utils/file_manager/task_metrics_generator.py`.                                     |
| M3  | VibeVoice-spezifische Generation-Params (`cfg_scale`, `diffusion_steps`, `voice_speed_factor`, `use_sampling`) erschienen nicht in `task_metrics.json`; stattdessen Default-0.0 für die Chatterbox-Felder. | `_get_selected_candidate_generation_params` hardcodete Chatterbox-Keys.                                    | Alle `generation_params` (außer `seed`, `language_id`) werden durchgereicht. |

## Umgesetzte Low-Hanging Fruits (Follow-up-Commit)

Alle zuvor offenen Empfehlungen aus dem Testrun wurden in einem Folge-Commit
adressiert. Die Tabelle fasst Änderung und Ort zusammen.

| ID  | Zustand jetzt | Wo? |
| --- | --- | --- |
| P1  | Flow-Subscore wird für Chunks mit < 30 Wörtern oder < 2 s auf 0.5 (neutral) gesetzt statt Band-Strafe. | `src/validation/prosody_scorer.py` |
| P2  | `intelligibility` ist jetzt Token-Count-Coverage (Dreieck um 1.0, Toleranz ±35 %) statt konstant 1.0. | `src/validation/prosody_scorer.py` |
| P3  | `candidates_metadata.json` schreibt `audio_duration` als ground truth; `task_metrics_generator` zieht primär aus dieser Datei, `quality_details` nur Fallback. | `src/utils/file_manager/io_handlers/candidate_io.py`, `src/utils/file_manager/task_metrics_generator.py` |
| P4/R3 | Nach jedem Candidate-Sync rechnet `whisper_io` `best_candidate`/`best_score` neu (argmax über `final_selection_score` → `final_score` → `overall_quality_score`). | `src/utils/file_manager/io_handlers/whisper_io.py` |
| P5  | VibeVoice-Language-Warnung wird einmal pro unbekanntem Code geloggt; `generation.vibevoice.language_strict=true` wirft jetzt `ValueError`. Qwen3-Warnung ebenfalls einmalig. | `src/generation/tts_generator.py`, `config/defaults/vibevoice.yaml` |
| A1  | `max_new_tokens` ist dynamisch auf `max(min_max_new_tokens, audio_s × tokens_per_second)` gekappt (Defaults: 256 / 40). | `src/validation/asr/vibevoice_asr_backend.py` |
| A2  | README der vendored VibeVoice-Kopie dokumentiert den `is_final_chunk`-Shim und die Tokenizer-Anpassung. | `src/third_party/vibevoice/README.md` |
| A3  | `ValidationHandler.execute_validation` ruft vor dem Candidate-Loop `torch.cuda.empty_cache()` + `ipc_collect()` auf. | `src/pipeline/task_executor/stage_handlers/validation_handler.py` |
| A4  | Beim VV-ASR-Load wird über `huggingface_hub.try_to_load_from_cache` einmalig ein Hinweis auf den anstehenden Download (~17 GB) geloggt. | `src/validation/asr/vibevoice_asr_backend.py` |
| R1  | In der aktuellen `task_metrics.json`-Struktur existieren die Felder `selection`/`assembly` nicht (mehr). Kein Code-Change nötig. | — |
| R2  | `MOSProvider.score_segmented` aggregiert Segment-Fehler und loggt einmal `MOS segment scoring: N/M segment(s) failed (first: …)`. | `src/validation/mos_providers/base.py` |
| Mypy | `mypy src/` ist clean (76 Module). Vendored VibeVoice-Code per `mypy.ini` ausgenommen, `audio_utils`, `file_manager`, `tts_generator`, `task_metrics_generator`, `assembly_handler` nachgezogen. | `mypy.ini`, diverse Module |

## Nicht abschließend getestet

* VibeVoice-ASR-Validation End-to-End am echten Pipeline-Run: nur ein
  isolierter Kandidat getestet (erfolgreich, 715 s). Ein vollständiger Lauf
  mit `asr_backend: auto` würde ≈ 60–90 min beanspruchen.
* Prosody-Scoring mit VibeVoice-ASR-Word-Alignments (VV-ASR liefert
  Segment- und Word-Timestamps, die der Prosody-Scorer heute nur aus
  WhisperX konsumiert). Perspektivisch kann das WhisperX obsolet machen.
