**User**

Ein Frage zu einem gap filling run mit "resume": Angenommen ich habe aus zwei Chunks sämtliche Audio-Kanidaten gelöscht. (Habe aber an den Text-Chunks nichts verändert und führe dann den resume-Gang aus. Welche Texte werden dann für die gelöschten Audio-Chunks beim neu-rendern herangezogen? Wird das Chunking neu durchlaufen auf den originalen Source-Input-Text? Oder auf den _processed Text im Task-Order unter /texts/? Oder auf die unproccessed input-text-Kopie im Task-Ordner? Oder wird kein neues Chunking durchgeführt und die bereitliegenden chunks_xxx.txt-Dateien werden verwendet? Zweite Frage, was würde passieren, wenn ich auch die entsprechenden chunks_xxx.txt Dateien im /texts/ Ordner löschen würde? Welchen Text würde es dann neu chunken: den Original Source-Input-Text oder die Kopie im /texts/-Ordner, oder dessen processed Variante?

---

**Cursor**

Ich prüfe jetzt direkt den `resume`-Ablauf im Code (`TaskExecutor`/`JobManager`/Chunk-IO), damit ich dir exakt sagen kann, welche Textquelle bei fehlenden Kandidaten bzw. fehlenden `chunks_*.txt` verwendet wird.

Sehr gute Frage. Im aktuellen Code ist das Verhalten klar:

- Bei `resume` wird **nur dann neu gechunkt**, wenn der Task-Status auf `PREPROCESSING` fällt (also effektiv: keine Chunks mehr vorhanden).
- Wenn Chunks vorhanden sind, nutzt Generation **ausschließlich die vorhandenen `texts/chunk_*.txt`** plus `chunks_metadata.json` für Speaker-Metadaten.
- Der `_processed`-Text in `texts/<stem>_processed.txt` (Legacy: `texts/original_<stem>_processed.txt`) ist nur ein Artefakt/Trace und **keine direkte Quelle** für Resume-Generation.
- Die unprocessed Backup-Kopie `texts/<filename>.txt` (Legacy: `texts/original_<filename>.txt`) ist die primäre Quelle für Preprocessing/Chunking, falls Preprocessing erneut läuft.

### 1) Nur Audio-Kandidaten aus 2 Chunks gelöscht (Chunks unverändert)
Dann passiert bei `resume`:

- Kein neues Chunking.
- `GenerationHandler` lädt bestehende Chunks über `get_chunks()` und rendert nur fehlende Kandidaten pro Chunk nach.
- Textquelle für Re-Render ist der jeweilige Inhalt aus `texts/chunk_XXX.txt`.

Das sieht man hier:

```82:89:src/pipeline/task_executor/stage_handlers/generation_handler.py
chunks = self.file_manager.get_chunks()
if not chunks:
    logger.error("No chunks found for generation")
    return False
```

und:

```101:119:src/utils/file_manager/io_handlers/chunk_io.py
# Load chunk files
chunk_files = sorted(self.texts_dir.glob("chunk_*.txt"))
...
with open(chunk_file, "r", encoding="utf-8") as f:
    text = f.read()
```

---

### 2) Wenn du zusätzlich `texts/chunk_*.txt` löschst
Hier gibt es zwei unterschiedliche Fälle:

- **Alle** `chunk_*.txt` gelöscht:  
  Dann `has_chunks=False` -> Preprocessing läuft neu -> Chunking läuft neu auf `get_input_text()`.

- **Nur einige** `chunk_*.txt` gelöscht:  
  Dann werden einfach weniger Chunks geladen; es gibt **keine Soll-Anzahl-Prüfung** gegen frühere Chunkzahl.  
  Das kann zu inkonsistentem Zustand führen (fehlende Chunks werden nicht „rekonstruiert“, weil sie nicht mehr im geladenen Chunk-Set existieren).

Wichtig: Bei neuem Preprocessing kommt der Text aus `get_input_text()`, und das bevorzugt task-lokal:

`get_input_text()` bevorzugt task-lokal über `_resolve_task_input_backup_path()`:

1. `texts/<filename>.txt` (aktuelle Backup-Kopie im Task)
2. `texts/original_<filename>.txt` (Legacy-Backup, alte Tasks)
3. sonst `data/input/texts/...` laut Config

Der `_processed`-Text wird nicht als Input geladen; er wird nur geschrieben nach `processed_text_path()` (`<stem>_processed.txt`, Legacy: `original_<stem>_processed.txt`).

---

**User**

Und wie geht der Rebuild Vorgang im Unterschied zum Resume vor? Oder verhält dieser sich gleich, nur das er automatisch alle audio-Chunks vorher entfernt? Oder wirft er auch alle Text Chunks weg und chunked neu?

---

**Cursor**

Ich schaue kurz in den `rebuild`-Pfad (`cbpipe.py`/`job_manager`) und bestätige dir exakt, welche Artefakte dabei gelöscht werden (nur Audio/Validation oder auch Text-Chunks).

Kurz: **`rebuild` ist nicht identisch zu `resume`**.

- `rebuild` setzt intern `rerender_all=True` **und** `force_final_generation=True`.
- Dadurch wird zu Beginn explizit gelöscht:
  - kompletter `candidates/`-Inhalt
  - kompletter `whisper/`-Inhalt inkl. `whisper_metrics.json`
  - `final/*_final.wav` + `final/*_final_metadata.json`
  - `task_metrics.json`

Siehe Löschlogik in `src/pipeline/task_executor/task_executor.py` (`_delete_all_candidates_and_validation`).

**Was nicht gelöscht wird bei `rebuild`:**
- `texts/chunk_*.txt`
- `texts/chunks_metadata.json`
- `texts/<input-filename>.txt` (bzw. Legacy `texts/original_*.txt`)
- `texts/<stem>_processed.txt` (bzw. Legacy `texts/original_*_processed.txt`)

Also: **Text-Chunks bleiben erhalten**. Danach wird neu gerendert/validiert/assembled auf Basis der vorhandenen Chunks.

### Unterschied zu `resume`
- `resume` füllt nur Lücken (fehlende Kandidaten/Whisper/Selections), möglichst selektiv.
- `rebuild` löscht Audio/Validation/Metriken hart und baut diese Stufen komplett neu auf.

### Nur falls Text-Chunks fehlen
Dann fällt der Run auf Preprocessing/Chunking zurück und erzeugt Chunks neu (Quelle wie zuvor erklärt: task-lokale Input-Kopie in `texts/` bevorzugt).

---

**User**

Ich verstehe. Wenn es also Probleme mit der yaml-Configuration gab und ich Korrektur-Läufe machen möchte, dann kann ich die task.yaml korrigieren und eien resume-Lauf verwenden. Oder wenn ich Texte Ändere und neu rendern lassen möchte, dann muss ich am besten die Kopie des Quelltexts im Task-Ornder in /texts/ ändern und alle(!) Text-Chunks-löschen, damit er neu ge-chunked wird, und dazu die entsprechenden audio-Kanidaten chunks in /candidates/ löschen (nur die von betroffenen audio-Kanidaten) und dann kann ich einen resume lauf machen lassen. Und wenn etwas mit der Sprecher-Konfiguration nicht gestimmt hat, dann korrigiere ich am besten gleich die job.yaml und lasse mit create lieber einen vollständg neuen Task ableiten.
Der Rebuild ist also im Grunde nur ein Resume mit vorausgegangen Audio-Chunk-Löschung, aber er verwendet immernoch die Task.yaml – er leitet NICHT erneut von der Job.Yaml ab.