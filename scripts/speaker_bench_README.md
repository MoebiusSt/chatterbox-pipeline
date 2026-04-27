# Speaker Benchmark Campaign — quick start

## Was ist hier drin

- `_template.yaml` — die Vorlage-YAML mit den Platzhaltern
- `generate_benchmark_yamls.py` — Generator-Script
- `test_text_DE.txt` / `test_text_EN.txt` — zwei sprachspezifische Testtexte mit variierender Absatzlänge (sehr kurz / kurz / kurz / mittel / lang)
- `generated/` — 25 fertige YAMLs, eine pro Sprecher, plus README mit Tabelle

## Sprecher-Sprachzuordnung

Nach Regel: alles mit `_DE` oder `DE_` im Dateinamen ist Deutsch, der Rest Englisch. Daraus ergeben sich **7 DE-Sprecher** und **18 EN-Sprecher** (nach Deduplizierung von `stephan_moebius_4`, der in der Eingabeliste doppelt war).

Alle `gesine_*`, `mike_kamp_1`, `maryjohn` wurden nach dieser Regel als EN klassifiziert. Wenn das für Gesine oder Mike Kamp 1 anders sein sollte, einfach die Zuordnung in `generate_benchmark_yamls.py` manuell korrigieren und neu generieren.

## Zu erwartendes Setup

1. Texte nach `data/input/texts/speaker_bench/` kopieren (oder welches Input-Verzeichnis cbpipe erwartet)
2. Die 25 YAMLs aus `generated/` in dein Config-Verzeichnis
3. Reference-Audio-Dateien müssen wie bei Erik üblich im Reference-Audio-Verzeichnis liegen
4. Seriell abarbeiten — ich würde mit einem Deutschen starten (kurz) um zu prüfen, dass das Chunking wie geplant 5-6 Chunks produziert

## Rendertime-Schätzung

~10 Kandidaten × 5-6 Chunks × 73 s ≈ 60-70 Minuten pro Sprecher.
25 Sprecher × 65 min ≈ 27 Stunden seriell.

Falls einzelne Sprecher in unter 45 Minuten fertig sind, vermutlich weil das Chunking nur 4 Chunks produziert — auch ok, aber gib mir später Bescheid welche das waren, dann passe ich die Auswertung an.

## Was du danach mir schickst

Pro Sprecher die `analysis_metrics.json` aus dem Task-Verzeichnis. Am einfachsten ist, die 25 Dateien mit einem Umbenennungsmuster zu versehen (z.B. `peter_yearsley_1_analysis.json`) und mir als ZIP oder einzeln zu schicken. Wenn's zu viele sind, kann ich auch in Batches von 5-10 arbeiten.

## Auswertung pro Sprecher

Für jeden Sprecher extrahiere ich dann:

1. **Halluzinations-Klippe** — ab welcher Position bricht `whisper_similarity` ein oder wird `is_valid=false`
2. **Einfrier-Klippe** — ab welcher Position kollabiert `wpm` oder `prosody_liveliness`
3. **Sichere Zone** — der Positionsbereich dazwischen
4. **Konkrete Parameter-Empfehlung** — mittlere 3-4 Positionen aus der sicheren Zone, in cbpipe-YAML-Syntax

Erik hat gezeigt, dass selbst im sicheren Bereich die automatischen Scores unzuverlässig für Lebendigkeit sind. Die Empfehlung zielt darum nicht auf "der beste Kandidat laut Score" sondern auf "der robuste Bereich laut Validierung" — genau das, was du wolltest.
