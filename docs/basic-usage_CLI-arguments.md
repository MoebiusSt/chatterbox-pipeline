Referenz:

---

## Grundaufruf

```bash
cd /home/stephan/projekte/chatterbox-pipeline
python src/cbpipe.py create config/test_turbo.yaml
```

Config-Dateien werden sowohl relativ zum CWD als auch relativ zum `config/`-Ordner gesucht – beide Varianten funktionieren:

```bash
python src/cbpipe.py create config/test_turbo.yaml
python src/cbpipe.py create test_turbo.yaml          # searches automatically in config/
```

---

## Alle Parameter

### Job selection (either config file OR `--job`, never both)

```bash
python src/cbpipe.py resume config/test_turbo.yaml         # one config file
python src/cbpipe.py resume config/a.yaml config/b.yaml    # multiple configs
python src/cbpipe.py resume --job "test_turbo"             # by job name
python src/cbpipe.py resume --job "test*"                  # wildcard: all jobs with "test..."
python src/cbpipe.py resume --job "test?job"               # wildcard: test1job, test2job, ...
```

### Verb commands

```bash
create       # Create new task(s) from job YAML
resume       # Fill gaps in the latest existing task per job
reassemble   # Regenerate final audio from existing candidates
rebuild      # Delete candidates and rerender everything
edit         # Open candidate editor for the latest task
```

`resume`, `reassemble`, and `rebuild` support `--all` to process all existing tasks in the selected job scope. Without a command, the interactive menu opens.

```bash
python src/cbpipe.py resume config/test_turbo.yaml
python src/cbpipe.py resume config/test_turbo.yaml --all
python src/cbpipe.py reassemble config/test_turbo.yaml
python src/cbpipe.py rebuild config/test_turbo.yaml
```

### Feature-Overrides (überschreiben die Config)

```bash
--enable-prosody        # Prosody-Scorer aktivieren (auch wenn in Config disabled)
--enable-tail-trim      # Tail-Trim-Preprocessing aktivieren
```

### Other options

```bash
--device auto|cpu|cuda|mps    # Gerät explizit setzen (Standard: auto)
--verbose / -v                # Ausführliches Logging (DEBUG-Level auf Konsole)
--explain-cache               # Model-Cache-Verhalten erklären und beenden
```

---

## Häufigste Aufrufe in der Praxis

```bash
# Neuen Turbo-Test anlegen und rendern
python src/cbpipe.py create config/test_turbo.yaml

# Letzten Task fortsetzen (fehlende Kandidaten ergänzen)
python src/cbpipe.py resume config/test_turbo.yaml

# Nur finales Audio neu zusammensetzen (Kandidaten bleiben)
python src/cbpipe.py reassemble config/test_turbo.yaml

# Komplett neu von vorne (alle Kandidaten löschen)
python src/cbpipe.py rebuild config/test_turbo.yaml

# Mit vollem Debug-Logging
python src/cbpipe.py resume config/test_turbo.yaml -v
```