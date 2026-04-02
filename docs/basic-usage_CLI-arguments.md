Referenz:

---

## Grundaufruf

```bash
cd /home/stephan/projekte/chatterbox-pipeline
python src/cbpipe.py config/test_turbo.yaml
```

Config-Dateien werden sowohl relativ zum CWD als auch relativ zum `config/`-Ordner gesucht – beide Varianten funktionieren:

```bash
python src/cbpipe.py config/test_turbo.yaml
python src/cbpipe.py test_turbo.yaml          # sucht automatisch in config/
```

---

## Alle Parameter

### Job-Auswahl (entweder Config-Datei ODER `--job`, nie beides)

```bash
python src/cbpipe.py config/test_turbo.yaml         # eine Config-Datei
python src/cbpipe.py config/a.yaml config/b.yaml    # mehrere auf einmal
python src/cbpipe.py --job "test_turbo"             # nach Job-Name
python src/cbpipe.py --job "test*"                  # Wildcard: alle Jobs mit "test..."
python src/cbpipe.py --job "test?job"               # Wildcard: test1job, test2job, ...
```

### `--mode` / `-m` — Ausführungsstrategie

```bash
--mode new       # Neuen Task anlegen und ausführen
--mode latest    # Letzten vorhandenen Task weiterführen/vervollständigen
--mode all       # Alle vorhandenen Tasks eines Jobs ausführen
--mode "job1:new,job2:all"   # Unterschiedlich pro Job
```

Ohne `--mode` erscheint ein interaktives Menü.

### Neurendering-Flags

```bash
--force-final-generation  (-f)   # Finales Audio neu assemblieren aus vorhandenen Kandidaten
                                  # (auch wenn final.wav schon existiert)
--rerender-all            (-r)   # Alle Kandidaten löschen und komplett neu rendern
                                  # (fragt sicherheitshalber nach Bestätigung)
```

### Feature-Overrides (überschreiben die Config)

```bash
--enable-prosody        # Prosody-Scorer aktivieren (auch wenn in Config disabled)
--enable-tail-trim      # Tail-Trim-Preprocessing aktivieren
```

### Sonstige

```bash
--device auto|cpu|cuda|mps    # Gerät explizit setzen (Standard: auto)
--verbose / -v                # Ausführliches Logging (DEBUG-Level auf Konsole)
--explain-cache               # Model-Cache-Verhalten erklären und beenden
--cli-menu-help               # CLI-Menü-Äquivalente und erweiterte Hilfe anzeigen
```

---

## Häufigste Aufrufe in der Praxis

```bash
# Neuen Turbo-Test anlegen und rendern
python src/cbpipe.py config/test_turbo.yaml --mode new

# Letzten Task fortsetzen (fehlende Kandidaten ergänzen)
python src/cbpipe.py config/test_turbo.yaml --mode latest

# Nur finales Audio neu zusammensetzen (Kandidaten bleiben)
python src/cbpipe.py config/test_turbo.yaml --mode latest -f

# Komplett neu von vorne (alle Kandidaten löschen)
python src/cbpipe.py config/test_turbo.yaml --mode new -r

# Mit vollem Debug-Logging
python src/cbpipe.py config/test_turbo.yaml --mode latest -v
```