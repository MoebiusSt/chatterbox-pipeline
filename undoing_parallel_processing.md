## Refactoring Plan: Undoing Parallel Processing

### 1. Analyse der betroffenen Komponenten

**Hauptkomponenten:**
- **BatchExecutor**: ThreadPoolExecutor, parallel_enabled, max_workers
- **TTSGenerator**: SerializedModelAccess, Thread-Logging  
- **ModelCache**: SerializedModelAccess, ConditionalCache, Threading-Locks
- **CLI**: --parallel flag, max_workers argument
- **Dokumentation**: README.md, Help-Texte

**Betroffene Dateien:**
1. `src/pipeline/batch_executor.py` - Hauptfokus
2. `src/generation/tts_generator.py` - Thread-Safety entfernen
3. `src/generation/model_cache.py` - Serialization entfernen
4. `src/main.py` - CLI-Argumente entfernen
5. `README.md` - Dokumentation aktualisieren

### 2. Refactoring-Reihenfolge

**Phase 1: CLI und BatchExecutor Vereinfachung**
1. **main.py**: Entfernung von --parallel und --max-workers
2. **batch_executor.py**: Entfernung von ThreadPoolExecutor und parallel_enabled
3. **Alle Aufrufe von BatchExecutor**: Parameter-Anpassung

**Phase 2: TTS-Generator Vereinfachung**
1. **tts_generator.py**: Entfernung von SerializedModelAccess
2. **model_cache.py**: Entfernung von Thread-Safety-Komponenten
3. **Direkter Modell-Zugriff**: Wiederherstellung der einfachen Logik

**Phase 3: Cleanup und Dokumentation**
1. **Threading-Imports**: Entfernung aller threading-bezogenen Imports
2. **Thread-Logging**: Entfernung von Thread-IDs aus Logging
3. **Dokumentation**: README.md und Help-Texte aktualisieren

### 3. Detaillierte Änderungen

**3.1 main.py**
- Entfernung von `--parallel` und `--max-workers` CLI-Argumenten
- Vereinfachung der BatchExecutor-Initialisierung
- Anpassung der Help-Texte

**3.2 batch_executor.py**
- Entfernung von `ThreadPoolExecutor` und `concurrent.futures` imports
- Entfernung von `parallel_enabled` und `max_workers` Parameter
- Vereinfachung zu rein serieller Ausführung mit for-loop
- Entfernung von `as_completed` Logik

**3.3 tts_generator.py**
- Entfernung von `SerializedModelAccess` Aufrufen
- Entfernung von `threading` imports und `_thread_local`
- Wiederherstellung der direkten `self.model` Nutzung
- Vereinfachung der `generate_single` Methode

**3.4 model_cache.py**
- Entfernung der gesamten `SerializedModelAccess` Klasse
- Entfernung der `ConditionalCache` Klasse (bereits deprecated)
- Entfernung aller Threading-Locks und RLock
- Vereinfachung zu einfacher Singleton-Implementierung

**3.5 README.md**
- Entfernung von parallel execution Beispielen
- Aktualisierung der Feature-Liste

### 4. Validierung und Testing

**4.1 Funktionalitätsprüfung**
- Alle bisherigen Funktionen müssen weiterhin funktionieren
- Race-Conditions dürfen nicht mehr auftreten (da seriell)
- Performance-Regression ist akzeptabel zugunsten von Stabilität

**4.2 Test-Strategie**
- Laufende Einzeltasks (wie bisher)
- Mehrere Tasks seriell (anstatt parallel)
- Verschiedene Job-Konfigurationen

**4.3 Ausführungsgeschwindigkeit**
- Länger ist akzeptabel - Stabilität ist wichtiger
- Übersichtlichkeit und Debugging-Freundlichkeit hat Priorität

### 5. Risiken und Mitigationen

**5.1 Risiken**
- Potentielle Änderungen in Aufruf-Patterns
- Mögliche Abhängigkeiten von Threading-Verhalten

**5.2 Mitigationen**
- Schrittweise Änderung mit Validierung nach jedem Schritt
- Beibehaltung aller bestehenden Funktionalitäten
- Vereinfachung der Codebasis für bessere Wartbarkeit

### 6. Erwartete Vorteile

- Deutlich weniger komplexer Code
- Bessere Nachvollziehbarkeit der Ausführung
- Keine Thread-Safety-Überlegungen mehr nötig
- Fokus auf Kernfunktionalität

### 7. Implementierungs-Checkliste

- [ ] **Phase 1: CLI und BatchExecutor**
  - [ ] main.py: CLI-Argumente entfernen
  - [ ] batch_executor.py: ThreadPoolExecutor entfernen
  - [ ] Alle BatchExecutor-Aufrufe anpassen
- [ ] **Phase 2: TTS-Generator**  
  - [ ] tts_generator.py: SerializedModelAccess entfernen
  - [ ] model_cache.py: Thread-Safety entfernen
  - [ ] Direkter Modell-Zugriff wiederherstellen
- [ ] **Phase 3: Cleanup**
  - [ ] Threading-Imports entfernen
  - [ ] Thread-Logging entfernen
  - [ ] Dokumentation aktualisieren
- [ ] **Validierung**
  - [ ] Funktionalitätstests
  - [ ] Race-Condition-Tests (sollten nicht mehr auftreten)
  - [ ] Performance-Vergleich (akzeptabel wenn länger)

### 8. Notizen

- Alle bestehenden Konfigurationsdateien sollen weiterhin funktionieren
- Die Programm-Architektur wird vereinfacht, nicht grundlegend verändert

