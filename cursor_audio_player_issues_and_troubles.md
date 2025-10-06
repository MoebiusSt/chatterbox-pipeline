- WSL2 Audio-System diagnostizieren und Audio-Packages installieren
- Audio-Code im chatterbox_tester.py analysieren
- chatterbox_tester.py auf ffplay-only Audio umstellen
- PulseAudio für WSL2 Audio-Support installieren
- WSL2 Audio ohne sudo mit PulseAudio-Software-Backend konfigurieren

## Audio-Problem behoben

**Diagnose:**
- WSL2 hat keine Audio-Hardware-Geräte
- pygame/mixer nutzt ALSA direkt und funktioniert nicht in WSL2
- ffplay nutzt WSLg-PulseAudio und funktioniert

**Änderungen:**
1. pygame/mixer deaktiviert (ALSA-Fehler)
2. ffplay als einziger Audio-Player aktiviert
3. pygame-Fallback entfernt

**Ergebnis:**
- ffplay läuft über WSLg-PulseAudio
- Audio-Test erfolgreich
- chatterbox_tester.py startet ohne pygame-Fehler

Der chatterbox_tester nutzt jetzt ausschließlich ffplay für Audio-Wiedergabe. Das behebt die Audio-Probleme in WSL2.

## Audio-Player-Stabilität bei langen Renderings verbessert

**Problem:**
- Bei langen Audio-Renderings wurde die Wiedergabe choppy und unzuverlässig
- Player blieb hängen und spielte nur noch kurze Bruchteile
- Neu-Rendern oder Reiter-Wechsel halfen nicht mehr
- Nur Programm-Neustart behebt das Problem

**Ursachen:**
1. Race Conditions in `_update_position()` - Timer wurden nicht korrekt gecancelt
2. Unvollständiges Prozess-Cleanup - ffplay-Prozesse wurden nicht sauber beendet
3. Memory-Leaks durch Timer-Akkumulation bei schnellen Start/Stop-Zyklen
4. Fehlende Synchronisation zwischen Threads

**Änderungen:**
1. Timer-Management verbessert - bestehende Timer werden vor neuen gecancelt
2. Prozess-Cleanup verstärkt - längere Timeouts für graceful shutdown
3. Race-Condition-Schutz - saubere State-Übergänge zwischen Play/Stop
4. Robustere Fehlerbehandlung - bessere Exception-Behandlung bei Prozess-Fehlern
5. Cleanup bei Programmende - alle Ressourcen werden ordnungsgemäß freigegeben

**Ergebnis:**
- Stabile Audio-Wiedergabe auch bei langen Renderings
- Keine hängenden ffplay-Prozesse mehr
- Saubere State-Übergänge ohne Race Conditions
- Robustes Cleanup verhindert Memory-Leaks

