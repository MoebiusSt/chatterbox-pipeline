# Task Executor Unification Plan

## Analyse der aktuellen Situation

### Identifizierte Kern-Logik

Die Task-Ausführung ist bereits im `TaskExecutor.execute_task()` vereinheitlicht. Die Duplikation liegt nur in der **Orchestrierung** um diese Kern-Logik herum.

**Gemeinsame Kern-Logik (identisch in beiden Pfaden):**
```python
# 1. Config-Loading
if task_config.preloaded_config:
    loaded_config = task_config.preloaded_config
else:
    loaded_config = config_manager.load_cascading_config(task_config.config_path)

# 2. FileManager-Erstellung
file_manager = FileManager(
    task_config, 
    preloaded_config=loaded_config, 
    config_manager=config_manager
)

# 3. TaskExecutor-Ausführung
task_executor = TaskExecutor(file_manager, task_config, config=loaded_config)
result = task_executor.execute_task()
```

### Unterschiede zwischen Single/Batch

| Aspekt | Single-Pfad (main.py) | Batch-Pfad (BatchTaskExecutor) |
|--------|----------------------|--------------------------------|
| **Logging** | Detaillierte Success-/Error-Logs | Progress-Tracking mit i/total |
| **Zeit-Formatierung** | HH:MM:SS Format | Sekunden-Anzeige |
| **Error-Handling** | Direkte TaskResult-Verarbeitung | Aggregierte Fehler-Sammlung |
| **Result-Verarbeitung** | `completion_stage` bei Fehlern | Batch-Statistiken |
| **Exit-Code** | `result.success ? 0 : 1` | `batch_result.failed_tasks == 0 ? 0 : 1` |

## Refactoring-Strategie

### Ansatz: Adaptives Logging im BatchTaskExecutor

Anstatt zwei separate Pfade zu haben, wird `BatchTaskExecutor` so erweitert, dass er je nach Anzahl der Tasks unterschiedliche Logging-Modi verwendet.

### Schritt 1: Erweiterte BatchTaskExecutor-Klasse

```python
class UnifiedTaskExecutor:
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.logging_mode = "auto"  # "single", "batch", "auto"
    
    def execute_tasks(self, task_configs: List[TaskConfig]) -> ExecutionResult:
        """Unified execution for single or multiple tasks."""
        is_single_task = len(task_configs) == 1
        
        if self.logging_mode == "auto":
            current_mode = "single" if is_single_task else "batch"
        else:
            current_mode = self.logging_mode
        
        # Adaptives Logging basierend auf Modus
        if current_mode == "single":
            return self._execute_single_mode(task_configs)
        else:
            return self._execute_batch_mode(task_configs)
```

### Schritt 2: Einheitliche Result-Struktur

```python
@dataclass
class ExecutionResult:
    """Unified result structure for both single and batch execution."""
    task_results: List[TaskResult]
    execution_time: float
    success: bool
    
    @property
    def is_single_task(self) -> bool:
        return len(self.task_results) == 1
    
    @property
    def failed_tasks(self) -> int:
        return sum(1 for r in self.task_results if not r.success)
    
    @property
    def success_rate(self) -> float:
        return (len(self.task_results) - self.failed_tasks) / len(self.task_results) * 100
```

### Schritt 3: Adaptive Logging-Implementierung

```python
def _execute_single_mode(self, task_configs: List[TaskConfig]) -> ExecutionResult:
    """Execute with single-task logging style."""
    result = self._execute_single_task(task_configs[0])
    
    # Single-task style logging
    self._log_single_task_result(result)
    
    return ExecutionResult(
        task_results=[result],
        execution_time=result.execution_time,
        success=result.success
    )

def _execute_batch_mode(self, task_configs: List[TaskConfig]) -> ExecutionResult:
    """Execute with batch-task logging style."""
    # Existing batch logic with progress tracking
    # ...
```

## Implementierungsschritte

### Phase 1: Unified Executor erstellen
1. **Neue Klasse**: `UnifiedTaskExecutor` in `src/pipeline/unified_task_executor.py`
2. **Adaptive Logging**: Implementierung der unterschiedlichen Logging-Modi
3. **Result-Struktur**: Einheitliche `ExecutionResult` Klasse

### Phase 2: Integration in main.py
1. **Ersetzen der Verzweigung**: Beide Pfade verwenden `UnifiedTaskExecutor`
2. **Rückwärtskompatibilität**: Identische Log-Ausgaben beibehalten
3. **Testing**: Alle bestehenden Tests müssen weiterhin funktionieren

### Phase 3: Cleanup
1. **BatchTaskExecutor**: Kann entfernt werden
2. **Code-Duplikation**: Entfernung der duplizierten Ausführungslogik
3. **Dokumentation**: Update der Dokumentation

## Risiken und Gegenmaßnahmen

### Risiko: Rückwärtskompatibilität
- **Problem**: Bestehende Tests und Logs könnten sich ändern
- **Lösung**: Bit-identische Reproduktion der bestehenden Ausgaben

## Erfolgskriterien

1. ✅ **Funktional**: Alle bestehenden Tests bestehen
2. ✅ **Logging**: Identische Ausgaben wie vorher
3. ✅ **Performance**: Keine messbare Verlangsamung
4. ✅ **Wartbarkeit**: Nur noch ein Ausführungspfad
5. ✅ **Erweiterbarkeit**: Neue Logging-Modi einfach hinzufügbar

## Nächste Schritte

1. **Implementierung**: `UnifiedTaskExecutor` erstellen
2. **Integration**: In `main.py` integrieren
3. **Testing**: Umfassende Tests für beide Modi
4. **Cleanup**: Alte Strukturen entfernen
5. **Dokumentation**: Updates und Beispiele

## Implementierungsstatus

### ✅ Drastische Vereinfachung abgeschlossen

Nach der Analyse stellte sich heraus: **Es gibt KEINE funktionalen Unterschiede zwischen Single/Batch außer Logging!**

Daher wurde eine drastische Vereinfachung implementiert:

1. **TaskOrchestrator**: Ersetzt alle Modi-Komplexität durch eine einfache Schleife
2. **Keine Modi**: Kein "single" vs "batch" - nur optionales Progress-Logging
3. **Minimale API**: Einfache `execute_tasks()` Methode ohne Abstraktionen
4. **Tests**: Erfolgreich getestet

### 🚀 Implementierte Vereinfachung

**Vorher: Komplexe Modi-Unterscheidung**
- UnifiedTaskExecutor mit adaptive logging modes
- ExecutionResult wrapper structure  
- Complex logging differentiation

**Nachher: Einfache Orchestrierung**
```python
def execute_tasks(self, task_configs: List[TaskConfig]) -> List[TaskResult]:
    results = []
    total_tasks = len(task_configs)
    
    for i, task_config in enumerate(task_configs, 1):
        # Optional progress log for multiple tasks
        if total_tasks > 1:
            logger.info(f"▶️ Task {i}/{total_tasks}: {task_config.job_name}:{task_config.task_name}")
        
        result = self._execute_single_task(task_config)
        results.append(result)
        
        # Optional completion log
        if total_tasks > 1:
            status = "✅" if result.success else "❌"
            logger.info(f"{status} {task_config.job_name}:{task_config.task_name}")
    
    return results
```

### 📊 Messergebnisse

- **Code-Reduktion**: ~150 Zeilen komplexe Abstraktion eliminiert
- **Wartbarkeit**: Einfache, verständliche Logik
- **Performance**: Kein Overhead durch unnötige Abstraktionen
- **Tests**: Erfolgreich verifiziert

### 💡 Wichtige Erkenntnis

Die ursprüngliche Unterscheidung zwischen "single" und "batch" execution war eine **unnötige Abstraktion**. Die einzigen Unterschiede waren:
- Progress-Logging bei mehreren Tasks
- Unterschiedliche Summary-Formate

Dies rechtfertigte NICHT die Komplexität von zwei separaten Ausführungspfaden.

### 🎯 Finale Lösung

**TaskOrchestrator ersetzt alle Komplexität:**
- Führt 1 Task aus → kein Progress-Log
- Führt mehrere Tasks aus → zeigt Progress an
- Immer derselbe Kern-Code: `TaskExecutor.execute_task()`

### 🏗️ Finale Architektur

```
TaskOrchestrator  (Orchestrierung für 1 oder mehrere Tasks)
└── TaskExecutor    (Echte Task-Ausführung - TTS Pipeline)
    ├── PreprocessingHandler
    ├── GenerationHandler  
    ├── ValidationHandler
    └── AssemblyHandler
```

---

*Erstellt: 2024-12-01*  
*Status: ✅ Drastische Vereinfachung abgeschlossen + Umbenennung zu TaskOrchestrator* 