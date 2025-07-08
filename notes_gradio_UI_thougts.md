# Gradio Integration Guide für TTS Pipeline CLI

## 🔍 **Aktuelle Situation**

### **Positive Erkenntnisse:**
- ✅ **Chatterbox verwendet tqdm für Progressbars** in `src/chatterbox/models/t3/t3.py:329`
-- ✅ **Strukturierte CLI-Architektur** mit JobManager, TaskOrchestrator, TaskExecutor
- ✅ **Klare Datenstrukturen** (TaskConfig, TaskResult, ExecutionPlan)
 ✅ **Gradio könnte als User Interface verwendet werden, es erkennt tqdm automatisch** - keine expliziten Callbacks nötig

### **Kritische Constraints:**
- ❌ **Chatterbox ist NICHT threadsafe** - kein AsyncJobExecutor möglich
- ❌ **Keine parallele Ausführung** - Tasks müssen seriell abgearbeitet werden

## 📋 **Implementierungsidee**

### **Neue Architektur: Job/Task-Verwaltung mit Run-Mode**

#### **Konzept:**
- **Job/Task-Liste**: Zeigt alle verfügbaren Jobs und deren Tasks, CRUD Operation mit Editor der YAML-Dateien
- **Status-Anzeige**: New, Partial: 48%, Completed (kleine Progress-Bars)
- **Run-Mode**: Toggle-Switch to Start or Stop Background-Task-Processing
- **Polling**: Regelmäßige Status-Updates ohne Blocking
- **Auto-Stop**: Run-Mode schaltet sich bei 100% completion aller Tasks ab

### **Phase 1: Background-Task-Manager **

#### **1.1 Background-Task-Queue**

#### **1.2 Gradio-Interface-Manager**

### **Phase 2: Gradio-Interface mit Job/Task-Verwaltung**

### **Phase 3: Polling-System und Status-Updates**

#### **3.1 Erweiterte Status-Logik Scribble**
```python
# Erweiterte Status-Behandlung für komplexe Tasks
class TaskStatusExtended:
    def __init__(self, task_manager):
        self.task_manager = task_manager
    
    def get_detailed_status(self, task_id):
        """Detaillierte Task-Informationen"""
        task_info = self.task_manager.task_status.get(task_id)
        if not task_info:
            return None
        
        return {
            "task_id": task_id,
            "job_name": task_info["config"].job_name,
            "status": task_info["status"].value,
            "progress": task_info["progress"],
            "message": task_info["message"],
            "start_time": task_info.get("start_time"),
            "estimated_completion": task_info.get("eta")
        }
```

## 🔧 **Technische Details**

### **tqdm-Integration (Bereits vorhanden)**
```python
# In chatterbox/models/t3/t3.py:329
for i in tqdm(range(max_new_tokens), desc="Sampling", dynamic_ncols=True):
    # Token-Generierung mit automatischem Progress-Feedback
    # Gradio erkennt das automatisch - keine Callbacks nötig!
```

### **Job/Task-Verwaltung mit Run-Mode**
```
Gradio UI (Polling) ←→ Interface Manager ←→ Background Task Manager
    ↓                         ↓                        ↓
Task-Tabelle              Job Queue              Worker Thread
Status-Updates         Task-Status-Tracking    (Seriell, eine Task)
Run-Mode-Toggle        Auto-Stop-Logic              ↓
                                                TaskExecutor
                                                (tqdm → Gradio)
```

### **Status-Flow**
1. **Job hinzufügen**: `New` → Task-Queue
2. **Run-Mode starten**: Background-Worker aktivieren
3. **Task-Ausführung**: `New` → `Running` → `Completed`/`Failed`
4. **Polling-Updates**: Interface aktualisiert alle 5s
5. **Auto-Stop**: Run-Mode stoppt bei 100% completion

- **New**: Task in Queue, noch nicht gestartet
- **Running: X%**: Task wird aktuell ausgeführt
- **Completed**: Task erfolgreich abgeschlossen
- **Failed**: Task mit Fehler abgebrochen

## ⚠️ **Wichtige Constraints**

1. **Keine parallele Ausführung**: Chatterbox ist nicht threadsafe
2. **Serieller Task-Flow**: Tasks müssen sequenziell abgearbeitet werden
3. **Blocking Model-Calls**: `model.generate()` blockiert die gesamte Ausführung. Aber das Interface ist eh nur ein Statusanzeige für die Dateibasierte Konfiguration und Dateibasierte Job-Architektur (Chunks completed als Progress-Meter).
4. **Single-User-Limitation**: Nur ein Job gleichzeitig pro Gradio-Instanz (NUr als Lokal laufende Anwendung)