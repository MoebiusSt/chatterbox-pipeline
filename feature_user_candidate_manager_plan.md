# User Candidate Manager - Implementierungsplan

## Übersicht
Implementierung eines Task-Editors zur nachträglichen Bearbeitung der Kandidatenauswahl für Audio-Chunks mit erweiterten CLI-Prompts und persistenter Speicherung in enhanced_metrics.json.

## 1. Architektur-Analyse

### 1.1 Betroffene Komponenten
- `src/main.py` - Haupt-CLI-Interface
- `src/pipeline/job_manager/user_interaction.py` - User-Prompts
- `src/utils/file_manager/` - Dateizugriff für enhanced_metrics.json
- `src/pipeline/task_executor/` - Task-Status-Prüfung
- Neue Komponente: `user_candidate_manager.py`

### 1.2 Datenstrukturen
- **enhanced_metrics.json**: `selected_candidates` Dictionary (chunk_id -> candidate_id)
- **final_metadata.json**: Finale Selektion (read-only nach Assembly)
- **Task State**: Erweiterte Zustandsprüfung für UI-Optionen

## 2. Implementierungsschritte

### Phase 1: Task-Status-Erweiterung
**Ziel**: Proaktive Task-Zustandsprüfung vor dem zweiten Prompt

#### 2.1 Task State Analyzer erweitern
- **Datei**: `src/utils/file_manager/state_analyzer.py`
- **Neue Funktionen**:
  ```python
  def get_comprehensive_task_state(task_path: str) -> TaskState
  def has_candidate_selection_data(task_path: str) -> bool
  def has_complete_candidates(task_path: str) -> bool
  def get_missing_candidates_info(task_path: str) -> Dict
  ```

#### 2.2 TaskState Dataclass erweitern
- **Datei**: `src/pipeline/job_manager/types.py`
- **Neue Felder**:
  ```python
  @dataclass
  class TaskState:
      # existing fields...
      has_candidate_selection: bool
      candidate_editor_available: bool
      missing_candidates: List[int]
      task_status_message: str
  ```

### Phase 2: User Interaction System überarbeiten

#### 2.1 Main.py - Erstes Prompt (unverändert)
- **Datei**: `src/main.py`
- Bleibt wie beschrieben, keine Änderungen

#### 2.2 Zweites Prompt erweitern - Task-Prompt
- **Datei**: `src/pipeline/job_manager/user_interaction.py`
- **Neue Funktion**: `show_task_options_with_state(task_info, task_state)`
- **Erweiterte Optionen**:
  - `[Enter]` - Run task, fill gaps, create new final audio
  - `r` - Run task, fill gaps, don't overwrite existing final audio  
  - `n` - Run task, re-render all candidates, create new final audio
  - `e` - Edit completed task (nur wenn candidate_selection verfügbar)
  - `c` - Cancel

#### 2.3 Task State Message Generation
- **Funktion**: `generate_task_state_message(task_state: TaskState) -> str`
- **Nachrichten**:
  - "Task is complete with final audio available"
  - "Task is complete but missing some candidates"
  - "Task is incomplete and needs to finish"

- Beispiel:
'''
Selected latest task: {job name} - {run_label} - 18.06.2025 06:15

Task state: {task state}

[Task is rendered complete and final audio is available.(Has no missing files, state is complete and final audio and candidate selection is present)|Task is complete with final audio but is missing some candidates and should be re-assembled(has Missing candidates, even though final audio and candidate selection present)|Task is incomplete and needs to finish. (any other state, like missing chunks, candidates, validation etc., and no final audio and candidate selection present)]

What to do with this task?
`[Enter]` - Run task, fill gaps, create new final audio
`r` - Run task, fill gaps, don't overwrite existing final audio  
`n` - Run task, re-render all candidates, create new final audio
`e` - Edit completed task (nur wenn candidate_selection verfügbar)
`c` - Cancel
'''

### Phase 3: User Candidate Manager Implementierung

#### 3.1 Neue Komponente erstellen
- **Datei**: `src/pipeline/user_candidate_manager.py`
- **Hauptklassen**:
  ```python
  class UserCandidateManager:
      def __init__(self, task_path: str)
      def load_candidate_data(self) -> Dict
      def save_candidate_selection(self, selections: Dict)
      def show_candidate_overview(self) -> None
      def edit_chunk_candidate(self, chunk_id: int) -> None
      def get_candidate_info(self, chunk_id: int) -> List[CandidateInfo]
  ```

#### 3.2 Candidate Info Datenstruktur
- **Datei**: `src/pipeline/user_candidate_manager.py`
- **Dataclass**:
  ```python
  @dataclass
  class CandidateInfo:
      candidate_id: int
      exaggeration: float
      cfg_weight: float
      temperature: float
      type: str
      similarity_score: float
      length_score: float
      quality_score: float
      validation_passed: bool
      is_selected: bool
  ```

### Phase 4: UI-Prompts implementieren

#### 4.1 User Candidate Editor Prompt
- **Funktion**: `show_candidate_editor_prompt(task_info, candidates_data)`
- **Features**:
  - Tabellarische Übersicht: Chunk | Cand. | Text | (changed)
  - Navigation: 1-XXX(Int) für Chunk-Auswahl, r für Re-run, c für Return
  - Change-Tracking mit "(changed)" Markierung

Beispiel:
'''
Selected latest task: {job name} - {run_label} - 18.06.2025 06:15

Candidates selected as best matching for the final audio assembly:

Chunk:  Cand.:  Text:
1       1       "Column. Raise Children to Life, not to War. ..."
2       3       "Seven years had passed since the miraculous ..."
3       2       "he event on October 4th was one of many  ..."
4       2       "Introduction ends. On October 7th, I arrive..."
5       5       "My friend Michal Halev flew to Israel to ..."
...usw...

Which Chunk would you like to edit/review?:
1-34    - Select chunk
c       - Return
'''


#### 4.2 User Candidate Selector Prompt  
- **Funktion**: `show_candidate_selector_prompt(chunk_id, candidates, current_text)`
- **Features**:
  - Chunk-Text anzeigen
  - Kandidaten-Tabelle mit Metriken
  - Aktuell gewählter Kandidat mit "<- sel" markiert
  - Sofortige Persistierung bei Auswahl

Beispiel:
'''
Selected latest task: {job name} - {run_label} - 18.06.2025 06:15

Select audio candidate for chunk: 001 / XXX

Text: "Column. Raise Children to Life, not to War. A text by Yael Deckelbaum. Sixteenth February 2024. Introduction. On October 4th, three days before the war broke out, I sang "Prayer of the Mothers" in a joint event of the Israeli Women Wage Peace and the Palestinian "Women of the Sun", two peace organizations led by women from two enemy nations."

Number of candidates: {total_candidates}
Current selected Candidate: 1

Candidate:  exageration:    cfg_weight  temp    type        sim_score   length_score    qty_score   passed
1 <- sel    0.4             0.2         0.9     EXPRESSIVE  0.72        .97             0.83        ✅
2           0.35            0.25        0.96    EXPRESSIVE  0.63        .97             0.78        ❌
3           0.30            0.20        0.98    EXPRESSIVE  0.53        .96             0.72        ❌
etc...

Select action:
1-XX    - Select candidate
c       - Return

'''


### Phase 5: Datenintegration

#### 5.1 Enhanced Metrics Manager
- **Datei**: `src/utils/file_manager/io_handlers/metrics_io.py`
- **Neue Funktionen**:
  ```python
  def load_enhanced_metrics(task_path: str) -> Dict
  def update_selected_candidates(task_path: str, selections: Dict)
  def backup_original_selections(task_path: str) -> Dict
  def get_changed_candidates(original: Dict, current: Dict) -> Set[int]
  ```

#### 5.2 Candidate Data Loader
- **Datei**: `src/utils/file_manager/io_handlers/candidate_io.py`
- **Erweiterte Funktionen**:
  ```python
  def get_candidate_metrics(task_path: str, chunk_id: int) -> List[CandidateInfo]
  def load_chunk_text(task_path: str, chunk_id: int) -> str
  def get_total_candidates_count(task_path: str, chunk_id: int) -> int
  ```

### Phase 6: Integration mit Task Execution

#### 6.1 Modified Task Execution Flow
- **Datei**: `src/pipeline/task_executor/task_executor.py`
- **Änderungen**:
  - Vor Final-Audio-Assembly: Check für custom selections in enhanced_metrics.json
  - Verwendung der User-Selections anstatt der ursprünglichen best candidates
  - Logging der verwendeten custom selections

#### 6.2 Assembly Handler Anpassung
- **Datei**: `src/pipeline/task_executor/stage_handlers/assembly_handler.py`
- **Neue Logik**:
  - Priorität: enhanced_metrics.json selected_candidates > original best candidates
  - Validation der User-Selections (Dateien existieren)
  - Fallback auf original selections bei fehlenden Kandidaten

## 3. Error Handling & Validation

### 3.1 Eingabevalidierung
- Chunk-IDs: 1-basierte User-Eingabe -> 0-basierte interne Verarbeitung
- Kandidaten-IDs: Existenz-Prüfung der Audio-Dateien
- Enhanced_metrics.json: Schema-Validation

### 3.2 Fehlerfälle
- Fehlende enhanced_metrics.json → Editor nicht verfügbar
- Fehlende Kandidaten-Dateien → Warnung + Fallback
- Unvollständige Tasks → Editor-Sperre

## 4. Testing

### 4.1 Unit Tests
- `test_user_candidate_manager.py`
- `test_enhanced_metrics_io.py`  
- `test_task_state_analyzer.py`

### 4.2 Integration Tests
- Vollständiger Workflow: Task-Auswahl → Editor → Re-Assembly
- CLI-Navigation: Alle Prompt-Pfade testen

## 5. Implementierungsreihenfolge

1. **TaskState & State Analyzer** (Phase 1)
2. **Enhanced Metrics I/O** (Phase 5.1)
3. **User Candidate Manager Core** (Phase 3.1)
4. **UI Prompts** (Phase 4)
5. **Main.py Integration** (Phase 2)
6. **Task Execution Integration** (Phase 6)
7. **Testing & Refinement** (Phase 4)

## 7. Technische Notizen

### 7.1 Konsistenz-Checks
- Chunk-Nummerierung: User-facing 1-based, intern 0-based
- File-naming: candidate_01.wav, candidate_02.wav etc.
- JSON-Keys: "0", "1", "2" (strings, nicht integers)

## 8. Dokumentation

### 8.1 User Documentation
- Neuer Abschnitt in README.md: "Task Candidate Editor"
- CLI-Hilfe: Erweiterte Prompt-Beschreibungen

### 8.2 Developer Documentation  
- API-Dokumentation für UserCandidateManager