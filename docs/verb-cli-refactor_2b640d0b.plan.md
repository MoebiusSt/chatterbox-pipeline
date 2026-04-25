---
name: verb-cli-refactor
overview: "Konzept B: Ersatz der `--mode`-Oberfläche durch Subkommandos (create/resume/reassemble/rebuild/edit) mit orthogonalem `--all`-Scope. Die interne TaskConfig-Semantik (`force_final_generation`, `rerender_all`) bleibt unverändert; nur die CLI-Oberfläche, die Intent-Resolution, die Menu-Vokabel und Doku werden umgebaut. Per-Job-Strategien entfallen ersatzlos."
todos:
  - id: verb_enum
    content: Verb-Enum in types.py einfuehren; ExecutionOptions & ExecutionIntent anpassen; alte Enums entfernen
    status: completed
  - id: menu_refactor
    content: menu_orchestrator.py auf neue Verben/Labels umbauen
    status: completed
  - id: cli_mapper
    content: cli_mapper.py vereinfachen (verb_to_options); parse_mode_argument entfernen
    status: completed
  - id: planner
    content: execution_planner.py auf args.command + neue Verben umstellen
    status: completed
  - id: cbpipe_parser
    content: "cbpipe.py: argparse auf Subparser umbauen, alte --mode/--force-final-generation/--rerender-all/--cli-menu-help entfernen"
    status: completed
  - id: scripts
    content: scripts/test_menu_orchestrator.py, test_task_system.py, test_regenerate_final.py, test_model_cache.py auf neue Verben anpassen
    status: completed
  - id: docs
    content: README.md, docs/basic-usage_CLI-arguments.md, docs/TECHNICAL_OVERVIEW.md, docs/readme_agent.md, docs/PERSISTENT_CACHE.md aktualisieren
    status: completed
  - id: future_doc
    content: docs/CLI_VERBS_FUTURE_IDEAS.md neu anlegen mit zurueckgestellten Ideen
    status: completed
  - id: smoke
    content: Lokalen Rauchtest mit create/resume/reassemble/rebuild/edit durchfuehren
    status: completed
isProject: false
---

# Konzept B: Verb-basierte CLI-Oberflaeche

## Zielbild

```
cbpipe create     [CONFIG...] [--job NAME]               # Neuer Task je Job-YAML
cbpipe resume     [CONFIG...] [--job NAME] [--all]       # Gap-fill, Final nur wenn fehlt
cbpipe reassemble [CONFIG...] [--job NAME] [--all]       # Nur Final-Audio neu
cbpipe rebuild    [CONFIG...] [--job NAME] [--all]       # Alle Kandidaten verwerfen + Final neu
cbpipe edit       [CONFIG]    [--job NAME]               # Interaktiver Candidate-Editor
cbpipe                                                   # Interaktives Menue (wie heute)
```

Scope-Default: neuester Task pro Job. `--all` = alle Tasks. Wildcards (YAML-Liste via Shell) bestimmen die Job-Menge.

## Internes Verb-Mapping (bleibt kompatibel zu vorhandenen TaskConfig-Feldern)

- `create`:    neue Task-YAML, `force_final_generation=False`, `rerender_all=False`
- `resume`:    `force_final_generation=False`, `rerender_all=False`
- `reassemble`: `force_final_generation=True`,  `rerender_all=False`
- `rebuild`:    `force_final_generation=True`,  `rerender_all=True`
- `edit`:      wie `resume`, aber Start im Editor vor Ausfuehrung

## Aenderungen

### 1) CLI-Parser: [src/cbpipe.py](src/cbpipe.py)

- `parse_arguments` komplett umbauen auf `argparse.add_subparsers(dest="command", required=False)`.
- Wenn `args.command is None` und keine YAML/`--job` -> interaktives Menue (wie heute bei leerem Aufruf).
- Subparser fuer `create`, `resume`, `reassemble`, `rebuild`, `edit`. Alle erhalten:
  - `config_files` (positional, `nargs="*"`, `type=Path`)
  - `--job/-j` (Pattern wie bisher)
  - Gemeinsame Flags: `--verbose`, `--device`, `--enable-prosody`, `--enable-tail-trim`.
- Nur `resume`/`reassemble`/`rebuild` bekommen `--all`.
- `--mode`, `--force-final-generation`, `--rerender-all`, `--cli-menu-help` fallen ersatzlos weg.
- `confirm_cli_rerender_action` bleibt; Aufruf wird an `rebuild`-Verb geknuepft.
- Fehlermeldung fuer `--job` + positional bleibt (identischer Check).

### 2) Datentypen: [src/pipeline/job_manager/types.py](src/pipeline/job_manager/types.py), [src/pipeline/job_manager/execution_types.py](src/pipeline/job_manager/execution_types.py)

- Neues Enum `Verb` in `types.py`: `CREATE, RESUME, REASSEMBLE, REBUILD, EDIT`.
- `ExecutionStrategy`, `UserChoice`, `TaskSelectionChoice`, `TaskOptionsChoice`, `AllTasksChoice` ersatzlos entfernen.
- `ExecutionIntent` erhaelt Feld `verb: Verb`, das bisherige `source`/`execution_mode` bleiben.
- `ExecutionOptions` vereinfacht (nur `force_final_generation`, `rerender_all`, `edit_mode: bool`).
- `ExecutionPlan.execution_mode`: weiterhin `"single" | "batch" | "cancelled"` (keine Aenderung an Downstream).

### 3) Intent-Resolution: [src/pipeline/job_manager/cli_mapper.py](src/pipeline/job_manager/cli_mapper.py)

- `CLIMapper` drastisch vereinfachen: Funktion `verb_to_options(verb, scope_all) -> (ExecutionOptions, task_selector)` statt Strategy-Map.
- `parse_cli_to_execution_intent`: liest `args.command` und `args.all`, waehlt Task-Liste per Helper:
  - `create` -> leere Task-Liste (Planner legt neue Tasks an).
  - `resume`/`reassemble`/`rebuild`: bei `--all` alle `context.existing_tasks`, sonst neuester Task pro Job (`tasks_by_job`-Logik bleibt).
  - `edit`: genau ein Task (neuester); bei `--all` Fehler.
- `menu_choice_to_cli_args`, `_parse_mode_argument`, `get_cli_help_text` werden entfernt (Parity-Text wird ersetzt durch Subparser-Hilfe).
- `StrategyResolver.requires_user_interaction` prueft kuenftig `args.command is None`.

### 4) Execution-Planner: [src/pipeline/job_manager/execution_planner.py](src/pipeline/job_manager/execution_planner.py)

- `_determine_execution_context`: `args.mode` -> `args.command`. Das bereits refactorierte Multi-YAML-Scanning bleibt.
- `args.mode != "new"`-Check wird zu `args.command != "create"`.
- Der Filter `args.mode in ["all", "last", "all-new"]` wird zu `args.command in ("resume","reassemble","rebuild")`.
- `_create_execution_plan`: `intent.execution_options.rerender_all` weiterhin auf `task.rerender_all` mappen; `force_final_generation` analog.

### 5) Menu-Vokabel: [src/pipeline/job_manager/menu_orchestrator.py](src/pipeline/job_manager/menu_orchestrator.py)

- Level-1-Auswahl umformulieren: Eintraege heissen kuenftig `Resume (fill gaps)`, `Reassemble final`, `Rebuild from scratch`, `Edit candidates`, `Create new task`, `Cancel`.
- Interne Enums (`TaskSelectionChoice`, `TaskOptionsChoice`, `AllTasksChoice`) durch das neue `Verb`-Enum ersetzen; `ExecutionIntent`-Erzeugung in `_create_*_intent` entsprechend anpassen.
- Confirm-Dialog fuer Rebuild (heute Rerender) bleibt inhaltlich; Titelzeile umbenennen.
- `_format_task_display` unveraendert.

### 6) JobManager-Wrapper: [src/pipeline/job_manager_wrapper.py](src/pipeline/job_manager_wrapper.py)

- `parse_mode_argument` entfernen (nicht mehr aufgerufen).
- `validate_execution_plan` / `print_execution_summary` unveraendert.

### 7) Scripts und interaktive Tests

- [scripts/test_menu_orchestrator.py](scripts/test_menu_orchestrator.py), [scripts/test_task_system.py](scripts/test_task_system.py), [scripts/test_regenerate_final.py](scripts/test_regenerate_final.py): String-Ersatz `--mode X` -> neue Subkommandos. Jeweils ein durchgehender Smoke-Lauf im Kopfkommentar.
- [scripts/test_model_cache.py](scripts/test_model_cache.py): Beispielzeile mit `--mode` ersetzen.

### 8) Doku-Updates (nur Verb-Begriffe/Beispiele)

- [README.md](README.md) Zeilen 201-233
- [docs/basic-usage_CLI-arguments.md](docs/basic-usage_CLI-arguments.md) Zeilen 33-87
- [docs/TECHNICAL_OVERVIEW.md](docs/TECHNICAL_OVERVIEW.md) Zeilen 129-136, 658-677
- [docs/readme_agent.md](docs/readme_agent.md) Zeilen 489-493
- [docs/PERSISTENT_CACHE.md](docs/PERSISTENT_CACHE.md) Zeilen 53-56

### 9) Neue Datei: docs/CLI_VERBS_FUTURE_IDEAS.md

Parkt die zurueckgestellten Ideen aus der Konzept-B-Diskussion:

- Verb-spezifische Flags (`--skip-whisper`, `--only-chunks`, `--keep-chunks`)
- `list` / `status` als eigene Verben
- Zusaetzliche Auto-Migration: konventioneller `jobs/`-Ordner
- Per-Job-Strategien (`--per-job job:verb,...`) als spaeteres Feature

### 10) Nicht angefasst

- `TaskExecutor` ([src/pipeline/task_executor/task_executor.py](src/pipeline/task_executor/task_executor.py)): liest weiterhin `task.force_final_generation` / `task.rerender_all`. Keine Aenderung noetig.
- `TaskConfig`-Felder in [src/utils/config_manager.py](src/utils/config_manager.py) bleiben.
- Audio-/Validation-/Prosody-Pfade sind von dem Refactor unabhaengig.

## Datenfluss (neue Variante)

```mermaid
flowchart LR
    CLI["cbpipe resume *.yaml --all"] --> Parse[parse_arguments]
    Parse --> Args["args.command='resume' args.all=True"]
    Args --> Planner[ExecutionPlanner]
    Planner --> Ctx["_determine_execution_context scans every YAML"]
    Ctx --> CLIMap[CLIMapper.parse_cli_to_execution_intent]
    CLIMap --> Intent["ExecutionIntent verb=RESUME tasks=[...]"]
    Intent --> Plan[_create_execution_plan]
    Plan --> Orch[TaskOrchestrator.execute_tasks]
    Orch --> Exec[TaskExecutor using force_final/rerender flags]
```

## Reihenfolge der Umsetzung

1. `Verb`-Enum und `ExecutionOptions` anpassen.
2. `menu_orchestrator.py` an neue Enums umbauen (Tests laufen wieder).
3. `cli_mapper.py` vereinfachen; `parse_mode_argument` entfernen.
4. `execution_planner.py` auf `args.command` umstellen.
5. `cbpipe.py` Subparser umbauen, alte Flags entfernen.
6. Scripts, Doku, neue Future-Ideas-Doc.
7. Rauchtest: `cbpipe`, `cbpipe create config/folker/erledigt/dem_algorithmus_die_hand_reichen.yaml`, `cbpipe resume --all`, `cbpipe edit`.

## Risiken

- **Breaking Change** fuer alle bestehenden Skripte/Dokumente, die `--mode` nutzen. Bewusst akzeptiert.
- Menue-Enum-Umbau aendert interne API von `menu_orchestrator`; andere Consumer (`scripts/test_menu_orchestrator.py`) muessen mit.
- `edit` + Multi-YAML ist inhaltlich ambig -> wird als Fehlermeldung im Parser abgefangen (nur 1 Config zulaessig).