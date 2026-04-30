# tts-pipeline-jobs

Private job orchestration for TTS audio production via [chatterbox-pipeline](https://github.com/MoebiusSt/chatterbox-pipeline).

## Structure

```
jobs/
├── pending/              ← active jobs, picked up by runner
│   ├── folker/           ← folker.world articles
│   ├── zwiefach/         ← zwiefach project
│   └── <project>/        ← add more projects as needed
└── completed/            ← finished jobs (auto-moved by workflow)
    ├── folker/
    ├── zwiefach/
    └── <project>/
```

## How It Works

```
Cloud Agent / You              GitHub Actions              WSL Self-Hosted Runner
       │                            │                            │
       ├─ push YAML to              │                            │
       │  jobs/pending/<project>/   │                            │
       │  ── or ──                  │                            │
       ├─ workflow_dispatch ────────┤                            │
       │                            ├── dispatch to runner ──────┤
       │                            │                            ├─ copy config → cbpipe/config/<project>/
       │                            │                            ├─ cbpipe create|resume|...
       │                            │                            ├─ WAV → MP3 (ffmpeg)
       │                            │                            ├─ upload MP3 artifact
       │                            │                            └─ move config → completed/
       ├─ download MP3 artifact ◄───┤                            │
       └─ upload to website         │                            │
```

## Trigger Methods

### 1. Push trigger (automatic)
Push a new `.yaml` to `jobs/pending/<project>/` → workflow detects it and runs.

### 2. Manual dispatch
Actions tab → "TTS Render" → fill in:
- **project:** `folker`, `zwiefach`, etc.
- **job_config:** filename (e.g. `dem_algorithmus.yaml`)
- **verb:** `create` | `resume` | `reassemble` | `rebuild`
- **device:** `cuda` | `cpu`

## Adding a New Project

1. Create directories:
   ```bash
   mkdir -p jobs/pending/<project> jobs/completed/<project>
   ```
2. Ensure cbpipe has matching config defaults at:
   ```
   /home/stephan/projekte/chatterbox-pipeline/config/<project>/defaults/
   ```
3. Job YAMLs use `parent: <project>/defaults/<model>.yaml`

## Artifact Naming

MP3 artifacts follow the pattern: `<project>-<job_label>-mp3`

Example: `folker-dem-algorithmus-die-hand-reichen-mp3`

Artifacts are retained for 30 days on GitHub.
