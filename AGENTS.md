# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

Pure CLI TTS pipeline (`src/cbpipe.py`). No web server, no database, no Docker.
See `README.md` for full documentation.

### Environment

- **Python 3.12**, venv at `/workspace/venv`. Always `source venv/bin/activate` first.
- **PYTHONPATH**: Must include `src/` for imports. Use `PYTHONPATH=src` when running scripts or tests.
- **No GPU** in Cloud Agent VMs — the TTS generation pipeline requires CUDA for actual audio rendering, but all non-GPU code paths (chunking, config, validation logic, tests) work on CPU.
- **Reference audio `.wav` files** are not in git (binary assets). Full pipeline runs (`create`/`resume`/`rebuild`) will fail without them.
- **`pkuseg`** (transitive dep of `chatterbox-tts==0.1.3`) does not compile on Python 3.12. Workaround: install `chatterbox-tts` with `--no-deps`, then install remaining chatterbox deps manually. The pipeline still works because `pkuseg` is only used for Chinese tokenization.

### Commands

| Task | Command |
|------|---------|
| Activate venv | `source venv/bin/activate` |
| Run tests | `PYTHONPATH=src python -m pytest tests/ -v` |
| Run CLI | `PYTHONPATH=src python src/cbpipe.py --help` |
| Black (check) | `black --check src/` |
| isort (check) | `isort --check-only src/` |
| flake8 | `flake8 src/` |
| mypy | `mypy src/` |

### Gotchas

- The update script installs `chatterbox-tts` with `--no-deps` to avoid the `pkuseg` build failure, then separately installs its other transitive dependencies. If `chatterbox-tts` adds new deps in a future version, they must be added manually.
- `numpy` must be installed before `chatterbox-tts --no-deps` because several packages need it at build time.
- spaCy model `en_core_web_sm` must be downloaded separately (`python -m spacy download en_core_web_sm`).

### Folker orchestration

- Folker WordPress-to-TTS orchestration notes live in `docs/folker-tts-orchestration.md`.
- `tts-pipeline-jobs` is expected to be a separate private repository cloned next to this repository.
- `TTS_JOBS_GITHUB_TOKEN` must have read/write access to that private jobs repository before agents can clone it, push job YAMLs, or monitor workflow runs.
