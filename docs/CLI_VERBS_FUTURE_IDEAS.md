# CLI Verb Future Ideas

These ideas are intentionally out of scope for the initial verb-based CLI refactor.

## Verb-Specific Flags

- `resume --skip-whisper`: continue generation without re-validating existing candidates.
- `reassemble --only-chunks 5,7`: rebuild final audio from a selected chunk subset.
- `rebuild --keep-chunks`: keep text chunking output but regenerate all candidates.

## Additional Verbs

- `list`: show available jobs and tasks without executing anything.
- `status`: show completion stage, missing files, selected candidates, and final audio paths.

## Job Layout Helpers

- Support a conventional `jobs/` directory as an additional job YAML discovery path.
- Add migration helpers for moving legacy generated configs into a clearer job layout.

## Per-Job Verb Overrides

An optional future feature could allow mixed operations in one invocation, for example:

```bash
python src/cbpipe.py --per-job "job1:resume,job2:reassemble"
```

For the initial verb-based CLI, mixed per-job operations are intentionally replaced by explicit repeated calls or shell loops.
