# RNG seeding (`global_seed`, `seed_fixed`, speaker `seed`)

TTS generation uses `torch.manual_seed(...)` so runs can be reproducible. Configuration lives under `generation:` in your job YAML (merged with [`config/default_config.yaml`](../config/default_config.yaml)).

## `global_seed`

- **`global_seed: 0`** — Random behaviour for task startup logging; per-candidate seeds are still derived unless `seed_fixed` applies (see below).
- **`global_seed: <positive integer>`** — Used as the **default base** for deriving per-candidate torch seeds when the active speaker does not override `seed`.

On startup, `TTSGenerator` calls `torch.manual_seed(global_seed)` once when `global_seed > 0`.

## Speaker-specific `seed`

Under `generation.speakers[].seed`:

- **`seed` omitted** — Use `global_seed` as the base for that speaker.
- **`seed: 0`** — Random / non-fixed base for that speaker (same meaning as elsewhere for “no fixed seed”).
- **`seed: <positive integer>`** — This value becomes the **effective base seed** for that speaker (`get_speaker_seed` in `src/generation/tts_generator.py`).

## Default per-candidate formula (`seed_fixed: false`)

Unless `seed_fixed` is enabled, each candidate index `i` and chunk text `text` use:

```text
torch_seed = base + (i * 1000) + (hash(text) % 10000)
```

where `base` is `get_speaker_seed(speaker_id)` (i.e. `global_seed` or the speaker’s `seed`).

Effects:

- Different **candidates** in the same chunk get different seeds (diversity).
- Different **chunks** (different `text`) get different hash terms (discernible runs).

Chatterbox **immediate retries** (artifact handling) use `base + attempt * 9973` per attempt when `seed_fixed` is false.

## `seed_fixed: true`

Set `generation.seed_fixed: true` when you want **one integer only** for every `torch.manual_seed` call tied to a candidate or retry — **no** `+ i*1000`, **no** `+ hash(text)%10000`, **no** `+ attempt*9973` inside the immediate-retry loop.

The value used is the **effective base seed** `base` from `get_speaker_seed(speaker_id)` (again: `global_seed` or `generation.speakers[].seed`).

Requirements:

- **`base > 0`**. If `global_seed` is `0` and no speaker sets a positive `seed`, fixed seeding does not apply; startup logs a warning.

Trade-offs:

- Multiple **candidates** in one chunk share the **same** RNG seed; variation comes mainly from **different synthesis parameters** (temperature ramps, etc.), not from different seeds.
- **Immediate retries** reuse the same seed on every attempt, which can reduce the effectiveness of retries that relied on seed changes.

## References

- Implementation: [`src/generation/tts_generator.py`](../src/generation/tts_generator.py) (`_torch_seed_per_candidate`, `_torch_seed_immediate_retry`, `TTSGenerator.__init__`).
- Defaults: [`config/default_config.yaml`](../config/default_config.yaml), [`config/defaults/qwen3.yaml`](../config/defaults/qwen3.yaml).
