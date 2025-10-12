# Prosody Validation (whisperx + Tail-Trim + UTMOS)

## Overview
Adds a multilingual prosody and MOS-based selection layer to candidate validation:
- Tail-end speech-aware trimming (VAD/whisperx) before any validation
- Prosody scoring (flow, liveliness, intelligibility, semantic proxy)
- MOS via UTMOS (VERSA), fallback NISQA (optional)
- Final selection score combines quality, prosody, and MOS

## Configuration
See `config/default_config.yaml` under `validation.preprocessing.tail_trim`, `validation.prosody`, `validation.mos`, `validation.alignment`, and `validation.selection.gating`.

Key fields:
- `validation.mos.min_mos`: 3.5 (UTMOS threshold)
- `validation.prosody.select`: weights for combining scores
- `validation.preprocessing.tail_trim`: lookback window, post-silence, fade-out

## Integration Points
- TailTrim applied in `validation_handler` before Whisper validation
- Prosody/MOS scored per candidate; persisted in whisper metrics as `prosody` and `final_selection_score`
- Assembly prefers `final_selection_score` (fallbacks to legacy overall scores)

## CLI
- `--enable-prosody` enables prosody scoring regardless of config
- `--enable-tail-trim` enables tail trimming regardless of config

## Testing
- `scripts/test_prosody_scorer.py` runs TailTrim and Prosody scoring on a dummy sample

## Notes
- UTMOS via `versa` is integrated as a stub: real API calls may need adapting to your exact VERSA version. If unavailable, MOS is skipped gracefully.
- Alignment with whisperx is optional and requires language support
- Current prosody features use low-cost heuristics; add true F0 via `praat-parselmouth` if needed.