# Prosody Validation (whisperx + Tail-Trim + UTMOS)

## Overview
Adds a multilingual prosody and MOS-based selection layer to candidate validation:
- Tail-end speech-aware trimming (Smart/WhisperX → VAD → Energy) before any validation
- Prosody scoring (flow, liveliness, intelligibility, semantic proxy)
- MOS via UTMOS (VERSA V1 or V2), fallback NISQA (optional)
- Final selection score combines quality, prosody, and MOS

## Configuration
See `config/default_config.yaml` under `validation.preprocessing.tail_trim`, `validation.prosody`, `validation.mos`, and `validation.selection.gating`. Note: `validation.alignment` has been removed; alignment language gating is controlled via `validation.preprocessing.tail_trim.smart_match.language_gate`.

Key fields:
- `validation.mos.min_mos`: 3.5 (UTMOS threshold)
- `validation.prosody.select`: weights for combining scores
- `validation.preprocessing.tail_trim`: Smart-Trim (WhisperX), lookback window, post-silence, fade-out, diagnostics/persistence

## Integration Points
- TailTrim applied in `validation_handler` before Whisper validation
- Trim metadata (`tail_trim`) persisted per candidate in `whisper/whisper_metrics.json` (method, match_type, cut_sample, matched_words, etc.)
- `candidate_YY_trimmed.wav` preferred during assembly when present
- Prosody/MOS scored per candidate; persisted in whisper metrics as `prosody` and `final_selection_score`
- Assembly prefers `final_selection_score` (fallbacks to legacy overall scores)

## CLI
- `--enable-prosody` enables prosody scoring regardless of config
- `--enable-tail-trim` enables tail trimming regardless of config

## Smart Tail-Trim details
- Smart search for last N words of the input text (
  `validation.preprocessing.tail_trim.smart_match.last_n_words`) in the last
  `search_window_words` tokens of WhisperX-aligned words; fuzzy threshold via `fuzzy_ratio`.
- Language gating via `language_gate` to limit Smart-Trim to supported languages.
- Fallbacks when Smart-Trim misses: WhisperX last word end → VAD → energy.
- Post-speech silence (`post_speech_silence_ms`) added, `fade_out_ms` applied to end.
- Debug artifacts: `*_removed_tail.wav` if `debug_save_removed_tail` is true.

## Testing
- `scripts/test_prosody_scorer.py` runs TailTrim and Prosody scoring on a dummy sample

## Notes
- UTMOS via `versa` is integrated as a stub: real API calls may need adapting to your exact VERSA version. If unavailable, MOS is skipped gracefully.
- Alignment with whisperx is optional and requires language support
- Current prosody features use low-cost heuristics; add true F0 via `praat-parselmouth` if needed.