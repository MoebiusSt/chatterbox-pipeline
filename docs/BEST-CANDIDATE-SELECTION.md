## Best Candidate Selection - Current Logic

This document explains how candidates are validated, gated, scored, and ultimately selected as the best candidate in the pipeline. It also summarizes the UI columns shown in `user_candidate_manager`.

### Components and Data Flow
- WhisperValidator: Transcribes audio and computes text similarity and a whisper quality score.
- QualityCalculator: Computes whisper quality score = 70% similarity + 30% length score.
- QualityScorer: Produces `overall_score` using weighted similarity and length components and penalty handling.
- ProsodyScorer: Computes prosody subscores `flow`, `liveliness`, `intelligibility`, `mos` and an aggregate `prosody_score`.
- ValidationHandler: Combines the above into enhanced metrics and computes `final_selection_score`.

Each candidate stored in enhanced metrics (whisper_metrics.json) contains at minimum:
- `quality_details.individual_scores.{similarity_score,length_score,overall_score}`
- `quality_details.validation_metrics.whisper_quality`
- `overall_quality_score` (alias of individual overall)
- `prosody` object with `subscores.{flow,liveliness,mos,...}` and `prosody_score`
- `final_selection_score` (combined score for selection)
- `is_valid` (whisper pass/fail), optional gating flags: `passes_mos_gate`, `passes_similarity_gate`

### Pass/Fail (Whisper Gate)
- Implemented in `validation/whisper_validator.py::validate_candidate`.
- Criteria:
  - Similarity must be above an effective threshold (dynamic based on text) derived from `validation.similarity_threshold`.
  - Quality must be above `validation.min_quality_score`.
  - Flexible rules allow small trade-offs between similarity and quality.
- Result is written as `is_valid` and included in enhanced metrics.

Note: No `min_prosody_score` gate is applied by Whisper validation.

### MOS Gate (Optional)
- Controlled via `validation.selection.gating.require_mos`.
- Uses `validation.mos.min_mos` to set `passes_mos_gate` based on raw MOS value.
- If `require_mos` is true, candidates with `raw_mos < min_mos` are effectively demoted during best selection.

### Prosody Gate
- Not currently enforced as a selection filter. There is a config threshold `validation.prosody.thresholds.min_prosody_score`, but it is not used to exclude candidates. It can be used for UI diagnostics.

### Final Selection Score
Final selection uses a combined score:

```
final_selection_score = alpha_quality * overall_quality_score
                        + beta_prosody * prosody_score
                        + gamma_mos   * mos_unit
```

Weights come from `validation.prosody.select`:
- `alpha_quality` (default 0.50)
- `beta_prosody` (default 0.35)
- `gamma_mos` (default 0.15)

`mos_unit` is the MOS normalized to [0..1] by the MOS provider.

### Best Candidate Determination
- Implemented in `ValidationHandler._create_enhanced_metrics`.
- The best candidate is chosen by highest `final_selection_score` if present; otherwise falls back to individual `overall_score`.
- If gating flags exist, candidates that fail required gates (`passes_mos_gate`, `passes_similarity_gate`) are demoted (effective score = -inf). If all are demoted, a fallback selection ignores gates and picks by score.
- In `AssemblyHandler.execute_assembly`, missing selections are filled using `final_selection_score` with the same fallback.

### Backfill Behavior
- If older runs saved whisper data without prosody/final scores, `ValidationHandler` recomputes prosody and `final_selection_score` on the fly and updates the stored result.

### Configuration Keys (defaults shown)
```yaml
validation:
  similarity_threshold: 0.64   # whisper similarity gate
  min_quality_score: 0.72      # whisper quality gate

  prosody:
    enabled: true
    weights:
      flow: 0.45
      liveliness: 0.25
      mos: 0.30
    select:
      alpha_quality: 0.50
      beta_prosody: 0.35
      gamma_mos: 0.15
    thresholds:
      min_prosody_score: 0.60   # not used as gate (documentation time)

  mos:
    provider: combined
    min_mos: 3.0

  selection:
    gating:
      require_mos: true
      require_similarity: true
```

### UI: `user_candidate_manager`
Columns now show:
- `val_score`: `validation_metrics.whisper_quality` (Whisper-specific quality)
- `qty_score`: `individual_scores.overall_score` (QualityScorer overall)
- `flow`, `live`, `mos`, `pros`: Prosody subscores and aggregate prosody
- `final`: `final_selection_score`
- `passed[sim,pros]`: Two symbols showing (1) Whisper similarity/quality pass (`is_valid`), and (2) whether `prosody_score ≥ min_prosody_score`. This is diagnostic-only and does not gate selection.

### Notes
- Best-candidate selection is correctly driven by the new `final_selection_score`.
- Whisper pass/fail uses similarity and min quality; MOS gate is optional; prosody threshold is not a gate but is surfaced in UI.


