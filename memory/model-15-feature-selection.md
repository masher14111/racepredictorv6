---
name: model-15-feature-selection
description: Feature selection (2026-06-16) — pruned price-free set 30→22 cols with zero accuracy/calibration loss; added per-prediction SHAP explanation API (models/explain.py)
metadata:
  type: project
---

# Model-15 Feature Selection + Per-Prediction Explanations (2026-06-16)

Builds on [[model-10-features]] / [[model-09-baseline-audit]]. Two deliverables:
(1) measured which of the expanded features carry signal and pruned the dead ones;
(2) added a reusable per-prediction SHAP API for the "why this horse" UI.

Analysis ran on the **regenerated full matrix** (248,173 labelled rows, all 37
FEATURE_COLS present — the on-disk `training.parquet` predated model-10, so it was
rebuilt from `data/historical/betsp.parquet` via `normalize()` → `build_training_matrix()`,
cached at `data/features/training_full.parquet`). Chronological 70/12/18
train/calib/test; **price-free**, isotonic-calibrated (the honest variant).
Importance uses exact CatBoost tree SHAP (`get_feature_importance(type="ShapValues")`)

- `PredictionValuesChange` — **no `shap` dependency added** (CatBoost has it built in).

## Global importance (won, price-free) — mean |SHAP|

`field_size` 0.29, `horse_speed_rank` 0.18, `horse_speed` 0.18, `speed_trend` 0.10,
`jockey_win_rate` 0.10, `jt_combo_win_rate` 0.09, `race_complexity` 0.09,
`distance_place_rate` 0.09, `trainer_win_rate` 0.08, `distance_furlongs` 0.05.
The model-10 fixes (per-race key) and additions (speed_trend, distance form)
dominate — confirms model-10's lift is real.

## Pruning result — drop 8, lose nothing (price-free 30 → 22 cols)

Dropped (in `models.features.EMPIRICALLY_DEAD_COLS`):

- **5 dead (100% null on betSP):** `timeform_rating`, `rating_rank`, `pace_bias`,
  `class_change`, `recent_form_avg`.
- **3 low-signal:** `recent_form_wins`, `recent_form_runs` (Timeform-figure proxies),
  `going_speed` (79% null, superseded by `going_pref_*` keyed on `going_band`).

Held-out tail (test 2025-12-20 → 2026-06-12, n=44,672), full → pruned:

| target   | AUC             | log-loss        | Brier           | ECE             |
| -------- | --------------- | --------------- | --------------- | --------------- |
| won      | 0.7005 → 0.7004 | 0.3353 → 0.3354 | 0.0982 → 0.0982 | 0.0024 → 0.0022 |
| placed_2 | 0.7114 → 0.7116 | 0.4938 → 0.4928 | 0.1608 → 0.1609 | 0.0052 → 0.0054 |
| showed   | 0.7283 → 0.7282 | 0.5684 → 0.5685 | 0.1934 → 0.1935 | 0.0060 → 0.0054 |

All deltas < 0.001 — accuracy and calibration unchanged, model smaller/faster.
**No strict redundancy found** (max pairwise |corr| 0.93: `historical_place_rate`↔
`distance_place_rate`, `field_size`↔`race_complexity`); both pairs are additive and
below the 0.95 prune threshold, so kept (pruned-22 already held all metrics).

### Wiring (forward-compatible, no retrain forced)

`models/features.py` adds `EMPIRICALLY_DEAD_COLS`, `LEAN_FEATURE_COLS` (37→29),
`LEAN_PRICE_FREE_FEATURE_COLS` (30→22). FEATURE_COLS is left as the **superset**:
the 5 null features revive automatically if a Timeform/racecard source is wired
(model-10 backlog), so they were not deleted. To train the lean variant:
`train(feature_cols=LEAN_PRICE_FREE_FEATURE_COLS, version_tag="v3nf")`.

## Per-prediction explanation API — `models/explain.py`

```python
from models.explain import get_explainer
ex = get_explainer(target="won", version_tag="v3nf")   # cached (lru_cache)
why = ex.explain(runner_row)            # dict / pd.Series of feature values
why["top_positive"]   # [{feature, label, value, contribution}, ...] margin-space
why["top_negative"]   # most-negative first
ex.explain_frame(field_df)              # one SHAP pass for a whole race
```

- `Explainer.explain(row, top_k=6)` / `explain_frame(rows, top_k)` → base_value,
  predicted_margin, predicted_prob (raw, = sigmoid(margin)), and top ± drivers.
  Contributions are **log-odds (margin) space** and sum to the margin
  (`base + Σ = predicted_margin`); calibration is monotone so it never reorders
  drivers. `FEATURE_LABELS` gives UI strings (fallback = title-cased col name).
- **Fast/live:** tree SHAP for one race (~10 runners) is sub-ms; the only real
  cost is loading the model, which `get_explainer` memoises. Missing feature
  columns are filled NaN (won't crash on sparse live rows).
- Demo (`backtest/_explain_demo.py`) on a real 13-runner Gowran Park race: the
  actual winner ranked #1, drivers = recent place rate / field size / trainer +
  jockey win rate. Tests: `tests/models/test_explain.py` (12).

## Status

- **Suite: 850 passed / 3 skipped** (reportlab-only) after adding explain tests.
- Scratch/repro scripts in `backtest/`: `_regen_matrix.py`, `_feature_selection.py`,
  `_explain_demo.py`. Full results JSON: `docs/calibration/feature_selection.json`.
- **NOT retrained** — analysis only; served v3/v3nf bins unchanged. To realize the
  smaller model, retrain with the LEAN\_\* whitelists. Still open from prior audits:
  C1/C2 (served-model calibration / market-feature leakage).
