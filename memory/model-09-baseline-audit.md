---
name: model-09-baseline-audit
description: Baseline model-pipeline audit (2026-06-16) — held-out metrics + ranked win-prob backlog; reference for prompts 10-17
metadata:
  type: project
---

# Model Pipeline Baseline Audit (2026-06-16)

Diagnostic baseline for the CatBoost win/placed_2/showed pipeline. **Reference for
prompts 10–17.** No code changed. Computed on the real held-out tail of
`data/features/training.parquet` (248,173 rows; chronological split, test = last
20% = 49,635 rows spanning **2025-12-02 → 2026-06-12**). Two models exist:
`v3` (priced, displayed in UI) and `v3nf` (price-free, drives value layer).

## Held-out metrics (exact, recomputed from saved models)

| target   | model             | AUC   | log-loss | Brier | ECE       | served calibrated?              |
| -------- | ----------------- | ----- | -------- | ----- | --------- | ------------------------------- |
| won      | v3 (priced)       | 0.789 | 0.808    | 0.296 | **0.418** | **NO calibrator on disk → RAW** |
| placed_2 | v3                | 0.786 | 0.732    | 0.263 | 0.324     | **NO → RAW**                    |
| showed   | v3                | 0.787 | 0.661    | 0.233 | 0.228     | **NO → RAW**                    |
| won      | v3nf (price-free) | 0.670 | 0.344    | 0.100 | **0.004** | yes (isotonic)                  |
| placed_2 | v3nf              | 0.678 | 0.511    | 0.167 | 0.006     | yes                             |
| showed   | v3nf              | 0.686 | 0.600    | 0.206 | 0.006     | yes                             |

Base rates: won 11.8%, placed_2 23.6%, showed 35.2%. v3nf-won Brier 0.100 ≈
base-rate Brier 0.104 → honest model has **modest sharpness** (good ranking, limited
probability resolution). v3's higher AUC is **optimistic** (leakage, see C2).

Per-(venue-day) win-prob sum: v3 RAW **32.9**, v3nf CALIB **7.3** (~7 real races/group
→ v3nf ≈1.0/race = sane; v3 ≈4.5× inflated).

## Ranked findings (by impact on win-probability quality)

### P0 — Critical (corrupt the served probabilities)

- **C1. v3 (the displayed model) serves UNCALIBRATED probs, ~3–4× inflated.** No
  `catboost_*_v3_calib.pkl` on disk (only v3nf has them). ECE_won=0.42, A/E≈0.2–0.3.
  Won/placed_2/showed shown in the UI are grossly overconfident. Fix: fit+save v3
  calibrators (`train --reuse-params`) **or** surface the calibrated v3nf probs in UI.
- **C2. Outcome-adjacent leakage + train/serve mismatch via `odds_finish`.**
  `odds_decimal` and `sp` are **100% null** in history, so **248,172/248,173** rows
  build EVERY market feature (implied_prob, overround_norm_prob, log_odds,
  market_rank, ew_value_index, odds_value_delta) from `odds_finish` = the returned/
  finishing SP — the exact "finishing odds" the project bans. Live inference uses
  `odds_decimal` (board price) instead → different distribution → v3's 0.789 AUC will
  NOT hold live. Fix: train market features on a pre-off price (`morningwap`, 0 null)
  to match what's available at prediction time. v3nf already avoids this.
- **C3. Every per-race feature is aggregated over the whole VENUE-DAY, not the race.**
  `race_time` is **100% null** → `_RACE_KEY=[race_date, venue]` collapses ~7 races
  into one group. `field_size` median = **72** (true ≈8–12); `market_rank`,
  `overround_norm_prob` (normalizes over ~72 runners), `rating_rank`,
  `horse_speed_rank`, `race_complexity` are all on the wrong unit. predictor.py groups
  by `[venue, race_time]` but race_time null → also degrades to venue-day. Fix: recover
  a per-race key (race_time / race_id) end-to-end.

### P1 — High

- **C4. Dead features in the whitelist.** `timeform_rating` 100% null (paywalled,
  empty `session_cookie`) → `rating_rank` all-null, `pace_bias` null, `horse_speed`
  always falls back to its proxy, `race_complexity` loses its ratings component.
  Populate Timeform or drop these from FEATURE_COLS.
- **C5. Double imbalance correction.** Both `auto_class_weights="Balanced"` AND
  odds-inverse `sample_weight` (cap 20) inflate positive-class probability — the root
  cause of raw A/E≈0.2–0.3. Recoverable via calibration (v3nf proves it) but makes raw
  probs meaningless and is the proximate cause of C1's inflation. Drop one; always calibrate.

### P2 — Medium / methodological

- **C6. Optuna CV is random, not time-aware** (`StratifiedKFold(shuffle=True)`): folds
  train on chronologically-future rows. Use `TimeSeriesSplit`. Modest impact (final
  fit/eval use a clean time split; point-in-time features limit per-row leakage).
- **C7. `_time_split` slices by row position**, so a venue-day on the boundary can
  straddle train/test. Split on a date boundary instead. Negligible at 20%.
- **C8. derive trailing rates use `strict=False`** → an entity's earlier same-day runs
  count toward that day's later races (results maybe unknown at the off). Low impact;
  engine features already use `strict=True`.

## Confirmed leak-safe (NOT issues)

- Trailing rates do **not** self-include the current row's outcome (verified
  `_trailing_fast._bounds`: `end` excludes position i for both strict modes).
- `horse_speed` proxy = strictly-prior trailing mean of finish percentile (self excluded).
- `class_change` shifts to the prior race only.
- Train/test split is strictly chronological.
- `v3nf` is genuinely price-free and excellently calibrated (ECE 0.004–0.006).

## Fix order for prompts 10–17

1. C2 (leakage → honest market features) and C3 (per-race key) — fix the data first.
2. C1 + C5 (calibrate the served model / unify on the calibrated probs).
3. C4 (prune/populate dead features), C6/C7/C8 (tighten methodology).
   Re-baseline after each: AUC + log-loss + Brier + ECE on the same held-out tail.
