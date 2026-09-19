---
name: model-10-features
description: Feature expansion (2026-06-16) — recovered per-race key (+0.023 AUC) + 6 leak-free features (+0.007 AUC); measured marginal lift, all kept
metadata:
  type: project
---

# Model-10 Feature Expansion (2026-06-16)

Expanded the win-probability feature set on top of the [[model-09-baseline-audit]]
baseline. Every feature is strictly point-in-time (reads only runs **before** each
row's race_date) → leak-free. Marginal lift measured on the real betSP history
(248,173 labelled rows, chronological 70/12/18 train/calib/test split, isotonic-
calibrated **price-free** model — the honest variant). AUC is calibration-invariant;
log-loss/Brier/ECE are post-calibration so the calibration confound is removed.
**Strong calibration held throughout (test ECE stayed ~0.002–0.006) — decisive
evidence the new features are not leaking.**

## Headline result (target = won, price-free, calibrated)

| stage                         | AUC    | log-loss | Brier  | ECE    |
| ----------------------------- | ------ | -------- | ------ | ------ |
| baseline (venue-day key, old) | 0.6729 | 0.3432   | 0.0997 | 0.0039 |
| **+ per-race key fix**        | 0.6960 | 0.3362   | 0.0980 | 0.0019 |
| + going_band preference       | 0.6962 | 0.3362   | 0.0979 | 0.0016 |
| + layoff & career             | 0.6985 | 0.3357   | 0.0978 | 0.0024 |
| + course suitability          | 0.6987 | 0.3356   | 0.0978 | 0.0032 |
| + distance suitability        | 0.6998 | 0.3354   | 0.0978 | 0.0024 |
| + speed trend                 | 0.7025 | 0.3344   | 0.0976 | 0.0024 |

Net **won AUC 0.673 → 0.703 (+0.030, +4.5% rel)**; log-loss −0.0088; calibration
intact. placed_2 **+0.0073 AUC** (0.7075→0.7149), showed **+0.0065 AUC**
(0.7253→0.7318), both with lower log-loss/Brier.

## THE biggest lever — per-race key recovery (audit C3), +0.023 AUC alone

The Betfair backbone (`data/historical/betsp.parquet`) stores the **race off-time
inside `race_date`** (e.g. `2024-03-01 15:45`). `normalizer._reindex_canonical`
floored it to `YYYY-MM-DD`, and `_from_betsp` never set `race_time` → the entire
history had null off-time. So `_RACE_KEY=[race_date, venue]` collapsed ~7 races
per venue-day into ONE group: **field_size median 72** (true ≈9–11), and
`market_rank`/`overround_norm_prob`/`rating_rank`/`horse_speed_rank`/
`race_complexity` were all on the wrong unit.

Fix (two parts):

1. `normalizer._from_betsp` now captures the off-time into `race_time` before the
   floor (off-times are published in advance → leak-free).
2. `features.derive.add_race_key` builds `race_uid = venue + off-time(minute)`,
   falling back to `[race_date, venue]` when no time is known. All per-race
   grouping/rank sites now use `_race_cols(df)` (→ `["race_uid"]`).

Result: **field_size median 72 → 10**, and the corrected features became the model's
**top three** (horse_speed_rank, race_complexity, field_size by PredictionValuesChange).
This single change is the largest accuracy gain in the whole exercise.

## New leak-free features (features/derive.py + features/engine.py)

All bounded by the configured lookback (12 months / 20 runs), strictly prior, via
the vectorized `_trailing_fast` primitives (no O(n²) scans, no denominator/as-of-date
pitfalls — confirmed by the calibration holding).

- **days_since_last_run** (derive) — gap in days to the horse's previous run.
  Layoff/freshness. Fill 86.8%. Importance rank **#8**.
- **horse_career_runs** (derive) — count of strictly-prior known-result runs
  (uncapped, unwindowed). Separates unexposed types from veterans. Fill 100%. #15.
- **course_win_rate / course_place_rate / course_runs** (engine) — trailing form at
  THIS venue (group on `[horse_id, venue]`). Sparse (33% have a prior course run);
  weakest group by AUC (+0.0003) but **improved** calibration and is net-positive,
  so kept. Importance #18–22.
- **distance_win_rate / distance_place_rate / distance_runs** (engine) — trailing
  form at a similar trip (furlong bands: sprint<7, mile 7–9.5, middle 9.5–13,
  staying ≥13). Fill 78%. `distance_place_rate` importance **#11**.
- **speed_trend** (engine, in add_speed_figures) — short 3-run mean of the speed
  proxy minus the long-window baseline: + = improving form. Fill 84%. The single
  strongest NEW feature, importance **#5** (after the four key-fixed features).
- **going_band** (derive) + going-preference rewrite — coarse ground band parsed
  from the ~100%-filled raw `going` string (heavy/soft/good/standard/firm), keying
  `add_going_preference` on `[horse_id, going_band]` instead of the config-mapped
  `going_speed` (only ~21% filled). **going_pref coverage 6.9% → 68.6%.** Modest
  model lift (+0.0003) but a big robustness/coverage win; `going_speed` retained as
  fallback for sources without a raw going string.

`going_band` and `race_uid` are internal grouping columns — NOT added to
FEATURE_COLS (they are strings; the model would coerce them to NaN). Their signal
enters via going_pref / field_size / the ranks.

## Rejected / not built

- **No raw columns** in betSP for: weight carried, weight-for-age, draw/stall, going
  is present but headgear/first-time-blinkers absent, official class/class-drop,
  pace/run-style sectionals, speed/class ratings (`timeform_rating` 100% null,
  paywalled). These remain dead in FEATURE*COLS (`class_change`, `rating_rank`,
  `pace_bias`, `recent_form*\*`, `timeform_rating`) — populate Timeform or a racecard
  source to revive them (next backlog).
- No feature was dropped for hurting — course was the only borderline group and it
  helped calibration, so all stay.

## Wiring & status

- `models/features.py` FEATURE_COLS extended with the 9 numeric new columns (all
  flow into PRICE_FREE_FEATURE_COLS automatically — none are price-derived).
- New tests: per-race key, going_band, layoff, career, course, distance, speed_trend
  (tests/features/test_derive.py, test_engine.py), off-time recovery
  (tests/utils/test_normalizer.py). Full suite **695 passed / 3 skipped**.
- **NOT yet retrained/regenerated.** `data/unified_races.parquet` is empty; the
  source of truth is `data/historical/betsp.parquet`. To realize the lift live:
  re-run the normalizer (→ unified, now with race_time) → `build_training_matrix`
  → `python -m models.train` (and `--price-free`). The measured numbers above were
  produced by exactly that pipeline on the betSP backbone.
- Still open from the audit: C1/C5 (served v3 uncalibrated — unify on v3nf or fit v3
  calibrators), C2 (train market features on a pre-off price, not `odds_finish`),
  C6/C7 (time-aware CV/split).
