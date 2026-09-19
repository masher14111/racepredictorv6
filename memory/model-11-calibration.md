---
name: model-11-calibration
description: Probability calibration (auto isotonic/sigmoid) + within-race normalization — fixes audit C1, v3 now served calibrated (2026-06-16)
metadata:
  type: project
---

# Model Calibration (2026-06-16)

Made the served win/place probabilities **well-calibrated** (a predicted 20%
wins ~20% of the time) — the priority for betting over raw AUC. Fixes audit C1
(v3 was served RAW, ECE 0.42). See [[model-09-baseline-audit]].

## Method chosen — `config.yaml: calibration_method: auto`

New module `models/calibration.py`: `SigmoidCalibrator` (Platt: stores only
`a,b` floats → pickles independent of sklearn version), `fit_calibrator(method,
p_raw, y)`, `brier_score`, `normalize_within_race`. `auto` fits isotonic **and**
sigmoid on the earlier 70% of the time-ordered calibration slice, keeps the
lower-Brier one on the held-out tail, then refits the winner on the full slice
(no leakage into model or test). Fit on a chronological calib slice carved from
the train tail (`calib_frac`), persisted as `catboost_<target>_<tag>_calib.pkl`,
loaded by `predictor._load_calibrator` via the shared `.predict(p)->p` interface.

`auto` selected **per target**:

| model | won      | placed_2 | showed  |
| ----- | -------- | -------- | ------- |
| v3    | isotonic | sigmoid  | sigmoid |
| v3nf  | sigmoid  | sigmoid  | sigmoid |

## Before → after (held-out tail, 49,635 rows; `docs/calibration/*.png` + metrics.csv)

| model | target   | Brier raw→cal   | ECE raw→cal     | AUC (unchanged) |
| ----- | -------- | --------------- | --------------- | --------------- |
| v3    | won      | 0.297→**0.090** | 0.419→**0.005** | 0.789           |
| v3    | placed_2 | 0.263→**0.145** | 0.321→**0.007** | 0.785           |
| v3    | showed   | 0.233→**0.175** | 0.226→**0.012** | 0.787           |
| v3nf  | won      | 0.271→**0.100** | 0.407→**0.002** | 0.670           |
| v3nf  | placed_2 | 0.255→**0.167** | 0.294→**0.006** | 0.678           |
| v3nf  | showed   | 0.240→**0.206** | 0.183→**0.008** | 0.686           |

Reliability curves: raw sits far **below** the diagonal (overconfident, the
auto_class_weights + odds-inverse sample-weight inflation); calibrated hugs the
diagonal. Calibration is monotone → AUC/ranking unchanged (small drift is
isotonic tie-binning only).

## Within-race normalization — kept as a SEPARATE column, not the headline

`normalize_within_race(probs, group_ids)` divides each runner's win prob by its
race-field total so the field sums to 1 (exactly one winner). Exposed as
`won_prob_normalized` in `predictor.py`; **`won_prob` (calibrated marginal) is
NOT overwritten**.

**Why not normalize in place:** on the held-out set normalization slightly
_worsened_ pooled Brier/ECE (v3 won 0.090→0.097, ECE 0.005→0.047). Two causes:
(1) the calibrated marginal is already near-perfect (ECE≈0.005), so the sum-to-1
constraint pulls _against_ marginal calibration; (2) audit C3 — historical
`race_time` is 100% null, so the offline race key is approximate (date, venue,
distance) and over-merges races (field-sum mean 1.66 ≫ median 1.24). At serve
time `live_odds.parquet` HAS real `race_time`, so the true (venue, race_time)
key gives a clean sum-to-1 — normalized field sum → 1.000 in the report.

**How to apply:** trust `won_prob` for calibrated absolute probability / value
math; use `won_prob_normalized` only when you need a coherent within-race
distribution (e.g. "who wins this race" ranking display). Don't feed normalized
probs into A/E or EV against fixed odds — that's what the calibrated marginal is
for. Refit calibrators after retrain via `python -m scripts.refit_calibrators`;
regenerate the report via `python -m scripts.calibration_report`. Suite: 729
pass / 3 skip.
