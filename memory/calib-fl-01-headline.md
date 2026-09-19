---
name: calib-fl-01-headline
description: Fixed the overconfident headline win prob — promoted won_prob_normalized (ECE 0.029) over the saturating raw marginal won_prob (ECE 0.222); UI/API/CLI/backtest repointed, won_prob kept as raw debug column (2026-06-17)
metadata:
  type: project
---

# calib-fl-01 — headline win prob fix (2026-06-17)

First prompt of the calibration programme ([[calib-fl-00-plan]]). Fixed **Bug 1**:
the UI headline `won_prob` was badly overconfident out-of-sample. **Option (a)
chosen** — promote `won_prob_normalized` to be the headline the UI/API present;
keep `won_prob` (raw calibrated marginal) as a clearly-named debug / EV-reference
column. No model retrain, no recalibration — purely a presentation/contract change.

## Why (a), not (b) refit-the-calibrator

The v3 `won` model is the _priced_ model (28 feats, built on leaked `odds_finish`,
audit C2 in [[model-09-baseline-audit]]). OOS its raw output **compresses** into a
tiny band — on 2026-06-06..12 raw ∈ [0.2825, 0.5085] (median 0.4515). The isotonic
calibrator (`catboost_won_v3_calib.pkl`, x-knots [0.252, 0.509] → y [0.007, 0.734])
maps that narrow band onto a steep range, so the field piles near the 0.734 ceiling
(698 runners at 0.306, 133 at 0.734). 100% of raw was _inside_ the knot span — so the
defect is **steepness / distribution-shift compression**, not out-of-range clamping
(corrects the plan's "saturates everything outside [0.252,0.509]" wording). Refitting
the marginal can't fix the compressed raw range and still can't enforce sum-to-1
coherence. `normalize_within_race` (already computed, [[model-11-calibration]])
re-anchors the field to base rate for free (sum-to-1 ⇒ field mean = base rate),
preserves within-race ranking, and is the simplest change. (b) was rejected as more
work for a worse result.

## Before → after (OOS, `python -m scripts.last_week_backtest`, 2,338 runners, base 0.109)

| column                             | mean  | AUC   | logloss | Brier | ECE          |
| ---------------------------------- | ----- | ----- | ------- | ----- | ------------ |
| **HEADLINE** `won_prob_normalized` | 0.109 | 0.778 | 0.300   | 0.088 | **0.029** ✅ |
| raw marginal `won_prob` (debug)    | 0.331 | 0.773 | 0.448   | 0.140 | **0.222** ❌ |

Both targets hit: **ECE 0.029 < 0.05** and **AUC 0.778 ≥ 0.773** (not worse — slightly
better). The model-11 note's "won_prob ECE≈0.005, keep as headline" was measured on the
_held-out historical tail_ (where the marginal was near-perfect and normalization
slightly hurt); it is **false OOS on live data** (where the marginal saturates and the
race key is clean because live racecards carry real `race_time`). The 2026-06-17 OOS
week is the regime that matters for serving.

## What changed (no commit yet — awaiting review)

- `models/predictor.py` — rewrote the false L405-414 comment + `RunnerPrediction`
  docstring + `_runner_dict` field comments to state normalized = headline, won_prob =
  raw debug/EV column. CLI `_main` win= now prints the headline (normalized, fallback).
  **Internal columns unchanged**: `df["won_prob"]` stays the calibrated marginal (drives
  composite + is the normalization basis); `won_prob_normalized` already emitted.
- `ui/_components.py` — new pure helper `headline_win_prob(sel)` → `won_prob_normalized`
  if present else `won_prob` (the `is not None` fallback keeps 0.0 valid and lets old
  caches / single-runner rows degrade gracefully — this is why all existing UI test
  fixtures, which set only `won_prob`, stayed green).
- Repointed every **headline "Win %" / win-bar** display to the helper: `app.py`,
  `predictions.py` (sel card + best-bets strip), `live_races.py`, `race_compare.py`
  (prob row + radar axis → `won_prob_normalized`), `bet_placer.py`, `_form.py`
  (`race_shape` top-pick %/gap), `pages/8_Race_Detail.py`, `pages/12_Horse_Detail.py`.
- `scripts/last_week_backtest.py` — `model_quality` now reports headline vs raw marginal
  side-by-side (refactored to `_quality_metrics`), so "record the new headline ECE/AUC"
  is built in.
- Tests: `tests/models/test_predictor.py` (`TestHeadlineWinProb`: field-sum =
  expected-winner-count; headline re-anchors a saturated marginal; runner-dict emits
  distinct headline vs raw), `tests/models/test_calibration.py` (saturated-field
  re-anchor), `tests/ui/test_components.py` (helper prefers normalized / falls back /
  0.0 honored). Suite: **1063 passed, 3 skipped** (reportlab-only). Live
  `data/predictions.json`: all 394 runners carry `won_prob_normalized` (e.g. raw 0.379
  → headline 0.223), so the UI shows the corrected value, not the fallback.

## Deliberately NOT changed (deferred to calib-fl-03/04, the value layer)

- **Kelly staking** (`recommend_stake`) and **edge/value badges** (`model_edge =
won_prob − implied`) in `bet_placer.py` / `predictions.py` / `_betting.py` still read
  the raw marginal. Per design these should use `value_win_prob` (price-free), and the
  F-L recalibration prompts own that layer; changing edge/EV math here would pre-empt
  them. **An overconfident marginal still inflates those edge badges / Kelly stakes** —
  flag this for calib-fl-03.
- `dashboard.py` / `performance*.py` read `won_prob` from the **bet_tracker DB** (a
  stored historical value on settled bets) — a separate schema concern, untouched.
- The losing bet strategies in the backtest (value −51/−77%, top-pick −25/−33% ROI) are
  **Bug 2** (favorite-longshot bias), the next prompt's job — not addressed here.

Next: **calib-fl-02** (quantify F-L bias on the full backtest OOS frame, fit an
odds-band recalibrator for the price-free model). Relates to [[calib-fl-00-plan]],
[[model-11-calibration]], [[model-09-baseline-audit]], [[wrap-29-yesterday-predictor]].
