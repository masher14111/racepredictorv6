---
name: calib-fl-02-fl-design
description: Designed + fit the favourite-longshot recalibrator (OddsBandCalibrator, 9 fixed odds bands, isotonic, log-odds blended) for the price-free v3nf prob; A/E flattened 2.80..0.17 -> ~1.0..0.79 on held-out OOS; artifact models/fl_oddsband_v3nf_calib.pkl; NOT wired into predictor yet (2026-06-17)
metadata:
  type: project
---

# calib-fl-02 — favourite-longshot recalibrator (design + artifact, OFFLINE) — 2026-06-17

Second prompt of the calibration programme ([[calib-fl-00-plan]]). Fixed **Bug 2**:
the price-free `value_win_prob` (v3nf) has a severe favourite-longshot bias. Built,
evaluated, and saved a recalibrator artifact. **Nothing wired into the live predictor
this prompt** (that is calib-fl-03). No commit yet.

## The key diagnostic insight

On the saved 77k-runner OOS frame (`data/backtests/20260616_195957/scored.parquet`,
walk-forward leak-free, bets at pre-off `ppwap`):

- **A/E by the model's own PROBABILITY bucket ≈ 1.0 everywhere** (ECE10 0.0015) — the
  model is _marginally_ calibrated.
- **A/E by ODDS band is wildly biased** — under-rates favourites, over-rates longshots:

| odds band      | n     | A/E before | A/E after (full frame) |
| -------------- | ----- | ---------- | ---------------------- |
| odds-on (<2.0) | 1476  | **2.803**  | 1.045                  |
| 2.0–4.0        | 7928  | **1.884**  | 0.968                  |
| 4.0–8.0        | 17823 | 1.239      | 0.983                  |
| 8.0–16.0       | 18643 | 0.739      | 0.971                  |
| 16.0–34.0      | 14364 | **0.437**  | 0.958                  |
| longshot (>34) | 17089 | **0.166**  | 0.801                  |

Because the model is already honest _by its own prob_, a calibrator that sees only
`prob` is ≈ identity and **cannot** fix this. **The correction must use the market
price.** (`bet_price`/`ppwap` is pre-off, legitimately available at inference — it is
_not_ the `odds_finish` close, which is leak-only / CLV-only.)

## Method chosen — `OddsBandCalibrator` (per-odds-band isotonic, log-odds blended)

Fits an independent prob→won calibrator **within each decimal-odds band**, then at
predict time **blends the two bands bracketing the runner's price in log-odds space**
(continuous across band edges). Each per-band map is monotone in prob and blend
weights depend only on odds ⇒ output is monotone in prob at any fixed price, so
**within-race ranking among similarly-priced runners is preserved** (removes band-level
bias, keeps the model's within-price discrimination = the value signal).
Default edges `[1,2,3,5,8,13,21,34,60,∞]` (dedicated odds-on band + finer mid/tail).

### Why not the 2-feature logistic (the main rejected alternative)

Evaluated `logistic2 = sigmoid(a·logit(p) + b·log(d) + c)` on the same split. It scored
marginally better AUC (0.7938 vs 0.7927, noise) BUT its **`coef(logit_p)=0.045 ≈ 0`** —
it structurally **discards the model probability and collapses onto the market price**.
For a _value_ model that is fatal: two horses at the same price get ~identical
calibrated probs ⇒ ~zero differentiated edge ⇒ it just re-bets the favourite. On the
task's own selection criteria the OBC also wins or ties everything else (held-out tail):

| method                                    | Brier       | ECE10       | ECEmass     | AUC    | max\|log A/E\| |
| ----------------------------------------- | ----------- | ----------- | ----------- | ------ | -------------- |
| RAW                                       | 0.09501     | 0.00507     | 0.00859     | 0.6749 | 1.743          |
| logistic2                                 | 0.08525     | 0.00683     | 0.00914     | 0.7938 | 0.251          |
| **OBC 9-band isotonic (chosen)**          | **0.08515** | **0.00429** | **0.00610** | 0.7927 | **0.238**      |
| OBC 9-band auto                           | 0.08510     | 0.00459     | 0.00656     | 0.7936 | 0.274          |
| OBC 6-band (`_ODDS_BANDS` edges) isotonic | 0.08526     | 0.00799     | 0.00737     | 0.7925 | 0.488          |

OBC isotonic wins Brier, both ECE definitions, and A/E flatness; the dedicated `[1,2)`
band + finer tail beat the coarse 6-band (which left longshot A/E at 0.61).

## Leak-free validation split

Train slice `race_date < 2026-03-26` (n=64440, folds 0–4); held-out tail
`race_date >= 2026-03-26` (n=12883, fold 5, base 0.1121). Fit on train only.
Tail A/E after: odds-on 1.006, 2–4 0.979, 4–8 0.999, 8–16 0.948, 16–34 0.948,
longshot 0.788 (vs before 2.52 / 1.88 / 1.25 / 0.72 / 0.43 / 0.18). Tail overall:
Brier 0.09501→0.08515, ECE10 0.00507→0.00429, AUC 0.6749→0.7927.
(The production artifact is fit on TRAIN only — same as the validation, not refit on
all data — so the saved A/E-after numbers are genuine held-out where tail is shown.)

## Artifacts / files (no commit yet)

- **`models/fl_oddsband_v3nf_calib.pkl`** — the recalibrator. 9 isotonic bands, plain
  Python floats only (`assert b"numpy" not in blob` passes ⇒ numpy-version portable,
  like [[model-11-calibration]]'s Isotonic/Sigmoid).
- `models/calibration.py` — new portable class `OddsBandCalibrator` (`.fit(prob, odds,
y, edges=, method=, min_rows=)`, `.predict(prob, odds) -> prob`).
- `tests/models/test_calibration.py::TestOddsBandCalibrator` — 9 tests: monotone-in-prob
  at fixed odds, A/E flattening on a synthetic biased fixture, edge-continuity, no-numpy
  pickle + reload, scalar I/O, sparse-band skip, raises-when-unfittable, single-band
  ignores odds. Suite: **tests/models 314 passed**.
- `docs/calibration/fit_fl_recalibrator.py` — fits + saves artifact + writes
  `docs/calibration/fl_recalibration_report.md` (before/after A/E tables).
- `docs/calibration/fl_prototype.py` — full method comparison (logistic2 vs OBC variants
  - value-signal check). Sanity: favourite 0.22@2.0 → 0.50 (revised UP); longshot
    0.08@50 → 0.022 (revised DOWN).

## Exact validation / reproduce command

```
python -m docs.calibration.fit_fl_recalibrator   # re-fit + regenerate the report
python -m docs.calibration.fl_prototype          # method comparison table
python -m pytest tests/models/test_calibration.py -q
```

## NOT done (next: calib-fl-03)

Wiring the recalibrator into `models/predictor.py` (recalibrate `value_win_prob` via
`.predict(prob, best_odds)`), backward-compatible + tests. Then **calib-fl-04**:
re-run `models.value.evaluate_filter` / `backtest` on the corrected probs and re-tune
the value gates on **CLV + A/E** (expect CLV to lift off the −7..−12% floor now that the
longshot over-prediction that drove it is gone). Caveat: recalibration injects market
price, so post-correction _band-level_ edge ≈ 0 — value now lives in within-price
disagreement; the value layer's edge gates ([[model-16-value-detection]]) must be
re-validated against that. Relates to [[calib-fl-00-plan]], [[calib-fl-01-headline]],
[[model-14-backtest]], [[model-09-baseline-audit]].
