---
name: model-13-ensemble
description: Ensemble vs single CatBoost on win prob (2026-06-16) — diverse-learner blend gives ~0 gain; market blend is the real lever but blocked on C2 leakage. NOT adopted.
metadata:
  type: project
---

# Ensemble experiment — win probability (2026-06-16)

Prompt 13: does blending diverse learners beat the single CatBoost on a
**calibrated** win probability? Harness: `models/ensemble_experiment.py`
(`python -m models.ensemble_experiment` → `docs/calibration/ensemble_experiment.json`).
Pure experiment — **nothing wired into the serving path**.

## Design (mirrors train.py so numbers are comparable)

- **Feature set:** price-free whitelist (the honest `v3nf` set — 21 cols present
  in `training.parquet`; 5 are 100% null, dropped only for the linear learner).
  The priced `v3` set is leakage-built (audit C2) so beating it is meaningless.
- **Split:** sort by `race_date`, last 20% = test (core 168,758 / cal 29,780 /
  test 49,635). Most-recent 15% of train = time-ordered calibration slice.
- **Weights:** odds-inverse sample weights, cap 20 (`train._sample_weights`).
- **Calibration:** `fit_calibrator("auto", …)` on the cal slice — every reported
  prob is _calibrated_ (all picked sigmoid). Monotone, so AUC unchanged.
- **Base learners:** CatBoost (tuned v3nf 'won' params, 600 iters, Balanced),
  XGBoost (depth 5, lr 0.03, subsample/colsample 0.8, scale_pos_weight=neg/pos),
  LogisticRegression (median-impute + standardise the 16 non-dead cols, L2 C=1,
  class_weight balanced). Fixed iters (no early stopping) → clean OOF/refit parity.
- **Blends:** simple average of the 3 calibrated probs; **stacking** = logistic
  meta-learner on **out-of-fold** base preds (TimeSeriesSplit, 4 folds, no leak);
  each `+market` = `implied_prob` added as an extra stack input.

## Held-out metrics (target=won, test n=49,635, base rate 0.118)

| model               | AUC    | logloss | Brier  | ECE    |
| ------------------- | ------ | ------- | ------ | ------ |
| **catboost (base)** | 0.6708 | 0.3433  | 0.0999 | 0.0016 |
| xgboost             | 0.6677 | 0.3440  | 0.1000 | 0.0028 |
| logreg              | 0.6657 | 0.3446  | 0.1001 | 0.0051 |
| avg (calibrated)    | 0.6722 | 0.3430  | 0.0998 | 0.0041 |
| stack (calibrated)  | 0.6716 | 0.3432  | 0.0998 | 0.0028 |
| market_only         | 0.7447 | 0.3560  | 0.1100 | 0.0801 |
| **stack+market**    | 0.7494 | 0.3202  | 0.0939 | 0.0085 |

Deltas vs base CatBoost — **model-only**: avg AUC **+0.0014**, stack **+0.0008**;
logloss/Brier essentially flat (≤0.0003); **ECE 1.5–2.5× worse** (0.0028–0.0041
vs 0.0016). AUC SE on this set ≈ 0.005–0.007, so the AUC "gain" is **noise**.
CatBoost alone is already the best-calibrated learner and ties the blend on
Brier/logloss. **+market**: AUC **+0.079**, logloss **−0.023**, Brier **−0.006**
— a large jump, the single biggest lever (as expected).

Meta-learner weights (logit): model-only `cb 1.88, lr 2.11, xgb 0.59, b −4.44`
(leans cb+lr, discounts xgb). With market `market 3.39, cb 2.48, xgb −0.39,
lr 0.35, b −4.31` — **the market dominates and washes out xgb/lr**.

Latency (ms / 1,000 runners): cb 1.86, xgb 7.61, lr 1.22, **stack total 8.74**
(~4.7× cb, still trivially fast). Whole experiment runs in ~50s on GPU.

## Decision — NOT adopted (and why)

1. **Diverse-learner ensemble (no market): rejected.** No genuine gain in
   calibrated probability — AUC delta is within noise, Brier/logloss flat, and
   ECE _degrades_ (cb is already ECE 0.0016). It would cost 2 extra model
   artefacts + a meta-learner + ~4.7× latency for nothing. Keep single CatBoost.
2. **Market blend: the real win, but BLOCKED on leakage.** The +market AUC 0.749
   is built on `implied_prob`, which in `training.parquet` derives from
   `odds_finish` (the finishing SP) — exactly the outcome-adjacent leakage of
   **audit C2**. `market_only` already scores AUC 0.745 because the finishing SP
   nearly _is_ the result; live inference only has a pre-off board price. So
   0.749 is an **upper bound that will not hold live** — not adoptable as-is.

## Actionable next step

Fix **C2 first** (rebuild market features from a pre-off price — `morningwap`,
0 null per audit — instead of `odds_finish`), then re-run this harness's
`stack+market` row on the honest price. A market-blended win model is the
highest-value direction once the price feature is leak-free; the model-only
ensemble is a dead end. See [[model-09-baseline-audit]] (C2, C3) and
[[calibration-value-work-handoff]].

Repro: `python -m models.ensemble_experiment`; metric helpers covered by
`tests/models/test_ensemble_experiment.py`. xgboost is in `requirements-dev.txt`
(experiment-only, imported lazily — never on the serving path).
