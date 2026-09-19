---
name: model-12-tuning
description: Hyperparameter tuning overhaul (2026-06-16) — chronological CV + log-loss objective + pruner + tunable class-weight; new search space, best params, metric deltas
metadata:
  type: project
---

# Hyperparameter Tuning Overhaul (2026-06-16)

Rewrote `models/tuner.py` (Optuna study for the CatBoost won/placed_2/showed
classifiers) to fix audit findings C5/C6 and optimise for calibrated probability
instead of raw ranking. Search space + budget are now config-driven via
`config.yaml` `model.tuning`. Full 50-trial × 3-target session run on GPU
(~108 min); v3 refit with the winning params (calibrators now persisted).

## What changed

1. **CV is now strictly chronological.** `StratifiedKFold(shuffle=True)` →
   `TimeSeriesSplit(n_splits=cv_folds)`. `X_train` arrives sorted by `race_date`
   (train.py guarantees), so each fold validates only on rows later than its
   training rows — no look-ahead (fixes **C6**). Single-class early folds (rare
   on `won`) are skipped, not crashed.
2. **Objective is calibration-aware.** Optimise **log-loss** (config: `logloss` |
   `brier` | `auc`), computed _unweighted_ on each validation fold, replacing
   raw AUC. Inflated-but-well-ranked probs (the A/E≈0.2 problem) score badly on
   log-loss, so the study is steered toward honest probabilities.
3. **Class imbalance is itself a tuned choice.** New categorical
   `class_weight_mode ∈ {Balanced, SqrtBalanced, none, scale_pos_weight}`; the
   last enables a tuned `scale_pos_weight` multiplier. `run_study` resolves this
   to a real CatBoost key (`auto_class_weights` or `scale_pos_weight`, with
   `none` → explicit `auto_class_weights=None`) — the synthetic selector never
   leaks into the final fit. Lets log-loss back off the double-correction
   (`auto_class_weights` + odds-inverse `sample_weight`) flagged in **C5**.
4. **Pruner + sensible budget.** `MedianPruner` (config: `median` | `hyperband` |
   `none`) with `pruner_warmup_trials`; `trial.report()` per fold + `should_prune`
   kills hopeless trials early. TPESampler is seeded from `model.random_seed`.
   CatBoost's own `early_stopping_rounds` still trims iterations per fit.
5. **Config-driven.** All bounds, metric, pruner, and trial count live in
   `config.yaml` `model.tuning` (+ existing `optuna_trials`). `train.py` passes
   `tuning` and `random_seed` through `_load_model_cfg`, and only injects the
   `auto_class_weights="Balanced"` fallback when tuning chose _no_ scheme (i.e.
   `--no-tune` / legacy `--reuse-params`), so a tuned `none` is honoured.

## Search space (config.yaml model.tuning)

| param               | old range      | new range       | sampling |
| ------------------- | -------------- | --------------- | -------- |
| learning_rate       | [1e-3, 0.3]    | [5e-4, 0.3]     | log      |
| depth               | [4, 10]        | [4, 10]         | int      |
| l2_leaf_reg         | [1.0, 10.0]    | **[1.0, 30.0]** | log      |
| bagging_temperature | [0.0, 1.0]     | **[0.0, 2.0]**  | uniform  |
| random_strength     | [1e-9, 10.0]   | [1e-9, 10.0]    | log      |
| border_count        | [32, 254]      | [32, 254]       | int      |
| class_weight_mode   | — (fixed Bal.) | **categorical** | —        |
| scale_pos_weight    | —              | **[1.0, 25.0]** | log      |

metric=`logloss`, pruner=`median` (warmup 5), trials=50, cv_folds=5.

## Best params (winning trials — all chose class_weight_mode = "none")

| target   | lr     | depth | l2_leaf_reg | bagging_temp | random_strength | border | class_wt |
| -------- | ------ | ----- | ----------- | ------------ | --------------- | ------ | -------- |
| won      | 0.0268 | 5     | 11.23       | 1.72         | 9.6e-4          | 214    | none     |
| placed_2 | 0.0250 | 4     | 2.70        | 0.733        | 3.6e-5          | 207    | none     |
| showed   | 0.0106 | 6     | 16.39       | 0.735        | 6.9e-4          | 213    | none     |

Notable: log-loss pushed every target to **drop class weighting** and toward
**stronger L2** (won 6.75→11.23, showed 2.11→16.39) and **higher bagging temp**
(won 0.83→1.72) — i.e. flatter, better-regularised models that calibrate cleanly.

## Metric delta vs previous v3 (held-out test, calibrated)

| target   | AUC old→new     | Brier old→new       | base  | CV log-loss (new) |
| -------- | --------------- | ------------------- | ----- | ----------------- |
| won      | 0.7876 → 0.7869 | 0.0904 → **0.0901** | 0.104 | 0.2914            |
| placed_2 | 0.7854 → 0.7856 | 0.1446 → **0.1442** | 0.180 | 0.4342            |
| showed   | 0.7868 → 0.7872 | 0.1750 → **0.1745** | 0.228 | 0.5164            |

AUC is essentially flat (≤0.0007 either way — expected, ranking was already
good); Brier improves slightly on all three. The real wins are **methodological**:
honest chronological CV, a calibration-aligned objective, no more double
imbalance-correction, and v3 calibrators are now persisted to disk (also clears
**C1**). The previous tuner optimised AUC, so there is no comparable prior CV
log-loss to diff against.

Caveat: v3 still trains the **leaked market features** from `odds_finish` (audit
**C2**) and venue-day-aggregated per-race features (**C3**) — tuning does not
touch those. The price-free **v3nf** remains the trustworthy value-layer model.
See [[model-09-baseline-audit]].

## Tests

`tests/models/test_tuner.py` rewritten (9 tests): config-driven space override,
no `class_weight_mode` leak into returned params, metric-direction map,
TimeSeriesSplit chronology. Full suite **733 passed, 3 skipped** (reportlab-only).
