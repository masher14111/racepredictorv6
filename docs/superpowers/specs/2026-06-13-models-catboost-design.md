# Design: CatBoost Models Layer

**Date:** 2026-06-13  
**Status:** Approved  
**Scope:** `models/` package — three independent CatBoost binary classifiers (won / placed_2 / showed) with Optuna hyperparameter tuning, stratified k-fold CV, time-based hold-out evaluation, and feature importance logging.

---

## 1. Context

The feature layer (`features/builder.py`) produces a labelled training matrix at `data/features/training.parquet` with ~52 columns. Labels `won` and `placed` (top-3) already exist. The training matrix is currently empty (no finishing positions yet), but the infrastructure must be ready to train automatically once historical results land.

Existing `model_weights.lightgbm/baseline` in `config.yaml` remains for future ensemble use; this spec adds a new `model:` block.

---

## 2. Module Structure

```
models/
  __init__.py
  features.py    — FEATURE_COLS whitelist (~30 input columns)
  targets.py     — derive won / placed_2 / showed from position
  tuner.py       — Optuna study + stratified k-fold objective
  evaluate.py    — AUC, confusion matrix, classification report, feature importance
  train.py       — CLI entry point; orchestrates the full pipeline

tests/models/
  test_targets.py
  test_tuner.py
  test_evaluate.py
  test_train.py  — integration test with synthetic DataFrame
```

---

## 3. Target Definitions (`targets.py`)

`add_targets(df) -> pd.DataFrame` derives three binary columns from `position` (Int64, null when unknown):

| Column     | Condition                    | Notes                                                                                      |
| ---------- | ---------------------------- | ------------------------------------------------------------------------------------------ |
| `won`      | `position == 1`              | already present in training matrix; re-derived for safety                                  |
| `placed_2` | `position <= 2`              | new column — top-2 finish; named `placed_2` to avoid overwriting existing `placed` (top-3) |
| `showed`   | `position <= show_positions` | new column — mirrors existing `placed` (top-3); `show_positions` from config (default 3)   |

`add_targets` never modifies the existing `placed` (top-3) column produced by `features/labels.py`. All three output columns are `Int64` (nullable). Null position → null label (row excluded from that target's training, not dropped globally).

---

## 4. Feature Columns (`features.py`)

`FEATURE_COLS` is a static whitelist of ~30 columns. Excluded:

- **IDs / metadata:** `race_id`, `horse_id`, `jockey_id`, `trainer_id`, `src_*`, `race_date`, `venue`, `horse_name`, `fetched_at`, `source`, `result_source`
- **Target leakage:** `position`, `won`, `placed`, `placed_2`, `showed`, `win_lose`, `sp`
- **Redundant after derivation:** raw `going`, raw `race_class`

Included inputs:

```
implied_prob, overround_norm_prob, log_odds, market_rank, field_size,
going_speed, class_change, distance_furlongs,
recent_form_avg, recent_form_wins, recent_form_runs,
timeform_rating, rating_rank, pace_rating,
horse_win_rate, horse_place_rate, horse_runs_in_window,
jockey_win_rate, jockey_place_rate,
trainer_win_rate, trainer_place_rate,
jt_combo_win_rate, jt_combo_runs,
going_pref_win_rate, going_pref_place_rate,
ew_value_index, odds_drift, odds_value_delta,
horse_speed, horse_speed_rank, race_complexity
```

CatBoost handles missing values natively — no imputation step required. Sparse trailing-rate columns will have low importance initially and gain weight as historical data fills in.

---

## 5. Data Flow (`train.py`)

```
build_training_matrix()                    # features/builder.py
  → targets.add_targets(df)               # adds placed_2 (top-2) + showed cols; re-derives won
  → sample_weight = clip(1 / implied_prob, 1, max_sample_weight)  # odds-inverse, computed once
  → time_split(df, sample_weight, test_size=0.2)  # sort race_date asc, last 20% → test
  → for target in cfg.targets:            # [won, placed_2, showed]
      drop rows where target is null (adjust sample_weight slice accordingly)
      X_train, y_train, w_train = train_df[FEATURE_COLS], train_df[target], sample_weight[train_idx]
      X_test,  y_test           = test_df[FEATURE_COLS],  test_df[target]
      best_params = tuner.run_study(X_train, y_train, w_train, cfg)
      fit final CatBoostClassifier(X_train, y_train, best_params, sample_weight=w_train,
                                   early_stopping on 10% of train)
      evaluate.report(model, X_test, y_test, test_df["implied_prob"], FEATURE_COLS, target)
      model.save_model(f"models/catboost_{target}_v3.bin")
  → write models/catboost_v3_meta.json
```

**Empty matrix guard:** if `build_training_matrix()` returns 0 rows, `train.py` logs an error and exits with code 1. No model is saved.

---

## 6. Tuner (`tuner.py`)

`run_study(X_train, y_train, sample_weight, cfg) -> dict`

**Underdog bias mitigation (applied before CV):**

Favorites dominate positives in training data (a 2/1 shot wins ~3× as often as a 10/1 shot), so without intervention the model becomes a glorified odds re-ranker. Three defences:

1. **Odds-inverse sample weights** — each row's weight = `1 / implied_prob` (clipped to `[1, max_sample_weight]`, default cap 20). This makes the model pay more attention to high-odds runners, penalising it more heavily when it misses an underdog winner. Computed in `train.py` from `implied_prob` before the train/test split; passed through to every CatBoost fit call.

2. **`auto_class_weights='Balanced'`** — CatBoost balances the positive/negative class ratio automatically on top of the sample weights. Tunable: Optuna can also search `scale_pos_weight` as an alternative (see search space below).

3. **Odds-band AUC in evaluation** — `evaluate.report` slices the test set into three odds bands (favourite: implied_prob ≥ 0.25; mid: 0.10–0.25; underdog: < 0.10) and logs AUC for each band separately. If underdog-band AUC collapses near 0.5, the model has learned nothing useful beyond the market.

**Objective function (per Optuna trial):**

1. `StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=42)`
2. Each fold: fit `CatBoostClassifier` with trial params + fold's slice of `sample_weight`; `early_stopping_rounds=cfg.early_stopping_rounds` against the fold's eval set
3. Score: `roc_auc_score(y_val, y_prob)` → mean AUC across folds
4. Direction: `maximize`

**Hyperparameter search space:**

| Param                 | Distribution                                                        |
| --------------------- | ------------------------------------------------------------------- |
| `learning_rate`       | log-uniform(1e-3, 0.3)                                              |
| `depth`               | int(4, 10)                                                          |
| `l2_leaf_reg`         | log-uniform(1, 10)                                                  |
| `bagging_temperature` | uniform(0, 1)                                                       |
| `random_strength`     | log-uniform(1e-9, 10)                                               |
| `border_count`        | int(32, 255)                                                        |
| `scale_pos_weight`    | log-uniform(1, 20) — searched only when `auto_class_weights` is off |

**Fixed CatBoost params:** `iterations=cfg.iterations` (1000), `loss_function='Logloss'`, `eval_metric='AUC'`, `auto_class_weights='Balanced'`, `verbose=False`, `allow_writing_files=False`.

Optuna logging set to `WARNING` to suppress per-trial noise; study progress logged via `get_logger`.

---

## 7. Evaluate (`evaluate.py`)

`report(model, X_test, y_test, feature_cols, target_name) -> dict`

Logs via `get_logger(__name__)` and returns a metrics dict:

- **AUC-ROC (overall):** `roc_auc_score(y_test, y_prob)`
- **AUC-ROC by odds band** — test set sliced into three bands using `implied_prob`; AUC logged per band:
  - Favourite: `implied_prob >= 0.25` (odds ≤ 4.0)
  - Mid: `0.10 <= implied_prob < 0.25` (odds 4–10)
  - Underdog: `implied_prob < 0.10` (odds > 10)
    If underdog-band AUC ≈ 0.5, the model has learned nothing beyond market odds for that segment.
- **Classification report:** precision / recall / F1 at threshold 0.5 (`classification_report`)
- **Confusion matrix:** plain-text 2×2 table
- **Feature importance:** top-20 by `get_feature_importance(type='PredictionValuesChange')`, logged as a ranked list

No plots written to disk — Streamlit UI handles visualisation.

---

## 8. Config (`config.yaml` additions)

```yaml
model:
  test_size: 0.2
  cv_folds: 5
  optuna_trials: 50
  early_stopping_rounds: 50
  iterations: 1000
  show_positions: 3 # position <= N counts as "showed"
  targets: [won, placed_2, showed]
  model_dir: models
  max_sample_weight: 20 # cap on odds-inverse sample weight (prevents extreme outliers)
```

---

## 9. Saved Artefacts

```
models/catboost_won_v3.bin
models/catboost_placed_2_v3.bin
models/catboost_showed_v3.bin
models/catboost_v3_meta.json   ← best_params per target, test AUC per target,
                                   feature_cols list, train/test row counts, timestamp
```

---

## 10. CLI

```
python -m models.train [options]

  --targets won placed_2 showed   subset of targets (default: all from config)
  --trials N                    override optuna_trials (e.g. 5 for smoke test)
  --no-tune                     skip Optuna, use CatBoost defaults
  --model-dir PATH              override model_dir from config
```

---

## 11. Testing

| File                            | Coverage                                                                                                                     |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `tests/models/test_targets.py`  | `add_targets`: top-1/2/3 derivation; null position → null label; non-null unaffected                                         |
| `tests/models/test_tuner.py`    | objective returns float AUC; `n_trials=2` study completes; stratified folds preserve positive class ratio                    |
| `tests/models/test_evaluate.py` | `report` on known arrays; confusion matrix is 2×2; feature importance list length == min(20, n_features)                     |
| `tests/models/test_train.py`    | integration: 200-row synthetic df → `train(targets=['won'], trials=1)` → `.bin` saved, `meta.json` written with correct keys |

---

## 12. Dependencies Added

```
catboost
optuna
```

Add to `requirements.txt`. No other new deps — `scikit-learn` (already present) provides `StratifiedKFold` and `roc_auc_score`.
