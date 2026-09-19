# CatBoost Models Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `models/` — three independent CatBoost binary classifiers (won / placed_2 / showed) with Optuna hyperparameter tuning, stratified k-fold CV, odds-inverse sample weighting for underdog bias mitigation, time-based hold-out evaluation, per-odds-band AUC reporting, and feature importance logging.

**Architecture:** `features.py` holds the 28-column input whitelist; `targets.py` derives labels from `position`; `tuner.py` runs an Optuna study using stratified k-fold as the objective; `evaluate.py` logs metrics including per-odds-band AUC; `train.py` orchestrates split → tune → fit → evaluate → save for each target. The `train()` function accepts an optional pre-loaded `df` so tests never touch disk.

**Tech Stack:** catboost, optuna, scikit-learn (StratifiedKFold, roc_auc_score, classification_report, confusion_matrix), pandas, numpy, pyyaml, pytest.

---

## File Map

| Action | Path                            |
| ------ | ------------------------------- |
| Create | `models/__init__.py`            |
| Create | `models/features.py`            |
| Create | `models/targets.py`             |
| Create | `models/evaluate.py`            |
| Create | `models/tuner.py`               |
| Create | `models/train.py`               |
| Create | `tests/models/__init__.py`      |
| Create | `tests/models/test_targets.py`  |
| Create | `tests/models/test_evaluate.py` |
| Create | `tests/models/test_tuner.py`    |
| Create | `tests/models/test_train.py`    |
| Modify | `requirements.txt`              |
| Modify | `config.yaml`                   |

---

## Task 1: Add dependencies and config

**Files:**

- Modify: `requirements.txt`
- Modify: `config.yaml`

- [ ] **Step 1: Add catboost and optuna to requirements.txt**

Open `requirements.txt` and append two lines after `pydantic`:

```
catboost
optuna
```

- [ ] **Step 2: Install the new deps**

```powershell
pip install catboost optuna
```

Expected: both install without error. catboost is large (~100 MB); this may take a minute.

- [ ] **Step 3: Add model block to config.yaml**

Open `config.yaml` and append at the end (after the `timeform:` block):

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
  max_sample_weight: 20 # cap on odds-inverse sample weight
```

- [ ] **Step 4: Verify catboost import works**

```powershell
python -c "from catboost import CatBoostClassifier; import optuna; print('OK')"
```

Expected output: `OK`

- [ ] **Step 5: Commit**

```bash
git add requirements.txt config.yaml
git commit -m "feat: add catboost + optuna deps; add model config block"
```

---

## Task 2: models package + features.py

**Files:**

- Create: `models/__init__.py`
- Create: `models/features.py`
- Create: `tests/models/__init__.py`

Column names are exact — derived by reading `features/derive.py` and `features/engine.py`. Notable differences from the spec prose:

- horse trailing rates → `historical_win_rate` / `historical_place_rate` (not `horse_*`)
- pace feature → `pace_bias` (output of `engine.add_pace_bias`, not `pace_rating`)
- jockey/trainer place rates are **dropped** internally — only win rates exist as columns

- [ ] **Step 1: Create models/**init**.py**

```python

```

(empty file)

- [ ] **Step 2: Create tests/models/**init**.py**

```python

```

(empty file)

- [ ] **Step 3: Write test for FEATURE_COLS**

Create `tests/models/test_features.py`:

```python
from models.features import FEATURE_COLS


def test_feature_cols_is_list_of_strings():
    assert isinstance(FEATURE_COLS, list)
    assert all(isinstance(c, str) for c in FEATURE_COLS)
    assert len(FEATURE_COLS) > 0


def test_no_leakage_cols():
    leakage = {"position", "won", "placed", "placed_2", "showed", "win_lose", "sp"}
    assert leakage.isdisjoint(set(FEATURE_COLS)), \
        f"Leakage columns found in FEATURE_COLS: {leakage & set(FEATURE_COLS)}"


def test_no_id_cols():
    id_cols = {"race_id", "horse_id", "jockey_id", "trainer_id",
               "horse_name", "venue", "fetched_at", "source"}
    assert id_cols.isdisjoint(set(FEATURE_COLS)), \
        f"ID columns found in FEATURE_COLS: {id_cols & set(FEATURE_COLS)}"


def test_no_duplicates():
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))
```

- [ ] **Step 4: Run test to verify it fails**

```powershell
python -m pytest tests/models/test_features.py -v
```

Expected: `ModuleNotFoundError: No module named 'models.features'`

- [ ] **Step 5: Create models/features.py**

```python
"""Input feature whitelist for CatBoost models.

Column names match the output of features/derive.py and features/engine.py exactly.
CatBoost handles NaN natively — columns present but all-null are fine.
"""

FEATURE_COLS = [
    # odds-market features
    "implied_prob",
    "overround_norm_prob",
    "log_odds",
    "market_rank",
    "field_size",
    # race context
    "going_speed",
    "class_change",
    "distance_furlongs",
    # recent form (from timeform free finishing figures)
    "recent_form_avg",
    "recent_form_wins",
    "recent_form_runs",
    # ratings
    "timeform_rating",
    "rating_rank",
    "pace_bias",
    # trailing rates — horse (historical_* from derive.add_all_trailing_rates)
    "historical_win_rate",
    "historical_place_rate",
    # trailing rates — jockey / trainer (place rates are dropped internally)
    "jockey_win_rate",
    "trainer_win_rate",
    # jockey-trainer combo
    "jt_combo_win_rate",
    "jt_combo_runs",
    # going preference
    "going_pref_win_rate",
    "going_pref_place_rate",
    # value / drift signals
    "ew_value_index",
    "odds_drift",
    "odds_value_delta",
    # speed / complexity
    "horse_speed",
    "horse_speed_rank",
    "race_complexity",
]
```

- [ ] **Step 6: Run tests to verify they pass**

```powershell
python -m pytest tests/models/test_features.py -v
```

Expected: 4 tests PASSED.

- [ ] **Step 7: Commit**

```bash
git add models/__init__.py models/features.py tests/models/__init__.py tests/models/test_features.py
git commit -m "feat: models package + FEATURE_COLS whitelist"
```

---

## Task 3: targets.py

**Files:**

- Create: `models/targets.py`
- Create: `tests/models/test_targets.py`

`add_targets` adds three Int64 nullable columns: `won` (pos==1), `placed_2` (pos<=2), `showed` (pos<=show_positions). It never modifies the existing `placed` column from `features/labels.py`.

- [ ] **Step 1: Write tests**

Create `tests/models/test_targets.py`:

```python
import pandas as pd
import pytest

from models.targets import add_targets


def _df(positions):
    """Minimal df with position (Int64) and existing placed (top-3) column."""
    pos = pd.array(positions, dtype="Int64")
    placed = pd.array(
        [pd.NA if p is pd.NA else int(p <= 3) for p in positions], dtype="Int64"
    )
    return pd.DataFrame({"position": pos, "placed": placed})


def test_won_derivation():
    out = add_targets(_df([1, 2, 3, 4, 5]), show_positions=3)
    assert list(out["won"].fillna(-1)) == [1, 0, 0, 0, 0]


def test_placed_2_derivation():
    out = add_targets(_df([1, 2, 3, 4, 5]), show_positions=3)
    assert list(out["placed_2"].fillna(-1)) == [1, 1, 0, 0, 0]


def test_showed_derivation_default_3():
    out = add_targets(_df([1, 2, 3, 4, 5]), show_positions=3)
    assert list(out["showed"].fillna(-1)) == [1, 1, 1, 0, 0]


def test_showed_derivation_custom_positions():
    out = add_targets(_df([1, 2, 3, 4, 5]), show_positions=4)
    assert list(out["showed"].fillna(-1)) == [1, 1, 1, 1, 0]


def test_null_position_gives_null_labels():
    out = add_targets(_df([pd.NA, pd.NA]), show_positions=3)
    assert out["won"].isna().all()
    assert out["placed_2"].isna().all()
    assert out["showed"].isna().all()


def test_mixed_null_and_non_null():
    out = add_targets(_df([1, pd.NA, 4]), show_positions=3)
    assert list(out["won"].fillna(-1)) == [1, -1, 0]
    assert list(out["placed_2"].fillna(-1)) == [1, -1, 0]
    assert list(out["showed"].fillna(-1)) == [1, -1, 0]


def test_existing_placed_col_untouched():
    """add_targets must never overwrite the existing placed (top-3) column."""
    df = _df([1, 4])
    out = add_targets(df, show_positions=3)
    # original placed column: pos 1 → 1, pos 4 → 0
    assert list(out["placed"].fillna(-1)) == [1, 0]


def test_output_dtype_is_Int64():
    out = add_targets(_df([1, 2, None]), show_positions=3)
    assert str(out["won"].dtype) == "Int64"
    assert str(out["placed_2"].dtype) == "Int64"
    assert str(out["showed"].dtype) == "Int64"
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
python -m pytest tests/models/test_targets.py -v
```

Expected: `ModuleNotFoundError: No module named 'models.targets'`

- [ ] **Step 3: Implement models/targets.py**

```python
"""Derive binary target columns from finishing position.

add_targets never overwrites the existing `placed` (top-3) column from
features/labels.py — it only adds won, placed_2, and showed.
"""
import pandas as pd


def add_targets(df: pd.DataFrame, show_positions: int = 3) -> pd.DataFrame:
    """Add won, placed_2, showed columns derived from position.

    All three are Int64 (nullable). Null position -> null label.
    Existing columns (including `placed`) are never modified.
    """
    out = df.copy()
    pos = pd.to_numeric(out["position"], errors="coerce")

    def _binary(mask_series) -> pd.array:
        result = pd.array([pd.NA] * len(out), dtype="Int64")
        known = pos.notna()
        result[known] = mask_series[known].astype("Int64")
        return result

    out["won"] = _binary(pos == 1)
    out["placed_2"] = _binary(pos <= 2)
    out["showed"] = _binary(pos <= show_positions)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
python -m pytest tests/models/test_targets.py -v
```

Expected: 8 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add models/targets.py tests/models/test_targets.py
git commit -m "feat: models/targets.py — derive won/placed_2/showed labels"
```

---

## Task 4: evaluate.py

**Files:**

- Create: `models/evaluate.py`
- Create: `tests/models/test_evaluate.py`

`report(model, X_test, y_test, implied_prob, feature_cols, target_name) -> dict` logs overall AUC, per-odds-band AUC (favourite/mid/underdog), classification report, confusion matrix, and top-20 feature importances. Returns all metrics as a dict.

- [ ] **Step 1: Write tests**

Create `tests/models/test_evaluate.py`:

```python
import numpy as np
import pytest
from catboost import CatBoostClassifier

from models.evaluate import report
from models.features import FEATURE_COLS


def _tiny_model(n_features=5, n_samples=120, seed=42):
    """Train a minimal CatBoost model on synthetic data for testing."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_samples, n_features)).astype(float)
    y = (X[:, 0] > 0.5).astype(int)
    model = CatBoostClassifier(
        iterations=10, verbose=False, allow_writing_files=False,
        auto_class_weights="Balanced",
    )
    model.fit(X, y)
    return model, X, y


def test_report_returns_dict():
    model, X, y = _tiny_model()
    cols = [f"f{i}" for i in range(X.shape[1])]
    implied_prob = np.full(len(y), 0.15)  # all mid-band
    result = report(model, X, y, implied_prob, cols, "won")
    assert isinstance(result, dict)


def test_report_has_auc_key():
    model, X, y = _tiny_model()
    cols = [f"f{i}" for i in range(X.shape[1])]
    implied_prob = np.full(len(y), 0.15)
    result = report(model, X, y, implied_prob, cols, "won")
    assert "auc" in result
    assert 0.0 <= result["auc"] <= 1.0


def test_report_has_band_aucs():
    model, X, y = _tiny_model()
    cols = [f"f{i}" for i in range(X.shape[1])]
    # spread across all three bands
    rng = np.random.default_rng(0)
    implied_prob = rng.uniform(0.01, 0.5, len(y))
    result = report(model, X, y, implied_prob, cols, "won")
    assert "band_aucs" in result
    assert set(result["band_aucs"].keys()) == {"favourite", "mid", "underdog"}


def test_report_confusion_matrix_shape():
    model, X, y = _tiny_model()
    cols = [f"f{i}" for i in range(X.shape[1])]
    implied_prob = np.full(len(y), 0.15)
    result = report(model, X, y, implied_prob, cols, "won")
    cm = result["confusion_matrix"]
    assert len(cm) == 2
    assert len(cm[0]) == 2


def test_report_feature_importance_length():
    model, X, y = _tiny_model(n_features=5)
    cols = [f"f{i}" for i in range(5)]
    implied_prob = np.full(len(y), 0.15)
    result = report(model, X, y, implied_prob, cols, "won")
    fi = result["feature_importance"]
    assert len(fi) == min(20, 5)  # 5 features → 5 entries


def test_report_feature_importance_sorted_descending():
    model, X, y = _tiny_model(n_features=5)
    cols = [f"f{i}" for i in range(5)]
    implied_prob = np.full(len(y), 0.15)
    result = report(model, X, y, implied_prob, cols, "won")
    importances = [v for _, v in result["feature_importance"]]
    assert importances == sorted(importances, reverse=True)


def test_band_auc_none_when_single_class():
    """Band with only one class label → AUC reported as None, not an error."""
    model, X, y = _tiny_model(n_samples=120)
    cols = [f"f{i}" for i in range(X.shape[1])]
    # force all to favourite band, but all labels = 0 (single class)
    implied_prob = np.full(len(y), 0.5)
    y_single = np.zeros(len(y), dtype=int)
    # retrain on single-class data (model will be trivial, but that's ok)
    model2 = CatBoostClassifier(
        iterations=5, verbose=False, allow_writing_files=False,
    )
    model2.fit(X, y_single)
    result = report(model2, X, y_single, implied_prob, cols, "test")
    # band AUC should be None, not raise
    assert result["band_aucs"]["favourite"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
python -m pytest tests/models/test_evaluate.py -v
```

Expected: `ModuleNotFoundError: No module named 'models.evaluate'`

- [ ] **Step 3: Implement models/evaluate.py**

```python
"""Evaluation metrics for a trained CatBoost binary classifier.

report() logs to the rotating logger and returns a metrics dict.
No plots are written — the Streamlit UI handles visualisation.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from utils.logger import get_logger

logger = get_logger(__name__)

_BANDS = [
    ("favourite", lambda p: p >= 0.25),
    ("mid",       lambda p: (p >= 0.10) & (p < 0.25)),
    ("underdog",  lambda p: p < 0.10),
]


def report(model, X_test, y_test, implied_prob, feature_cols, target_name):
    """Evaluate model on hold-out test set and log all metrics.

    Parameters
    ----------
    model        : trained CatBoostClassifier
    X_test       : np.ndarray or pd.DataFrame of shape (n, n_features)
    y_test       : 1-D array of int labels (0/1)
    implied_prob : 1-D array of float, same length as y_test
    feature_cols : list[str], same order as X_test columns
    target_name  : str, used in log prefixes

    Returns
    -------
    dict with keys: auc, band_aucs, classification_report, confusion_matrix,
                    feature_importance
    """
    X = np.asarray(X_test, dtype=float)
    y = np.asarray(y_test, dtype=int)
    ip = pd.to_numeric(pd.Series(implied_prob), errors="coerce").to_numpy(dtype=float)

    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {}

    # --- overall AUC --------------------------------------------------------
    auc = roc_auc_score(y, y_prob)
    metrics["auc"] = float(auc)
    logger.info("[%s] AUC=%.4f  n=%d  pos=%.1f%%",
                target_name, auc, len(y), 100 * y.mean())

    # --- per-odds-band AUC --------------------------------------------------
    band_aucs = {}
    for band_name, mask_fn in _BANDS:
        valid = ~np.isnan(ip)
        mask = mask_fn(ip) & valid
        if mask.sum() >= 2 and len(np.unique(y[mask])) > 1:
            b_auc = float(roc_auc_score(y[mask], y_prob[mask]))
            band_aucs[band_name] = b_auc
            logger.info("[%s] %s-band AUC=%.4f (n=%d)",
                        target_name, band_name, b_auc, int(mask.sum()))
        else:
            band_aucs[band_name] = None
            logger.info("[%s] %s-band: insufficient data (n=%d, classes=%s)",
                        target_name, band_name, int(mask.sum()),
                        list(np.unique(y[mask])) if mask.sum() else [])
    metrics["band_aucs"] = band_aucs

    # --- classification report ----------------------------------------------
    cr = classification_report(y, y_pred, zero_division=0)
    logger.info("[%s] classification report:\n%s", target_name, cr)
    metrics["classification_report"] = cr

    # --- confusion matrix ---------------------------------------------------
    cm = confusion_matrix(y, y_pred)
    cm_str = f"[[{cm[0,0]} {cm[0,1]}]\n [{cm[1,0]} {cm[1,1]}]]"
    logger.info("[%s] confusion matrix:\n%s", target_name, cm_str)
    metrics["confusion_matrix"] = cm.tolist()

    # --- feature importance (top-20) ----------------------------------------
    importances = model.get_feature_importance(type="PredictionValuesChange")
    top_n = min(20, len(feature_cols))
    pairs = sorted(zip(feature_cols, importances.tolist()),
                   key=lambda x: x[1], reverse=True)[:top_n]
    logger.info("[%s] top features: %s", target_name, pairs)
    metrics["feature_importance"] = pairs

    return metrics
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
python -m pytest tests/models/test_evaluate.py -v
```

Expected: 7 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add models/evaluate.py tests/models/test_evaluate.py
git commit -m "feat: models/evaluate.py — AUC, band-AUC, confusion matrix, feature importance"
```

---

## Task 5: tuner.py

**Files:**

- Create: `models/tuner.py`
- Create: `tests/models/test_tuner.py`

`run_study(X_train, y_train, sample_weight, cfg) -> dict` runs an Optuna study using stratified k-fold AUC as the objective. Returns the best hyperparams dict (without fixed params — caller merges them).

- [ ] **Step 1: Write tests**

Create `tests/models/test_tuner.py`:

```python
import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from models.tuner import run_study

_TINY_CFG = {
    "cv_folds": 2,
    "optuna_trials": 2,
    "iterations": 10,
    "early_stopping_rounds": 5,
}


def _data(seed=0, n=100, n_features=5, pos_rate=0.3):
    rng = np.random.default_rng(seed)
    X = rng.random((n, n_features)).astype(float)
    y = (X[:, 0] > (1 - pos_rate)).astype(int)
    w = np.ones(n)
    return X, y, w


def test_run_study_returns_dict():
    X, y, w = _data()
    params = run_study(X, y, w, _TINY_CFG)
    assert isinstance(params, dict)


def test_run_study_has_learning_rate():
    X, y, w = _data()
    params = run_study(X, y, w, _TINY_CFG)
    assert "learning_rate" in params
    assert 1e-3 <= params["learning_rate"] <= 0.3


def test_run_study_has_depth():
    X, y, w = _data()
    params = run_study(X, y, w, _TINY_CFG)
    assert "depth" in params
    assert 4 <= params["depth"] <= 10


def test_run_study_with_sample_weight():
    """Underdog weighting (non-uniform weights) should not crash the study."""
    X, y, w = _data()
    # simulate underdog weights: high-index runners get higher weight
    w = np.linspace(1.0, 20.0, len(y))
    params = run_study(X, y, w, _TINY_CFG)
    assert params is not None


def test_stratified_folds_preserve_class_ratio():
    """Verify StratifiedKFold keeps positive rate within 5pp of true rate."""
    rng = np.random.default_rng(42)
    y = np.zeros(100, dtype=int)
    y[:10] = 1
    rng.shuffle(y)
    X = rng.random((100, 3))

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for _, val_idx in kf.split(X, y):
        val_rate = y[val_idx].mean()
        assert abs(val_rate - 0.10) < 0.05
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
python -m pytest tests/models/test_tuner.py -v
```

Expected: `ModuleNotFoundError: No module named 'models.tuner'`

- [ ] **Step 3: Implement models/tuner.py**

```python
"""Optuna hyperparameter study for CatBoost binary classifiers.

run_study returns only the tuned hyperparams (not fixed params like loss_function).
The caller merges them with fixed params before fitting the final model.
"""
import logging

import numpy as np
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from utils.logger import get_logger

logger = get_logger(__name__)
optuna.logging.set_verbosity(logging.WARNING)


def _objective(trial, X, y, sample_weight, cfg):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-9, 10.0, log=True),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "iterations": cfg["iterations"],
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "auto_class_weights": "Balanced",
        "verbose": False,
        "allow_writing_files": False,
    }
    kf = StratifiedKFold(n_splits=cfg["cv_folds"], shuffle=True, random_state=42)
    aucs = []
    for train_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        w_tr = sample_weight[train_idx]
        model = CatBoostClassifier(**params)
        model.fit(
            X_tr, y_tr,
            sample_weight=w_tr,
            eval_set=(X_val, y_val),
            early_stopping_rounds=cfg["early_stopping_rounds"],
            verbose=False,
        )
        y_prob = model.predict_proba(X_val)[:, 1]
        aucs.append(roc_auc_score(y_val, y_prob))
    return float(np.mean(aucs))


def run_study(X_train, y_train, sample_weight, cfg):
    """Run an Optuna study; return the best hyperparams dict.

    Parameters
    ----------
    X_train       : np.ndarray (n, n_features)
    y_train       : np.ndarray (n,) int 0/1
    sample_weight : np.ndarray (n,) float, odds-inverse weights
    cfg           : dict with keys cv_folds, optuna_trials, iterations,
                    early_stopping_rounds

    Returns
    -------
    dict of tuned hyperparams (no fixed params — caller merges)
    """
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=int)
    w = np.asarray(sample_weight, dtype=float)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _objective(trial, X, y, w, cfg),
        n_trials=cfg["optuna_trials"],
        show_progress_bar=False,
    )
    logger.info("tuner: best CV AUC=%.4f  params=%s",
                study.best_value, study.best_params)
    return study.best_params
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
python -m pytest tests/models/test_tuner.py -v
```

Expected: 5 tests PASSED. The two `run_study` tests each run 2 Optuna trials; each trial fits 2 CV folds with 10 CatBoost iterations — should complete in under 30 seconds total.

- [ ] **Step 5: Commit**

```bash
git add models/tuner.py tests/models/test_tuner.py
git commit -m "feat: models/tuner.py — Optuna stratified k-fold study with underdog sample weights"
```

---

## Task 6: train.py

**Files:**

- Create: `models/train.py`
- Create: `tests/models/test_train.py`

`train(df=None, targets=None, trials=None, no_tune=False, model_dir=None) -> dict|None` is the programmatic entry point. When `df=None` it calls `build_training_matrix(write=False)`. The CLI `main()` parses args and calls `train()`. Returns `None` and exits 1 when the training matrix is empty.

Key design:

- Sample weights computed on full df before the time split, so test rows have no weights computed from them.
- FEATURE_COLS filtered to columns actually present in df (avoids KeyError when features are sparse).
- Final model uses a 10% temporal hold-out carved from the training split for early stopping.

- [ ] **Step 1: Write tests**

Create `tests/models/test_train.py`:

```python
import json
import os

import numpy as np
import pandas as pd
import pytest

from models.features import FEATURE_COLS
from models.train import _sample_weights, _time_split, train


# ── helpers ─────────────────────────────────────────────────────────────────

def _synthetic_df(n=300, seed=42):
    """Labelled df with all FEATURE_COLS present and realistic dtypes."""
    rng = np.random.default_rng(seed)
    data = {col: rng.random(n) for col in FEATURE_COLS}
    data["race_date"] = pd.date_range("2023-01-01", periods=n, freq="D")
    data["implied_prob"] = rng.uniform(0.05, 0.5, n)
    data["position"] = pd.array(rng.integers(1, 10, n), dtype="Int64")
    data["placed"] = pd.array((rng.integers(1, 10, n) <= 3).astype(int), dtype="Int64")
    return pd.DataFrame(data)


# ── unit: _time_split ────────────────────────────────────────────────────────

def test_time_split_sizes():
    df = _synthetic_df(100)
    train_df, test_df = _time_split(df, 0.2)
    assert len(train_df) == 80
    assert len(test_df) == 20


def test_time_split_train_before_test():
    df = _synthetic_df(100)
    train_df, test_df = _time_split(df, 0.2)
    assert train_df["race_date"].max() <= test_df["race_date"].min()


# ── unit: _sample_weights ────────────────────────────────────────────────────

def test_sample_weights_range():
    df = _synthetic_df(50)
    w = _sample_weights(df, max_weight=20)
    assert w.min() >= 1.0
    assert w.max() <= 20.0


def test_sample_weights_underdog_higher():
    df = _synthetic_df(10)
    df["implied_prob"] = [0.5, 0.1, 0.05, 0.5, 0.1, 0.05, 0.5, 0.1, 0.05, 0.5]
    w = _sample_weights(df, max_weight=20)
    # runner with implied_prob=0.05 should get higher weight than 0.5
    ip = df["implied_prob"].values
    for i in range(len(ip)):
        for j in range(len(ip)):
            if ip[i] < ip[j]:
                assert w[i] >= w[j]


def test_sample_weights_nan_gets_neutral():
    df = _synthetic_df(5)
    df.loc[2, "implied_prob"] = float("nan")
    w = _sample_weights(df, max_weight=20)
    assert w[2] == pytest.approx(1.0)


# ── integration: train() ─────────────────────────────────────────────────────

def test_train_returns_none_on_empty_df(tmp_path):
    empty = pd.DataFrame(columns=list(_synthetic_df(1).columns))
    result = train(df=empty, targets=["won"], trials=1,
                   no_tune=True, model_dir=str(tmp_path))
    assert result is None


def test_train_saves_model_bin(tmp_path):
    df = _synthetic_df(300)
    result = train(df=df, targets=["won"], trials=1,
                   no_tune=True, model_dir=str(tmp_path))
    assert result is not None
    assert os.path.exists(os.path.join(str(tmp_path), "catboost_won_v3.bin"))


def test_train_saves_meta_json(tmp_path):
    df = _synthetic_df(300)
    result = train(df=df, targets=["won"], trials=1,
                   no_tune=True, model_dir=str(tmp_path))
    meta_path = os.path.join(str(tmp_path), "catboost_v3_meta.json")
    assert os.path.exists(meta_path)
    with open(meta_path) as fh:
        meta = json.load(fh)
    assert "won" in meta["targets"]
    assert "test_auc" in meta["targets"]["won"]
    assert "feature_cols" in meta
    assert "train_rows" in meta
    assert "test_rows" in meta


def test_train_meta_band_aucs_present(tmp_path):
    df = _synthetic_df(300)
    result = train(df=df, targets=["won"], trials=1,
                   no_tune=True, model_dir=str(tmp_path))
    assert "band_aucs" in result["targets"]["won"]
    band_aucs = result["targets"]["won"]["band_aucs"]
    assert set(band_aucs.keys()) == {"favourite", "mid", "underdog"}
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
python -m pytest tests/models/test_train.py -v
```

Expected: `ModuleNotFoundError: No module named 'models.train'`

- [ ] **Step 3: Implement models/train.py**

```python
"""CLI entry point and programmatic API for training CatBoost race models.

Usage:
    python -m models.train [--targets won placed_2 showed]
                           [--trials N] [--no-tune] [--model-dir PATH]

The train() function accepts an optional pre-loaded df so callers (including
tests) never need to touch disk.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier

from features.builder import build_training_matrix
from models.evaluate import report
from models.features import FEATURE_COLS
from models.targets import add_targets
from models.tuner import run_study
from utils.logger import get_logger

logger = get_logger(__name__)

_BASE = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_CONFIG_PATH = os.path.join(_BASE, "config.yaml")


def _load_model_cfg():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    m = raw.get("model", {})
    return {
        "test_size": float(m.get("test_size", 0.2)),
        "cv_folds": int(m.get("cv_folds", 5)),
        "optuna_trials": int(m.get("optuna_trials", 50)),
        "early_stopping_rounds": int(m.get("early_stopping_rounds", 50)),
        "iterations": int(m.get("iterations", 1000)),
        "show_positions": int(m.get("show_positions", 3)),
        "targets": list(m.get("targets", ["won", "placed_2", "showed"])),
        "model_dir": str(m.get("model_dir", "models")),
        "max_sample_weight": float(m.get("max_sample_weight", 20.0)),
    }


def _time_split(df, test_size):
    """Sort by race_date asc; last test_size fraction -> test."""
    df = df.sort_values("race_date").reset_index(drop=True)
    split = int(len(df) * (1.0 - test_size))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def _sample_weights(df, max_weight):
    """Odds-inverse weights clipped to [1, max_weight]. NaN implied_prob -> 1.0."""
    ip = pd.to_numeric(df["implied_prob"], errors="coerce")
    ip = ip.fillna(1.0)                       # NaN -> neutral weight
    ip = ip.clip(lower=1.0 / max_weight)      # floor avoids weight > max_weight
    return (1.0 / ip).clip(upper=max_weight).to_numpy(dtype=float)


def train(df=None, targets=None, trials=None, no_tune=False, model_dir=None):
    """Train CatBoost models and save artefacts.

    Parameters
    ----------
    df         : pre-loaded labelled DataFrame. If None, calls build_training_matrix().
    targets    : list of target names to train. Defaults to config value.
    trials     : override optuna_trials from config.
    no_tune    : if True, skip Optuna and use CatBoost defaults.
    model_dir  : override model_dir from config.

    Returns
    -------
    dict of metadata, or None if the training matrix is empty.
    """
    cfg = _load_model_cfg()
    if targets is not None:
        cfg["targets"] = targets
    if trials is not None:
        cfg["optuna_trials"] = trials
    if model_dir is not None:
        cfg["model_dir"] = model_dir

    if df is None:
        logger.info("train: loading training matrix from disk")
        df = build_training_matrix(write=False)

    if len(df) == 0:
        logger.error(
            "train: training matrix is empty — no labelled data (finishing positions) "
            "available yet. Fetch historical results first."
        )
        return None

    df = add_targets(df, cfg["show_positions"])

    # compute odds-inverse sample weights once on the full df (before split)
    w_full = _sample_weights(df, cfg["max_sample_weight"])

    train_df, test_df = _time_split(df, cfg["test_size"])
    n_train = len(train_df)
    w_train_full = w_full[:n_train]

    out_dir = os.path.join(_BASE, cfg["model_dir"])
    os.makedirs(out_dir, exist_ok=True)

    # filter FEATURE_COLS to columns actually present in the df
    avail_cols = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(avail_cols)
    if missing:
        logger.warning("train: %d FEATURE_COLS not in df, skipping: %s",
                       len(missing), sorted(missing))

    meta = {
        "targets": {},
        "feature_cols": avail_cols,
        "train_rows": n_train,
        "test_rows": len(test_df),
    }

    for target in cfg["targets"]:
        if target not in train_df.columns:
            logger.warning("train: target '%s' not in dataframe, skipping", target)
            continue

        logger.info("train: ── target=%s ──", target)

        # drop rows where this target is null
        tr_mask = train_df[target].notna().values
        te_mask = test_df[target].notna().values

        X_tr_raw = train_df.loc[tr_mask, avail_cols].copy()
        y_tr = train_df.loc[tr_mask, target].astype(int).values
        w_tr = w_train_full[tr_mask]

        X_te_raw = test_df.loc[te_mask, avail_cols].copy()
        y_te = test_df.loc[te_mask, target].astype(int).values
        ip_te = pd.to_numeric(test_df.loc[te_mask, "implied_prob"],
                              errors="coerce").to_numpy(dtype=float)

        # CatBoost handles NaN, but arrays must be float
        X_tr = X_tr_raw.fillna(np.nan).astype(float).values
        X_te = X_te_raw.fillna(np.nan).astype(float).values

        if no_tune:
            best_params = {}
            logger.info("train: --no-tune: using CatBoost defaults")
        else:
            logger.info("train: running Optuna (%d trials)", cfg["optuna_trials"])
            best_params = run_study(X_tr, y_tr, w_tr, cfg)

        # fit final model; hold out last 10% of train for early stopping
        n_val = max(1, int(len(X_tr) * 0.10))
        X_fit, X_es = X_tr[:-n_val], X_tr[-n_val:]
        y_fit, y_es = y_tr[:-n_val], y_tr[-n_val:]
        w_fit = w_tr[:-n_val]

        final_params = {
            "iterations": cfg["iterations"],
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "auto_class_weights": "Balanced",
            "early_stopping_rounds": cfg["early_stopping_rounds"],
            "verbose": False,
            "allow_writing_files": False,
            **best_params,
        }
        model = CatBoostClassifier(**final_params)
        model.fit(X_fit, y_fit, sample_weight=w_fit,
                  eval_set=(X_es, y_es), verbose=False)

        metrics = report(model, X_te, y_te, ip_te, avail_cols, target)

        model_path = os.path.join(out_dir, f"catboost_{target}_v3.bin")
        model.save_model(model_path)
        logger.info("train: saved %s", model_path)

        meta["targets"][target] = {
            "best_params": best_params,
            "test_auc": metrics["auc"],
            "band_aucs": metrics["band_aucs"],
        }

    meta_path = os.path.join(out_dir, "catboost_v3_meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    logger.info("train: saved meta to %s", meta_path)

    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Train CatBoost race outcome models (won / placed_2 / showed)"
    )
    parser.add_argument(
        "--targets", nargs="+", choices=["won", "placed_2", "showed"], default=None,
        help="Subset of targets to train (default: all from config)"
    )
    parser.add_argument(
        "--trials", type=int, default=None,
        help="Override optuna_trials from config (e.g. 5 for a smoke test)"
    )
    parser.add_argument(
        "--no-tune", action="store_true",
        help="Skip Optuna; use CatBoost default hyperparameters"
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="Override model_dir from config"
    )
    args = parser.parse_args()
    result = train(
        targets=args.targets,
        trials=args.trials,
        no_tune=args.no_tune,
        model_dir=args.model_dir,
    )
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
python -m pytest tests/models/test_train.py -v
```

Expected: 9 tests PASSED. Integration tests use `no_tune=True` so they skip Optuna — each runs in a few seconds.

- [ ] **Step 5: Commit**

```bash
git add models/train.py tests/models/test_train.py
git commit -m "feat: models/train.py — CLI + train() with underdog weighting, time split, Optuna"
```

---

## Task 7: Full test suite verification

**Files:** none new — runs all existing tests to confirm no regressions.

- [ ] **Step 1: Run the full test suite**

```powershell
python -m pytest --tb=short -q
```

Expected: all tests pass. The previous full-suite count was 233; this adds ~30 new tests → expected ≥ 263 PASSED, 0 failed.

- [ ] **Step 2: Smoke-test the CLI**

```powershell
python -m models.train --no-tune --targets won --trials 1
```

Expected: logs an error `training matrix is empty` and exits with code 1 (no historical positions yet). This is correct behaviour — the model infrastructure is ready; it populates automatically once `betsp_historical` or timeform results land.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat: models layer complete — CatBoost won/placed_2/showed with underdog weighting"
```
