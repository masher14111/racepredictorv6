import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from models.split_utils import race_group_overlap
from models.tuner import _cv_splits, _metric_meta, _space, run_study

_TINY_CFG = {
    "cv_folds": 2,
    "optuna_trials": 2,
    "iterations": 10,
    "early_stopping_rounds": 5,
}


def _data(seed=0, n=120, n_features=5, pos_rate=0.3):
    rng = np.random.default_rng(seed)
    X = rng.random((n, n_features)).astype(float)
    y = (X[:, 0] > (1 - pos_rate)).astype(int)
    w = np.ones(n)
    return X, y, w


def _race_data(seed=0, n_races=40, races_per_day=2, min_size=2, max_size=6,
               n_features=5, pos_rate=0.3):
    """Like _data, but with race_uid/race_date so grouped CV can be exercised —
    several races per day, several rows per race (the shape that exposed the
    row-indexed TimeSeriesSplit's group-splitting risk, step 02 scope item 3)."""
    rng = np.random.default_rng(seed)
    race_uid, race_date, rows = [], [], []
    for i in range(n_races):
        day = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=i // races_per_day)
        size = int(rng.integers(min_size, max_size + 1))
        for _ in range(size):
            race_uid.append(f"race_{i}")
            race_date.append(day)
            rows.append(rng.random(n_features))
    X = np.asarray(rows, dtype=float)
    y = (X[:, 0] > (1 - pos_rate)).astype(int)
    w = np.ones(len(X))
    return X, y, w, np.array(race_uid), np.array(race_date)


def test_run_study_returns_dict():
    X, y, w = _data()
    params = run_study(X, y, w, _TINY_CFG)
    assert isinstance(params, dict)


def test_run_study_has_learning_rate():
    X, y, w = _data()
    params = run_study(X, y, w, _TINY_CFG)
    assert "learning_rate" in params
    assert 5e-4 <= params["learning_rate"] <= 0.3


def test_run_study_has_depth():
    X, y, w = _data()
    params = run_study(X, y, w, _TINY_CFG)
    assert "depth" in params
    assert 4 <= params["depth"] <= 10


def test_run_study_with_sample_weight():
    """Underdog weighting (non-uniform weights) should not crash the study."""
    X, y, w = _data()
    w = np.linspace(1.0, 20.0, len(y))
    params = run_study(X, y, w, _TINY_CFG)
    assert params is not None


def test_no_class_weight_mode_leaks_into_params():
    """The synthetic selector must never reach the final fit; only resolved
    CatBoost keys (auto_class_weights or scale_pos_weight) may appear."""
    X, y, w = _data()
    params = run_study(X, y, w, _TINY_CFG)
    assert "class_weight_mode" not in params
    # at most one of the two real keys is set, never both (CatBoost rejects both)
    assert not ("auto_class_weights" in params and "scale_pos_weight" in params)


def test_config_driven_search_space_narrows_range():
    """cfg['tuning'] should override the default bounds."""
    X, y, w = _data()
    cfg = {**_TINY_CFG, "tuning": {"depth": [4, 4], "metric": "logloss"}}
    params = run_study(X, y, w, cfg)
    assert params["depth"] == 4


def test_metric_meta_directions():
    assert _metric_meta("auc") == ("maximize", "AUC", False)
    assert _metric_meta("logloss")[0] == "minimize"
    assert _metric_meta("brier")[0] == "minimize"


def test_space_merges_defaults():
    """A partial cfg['tuning'] keeps defaults for unspecified keys."""
    space = _space({"tuning": {"metric": "brier"}})
    assert space["metric"] == "brier"
    assert space["depth"] == [4, 10]  # default preserved


def test_timeseries_split_folds_are_chronological():
    """Each TimeSeriesSplit fold trains only on rows before its validation slice."""
    X, _, _ = _data(n=100)
    kf = TimeSeriesSplit(n_splits=3)
    for train_idx, val_idx in kf.split(X):
        assert train_idx.max() < val_idx.min()


# ── grouped CV (step 02: whole-race-group-safe tuning) ──────────────────────

def test_run_study_with_race_ids_does_not_crash():
    X, y, w, race_uid, race_date = _race_data(n_races=60, races_per_day=3)
    cfg = {**_TINY_CFG, "cv_folds": 3}
    params = run_study(X, y, w, cfg, race_ids=race_uid, order_keys=race_date)
    assert isinstance(params, dict) and "learning_rate" in params


def test_cv_splits_grouped_are_race_safe():
    X, y, w, race_uid, race_date = _race_data(n_races=60, races_per_day=3, seed=2)
    splits = _cv_splits(X, {"cv_folds": 4}, race_uid, race_date)
    assert len(splits) == 4
    for train_idx, val_idx in splits:
        assert race_group_overlap((train_idx, race_uid), (val_idx, race_uid)) == 0
        assert race_date[train_idx].max() <= race_date[val_idx].min()


def test_cv_splits_falls_back_when_too_few_races():
    """Fewer unique races than cv_folds+1: grouped split can't form the
    requested folds, so it falls back to row-indexed TimeSeriesSplit rather
    than raising out of run_study."""
    X, y, w, race_uid, race_date = _race_data(n_races=3, races_per_day=1, min_size=20,
                                               max_size=20)
    splits = _cv_splits(X, {"cv_folds": 5}, race_uid, race_date)
    assert len(splits) == 5  # TimeSeriesSplit(n_splits=5) always yields 5 folds


def test_cv_splits_without_race_ids_uses_row_fallback():
    X, _, _ = _data(n=100)
    splits = _cv_splits(X, {"cv_folds": 3}, None, None)
    assert len(splits) == 3
    for train_idx, val_idx in splits:
        assert train_idx.max() < val_idx.min()
