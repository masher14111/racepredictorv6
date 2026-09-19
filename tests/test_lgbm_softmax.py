"""Tests for the LightGBM grouped-softmax win-probability model line.

Covers the acceptance criteria from the porting prompt:
  * probabilities sum to 1.0 within each race;
  * save -> load -> predict reproduces probabilities exactly;
plus the explicit robustness requirements:
  * extra / reordered prediction columns are handled by name (the 43->45 bug);
  * a missing trained feature errors clearly;
  * single-runner races return probability 1.0;
  * a post-result leakage feature is refused at fit time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.lgbm_softmax import LGBMSoftmaxModel, softmax_by_race

FEATURES = ["f_market", "f_form", "f_speed", "f_class"]


def _synthetic_frame(n_races: int = 30, seed: int = 7):
    """Build a tiny labelled frame: one winner per race, mild learnable signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        field = int(rng.integers(4, 11))  # 4-10 runners
        feats = rng.normal(size=(field, len(FEATURES)))
        # A latent utility the model can partially recover -> a real winner.
        utility = feats @ np.array([1.2, 0.8, 0.5, -0.4]) + rng.normal(scale=0.5, size=field)
        winner = int(np.argmax(utility))
        for i in range(field):
            row = {f: feats[i, j] for j, f in enumerate(FEATURES)}
            row["race_id"] = f"R{r:03d}"
            row["won"] = 1 if i == winner else 0
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


def _split_xy(df):
    X = df[FEATURES].copy()
    y = df["won"].to_numpy()
    race_ids = df["race_id"].to_numpy()
    return X, y, race_ids


@pytest.fixture(scope="module")
def trained_model():
    df = _synthetic_frame()
    X, y, race_ids = _split_xy(df)
    model = LGBMSoftmaxModel(n_estimators=60, num_leaves=7, min_data_in_leaf=2)
    model.fit(X, y, race_ids, race_dates=pd.to_datetime("2026-01-01"))
    return model, df


def test_probs_sum_to_one_per_race(trained_model):
    model, df = trained_model
    X, _, race_ids = _split_xy(df)
    proba = model.predict_proba(X, race_ids)

    assert proba.shape == (len(df),)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)

    sums = pd.Series(proba).groupby(df["race_id"].to_numpy()).sum()
    np.testing.assert_allclose(sums.to_numpy(), 1.0, atol=1e-9)


def test_save_load_reproduces_predictions(trained_model, tmp_path):
    model, df = trained_model
    X, _, race_ids = _split_xy(df)
    before = model.predict_proba(X, race_ids)

    path = tmp_path / "lgbm_softmax_test.txt"
    model.save(path)
    assert path.exists()
    assert path.with_suffix(".meta.json").exists()

    reloaded = LGBMSoftmaxModel.load(path)
    after = reloaded.predict_proba(X, race_ids)

    np.testing.assert_allclose(before, after, rtol=0, atol=0)
    assert reloaded.feature_name == FEATURES
    assert reloaded.metadata["n_races"] == df["race_id"].nunique()
    assert reloaded.metadata["n_rows"] == len(df)
    assert reloaded.metadata["env"].endswith("-native")


def test_predict_selects_features_by_name(trained_model):
    """Extra and reordered columns must not break prediction (the 43->45 bug)."""
    model, df = trained_model
    X, _, race_ids = _split_xy(df)
    baseline = model.predict_proba(X, race_ids)

    scrambled = X[FEATURES[::-1]].copy()          # reverse column order
    scrambled["junk_extra"] = 999.0               # unseen extra column
    scrambled["another_extra"] = -1.0
    out = model.predict_proba(scrambled, race_ids)

    np.testing.assert_allclose(out, baseline, atol=1e-12)


def test_missing_feature_raises(trained_model):
    model, df = trained_model
    X, _, race_ids = _split_xy(df)
    with pytest.raises(ValueError, match="missing"):
        model.predict_proba(X.drop(columns=["f_speed"]), race_ids)


def test_single_runner_race():
    raw = np.array([3.14])
    proba = softmax_by_race(raw, np.array(["solo"]))
    assert proba.tolist() == [1.0]


def test_softmax_by_race_preserves_row_order():
    # Interleaved race ids (not consecutive) must still normalise correctly.
    raw = np.array([1.0, 2.0, 1.0, 0.0])
    race_ids = np.array(["A", "B", "A", "B"])
    proba = softmax_by_race(raw, race_ids)
    np.testing.assert_allclose(proba[[0, 2]].sum(), 1.0)
    np.testing.assert_allclose(proba[[1, 3]].sum(), 1.0)
    # Within race A, the higher raw score (index 0 == index 2 here) -> equal split.
    np.testing.assert_allclose(proba[0], proba[2])


def test_leakage_feature_refused():
    df = _synthetic_frame(n_races=5)
    X, y, race_ids = _split_xy(df)
    X = X.assign(odds_finish=2.5)  # post-result column
    model = LGBMSoftmaxModel(n_estimators=10)
    with pytest.raises(ValueError, match="pre-off price"):
        model.fit(X, y, race_ids)
