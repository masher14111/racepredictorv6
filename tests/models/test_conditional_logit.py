"""Tests for the simple race conditional-logit baseline (step 10)."""
import json

import numpy as np
import pandas as pd
import pytest

from models.conditional_logit import ConditionalLogitModel, cv_race_log_loss, tune_l2


def _synthetic_races(n_races=200, field_size=6, seed=0):
    """One informative feature (higher -> more likely to win) + pure noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        n = field_size
        signal = rng.normal(size=n)
        noise = rng.normal(size=n)
        util = 1.5 * signal + rng.normal(scale=0.1, size=n)
        p = np.exp(util - util.max())
        p = p / p.sum()
        winner = rng.choice(n, p=p)
        for i in range(n):
            rows.append({
                "race_uid": f"race_{r}",
                "race_date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=r),
                "signal": signal[i],
                "noise": noise[i],
                "won": int(i == winner),
            })
    return pd.DataFrame(rows)


def test_predict_proba_sums_to_one_per_race():
    df = _synthetic_races(n_races=30)
    X = df[["signal", "noise"]]
    model = ConditionalLogitModel(l2=0.1).fit(X, df["won"], df["race_uid"])
    p = model.predict_proba(X, df["race_uid"].to_numpy())
    totals = pd.Series(p).groupby(df["race_uid"].to_numpy()).sum()
    assert np.allclose(totals.to_numpy(), 1.0, atol=1e-8)


def test_learns_the_informative_feature_direction():
    df = _synthetic_races(n_races=300, seed=1)
    X = df[["signal", "noise"]]
    model = ConditionalLogitModel(l2=0.1).fit(X, df["won"], df["race_uid"])
    idx = model.feature_name.index("signal")
    noise_idx = model.feature_name.index("noise")
    assert model.coef_[idx] > 0
    assert abs(model.coef_[idx]) > abs(model.coef_[noise_idx])


def test_beats_uniform_baseline_on_held_out_races():
    train = _synthetic_races(n_races=250, seed=2)
    test = _synthetic_races(n_races=60, seed=3)
    X_tr, X_te = train[["signal", "noise"]], test[["signal", "noise"]]
    model = ConditionalLogitModel(l2=0.1).fit(X_tr, train["won"], train["race_uid"])
    p = model.predict_proba(X_te, test["race_uid"].to_numpy())

    def race_log_loss(probs, y, rid):
        per_race = []
        for r in pd.unique(rid):
            m = rid == r
            w = probs[m][y[m].astype(bool)]
            if w.size:
                per_race.append(-np.log(np.clip(w, 1e-12, 1.0)).sum())
        return float(np.mean(per_race))

    model_ll = race_log_loss(p, test["won"].to_numpy(), test["race_uid"].to_numpy())
    uniform_ll = float(np.log(6))  # field_size=6, uniform 1/6 per runner
    assert model_ll < uniform_ll


def test_single_runner_race_gets_probability_one():
    df = pd.DataFrame({
        "race_uid": ["a", "a", "b"],
        "x": [0.1, -0.2, 0.5],
        "won": [1, 0, 1],
    })
    model = ConditionalLogitModel(l2=1.0).fit(df[["x"]], df["won"], df["race_uid"])
    p = model.predict_proba(df[["x"]], df["race_uid"].to_numpy())
    assert p[2] == pytest.approx(1.0)


def test_save_load_round_trip(tmp_path):
    df = _synthetic_races(n_races=40, seed=4)
    X = df[["signal", "noise"]]
    model = ConditionalLogitModel(l2=0.5).fit(X, df["won"], df["race_uid"])
    path = tmp_path / "condlogit.json"
    model.save(path)
    assert json.loads(path.read_text())["l2"] == 0.5

    loaded = ConditionalLogitModel.load(path)
    p1 = model.predict_proba(X, df["race_uid"].to_numpy())
    p2 = loaded.predict_proba(X, df["race_uid"].to_numpy())
    assert np.allclose(p1, p2)


def test_standardization_uses_train_stats_only_not_scoring_batch():
    """A scoring batch with a wildly different mean/scale must not change the
    model's predictions — regression guard against re-fitting stats at score
    time (which would make a race's probability depend on its scoring batch)."""
    train = _synthetic_races(n_races=200, seed=5)
    model = ConditionalLogitModel(l2=0.1).fit(
        train[["signal", "noise"]], train["won"], train["race_uid"])

    small_batch = _synthetic_races(n_races=2, seed=6)
    shifted = small_batch.copy()
    shifted["signal"] = shifted["signal"] * 100 + 500  # different scale/location

    raw_small = model.predict_raw(small_batch[["signal", "noise"]])
    # Re-deriving raw scores manually from the model's OWN frozen mean/std
    # must reproduce predict_raw exactly, regardless of the batch's own stats.
    Xs = (small_batch[["signal", "noise"]].to_numpy() - model.mean_) / model.std_
    expected = Xs @ model.coef_
    assert np.allclose(raw_small, expected)


def test_sample_weight_none_matches_uniform_weight_one():
    df = _synthetic_races(n_races=80, seed=8)
    X = df[["signal", "noise"]]
    m1 = ConditionalLogitModel(l2=0.3).fit(X, df["won"], df["race_uid"])
    m2 = ConditionalLogitModel(l2=0.3).fit(
        X, df["won"], df["race_uid"], sample_weight=np.ones(len(df)))
    assert np.allclose(m1.coef_, m2.coef_)


def test_sample_weight_changes_the_fit():
    df = _synthetic_races(n_races=150, seed=9)
    X = df[["signal", "noise"]]
    uniform = ConditionalLogitModel(l2=0.1).fit(X, df["won"], df["race_uid"])
    # Zero out half the races' influence entirely -> a materially different fit.
    races = df["race_uid"].unique()
    zeroed = set(races[: len(races) // 2])
    w = df["race_uid"].isin(zeroed).astype(float).to_numpy()
    weighted = ConditionalLogitModel(l2=0.1).fit(
        X, df["won"], df["race_uid"], sample_weight=w)
    assert not np.allclose(uniform.coef_, weighted.coef_)
    assert weighted.metadata["weighted"] is True
    assert uniform.metadata["weighted"] is False


def test_cv_race_log_loss_and_tune_l2_are_bounded_and_finite():
    df = _synthetic_races(n_races=120, seed=7)
    X = df[["signal", "noise"]]
    ll = cv_race_log_loss(X, df["won"], df["race_uid"].to_numpy(),
                          df["race_date"].to_numpy(), l2=0.5, n_splits=3)
    assert np.isfinite(ll)

    result = tune_l2(X, df["won"], df["race_uid"].to_numpy(),
                     df["race_date"].to_numpy(), n_trials=5, n_splits=3)
    assert "l2" in result and "cv_log_loss" in result
    assert np.isfinite(result["cv_log_loss"])
    assert result["l2"] > 0
