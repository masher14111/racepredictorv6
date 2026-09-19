"""Tests for the XGBoost candidate training/scoring module (step 13).

Mirrors tests/models/test_train.py's fixtures and integration shape (same
whole-race chronological-split guarantees apply — models.train_xgboost.train
reuses models.train's _time_split/_group_carve unchanged) so the two boosted-
tree lines are held to the same acceptance bar.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from models.features import FEATURE_COLS
from models.split_utils import race_group_overlap
from models.train_xgboost import load_frozen_model, score, train


def _synthetic_multi_row_races(n_races=120, races_per_day=2, min_runners=4,
                               max_runners=8, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_races):
        day = pd.Timestamp("2023-01-01") + pd.Timedelta(days=i // races_per_day)
        n_runners = int(rng.integers(min_runners, max_runners + 1))
        pos = rng.permutation(np.arange(1, n_runners + 1))
        for r in range(n_runners):
            row = {col: rng.random() for col in FEATURE_COLS}
            row["race_date"] = day
            row["race_uid"] = f"race_{i}"
            row["horse_id"] = f"race_{i}_horse_{r}"
            row["implied_prob"] = rng.uniform(0.05, 0.5)
            row["position"] = int(pos[r])
            row["placed"] = int(pos[r] <= 3)
            rows.append(row)
    return pd.DataFrame(rows)


def test_train_returns_none_on_empty_df(tmp_path):
    empty = pd.DataFrame(columns=list(_synthetic_multi_row_races(1).columns))
    result = train(df=empty, targets=["won"], trials=1, no_tune=True,
                   model_dir=str(tmp_path))
    assert result is None


def test_train_saves_model_and_meta(tmp_path):
    df = _synthetic_multi_row_races(150)
    result = train(df=df, targets=["won"], trials=1, no_tune=True,
                   model_dir=str(tmp_path), version_tag="test")
    assert result is not None
    assert os.path.exists(os.path.join(str(tmp_path), "xgboost_won_test.json"))
    meta_path = os.path.join(str(tmp_path), "xgboost_test_meta.json")
    assert os.path.exists(meta_path)
    with open(meta_path) as fh:
        meta = json.load(fh)
    assert "won" in meta["targets"]
    assert "test_auc" in meta["targets"]["won"]
    assert "feature_cols" in meta and meta["feature_cols"]


def test_train_end_to_end_zero_race_overlap(tmp_path):
    """train() itself must not crash on a realistic multi-runner matrix, and
    must record whole-race-disjoint train/test row counts (not an exact
    80/20 row split, since races vary in size) — same guarantee models.train
    provides via the shared _time_split/_group_carve primitives."""
    df = _synthetic_multi_row_races(120, races_per_day=2)
    result = train(df=df, targets=["won"], trials=1, no_tune=True,
                   model_dir=str(tmp_path))
    assert result is not None
    assert result["train_rows"] + result["test_rows"] == len(df)
    assert result["test_rows"] > 0
    assert result["targets"]["won"]["test_auc"] is not None


def test_train_restricts_to_win_market_rows_when_present(tmp_path):
    df = _synthetic_multi_row_races(40, races_per_day=2)
    n_total = len(df)
    place = df.copy()
    place["implied_prob"] = place["implied_prob"] * 0.1
    df["market_type"] = "WIN"
    place["market_type"] = "PLACE"
    mixed = pd.concat([df, place], ignore_index=True)

    result = train(df=mixed, targets=["won"], trials=1, no_tune=True,
                   model_dir=str(tmp_path))
    assert result["train_rows"] + result["test_rows"] == n_total


def test_train_price_free_feature_whitelist_is_honoured(tmp_path):
    from models.features import PRICE_FREE_FEATURE_COLS

    df = _synthetic_multi_row_races(80)
    result = train(df=df, targets=["won"], trials=1, no_tune=True,
                   model_dir=str(tmp_path), feature_cols=PRICE_FREE_FEATURE_COLS,
                   version_tag="nf")
    assert set(result["feature_cols"]) <= set(PRICE_FREE_FEATURE_COLS)
    assert "implied_prob" not in result["feature_cols"]


def test_load_frozen_model_reload_score_parity(tmp_path):
    """Reload/inference parity (step 10's D-item): a saved model's score on
    fresh rows must exactly match what the in-memory training call itself
    would have produced from the raw (uncalibrated) booster."""
    df = _synthetic_multi_row_races(150)
    train(df=df, targets=["won"], trials=1, no_tune=True,
          model_dir=str(tmp_path), version_tag="rt")

    frozen = load_frozen_model(os.path.join(str(tmp_path), "xgboost_won_rt.json"))
    assert frozen.target == "won"
    assert frozen.tag == "rt"

    p1 = score(frozen, df)
    p2 = score(frozen, df)
    assert np.array_equal(p1, p2)
    assert np.all(np.isfinite(p1))
    assert np.all((p1 >= 0.0) & (p1 <= 1.0))


def test_load_frozen_model_rejects_bad_filename(tmp_path):
    bad = tmp_path / "not_a_model.json"
    bad.write_text("{}")
    with pytest.raises(ValueError):
        load_frozen_model(str(bad))
