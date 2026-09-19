import json
import os

import numpy as np
import pandas as pd
import pytest

from models.features import FEATURE_COLS
from models.split_utils import race_group_overlap
from models.train import _group_carve, _sample_weights, _time_split, train


# ── helpers ─────────────────────────────────────────────────────────────────

def _synthetic_df(n=300, seed=42):
    """Labelled df with all FEATURE_COLS present and realistic dtypes."""
    rng = np.random.default_rng(seed)
    data = {col: rng.random(n) for col in FEATURE_COLS}
    data["race_date"] = pd.date_range("2023-01-01", periods=n, freq="D")
    data["race_uid"] = [f"race_{i}" for i in range(n)]
    data["implied_prob"] = rng.uniform(0.05, 0.5, n)
    data["position"] = pd.array(rng.integers(1, 10, n), dtype="Int64")
    data["placed"] = pd.array((rng.integers(1, 10, n) <= 3).astype(int), dtype="Int64")
    return pd.DataFrame(data)


def _synthetic_multi_row_races(n_races=200, races_per_day=3, min_runners=4,
                                max_runners=10, seed=7):
    """Labelled df with several races per day and several rows (runners) per
    race — the exact shape (multiple races sharing a race_date) that exposed
    the pre-step-02 row-count split defect (stages/01.md: shared race_uid
    across train/test, fit/calibration and fit/early-stop)."""
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
            row["implied_prob"] = rng.uniform(0.05, 0.5)
            row["position"] = int(pos[r])
            row["placed"] = int(pos[r] <= 3)
            rows.append(row)
    return pd.DataFrame(rows)


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
    train(df=df, targets=["won"], trials=1,
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


# ── whole-race chronological split integrity (step 02) ─────────────────────
#
# Stage 01 (memory/improvement/stages/01.md) reproduced live, on the real
# training matrix: 28 shared race_uid across train/test, 13 across
# fit-core/calibration, 45 across fit/early-stop, because the pre-step-02
# `_time_split` cut by absolute row count on date-sorted rows while several
# races shared a race_date. These tests reproduce that exact shape (several
# races per day, several rows per race) and assert the overlap is now zero.

def test_time_split_zero_overlap_with_multiple_races_per_day():
    df = _synthetic_multi_row_races(n_races=150, races_per_day=4)
    train_df, test_df = _time_split(df, 0.2)
    overlap = race_group_overlap(
        (np.arange(len(train_df)), train_df["race_uid"].to_numpy()),
        (np.arange(len(test_df)), test_df["race_uid"].to_numpy()),
    )
    assert overlap == 0
    assert train_df["race_date"].max() <= test_df["race_date"].min()


def test_calibration_and_early_stop_carve_zero_overlap():
    """Reproduces train()'s full boundary chain for one target: outer
    train/test, then core/calibration, then fit/early-stop — every pair must
    share zero race_uid, matching the acceptance gate in
    docs/improvement/prompts/02-whole-race-splits.md."""
    df = _synthetic_multi_row_races(n_races=200, races_per_day=3)
    train_df, test_df = _time_split(df, 0.2)
    core_df, calib_df = _group_carve(train_df, 0.15)
    fit_df, es_df = _group_carve(core_df, 0.10)

    partitions = {
        "test": (np.arange(len(test_df)), test_df["race_uid"].to_numpy()),
        "calib": (np.arange(len(calib_df)), calib_df["race_uid"].to_numpy()),
        "fit": (np.arange(len(fit_df)), fit_df["race_uid"].to_numpy()),
        "es": (np.arange(len(es_df)), es_df["race_uid"].to_numpy()),
    }
    names = list(partitions)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert race_group_overlap(partitions[names[i]], partitions[names[j]]) == 0, (
                f"shared race_uid between {names[i]} and {names[j]}")

    # strictly chronological: fit <= early-stop <= calibration <= test
    assert fit_df["race_date"].max() <= es_df["race_date"].min()
    assert es_df["race_date"].max() <= calib_df["race_date"].min()
    assert calib_df["race_date"].max() <= test_df["race_date"].min()


def test_train_end_to_end_multi_row_races_zero_overlap(tmp_path):
    """train() itself, on a realistic multi-runner-per-race matrix, must not
    crash and must record disjoint train/test row counts consistent with
    whole-race splitting (not an exact 80/20 row split, since races vary in
    size)."""
    df = _synthetic_multi_row_races(n_races=120, races_per_day=2)
    result = train(df=df, targets=["won"], trials=1,
                   no_tune=True, model_dir=str(tmp_path))
    assert result is not None
    assert result["train_rows"] + result["test_rows"] == len(df)
    assert result["test_rows"] > 0
    assert result["targets"]["won"]["test_auc"] is not None


# ── WIN-market population parity with models.train_lgbm (step 03) ──────────
#
# features/fuse.py now emits one row per runner PER MARKET (a runner racing
# under both books gets a WIN row and a PLACE row — DESIGN.md: "WIN and PLACE
# are different price books"). Live inference only ever serves WIN rows, and
# models.train_lgbm's _load_win_market already restricts to market_type=="WIN"
# — CatBoost must train/test on that same population, not an arbitrary mix of
# both books' price scales.

def test_train_restricts_to_win_market_rows_when_present(tmp_path):
    df = _synthetic_multi_row_races(n_races=40, races_per_day=2)
    n_total = len(df)
    # Duplicate every row as a PLACE-market sibling with a different price
    # scale — the shape that would previously have been silently mixed in.
    place = df.copy()
    place["implied_prob"] = place["implied_prob"] * 0.1
    df["market_type"] = "WIN"
    place["market_type"] = "PLACE"
    mixed = pd.concat([df, place], ignore_index=True)

    result = train(df=mixed, targets=["won"], trials=1,
                   no_tune=True, model_dir=str(tmp_path))
    assert result["train_rows"] + result["test_rows"] == n_total


def test_train_market_filter_is_noop_without_market_type_column(tmp_path):
    """Older/synthetic frames that never carried market rows must train
    exactly as before (no market_type column -> no filtering)."""
    df = _synthetic_df(300)
    assert "market_type" not in df.columns
    result = train(df=df, targets=["won"], trials=1,
                   no_tune=True, model_dir=str(tmp_path))
    assert result["train_rows"] + result["test_rows"] == len(df)
