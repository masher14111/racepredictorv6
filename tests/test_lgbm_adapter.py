"""Tests for the LightGBM feature adapter (features/lgbm_adapter.py).

Acceptance focus (porting prompt, step 6):
  * odds_finish / SP / finishing-position derived columns are ABSENT from X;
  * market features are rebuilt from a PRE-OFF price (morningwap -> ppwap), i.e.
    implied_prob == 1/pre-off price and provably NOT 1/odds_finish;
  * the matrix is fully numeric (LightGBM-ready) and trains end-to-end;
  * train vs inference paths behave per the contract;
  * the audit's empirical leakage guard actually fires on a planted post-off leak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.lgbm_adapter import (
    FINAL_FEATURE_COLS,
    INDEPENDENT_FEATURE_COLS,
    POST_OFF_COLS,
    SELECTED_V4_COLS,
    build_lgbm_matrix,
    feature_provenance,
)

# Columns the LightGBM line must never see (the prompt's explicit denylist).
_FORBIDDEN = ["odds_finish", "sp", "odds_drift", "ew_value_index", "position"]


def _frame(n_races: int = 25, seed: int = 3) -> pd.DataFrame:
    """Synthetic v4-derived matrix: real pre-off prices, a finishing SP that
    DIFFERS from the morning price, one winner per race, plus the v4 feature
    columns and a couple of sneaky post-off columns that must be dropped."""
    rng = np.random.default_rng(seed)
    bands = ["unknown", "first_time", "long_absence", "layoff", "normal", "fresh"]
    rows = []
    for r in range(n_races):
        field = int(rng.integers(4, 11))
        mw = rng.uniform(2.0, 30.0, size=field)
        winner = int(rng.integers(field))
        for i in range(field):
            row = {
                "race_uid": f"VEN|2026-01-01 13:{r:02d}",
                "horse_id": f"h{r}_{i}",
                "morningwap": float(mw[i]),
                # ppwap drifts a little off the morning line (still pre-off)
                "ppwap": float(mw[i] * rng.uniform(0.92, 1.08)),
                # finishing SP moves further and is materially different
                "odds_finish": float(mw[i] * rng.uniform(0.6, 1.5)),
                "sp": np.nan,  # nulled for the betSP backbone (see normalizer C2)
                "position": (1 if i == winner else i + 1),
                "won": (1 if i == winner else 0),
                # sneaky post-off columns that MUST be excluded:
                "odds_drift": float(rng.normal()),
                "ew_value_index": float(rng.normal()),
                # the one categorical feature, stored as a STRING label:
                "freshness_band": str(rng.choice(bands)),
                "is_steaming": bool(rng.integers(0, 2)),
                "is_drifting": bool(rng.integers(0, 2)),
            }
            for c in SELECTED_V4_COLS:
                row.setdefault(c, float(rng.normal()))
            rows.append(row)
    return pd.DataFrame(rows)


# ── leakage / provenance ──────────────────────────────────────────────────────


def test_forbidden_columns_absent_from_matrix():
    df = _frame()
    X, _, _ = build_lgbm_matrix(df, inference=False)
    for col in _FORBIDDEN:
        assert col in df.columns, f"fixture should carry {col} to prove it's dropped"
        assert col not in X.columns, f"post-off column {col} leaked into the matrix"
    assert set(X.columns).isdisjoint(POST_OFF_COLS)


def test_final_cols_disjoint_from_postoff_denylist():
    assert set(FINAL_FEATURE_COLS).isdisjoint(POST_OFF_COLS)


def test_implied_prob_is_preoff_not_finishing_price():
    df = _frame()
    X, _, _ = build_lgbm_matrix(df, inference=False)
    mw = pd.to_numeric(df["morningwap"], errors="coerce").where(lambda s: s > 1.0)
    pp = pd.to_numeric(df["ppwap"], errors="coerce").where(lambda s: s > 1.0)
    price = mw.fillna(pp).reset_index(drop=True)
    fin = pd.to_numeric(df["odds_finish"], errors="coerce").reset_index(drop=True)

    # exactly the pre-off implied probability
    np.testing.assert_allclose(X["implied_prob"].to_numpy(), (1.0 / price).to_numpy(),
                               rtol=0, atol=1e-12)
    # and clearly NOT the finishing-price implied probability
    assert float((X["implied_prob"] - 1.0 / fin).abs().max()) > 0.05


def test_market_block_rebuilt_per_race():
    df = _frame()
    X, _, race_ids = build_lgbm_matrix(df, inference=False)
    rid = pd.Series(race_ids)
    # within each race the overround-normalised probs sum to 1
    sums = X["overround_norm_prob"].groupby(rid.to_numpy()).sum()
    np.testing.assert_allclose(sums.to_numpy(), 1.0, atol=1e-9)
    # market_rank 1 == shortest price == highest implied prob, per race
    for _, idx in pd.Series(range(len(X))).groupby(rid.to_numpy()):
        block = X.iloc[idx.to_numpy()]
        fav = block["implied_prob"].idxmax()
        assert block.loc[fav, "market_rank"] == 1.0


# ── matrix shape / dtype / contract ────────────────────────────────────────────


def test_matrix_is_fully_numeric_and_ordered():
    df = _frame()
    X, _, _ = build_lgbm_matrix(df, inference=False)
    assert list(X.columns) == FINAL_FEATURE_COLS
    assert (X.dtypes == float).all(), "LightGBM needs an all-numeric matrix"


def test_freshness_band_string_encoded_to_ordinal():
    df = _frame()
    X, _, _ = build_lgbm_matrix(df, inference=False)
    vals = set(X["freshness_band"].dropna().unique())
    assert vals.issubset({0.0, 1.0, 2.0, 3.0, 4.0, 5.0})


def test_train_path_returns_labels_and_aligned_race_ids():
    df = _frame()
    X, y, race_ids = build_lgbm_matrix(df, inference=False)
    assert len(X) == len(y) == len(race_ids) == len(df)
    assert set(np.unique(y)).issubset({0.0, 1.0})
    # exactly one winner per race
    winners = pd.Series(y).groupby(pd.Series(race_ids).to_numpy()).sum()
    assert (winners == 1).all()


def test_inference_path_has_no_labels_and_keeps_all_rows():
    df = _frame().drop(columns=["won"])  # live racecards carry no result
    X, y, race_ids = build_lgbm_matrix(df, inference=True)
    assert y is None
    assert len(X) == len(df) == len(race_ids)


def test_train_path_drops_unlabelled_rows():
    df = _frame()
    df.loc[df.index[:5], "won"] = np.nan  # 5 rows with no known result
    X, y, race_ids = build_lgbm_matrix(df, inference=False)
    assert len(X) == len(y) == len(race_ids) == len(df) - 5


def test_train_path_requires_won_column():
    df = _frame().drop(columns=["won"])
    with pytest.raises(KeyError, match="won"):
        build_lgbm_matrix(df, inference=False)


def test_missing_v4_feature_filled_nan():
    df = _frame().drop(columns=["course_win_rate"])
    X, _, _ = build_lgbm_matrix(df, inference=False)
    assert "course_win_rate" in X.columns
    assert X["course_win_rate"].isna().all()


def test_empty_frame_returns_empty_matrix():
    X, y, race_ids = build_lgbm_matrix(pd.DataFrame(), inference=False)
    assert list(X.columns) == FINAL_FEATURE_COLS
    assert len(X) == 0 and len(y) == 0 and len(race_ids) == 0


# ── provenance metadata ────────────────────────────────────────────────────────


def test_independent_feature_cols_excludes_price_and_market_derived():
    from models.features import MARKET_DERIVED_FEATURE_COLS, PRICE_FEATURE_COLS

    assert set(INDEPENDENT_FEATURE_COLS).isdisjoint(PRICE_FEATURE_COLS)
    assert set(INDEPENDENT_FEATURE_COLS).isdisjoint(MARKET_DERIVED_FEATURE_COLS)
    assert set(INDEPENDENT_FEATURE_COLS).isdisjoint(POST_OFF_COLS)
    assert "historical_win_rate" in INDEPENDENT_FEATURE_COLS  # a genuine price-free signal survives


def test_build_lgbm_matrix_honours_independent_feature_cols():
    df = _frame()
    X, _, _ = build_lgbm_matrix(df, inference=False, feature_cols=INDEPENDENT_FEATURE_COLS)
    assert list(X.columns) == INDEPENDENT_FEATURE_COLS
    assert "implied_prob" not in X.columns
    assert (X.dtypes == float).all()


def test_build_lgbm_matrix_rejects_blocked_feature_cols():
    with pytest.raises(ValueError, match="odds_finish"):
        build_lgbm_matrix(_frame(), inference=False, feature_cols=["odds_finish"])


def test_provenance_lists_excluded_with_reasons():
    prov = feature_provenance()
    assert set(prov["excluded"]) == {"odds_drift", "ew_value_index"}
    assert "sp" in prov["excluded"]["odds_drift"]
    assert prov["preoff_price_priority"] == ["morningwap", "ppwap"]
    assert "odds_finish" in prov["post_off_denylist"]


# ── integration with the model line + audit guard ──────────────────────────────


def test_matrix_trains_in_lgbm_softmax_model():
    lgb = pytest.importorskip("lightgbm")  # noqa: F841
    from models.lgbm_softmax import LGBMSoftmaxModel

    df = _frame(n_races=60)
    X, y, race_ids = build_lgbm_matrix(df, inference=False)
    model = LGBMSoftmaxModel(n_estimators=40, num_leaves=7, min_data_in_leaf=2)
    model.fit(X, y, race_ids)  # would raise if any _LEAKAGE_FEATURES were present
    proba = model.predict_proba(X, race_ids)
    sums = pd.Series(proba).groupby(pd.Series(race_ids).to_numpy()).sum()
    np.testing.assert_allclose(sums.to_numpy(), 1.0, atol=1e-9)


def test_audit_guard_flags_a_planted_postoff_leak():
    """The audit's closing-move partial correlation must catch a feature built
    from odds_finish (here: its implied probability)."""
    from tools.audit_lgbm_features import (
        MOVE_PCORR_THRESH,
        _partial_corr_with_move,
    )

    df = _frame(n_races=400, seed=11)
    leak = 1.0 / pd.to_numeric(df["odds_finish"], errors="coerce")
    pc, n = _partial_corr_with_move(leak, df)
    assert pc is not None and n > 0
    assert abs(pc) > MOVE_PCORR_THRESH, (
        f"planted odds_finish leak slipped the guard (|pcorr|={abs(pc):.3f})"
    )

    # a genuine pre-off feature (the morning implied prob) must NOT trip it
    clean = 1.0 / pd.to_numeric(df["morningwap"], errors="coerce").where(lambda s: s > 1)
    pc_clean, _ = _partial_corr_with_move(clean, df)
    assert abs(pc_clean) <= MOVE_PCORR_THRESH
