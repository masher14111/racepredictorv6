"""Tests for models.head_to_head — the model-vs-de-vigged-market GO gate.

The decisive acceptance cases (from the build spec):
  * a model that EQUALS the de-vigged market scores ~equal log-loss and
    therefore does NOT beat the market (``model_beats_market_logloss`` False);
  * a model deliberately sharpened toward the actual winners DOES beat the
    market (True), with a positive log-loss gap.

Plus the invariants that keep the comparison honest: complete-odds-only
restriction, within-race normalisation of the model probs, the odds-band table
shape, and input validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.devig import devig
from models.head_to_head import ODDS_BANDS, head_to_head, odds_band_table


# ── Synthetic race builder ────────────────────────────────────────────────────


def _make_races(n_races: int = 40, seed: int = 0) -> pd.DataFrame:
    """A deterministic set of complete-odds races.

    Each race has 5–10 runners with a realistic favourite–longshot odds ladder;
    the winner is drawn from the de-vigged market probabilities so the market is a
    genuinely informative (but beatable) benchmark. Returns one row per runner
    with ``race_id`` / ``won`` / ``odds``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        field = int(rng.integers(5, 11))
        implied = np.sort(rng.uniform(0.05, 0.5, size=field))[::-1]
        odds = 1.0 / implied
        market = (1.0 / odds) / (1.0 / odds).sum()
        winner = int(rng.choice(field, p=market))
        for i in range(field):
            rows.append(
                {"race_id": f"R{r}", "won": 1 if i == winner else 0, "odds": float(odds[i])}
            )
    return pd.DataFrame(rows)


def _market_model(df: pd.DataFrame, method: str = "proportional") -> pd.Series:
    """Model probs set EQUAL to the de-vigged market (the break-even model)."""
    return pd.Series(devig(df["odds"], df["race_id"], method=method), index=df.index)


def _sharpened_model(df: pd.DataFrame, market: pd.Series, alpha: float) -> pd.Series:
    """Market line blended toward the winner one-hot within each race.

    ``alpha`` of probability mass is moved onto the actual winner, lowering
    −log(p_winner) in every race — a model that genuinely beats the market on
    log-loss (with hindsight, which is exactly what we want to detect here).
    """
    out = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby("race_id"):
        p = market.loc[g.index].to_numpy()
        onehot = g["won"].to_numpy(dtype=float)
        blended = (1.0 - alpha) * p + alpha * onehot
        out.loc[g.index] = blended / blended.sum()
    return out


# ── Acceptance: equal model does not beat the market ──────────────────────────


def test_model_equal_to_market_does_not_beat_it():
    df = _make_races()
    df["model_prob"] = _market_model(df)

    res = head_to_head(df, "model_prob", "odds", "race_id")

    assert res["model_beats_market_logloss"] is False
    # Identical probability vectors → identical log-loss / Brier / ECE.
    assert res["model"]["log_loss"] == pytest.approx(res["market"]["log_loss"], rel=1e-9)
    assert res["model"]["brier_runner_level"] == pytest.approx(
        res["market"]["brier_runner_level"], rel=1e-9
    )
    assert res["gaps"]["log_loss"] == pytest.approx(0.0, abs=1e-9)
    assert res["n_races_kept"] == res["n_races_total"] == 40


# ── Acceptance: a deliberately better model beats the market ──────────────────


def test_sharpened_model_beats_market():
    df = _make_races()
    market = _market_model(df)
    df["model_prob"] = _sharpened_model(df, market, alpha=0.3)

    res = head_to_head(df, "model_prob", "odds", "race_id")

    assert res["model_beats_market_logloss"] is True
    # Lower (better) log-loss for the model → positive market−model gap.
    assert res["model"]["log_loss"] < res["market"]["log_loss"]
    assert res["gaps"]["log_loss"] > 0.0


def test_anti_model_loses_to_market():
    # Mass shifted toward the winner with negative alpha pulls probability AWAY
    # from winners → worse log-loss → must not beat the market.
    df = _make_races()
    market = _market_model(df)
    df["model_prob"] = _sharpened_model(df, market, alpha=-0.15)

    res = head_to_head(df, "model_prob", "odds", "race_id")

    assert res["model_beats_market_logloss"] is False
    assert res["gaps"]["log_loss"] < 0.0


# ── Invariants ────────────────────────────────────────────────────────────────


def test_model_probs_are_normalised_within_race():
    # Scaling the model column by a constant must not change the verdict or the
    # metrics: head_to_head normalises within race before scoring.
    df = _make_races()
    df["model_prob"] = _market_model(df)
    base = head_to_head(df, "model_prob", "odds", "race_id")

    df["model_prob_scaled"] = df["model_prob"] * 7.0
    scaled = head_to_head(df, "model_prob_scaled", "odds", "race_id")

    assert scaled["model"]["log_loss"] == pytest.approx(base["model"]["log_loss"], rel=1e-12)
    assert scaled["model"]["brier_runner_level"] == pytest.approx(
        base["model"]["brier_runner_level"], rel=1e-12
    )


def test_incomplete_odds_races_are_dropped():
    df = _make_races(n_races=10)
    df["model_prob"] = _market_model(df)
    # Knock out one runner's price in race R0 → its whole book is incomplete.
    df.loc[(df["race_id"] == "R0") & (df["won"] == 0), "odds"] = np.nan
    # devig of the surviving races is unaffected; recompute on the kept rows only.

    res = head_to_head(df, "model_prob", "odds", "race_id")

    assert res["n_races_total"] == 10
    assert res["n_races_kept"] == 9  # R0 dropped from BOTH sides


def test_no_complete_races_returns_nan_and_false():
    # One race, but a runner is unpriced → nothing to score.
    df = pd.DataFrame(
        {
            "race_id": ["A", "A", "A"],
            "won": [1, 0, 0],
            "odds": [2.0, np.nan, 5.0],
            "model_prob": [0.5, 0.3, 0.2],
        }
    )
    res = head_to_head(df, "model_prob", "odds", "race_id")

    assert res["n_races_kept"] == 0
    assert res["model_beats_market_logloss"] is False
    assert np.isnan(res["model"]["log_loss"])
    assert np.isnan(res["gaps"]["log_loss"])


def test_devig_method_is_threaded_through():
    df = _make_races(seed=3)
    df["model_prob"] = _market_model(df, method="shin")
    res = head_to_head(df, "model_prob", "odds", "race_id", devig_method="shin")
    assert res["devig_method"] == "shin"
    # Model == shin market under the same method → tie, no beat.
    assert res["model_beats_market_logloss"] is False
    assert res["gaps"]["log_loss"] == pytest.approx(0.0, abs=1e-9)


def test_missing_column_raises():
    df = _make_races(n_races=3)
    df["model_prob"] = _market_model(df)
    with pytest.raises(KeyError, match="nope"):
        head_to_head(df, "nope", "odds", "race_id")


# ── Odds-band table ───────────────────────────────────────────────────────────


def test_odds_band_table_shape_and_aggregates():
    df = _make_races()
    df["model_prob"] = _market_model(df)
    res = head_to_head(df, "model_prob", "odds", "race_id")
    table = res["odds_band_table"]

    # One row per band plus the two aggregates (ALL, 10/1+).
    assert len(table) == len(ODDS_BANDS) + 2
    assert {"band", "n_runners", "actual_rate", "model_mean", "market_mean",
            "model_gap", "market_gap", "model_brier", "market_brier"} <= set(table.columns)

    all_row = table.loc[table["band"] == "ALL"].iloc[0]
    assert all_row["n_runners"] == res["n_runners"]
    assert int(all_row["n_wins"]) == res["n_races_kept"]  # one winner per kept race


def test_odds_band_table_length_mismatch_raises():
    with pytest.raises(ValueError, match="lengths differ"):
        odds_band_table(np.array([1.0, 0.0]), np.array([0.5]), np.array([0.5, 0.5]),
                        np.array([2.0, 3.0]))
