"""Regression tests for models.blend (step 12: market combination).

These pin the three properties the stage-12 acceptance turns on:

1. an executable bookmaker quote can never move the blended probability or the
   reference probability (the blend has no price parameter at all);
2. an incomplete reference book NaNs the whole race, never a partial blend;
3. a non-runner leaving the field renormalises the survivors without changing
   their relative standing.
"""
import json

import numpy as np
import pandas as pd
import pytest

from models.blend import (
    ALPHA_GRID,
    BETA_GRID,
    PowerBlend,
    power_blend,
    race_level_log_loss,
    select_power_blend,
)
from models.calibration import normalize_within_race
from models.devig import devig
from models.head_to_head import _race_level_log_loss


def _frame(seed=0, n_races=40, size=8):
    """Synthetic panel with a reference book, an executable book and outcomes."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        strength = rng.gamma(2.0, 1.0, size=size)
        true_p = strength / strength.sum()
        winner = rng.choice(size, p=true_p)
        # Reference book: fair prices with a 1.15 overround.
        ref_odds = 1.0 / (true_p * 1.15)
        # Executable book: an independent, LONGER best-of-N overlay. Nothing
        # downstream of the reference line may react to it.
        exe_odds = ref_odds * rng.uniform(1.02, 1.20, size=size)
        model_p = np.clip(true_p + rng.normal(0, 0.03, size=size), 1e-3, None)
        for i in range(size):
            rows.append({
                "race_uid": f"race_{r}",
                "won": int(i == winner),
                "ref_odds": ref_odds[i],
                "exe_odds": exe_odds[i],
                "p_model": model_p[i],
            })
    df = pd.DataFrame(rows)
    df["p_ref"] = devig(df["ref_odds"], df["race_uid"], method="proportional")
    return df


# ── normalisation / boundary controls ────────────────────────────────────────


def test_blend_sums_to_one_within_each_race():
    df = _frame()
    p = power_blend(df["p_model"], df["p_ref"], df["race_uid"], 0.4, 0.9)
    totals = pd.Series(p).groupby(df["race_uid"].to_numpy()).sum()
    assert np.allclose(totals.to_numpy(), 1.0)


def test_alpha_one_beta_zero_reproduces_the_normalised_model_line():
    df = _frame(seed=1)
    p = power_blend(df["p_model"], df["p_ref"], df["race_uid"], 1.0, 0.0)
    expected = normalize_within_race(df["p_model"].to_numpy(), df["race_uid"].to_numpy())
    assert np.allclose(p, expected)


def test_alpha_zero_beta_one_reproduces_the_reference_market_line():
    df = _frame(seed=2)
    p = power_blend(df["p_model"], df["p_ref"], df["race_uid"], 0.0, 1.0)
    assert np.allclose(p, df["p_ref"].to_numpy())


def test_single_runner_race_gets_probability_one():
    p = power_blend([0.3], [0.9], ["solo"], 0.5, 0.5)
    assert p == pytest.approx(1.0)


def test_scale_of_exponents_only_changes_sharpness_not_a_tie():
    # Equal model and reference probabilities → uniform blend for any exponents.
    rid = ["r"] * 4
    p = power_blend([0.25] * 4, [0.25] * 4, rid, 0.7, 1.3)
    assert np.allclose(p, 0.25)


# ── executable-quote invariance (stage-12 acceptance) ────────────────────────


def test_executable_quote_cannot_change_the_blend_or_the_reference_line():
    df = _frame(seed=3)
    base_ref = df["p_ref"].to_numpy().copy()
    base_blend = power_blend(df["p_model"], df["p_ref"], df["race_uid"], 0.5, 1.0)

    moved = df.copy()
    moved["exe_odds"] = moved["exe_odds"] * 3.0          # a violent quote move
    moved.loc[moved.index[:20], "exe_odds"] = 1.01        # and a collapse
    # Reference line is rebuilt from the REFERENCE book only.
    moved["p_ref"] = devig(moved["ref_odds"], moved["race_uid"], method="proportional")
    after_blend = power_blend(moved["p_model"], moved["p_ref"], moved["race_uid"], 0.5, 1.0)

    assert np.array_equal(base_ref, moved["p_ref"].to_numpy())
    assert np.array_equal(base_blend, after_blend)


def test_power_blend_has_no_price_parameter():
    import inspect

    params = set(inspect.signature(power_blend).parameters)
    assert params == {"p_model", "p_reference", "race_ids", "alpha", "beta"}


# ── incomplete reference books ───────────────────────────────────────────────


def test_incomplete_reference_book_nans_the_whole_race():
    df = _frame(seed=4)
    df.loc[df.index[3], "ref_odds"] = np.nan          # one unpriced runner
    df["p_ref"] = devig(df["ref_odds"], df["race_uid"], method="proportional")
    p = power_blend(df["p_model"], df["p_ref"], df["race_uid"], 0.5, 1.0)

    bad = df["race_uid"] == df.loc[df.index[3], "race_uid"]
    assert np.isnan(p[bad.to_numpy()]).all()
    assert np.isfinite(p[~bad.to_numpy()]).all()


def test_missing_model_probability_also_nans_the_whole_race():
    df = _frame(seed=5)
    df.loc[df.index[10], "p_model"] = np.nan
    p = power_blend(df["p_model"], df["p_ref"], df["race_uid"], 0.5, 1.0)
    bad = (df["race_uid"] == df.loc[df.index[10], "race_uid"]).to_numpy()
    assert np.isnan(p[bad]).all()
    assert np.isfinite(p[~bad]).all()


def test_non_positive_probability_is_treated_as_an_incomplete_book():
    df = _frame(seed=6)
    df.loc[df.index[0], "p_model"] = 0.0
    p = power_blend(df["p_model"], df["p_ref"], df["race_uid"], 0.5, 1.0)
    bad = (df["race_uid"] == df.loc[df.index[0], "race_uid"]).to_numpy()
    assert np.isnan(p[bad]).all()


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        power_blend([0.5, 0.5], [0.5], ["a", "a"], 1.0, 1.0)


def test_empty_input_returns_empty():
    assert power_blend([], [], [], 1.0, 1.0).size == 0


# ── non-runners ──────────────────────────────────────────────────────────────


def test_non_runner_removal_renormalises_without_reordering_survivors():
    df = _frame(seed=7, n_races=1, size=9)
    full = power_blend(df["p_model"], df["p_ref"], df["race_uid"], 0.6, 1.0)

    # A non-runner carries no reference price and no outcome: it leaves the
    # field entirely rather than being blended at some imputed probability.
    kept = df.drop(index=df.index[4]).reset_index(drop=True)
    kept["p_ref"] = devig(kept["ref_odds"], kept["race_uid"], method="proportional")
    reduced = power_blend(kept["p_model"], kept["p_ref"], kept["race_uid"], 0.6, 1.0)

    assert reduced.sum() == pytest.approx(1.0)
    survivors = [i for i in range(len(df)) if i != 4]
    # Ranking among the survivors is untouched by the withdrawal.
    assert list(np.argsort(full[survivors])) == list(np.argsort(reduced))


# ── scoring parity with the scorecard ────────────────────────────────────────


def test_race_log_loss_matches_head_to_head_definition():
    df = _frame(seed=8)
    p = power_blend(df["p_model"], df["p_ref"], df["race_uid"], 0.5, 1.0)
    mine = race_level_log_loss(df["won"].to_numpy(), p, df["race_uid"].to_numpy())
    theirs = _race_level_log_loss(
        df["won"].to_numpy(float), p, df["race_uid"].to_numpy()
    )
    assert mine == pytest.approx(theirs, abs=1e-12)


def test_race_log_loss_ignores_unscoreable_races():
    y = np.array([1, 0, 0, 0])
    p = np.array([np.nan, np.nan, 0.5, 0.5])
    rid = np.array(["a", "a", "b", "b"])
    # Race "a" has a NaN winner probability and "b" has no winner → nothing to score.
    assert np.isnan(race_level_log_loss(y, p, rid))


# ── selection + persistence ──────────────────────────────────────────────────


def test_selection_prefers_the_market_when_the_model_is_noise():
    df = _frame(seed=9, n_races=200)
    rng = np.random.default_rng(99)
    df["p_model"] = rng.uniform(0.01, 0.99, len(df))    # pure noise
    blend, table = select_power_blend(
        df["p_model"], df["p_ref"], df["race_uid"], df["won"]
    )
    assert len(table) == len(ALPHA_GRID) * len(BETA_GRID)
    assert blend.alpha <= 0.2
    assert blend.beta > 0
    market_only = table[(table.alpha == 0.0) & (table.beta == 1.0)]["race_log_loss"].iloc[0]
    assert blend.metadata["dev_race_log_loss"] <= market_only + 1e-9


def test_grid_contains_both_boundary_controls():
    assert 0.0 in ALPHA_GRID and 1.0 in ALPHA_GRID
    assert 0.0 in BETA_GRID and 1.0 in BETA_GRID


def test_powerblend_json_round_trip(tmp_path):
    b = PowerBlend(0.35, 1.25, metadata={"selected_by": "race_level_log_loss"})
    path = tmp_path / "blend.json"
    b.save(path)
    loaded = PowerBlend.load(path)
    assert (loaded.alpha, loaded.beta) == (b.alpha, b.beta)
    assert loaded.metadata == b.metadata
    assert json.loads(path.read_text())["model_type"] == "power_blend"

    df = _frame(seed=11)
    assert np.array_equal(
        b.transform(df["p_model"], df["p_ref"], df["race_uid"]),
        loaded.transform(df["p_model"], df["p_ref"], df["race_uid"]),
    )
