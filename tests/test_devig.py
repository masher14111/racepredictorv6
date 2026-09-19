"""Tests for models.devig — market de-vigging into margin-free probabilities.

Covers all three methods (proportional / power / shin):
  * each race's probabilities sum to 1.0,
  * power and shin remove MORE longshot margin than proportional
    (favourite–longshot bias correction),
  * partial books (any NaN price) are flagged NaN, never de-vigged,
  * single-runner races resolve to probability 1.0,
  * sub-1.0 / 1.0 odds are clipped defensively.

The hand cases pin the numeric behaviour against the source implementation
in racing_ingestion/ev/market_calibration.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.devig import (
    MARGIN_METHODS,
    de_vig_power,
    de_vig_proportional,
    de_vig_shin,
    devig,
    implied_probs,
)

METHODS = ["proportional", "power", "shin"]

# A realistic margin book: clear favourite + mid + two longshots. The booksum
# (Σ 1/o_i) is > 1 — there is a real bookmaker overround to strip out.
_ODDS = np.array([1.8, 3.5, 6.0, 9.0])


# ── Per-race math ─────────────────────────────────────────────────────────────


def test_implied_probs_are_raw_overround():
    pi = implied_probs(_ODDS)
    np.testing.assert_allclose(pi, 1.0 / _ODDS)
    # A real margin book sums to more than 1 before de-vigging.
    assert pi.sum() > 1.0


@pytest.mark.parametrize("method", METHODS)
def test_per_race_sums_to_one(method):
    p = MARGIN_METHODS[method](_ODDS)
    assert p.shape == _ODDS.shape
    assert np.all(p > 0)
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-12)


def test_proportional_hand_case():
    # p_i = (1/o_i) / Σ(1/o_j), exact closed form.
    pi = 1.0 / _ODDS
    np.testing.assert_allclose(de_vig_proportional(_ODDS), pi / pi.sum(), atol=1e-12)


def test_proportional_preserves_implied_ranking():
    # Proportional is a pure rescale, so it must keep the market's order.
    p = de_vig_proportional(_ODDS)
    assert np.all(np.argsort(p) == np.argsort(1.0 / _ODDS))


def test_power_exponent_above_one_for_margin_book():
    # With a positive overround, k > 1 is needed to shrink the implied probs to 1.
    _, k = de_vig_power(_ODDS, return_exponent=True)
    assert k > 1.0


def _longshot_margin_removed(method: str) -> float:
    """How much the de-vig lengthens the longshot vs its raw implied prob.

    Positive = probability pushed DOWN (more margin stripped from the longshot).
    """
    p = MARGIN_METHODS[method](_ODDS)
    pi_norm = (1.0 / _ODDS) / (1.0 / _ODDS).sum()  # proportional reference
    longshot = int(np.argmax(_ODDS))  # the 21.0 runner
    return pi_norm[longshot] - p[longshot]


def test_power_and_shin_remove_more_longshot_margin_than_proportional():
    # Proportional is its own reference, so removes nothing extra.
    assert _longshot_margin_removed("proportional") == pytest.approx(0.0, abs=1e-12)
    # Power and shin both push the longshot's probability below the proportional
    # line — the favourite–longshot bias correction.
    assert _longshot_margin_removed("power") > 1e-6
    assert _longshot_margin_removed("shin") > 1e-6


# ── Edge cases ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", METHODS)
def test_single_runner_is_certain(method):
    np.testing.assert_allclose(MARGIN_METHODS[method](np.array([3.7])), [1.0])


@pytest.mark.parametrize("method", METHODS)
def test_two_runner_sums_to_one(method):
    # n == 2 is the shin fallback boundary; must still be a valid simplex.
    p = MARGIN_METHODS[method](np.array([1.5, 3.0]))
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_odds_at_or_below_one_are_clipped(method):
    # Odds <= 1.0 are data errors; clip (to _MIN_ODDS) rather than divide-by-~0
    # or go negative. Result stays a valid simplex.
    p = MARGIN_METHODS[method](np.array([1.0, 0.5, 4.0]))
    assert np.all(np.isfinite(p))
    assert np.all(p >= 0)
    np.testing.assert_allclose(p.sum(), 1.0, atol=1e-12)


def test_shin_falls_back_to_power_on_two_runners():
    odds = np.array([1.8, 2.2])
    np.testing.assert_allclose(de_vig_shin(odds), de_vig_power(odds), atol=1e-12)


# ── Public DataFrame-level API ────────────────────────────────────────────────


def _two_race_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "race_id": ["A", "A", "A", "B", "B"],
            "odds": [2.5, 4.0, 8.0, 1.8, 2.2],
        }
    )


@pytest.mark.parametrize("method", METHODS)
def test_devig_sums_to_one_per_race(method):
    df = _two_race_frame()
    p = devig(df["odds"], df["race_id"], method=method)
    assert p.shape == (len(df),)
    for rid in df["race_id"].unique():
        mask = (df["race_id"] == rid).to_numpy()
        np.testing.assert_allclose(p[mask].sum(), 1.0, atol=1e-12)


def test_devig_aligns_to_input_rows():
    # Interleaved race ids must still scatter back to the right rows.
    odds = pd.Series([2.5, 1.8, 4.0, 2.2, 8.0])
    rids = pd.Series(["A", "B", "A", "B", "A"])
    p = devig(odds, rids, method="proportional")
    # Race A rows: indices 0,2,4 ; race B rows: 1,3.
    np.testing.assert_allclose(p[[0, 2, 4]].sum(), 1.0, atol=1e-12)
    np.testing.assert_allclose(p[[1, 3]].sum(), 1.0, atol=1e-12)
    # Row 0 should equal the standalone per-race result for race A.
    expected_a = de_vig_proportional(np.array([2.5, 4.0, 8.0]))
    np.testing.assert_allclose(p[[0, 2, 4]], expected_a, atol=1e-12)


def test_devig_flags_partial_book_with_nan():
    # Race A has a missing price → the WHOLE race is NaN (never de-vig a
    # partial book). Race B is complete and unaffected.
    df = pd.DataFrame(
        {
            "race_id": ["A", "A", "A", "B", "B"],
            "odds": [2.5, np.nan, 8.0, 1.8, 2.2],
        }
    )
    p = devig(df["odds"], df["race_id"], method="proportional")
    a = (df["race_id"] == "A").to_numpy()
    b = (df["race_id"] == "B").to_numpy()
    assert np.all(np.isnan(p[a]))           # entire incomplete book flagged
    assert np.all(np.isfinite(p[b]))        # complete book still de-vigged
    np.testing.assert_allclose(p[b].sum(), 1.0, atol=1e-12)


def test_devig_single_runner_race():
    odds = pd.Series([3.7])
    rids = pd.Series(["solo"])
    np.testing.assert_allclose(devig(odds, rids, method="shin"), [1.0])


def test_devig_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown method"):
        devig(pd.Series([2.0, 3.0]), pd.Series(["A", "A"]), method="nope")


def test_devig_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        devig(pd.Series([2.0, 3.0]), pd.Series(["A"]))


def test_devig_empty_input():
    p = devig(pd.Series([], dtype=float), pd.Series([], dtype=object))
    assert p.shape == (0,)
