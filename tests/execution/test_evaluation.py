"""execution/evaluation.py — hand-computed metrics + the anti-false-significance
property that Stage 4 paid for.

The load-bearing test here is
``test_race_clustered_ci_is_wider_than_row_bootstrap``: if that ever passes only
marginally, someone has quietly reintroduced row-level resampling and every
interval in every report is too narrow.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from execution.evaluation import (
    COMPARISON_COLUMNS,
    Interval,
    StrategyMetrics,
    compare_strategies,
    evaluate_ledger,
    expected_calibration_error,
    race_bootstrap_ci,
)

# ── the hand-computed ledger ─────────────────────────────────────────────────
# race  stake  odds  prob  won   profit
# R1     10    3.0   0.40   1    +20
# R1     10    5.0   0.20   0    -10
# R2     10    2.0   0.50   0    -10
# R3     10    4.0   0.25   0    -10
# R4     10    6.0   0.20   1    +50
# turnover 50 | profit 40 | roi 0.8 | hit 2/5 | E[wins] 1.55 | A 2
HAND_LEDGER = pd.DataFrame({
    "race_uid": ["R1", "R1", "R2", "R3", "R4"],
    "race_date": pd.to_datetime(
        ["2026-06-01", "2026-06-01", "2026-06-01", "2026-06-02", "2026-06-03"]),
    "horse_key": ["a", "b", "c", "d", "e"],
    "stake": [10.0, 10.0, 10.0, 10.0, 10.0],
    "decimal_odds": [3.0, 5.0, 2.0, 4.0, 6.0],
    "model_prob": [0.40, 0.20, 0.50, 0.25, 0.20],
    "won": [1, 0, 0, 0, 1],
    "closing_odds": [2.5, 6.0, 2.0, 5.0, 5.0],
})


@pytest.fixture
def hand_metrics() -> StrategyMetrics:
    return evaluate_ledger(HAND_LEDGER, name="hand", initial_bankroll=1000.0, n_boot=200)


def test_hand_computed_money_and_sample_size(hand_metrics):
    m = hand_metrics
    assert (m.n_bets, m.n_races, m.n_days) == (5, 4, 3)
    assert m.turnover == pytest.approx(50.0)
    assert m.profit == pytest.approx(40.0)
    assert m.total_return == pytest.approx(90.0)   # turnover + profit
    assert m.roi == pytest.approx(0.8)
    assert m.yield_pct == pytest.approx(80.0)


def test_hand_computed_hit_rate_and_ae(hand_metrics):
    m = hand_metrics
    assert m.hit_rate == pytest.approx(0.4)
    assert m.expected_wins == pytest.approx(1.55)
    assert m.actual_wins == pytest.approx(2.0)
    assert m.ae_ratio == pytest.approx(2.0 / 1.55)


def test_hand_computed_drawdown_and_streak(hand_metrics):
    # equity 1000 -> 1020 -> 1010 -> 1000 -> 990 -> 1040; trough is 30 below the
    # 1020 peak.
    m = hand_metrics
    assert m.max_drawdown == pytest.approx(30.0)
    assert m.max_drawdown_pct == pytest.approx(30.0 / 1020.0)
    # won sequence 1,0,0,0,1
    assert m.longest_losing_streak == 3


def test_hand_computed_clv(hand_metrics):
    m = hand_metrics
    expected_log = np.mean([
        math.log(3.0 / 2.5), math.log(5.0 / 6.0), math.log(2.0 / 2.0),
        math.log(4.0 / 5.0), math.log(6.0 / 5.0),
    ])
    expected_pct = np.mean([3 / 2.5 - 1, 5 / 6 - 1, 0.0, 4 / 5 - 1, 6 / 5 - 1])
    assert m.mean_clv_log == pytest.approx(expected_log)
    assert m.mean_clv_pct == pytest.approx(expected_pct)
    assert m.beat_close_rate == pytest.approx(0.4)   # 3>2.5 and 6>5


def test_missing_closing_odds_gives_none_not_zero():
    ledger = HAND_LEDGER.drop(columns=["closing_odds"])
    m = evaluate_ledger(ledger, name="no_close", n_boot=100)
    assert m.mean_clv_log is None
    assert m.mean_clv_pct is None
    assert m.beat_close_rate is None
    assert m.clv_ci is None
    # ...while everything that does not depend on the close is still reported.
    assert m.roi == pytest.approx(0.8)


def test_voided_bets_excluded_from_hit_rate_and_ae_but_counted_in_n_bets():
    ledger = pd.concat([
        HAND_LEDGER,
        pd.DataFrame({
            "race_uid": ["R5"],
            "race_date": pd.to_datetime(["2026-06-04"]),
            "horse_key": ["f"],
            "stake": [10.0],
            "decimal_odds": [8.0],
            "model_prob": [0.90],   # would wreck A/E if it counted
            "won": [0],
            "closing_odds": [8.0],
        }),
    ], ignore_index=True)
    ledger["voided"] = [False] * 5 + [True]

    m = evaluate_ledger(ledger, name="with_void", n_boot=100)
    assert m.n_bets == 6                       # the void consumed a ticket
    assert m.hit_rate == pytest.approx(0.4)    # 2/5, not 2/6
    assert m.expected_wins == pytest.approx(1.55)   # 0.90 excluded
    assert m.ae_ratio == pytest.approx(2.0 / 1.55)
    assert m.turnover == pytest.approx(50.0)   # stake returned, no exposure
    assert m.profit == pytest.approx(40.0)
    assert m.longest_losing_streak == 3        # a void does not extend a streak


def test_empty_ledger_is_zero_bets_and_no_intervals():
    m = evaluate_ledger(pd.DataFrame(), name="empty")
    assert m.n_bets == 0 and m.n_races == 0
    assert m.roi_ci is None and m.clv_ci is None and m.ae_ci is None
    assert m.to_dict()["n_bets"] == 0


def test_missing_cluster_column_suppresses_every_interval():
    """Fail-closed: no race key means no honest interval, so we publish none."""
    ledger = HAND_LEDGER.drop(columns=["race_uid"])
    m = evaluate_ledger(ledger, name="no_cluster", n_boot=100)
    assert m.roi_ci is None
    assert m.hit_rate_ci is None
    assert m.clv_ci is None
    assert m.ae_ci is None
    assert m.roi == pytest.approx(0.8)   # point estimates are still fine


# ── the bootstrap ────────────────────────────────────────────────────────────


def _correlated_races(n_races: int = 25, per_race: int = 5, seed: int = 7):
    """Perfectly within-race-correlated outcomes: every runner shares the race's
    value. Row resampling then sees ``n_races * per_race`` "independent" draws
    where only ``n_races`` exist."""
    rng = np.random.default_rng(seed)
    race_values = rng.normal(0.0, 1.0, n_races)
    values = np.repeat(race_values, per_race)
    clusters = np.repeat([f"R{i}" for i in range(n_races)], per_race)
    return values, clusters


def test_race_bootstrap_is_deterministic_for_a_seed():
    values, clusters = _correlated_races()
    a = race_bootstrap_ci(values, clusters, n_boot=300, seed=20260727)
    b = race_bootstrap_ci(values, clusters, n_boot=300, seed=20260727)
    assert (a.lower, a.upper) == (b.lower, b.upper)


def test_different_seed_gives_a_different_interval():
    values, clusters = _correlated_races()
    a = race_bootstrap_ci(values, clusters, n_boot=300, seed=20260727)
    c = race_bootstrap_ci(values, clusters, n_boot=300, seed=1)
    assert (a.lower, a.upper) != (c.lower, c.upper)


def test_race_clustered_ci_is_wider_than_row_bootstrap():
    """The anti-false-significance property. Row bootstrap = every row its own
    cluster; with 5 correlated runners per race it should be roughly
    ``sqrt(5)`` times too narrow."""
    values, clusters = _correlated_races(n_races=25, per_race=5)
    clustered = race_bootstrap_ci(values, clusters, n_boot=800, seed=20260727)
    row_level = race_bootstrap_ci(
        values, np.arange(values.size), n_boot=800, seed=20260727)

    assert clustered.width > row_level.width
    # Not marginally wider — materially so, which is the whole point.
    assert clustered.width > 1.5 * row_level.width


def test_row_bootstrap_manufactures_significance_that_clustering_denies():
    """A shifted version of the same data: the row bootstrap excludes zero, the
    honest race-clustered one does not."""
    values, clusters = _correlated_races(n_races=25, per_race=6, seed=3)
    values = values + 0.35 * np.std(values)

    clustered = race_bootstrap_ci(values, clusters, n_boot=800, seed=20260727)
    row_level = race_bootstrap_ci(
        values, np.arange(values.size), n_boot=800, seed=20260727)
    assert row_level.excludes_zero()
    assert not clustered.excludes_zero()


def test_race_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        race_bootstrap_ci([1.0, 2.0, 3.0], ["a", "b"])


def test_interval_excludes_zero_and_unknown_never_does():
    assert Interval(0.1, 0.4).excludes_zero()
    assert Interval(-0.4, -0.1).excludes_zero()
    assert not Interval(-0.1, 0.4).excludes_zero()
    assert not Interval(float("nan"), 0.4).excludes_zero()
    d = Interval(0.1, 0.4).to_dict()
    assert d["excludes_zero"] is True and d["level"] == 0.95


# ── calibration ──────────────────────────────────────────────────────────────


def test_expected_calibration_error_hand_computed():
    # Two bins used: [0.0,0.1) holds four 0.05 predictions with one winner
    # (|0.05 - 0.25| = 0.20); [0.5,0.6) holds four 0.50 predictions with two
    # winners (|0.50 - 0.50| = 0.0). ECE = 0.5*0.20 + 0.5*0.0 = 0.10.
    prob = [0.05, 0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 0.5]
    won = [1, 0, 0, 0, 1, 1, 0, 0]
    assert expected_calibration_error(prob, won, bins=10) == pytest.approx(0.10)


def test_expected_calibration_error_is_nan_when_unknown():
    assert math.isnan(expected_calibration_error([], []))


def test_perfect_calibration_scores_zero():
    prob = [0.5] * 4 + [1.0] * 2
    won = [1, 1, 0, 0, 1, 1]
    assert expected_calibration_error(prob, won, bins=10) == pytest.approx(0.0)


# ── comparison table ─────────────────────────────────────────────────────────


def test_compare_strategies_puts_sample_size_first():
    a = evaluate_ledger(HAND_LEDGER, name="model_only", n_boot=100)
    b = evaluate_ledger(HAND_LEDGER.iloc[:3], name="favourite", n_boot=100)
    table = compare_strategies({"model_only": a, "favourite": b})

    assert list(table.columns) == list(COMPARISON_COLUMNS)
    assert table.columns[0] == "n_bets"
    # ...and no ROI-ish column may precede it.
    assert table.columns.get_loc("n_bets") < table.columns.get_loc("roi")
    assert table.index.name == "strategy"
    assert list(table.index) == ["model_only", "favourite"]
    assert table.loc["model_only", "n_bets"] == 5
    assert table.loc["model_only", "roi"] == pytest.approx(0.8)


def test_compare_strategies_flattens_intervals_and_survives_none():
    m = evaluate_ledger(HAND_LEDGER.drop(columns=["closing_odds"]),
                        name="s", n_boot=100)
    table = compare_strategies({"s": m})
    assert table.loc["s", "clv_ci_lower"] is None
    assert table.loc["s", "clv_ci_excludes_zero"] is None
    assert table.loc["s", "roi_ci_lower"] is not None


def test_compare_strategies_on_empty_mapping():
    table = compare_strategies({})
    assert list(table.columns) == list(COMPARISON_COLUMNS)
    assert len(table) == 0


def test_to_dict_is_json_safe():
    import json

    m = evaluate_ledger(HAND_LEDGER, name="hand", n_boot=100)
    payload = json.dumps(m.to_dict())
    assert '"n_bets": 5' in payload
    # first key is the name, then sample size — the reading order the report keeps
    assert list(m.to_dict())[:4] == ["name", "n_bets", "n_races", "n_days"]
