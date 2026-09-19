"""Tests for backtest.staking — selection gates and stake schemes.

Each gate is checked in isolation (others left None) so a passing test pins the
exact predicate, and the gates are confirmed to AND together.
"""
import numpy as np
import pandas as pd
import pytest

from backtest.staking import FlatStake, KellyStake, Strategy


def _scored(prob, bet_price):
    return pd.DataFrame({"prob": prob, "bet_price": bet_price})


# ── Strategy selection gates ──────────────────────────────────────────────────

class TestStrategyGates:
    def test_no_gates_keeps_all_backable(self):
        # Only a valid price is required; invalid price row is dropped.
        s = Strategy("all")
        df = _scored([0.3, 0.3, 0.3], [4.0, 2.0, 1.0])
        mask = s.select(df)
        assert mask.tolist() == [True, True, False]

    def test_min_ev_gate(self):
        # p=0.30: EV @4.0 = +0.2 (keep), EV @3.0 = -0.1 (drop) at min_ev=0.05.
        s = Strategy("ev", min_ev=0.05)
        df = _scored([0.30, 0.30], [4.0, 3.0])
        assert s.select(df).tolist() == [True, False]

    def test_min_edge_pct_gate(self):
        # min_edge_pct=0.10 means p >= implied*1.10.
        # @4.0 implied=0.25 -> threshold 0.275. p=0.30 keeps, p=0.26 drops.
        s = Strategy("edge%", min_edge_pct=0.10)
        df = _scored([0.30, 0.26], [4.0, 4.0])
        assert s.select(df).tolist() == [True, False]

    def test_min_abs_edge_gate(self):
        # absolute edge p - 1/d >= 0.05. @4.0 implied 0.25.
        # Stay off the exact 0.05 boundary (0.30-0.25 floats to 0.0499...).
        s = Strategy("absedge", min_abs_edge=0.05)
        df = _scored([0.32, 0.28], [4.0, 4.0])  # edges 0.07, 0.03
        assert s.select(df).tolist() == [True, False]

    def test_odds_band_gates_longshots(self):
        # min_odds/max_odds is the longshot gate on the execution price.
        s = Strategy("band", min_odds=2.0, max_odds=26.0)
        df = _scored([0.3, 0.3, 0.3], [1.5, 10.0, 40.0])
        assert s.select(df).tolist() == [False, True, False]

    def test_min_prob_gate(self):
        s = Strategy("minp", min_prob=0.20)
        df = _scored([0.10, 0.25], [4.0, 4.0])
        assert s.select(df).tolist() == [False, True]

    def test_gates_are_anded(self):
        # EV and band both must pass.
        s = Strategy("combo", min_ev=0.05, min_odds=2.0, max_odds=8.0)
        df = _scored(
            [0.30, 0.30, 0.30],
            [4.0,  40.0, 1.5],   # row0: EV+band ok; row1: band fail; row2: band fail
        )
        assert s.select(df).tolist() == [True, False, False]

    def test_custom_selector_overrides_gates(self):
        s = Strategy("custom", min_ev=99.0,  # gate that would reject everything
                     selector=lambda d: d["prob"] > 0.25)
        df = _scored([0.30, 0.20], [4.0, 4.0])
        assert s.select(df).tolist() == [True, False]

    def test_mask_index_aligned(self):
        s = Strategy("all")
        df = _scored([0.3, 0.3], [4.0, 2.0])
        df.index = [7, 9]
        mask = s.select(df)
        assert list(mask.index) == [7, 9]

    def test_nan_inputs_stay_false(self):
        s = Strategy("ev", min_ev=0.05)
        df = _scored([np.nan, 0.3], [4.0, np.nan])
        assert s.select(df).tolist() == [False, False]


# ── FlatStake ─────────────────────────────────────────────────────────────────

class TestFlatStake:
    def test_constant_unit(self):
        f = FlatStake(unit=10.0)
        assert f.stake(1000.0, 0.3, 4.0) == 10.0
        assert f.stake(500.0, 0.9, 1.5) == 10.0  # independent of prob/odds

    def test_capped_at_bankroll(self):
        f = FlatStake(unit=10.0)
        assert f.stake(5.0, 0.3, 4.0) == 5.0

    def test_never_negative(self):
        f = FlatStake(unit=10.0)
        assert f.stake(-50.0, 0.3, 4.0) == 0.0


# ── KellyStake ────────────────────────────────────────────────────────────────

class TestKellyStake:
    def test_fractional_kelly_stake(self):
        # full kelly @ p=0.30,d=4.0 = (0.2)/3 = 0.0667; quarter = 0.01667.
        # cap=0.05 not binding -> stake = 0.01667 * 1000 = 16.67.
        k = KellyStake(fraction=0.25, cap=0.05)
        assert k.stake(1000.0, 0.30, 4.0) == pytest.approx(0.25 * (0.2 / 3.0) * 1000.0)

    def test_cap_binds_on_huge_edge(self):
        # Big edge would blow past cap; stake limited to cap*bankroll.
        k = KellyStake(fraction=1.0, cap=0.05)
        stake = k.stake(1000.0, 0.95, 4.0)
        assert stake == pytest.approx(0.05 * 1000.0)

    def test_no_bet_on_negative_edge(self):
        k = KellyStake(fraction=0.25, cap=0.05)
        assert k.stake(1000.0, 0.20, 4.0) == 0.0  # kelly floored at 0

    def test_invalid_price_is_zero(self):
        k = KellyStake(fraction=0.25, cap=0.05)
        assert k.stake(1000.0, 0.30, 1.0) == 0.0

    def test_scales_with_bankroll(self):
        k = KellyStake(fraction=0.25, cap=0.5)
        assert k.stake(2000.0, 0.30, 4.0) == pytest.approx(2.0 * k.stake(1000.0, 0.30, 4.0))
