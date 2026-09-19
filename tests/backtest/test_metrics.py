"""Tests for backtest.metrics — the betting maths and run summaries.

The EV/edge/Kelly/CLV identities are the contract every staking rule and the
engine depend on, so they're checked against hand-worked closed forms (not just
"runs without error"). NaN-handling and scalar-in/scalar-out are part of the
contract too — the per-bet settle path calls these with scalars.
"""
import numpy as np
import pandas as pd
import pytest

from backtest import metrics


# ── core identities (the value-betting maths) ────────────────────────────────

class TestImpliedProb:
    def test_basic(self):
        assert metrics.implied_prob(4.0) == pytest.approx(0.25)
        assert metrics.implied_prob(2.0) == pytest.approx(0.5)

    def test_invalid_price_is_nan(self):
        # <= 1.0 is the feeds' sentinel for "no price".
        assert np.isnan(metrics.implied_prob(1.0))
        assert np.isnan(metrics.implied_prob(0.5))
        assert np.isnan(metrics.implied_prob(np.nan))

    def test_vectorised(self):
        out = metrics.implied_prob([2.0, 5.0, 1.0])
        assert out[0] == pytest.approx(0.5)
        assert out[1] == pytest.approx(0.2)
        assert np.isnan(out[2])


class TestExpectedValue:
    def test_fair_bet_is_zero(self):
        # p == 1/d  ->  p*d - 1 == 0.
        assert metrics.expected_value(0.25, 4.0) == pytest.approx(0.0)

    def test_positive_and_negative(self):
        # 30% @ 4.0 -> 0.3*4 - 1 = 0.2 ; 20% @ 4.0 -> -0.2.
        assert metrics.expected_value(0.30, 4.0) == pytest.approx(0.20)
        assert metrics.expected_value(0.20, 4.0) == pytest.approx(-0.20)

    def test_invalid_price_is_nan(self):
        assert np.isnan(metrics.expected_value(0.5, 1.0))


class TestEdge:
    def test_edge_is_prob_minus_implied(self):
        # 0.30 - 1/4 = 0.05.
        assert metrics.edge(0.30, 4.0) == pytest.approx(0.05)

    def test_relationship_to_ev(self):
        # EV = p*d - 1 = d * (p - 1/d) = d * edge.
        p, d = 0.30, 4.0
        assert metrics.expected_value(p, d) == pytest.approx(d * metrics.edge(p, d))


class TestKellyFraction:
    def test_closed_form(self):
        # (p*d - 1)/(d - 1) = 0.2/3 for p=0.30, d=4.0.
        assert metrics.kelly_fraction(0.30, 4.0) == pytest.approx(0.2 / 3.0)

    def test_equals_ev_over_d_minus_1(self):
        # Kelly numerator IS the EV: (p*d-1)/(d-1) = EV/(d-1).
        p, d = 0.30, 4.0
        assert metrics.kelly_fraction(p, d) == pytest.approx(
            metrics.expected_value(p, d) / (d - 1.0))

    def test_floored_at_zero_for_negative_edge(self):
        # No bet when the edge is negative -> Kelly 0, never negative.
        assert metrics.kelly_fraction(0.20, 4.0) == 0.0

    def test_fair_bet_is_zero(self):
        assert metrics.kelly_fraction(0.25, 4.0) == pytest.approx(0.0)

    def test_invalid_price_is_nan(self):
        assert np.isnan(metrics.kelly_fraction(0.5, 1.0))


class TestSettle:
    def test_win_pays_net_profit(self):
        # €10 @ 4.0 win -> 10*(4-1) = 30 profit.
        assert metrics.settle(10.0, 4.0, 1) == pytest.approx(30.0)

    def test_loss_returns_negative_stake(self):
        assert metrics.settle(10.0, 4.0, 0) == pytest.approx(-10.0)

    def test_commission_only_on_winnings(self):
        # 5% commission on the 30 net win -> 28.5; losers unaffected.
        assert metrics.settle(10.0, 4.0, 1, commission=0.05) == pytest.approx(28.5)
        assert metrics.settle(10.0, 4.0, 0, commission=0.05) == pytest.approx(-10.0)

    def test_zero_stake_is_zero_pnl(self):
        assert metrics.settle(0.0, 4.0, 1) == 0.0

    def test_invalid_price_is_zero_pnl(self):
        # No real price -> no bet -> no P&L (not a loss).
        assert metrics.settle(10.0, 1.0, 0) == 0.0
        assert metrics.settle(10.0, np.nan, 1) == 0.0

    def test_vectorised(self):
        out = metrics.settle([10.0, 10.0], [4.0, 4.0], [1, 0])
        assert out[0] == pytest.approx(30.0)
        assert out[1] == pytest.approx(-10.0)


class TestCLV:
    def test_beat_the_close(self):
        # Backed 5.0, closed 4.0 -> 5/4 - 1 = +0.25 (you got a bigger price).
        assert metrics.clv_pct(5.0, 4.0) == pytest.approx(0.25)

    def test_worse_than_close(self):
        assert metrics.clv_pct(4.0, 5.0) == pytest.approx(4.0 / 5.0 - 1.0)

    def test_invalid_price_is_nan(self):
        assert np.isnan(metrics.clv_pct(5.0, 1.0))
        assert np.isnan(metrics.clv_pct(1.0, 4.0))


# ── scalar-in/scalar-out contract (the per-bet settle path relies on it) ──────

class TestScalarSqueeze:
    @pytest.mark.parametrize("fn,args", [
        (metrics.implied_prob, (4.0,)),
        (metrics.expected_value, (0.3, 4.0)),
        (metrics.edge, (0.3, 4.0)),
        (metrics.kelly_fraction, (0.3, 4.0)),
        (metrics.settle, (10.0, 4.0, 1)),
        (metrics.clv_pct, (5.0, 4.0)),
    ])
    def test_scalar_inputs_return_python_float(self, fn, args):
        out = fn(*args)
        assert isinstance(out, float)

    def test_array_inputs_return_array(self):
        out = metrics.expected_value([0.3, 0.4], [4.0, 3.0])
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)


# ── de-vig ────────────────────────────────────────────────────────────────────

class TestDevig:
    def test_fair_market_has_zero_overround(self):
        fair, overround = metrics.devig([0.5, 0.5])
        assert overround == pytest.approx(0.0)
        assert fair.sum() == pytest.approx(1.0)

    def test_strips_overround_and_normalises(self):
        # Two runners each priced 2.0 -> implied 0.5+0.5 = 1.0 overround 0,
        # but 1/1.5 each (0.667*2 = 1.333) -> overround 0.333, fair 0.5/0.5.
        q = metrics.implied_prob([1.5, 1.5])
        fair, overround = metrics.devig(q)
        assert overround == pytest.approx(1.0 / 1.5 * 2 - 1.0)
        assert fair.sum() == pytest.approx(1.0)
        assert fair[0] == pytest.approx(0.5)

    def test_all_invalid_returns_nan_overround(self):
        fair, overround = metrics.devig([np.nan, np.nan])
        assert np.isnan(overround)


# ── drawdown / sharpe ─────────────────────────────────────────────────────────

class TestMaxDrawdown:
    def test_monotone_up_has_no_drawdown(self):
        assert metrics.max_drawdown([100, 110, 120]) == pytest.approx(0.0)

    def test_simple_drawdown(self):
        # Peak 120 -> trough 90 = 25% drawdown.
        assert metrics.max_drawdown([100, 120, 90, 110]) == pytest.approx(0.25)

    def test_empty_is_zero(self):
        assert metrics.max_drawdown([]) == 0.0


class TestSharpe:
    def test_needs_two_points(self):
        assert metrics._sharpe([0.1]) is None

    def test_zero_variance_is_none(self):
        assert metrics._sharpe([0.1, 0.1, 0.1]) is None

    def test_positive_mean(self):
        s = metrics._sharpe([0.1, 0.2, 0.3])
        assert s is not None and s > 0


# ── A/E calibration table ─────────────────────────────────────────────────────

class TestAETable:
    def test_perfect_calibration_gives_ae_one(self):
        # 1000 runners at p=0.10, exactly 100 winners -> A/E == 1.0 in that bucket.
        n = 1000
        prob = np.full(n, 0.10)
        won = np.zeros(n)
        won[:100] = 1
        rows = metrics.ae_table(prob, won)
        assert len(rows) == 1
        r = rows[0]
        assert r["expected"] == pytest.approx(100.0)
        assert r["actual"] == pytest.approx(100.0)
        assert r["ae"] == pytest.approx(1.0)

    def test_overconfident_bucket_ae_below_one(self):
        # Predict 0.5 but only 25% actually win -> A/E ~ 0.5.
        n = 400
        prob = np.full(n, 0.5)
        won = np.zeros(n)
        won[:100] = 1
        rows = metrics.ae_table(prob, won)
        r = [x for x in rows if x["n"] == n][0]
        assert r["ae"] == pytest.approx(0.5, abs=1e-6)

    def test_odds_band_grouping(self):
        prob = np.array([0.5, 0.5, 0.1])
        won = np.array([1, 0, 0])
        price = np.array([1.8, 1.9, 20.0])  # two odds-on, one longshot
        rows = metrics.ae_table(prob, won, group_values=price, bands=metrics._ODDS_BANDS)
        labels = {r["bucket"]: r for r in rows}
        assert labels["odds-on (<2.0)"]["n"] == 2
        assert any("16.0-34.0" in b for b in labels)

    def test_empty_buckets_dropped(self):
        rows = metrics.ae_table([0.5], [1])
        # Only the 0.50-1.00 bucket is populated.
        assert all(r["n"] > 0 for r in rows)


# ── summarize_bets ────────────────────────────────────────────────────────────

def _ledger(rows):
    return pd.DataFrame(rows)


class TestSummarizeBets:
    def test_empty_ledger(self):
        out = metrics.summarize_bets(pd.DataFrame(), initial_bankroll=1000.0)
        assert out["n_bets"] == 0
        assert out["yield_pct"] is None
        assert out["final_bankroll"] == 1000.0

    def test_known_pnl_and_yield(self):
        # Two €10 flat bets: one wins @4.0 (+30), one loses (-10).
        # staked 20, profit 20, yield = 20/20 = 100%.
        led = _ledger([
            {"race_date": "2024-01-01", "stake": 10.0, "profit": 30.0, "won": 1,
             "bet_price": 4.0, "close_price": 3.5, "prob": 0.30, "bankroll_after": 1030.0},
            {"race_date": "2024-01-02", "stake": 10.0, "profit": -10.0, "won": 0,
             "bet_price": 4.0, "close_price": 4.5, "prob": 0.20, "bankroll_after": 1020.0},
        ])
        out = metrics.summarize_bets(led, 1000.0)
        assert out["n_bets"] == 2
        assert out["staked"] == pytest.approx(20.0)
        assert out["profit"] == pytest.approx(20.0)
        assert out["yield_pct"] == pytest.approx(100.0)
        assert out["roi_pct"] == out["yield_pct"]
        assert out["hit_rate"] == pytest.approx(0.5)
        assert out["final_bankroll"] == pytest.approx(1020.0)
        assert out["bankroll_growth_pct"] == pytest.approx(2.0)

    def test_clv_and_beat_close(self):
        # bet 4.0 vs close 3.5 -> beat the close; bet 4.0 vs close 4.5 -> didn't.
        led = _ledger([
            {"race_date": "2024-01-01", "stake": 10.0, "profit": 30.0, "won": 1,
             "bet_price": 4.0, "close_price": 3.5, "prob": 0.30, "bankroll_after": 1030.0},
            {"race_date": "2024-01-02", "stake": 10.0, "profit": -10.0, "won": 0,
             "bet_price": 4.0, "close_price": 4.5, "prob": 0.20, "bankroll_after": 1020.0},
        ])
        out = metrics.summarize_bets(led, 1000.0)
        assert out["beat_close_rate"] == pytest.approx(0.5)
        # mean CLV = mean(4/3.5-1, 4/4.5-1) * 100.
        expected = np.mean([4.0 / 3.5 - 1.0, 4.0 / 4.5 - 1.0]) * 100.0
        assert out["clv_pct_mean"] == pytest.approx(expected, abs=1e-3)
