"""End-to-end tests for backtest.engine with a deterministic stub model.

The headline test is the **no-leakage guarantee**: a spy factory records the
exact rows it was fitted on and asserts, at predict time, that no test row was
ever in a training slice and that every test race falls strictly after the
fold's training max date. The rest pin the bankroll simulation maths (flat
stake, settle, cumulative bankroll, stop-loss) and the dedup of overlapping
fold windows — all without booting CatBoost.
"""
import numpy as np
import pandas as pd
import pytest

from backtest import metrics
from backtest.engine import Backtester, BacktestConfig
from backtest.model import CallableModelFactory
from backtest.splitter import WalkForwardConfig
from backtest.staking import FlatStake, KellyStake, Strategy


# ── synthetic, leak-detectable panel ─────────────────────────────────────────

def _panel(days: int = 80, n_runners: int = 5) -> pd.DataFrame:
    """One venue/day, ``n_runners`` horses, horse 0 always wins.

    ``signal`` is a leak-safe feature the stub turns into a probability; the
    winner carries the highest signal so the model is informative but the data
    is fully deterministic.
    """
    dates = pd.date_range("2024-01-01", periods=days, freq="D", tz="UTC")
    sig = [0.40, 0.20, 0.15, 0.15, 0.10][:n_runners]
    odds = [2.5, 5.0, 6.0, 7.0, 10.0][:n_runners]
    rows = []
    for d in dates:
        for h in range(n_runners):
            rows.append({
                "race_date": d,
                "venue": "V",
                "horse_id": h,
                "horse_name": f"H{h}",
                "signal": sig[h],
                "won": 1 if h == 0 else 0,
                "bet_price": odds[h],
                "close_price": odds[h] * 0.95,  # closed shorter -> we beat the close
            })
    return pd.DataFrame(rows)


class _LeakSpy:
    """A fit/predict pair that audits the train/test boundary on every fold."""

    def __init__(self):
        self.fit_calls = 0
        self.violations: list[str] = []

    def fit(self, train_df: pd.DataFrame):
        self.fit_calls += 1
        return {
            "train_max": train_df["race_date"].max(),
            "train_keys": set(map(tuple, train_df[["race_date", "venue", "horse_id"]]
                                  .to_numpy().tolist())),
        }

    def predict(self, state, df: pd.DataFrame) -> np.ndarray:
        test_keys = set(map(tuple, df[["race_date", "venue", "horse_id"]].to_numpy().tolist()))
        if test_keys & state["train_keys"]:
            self.violations.append("train/test row overlap")
        if (df["race_date"] <= state["train_max"]).any():
            self.violations.append("test race on/before train_max (look-ahead)")
        return np.clip(df["signal"].to_numpy(dtype=float), 0.01, 0.99)


def _factory(spy: _LeakSpy) -> CallableModelFactory:
    return CallableModelFactory(fit_fn=spy.fit, predict_fn=spy.predict, feature_cols=["signal"])


# ── the no-leakage guarantee ──────────────────────────────────────────────────

class TestNoLeakage:
    def test_model_never_trains_on_test_rows(self):
        spy = _LeakSpy()
        bt = Backtester(_panel(), _factory(spy),
                        WalkForwardConfig(min_train_days=20, test_window_days=10))
        bt.run({"all": (Strategy("all"), FlatStake(10))})
        assert spy.fit_calls >= 2, "expected several walk-forward folds"
        assert spy.violations == [], f"leakage detected: {spy.violations}"

    def test_scored_frame_is_strictly_out_of_sample(self):
        # Every scored row's date must exceed the earliest possible origin
        # (min date + min_train_days) — nothing from the initial train block.
        spy = _LeakSpy()
        panel = _panel()
        bt = Backtester(panel, _factory(spy),
                        WalkForwardConfig(min_train_days=20, test_window_days=10))
        run = bt.run({"all": (Strategy("all"), FlatStake(10))})
        first_origin = panel["race_date"].min() + pd.Timedelta(days=20)
        assert (run.scored["race_date"] > first_origin).all()


# ── value signals attached to the scored frame ───────────────────────────────

class TestSignals:
    def test_ev_edge_kelly_match_formulas(self):
        spy = _LeakSpy()
        bt = Backtester(_panel(), _factory(spy),
                        WalkForwardConfig(min_train_days=20, test_window_days=10))
        run = bt.run({"all": (Strategy("all"), FlatStake(10))})
        s = run.scored
        p = s["prob"].to_numpy()
        d = s["bet_price"].to_numpy()
        assert np.allclose(s["ev"], p * d - 1.0)
        assert np.allclose(s["implied"], 1.0 / d)
        assert np.allclose(s["edge"], p - 1.0 / d)


# ── bankroll simulation maths ─────────────────────────────────────────────────

class TestSimulation:
    def test_flat_ledger_pnl_and_bankroll_are_consistent(self):
        spy = _LeakSpy()
        bt = Backtester(_panel(), _factory(spy),
                        WalkForwardConfig(min_train_days=20, test_window_days=10),
                        BacktestConfig(initial_bankroll=1000.0))
        run = bt.run({"flat": (Strategy("all"), FlatStake(10))})
        led = run.strategies["flat"]["ledger"]
        assert not led.empty
        # Flat stake is constant.
        assert (led["stake"] == 10.0).all()
        # Each profit equals settle(stake, price, won).
        expected = metrics.settle(led["stake"].to_numpy(), led["bet_price"].to_numpy(),
                                  led["won"].to_numpy())
        assert np.allclose(led["profit"].to_numpy(), expected)
        # bankroll_after is the running cumulative sum off the initial bankroll.
        assert np.allclose(led["bankroll_after"].to_numpy(),
                           1000.0 + np.cumsum(led["profit"].to_numpy()))

    def test_summary_profit_matches_ledger(self):
        spy = _LeakSpy()
        bt = Backtester(_panel(), _factory(spy),
                        WalkForwardConfig(min_train_days=20, test_window_days=10))
        run = bt.run({"flat": (Strategy("all"), FlatStake(10))})
        led = run.strategies["flat"]["ledger"]
        summary = run.strategies["flat"]["summary"]
        assert summary["profit"] == pytest.approx(led["profit"].sum(), abs=0.01)
        assert summary["n_bets"] == len(led)

    def test_stop_loss_halts_betting(self):
        # Backing ALL five runners flat (one short-priced winner, four losers)
        # bleeds the bankroll ~25/race, so it eventually hits the floor. The
        # stop-loss run must place fewer bets and preserve more capital than the
        # otherwise-identical run with no stop-loss. Training still has both
        # classes (horse 0 wins every race), so every fold scores.
        panel = _panel(days=200)
        wf = WalkForwardConfig(min_train_days=20, test_window_days=20)
        no_sl = Backtester(panel, _factory(_LeakSpy()), wf,
                           BacktestConfig(initial_bankroll=1000.0)).run(
            {"flat": (Strategy("all"), FlatStake(10))})
        with_sl = Backtester(panel, _factory(_LeakSpy()), wf,
                             BacktestConfig(initial_bankroll=1000.0, stop_loss_pct=0.20)).run(
            {"flat": (Strategy("all"), FlatStake(10))})

        led_no = no_sl.strategies["flat"]["ledger"]
        led_sl = with_sl.strategies["flat"]["ledger"]
        fin_no = no_sl.strategies["flat"]["summary"]["final_bankroll"]
        fin_sl = with_sl.strategies["flat"]["summary"]["final_bankroll"]

        assert len(led_sl) < len(led_no), "stop-loss should halt betting early"
        assert fin_sl > fin_no, "stop-loss should preserve capital"
        # Halts near the 800 floor; can't overshoot below by more than one race
        # of stakes (5 x €10).
        assert fin_sl >= 800.0 - 50.0

    def test_one_bet_per_runner_dedupes_overlapping_folds(self):
        # step < window => overlapping test windows => same runner scored twice.
        # The engine must keep one row per (race_date, venue, horse_id).
        spy = _LeakSpy()
        bt = Backtester(_panel(), _factory(spy),
                        WalkForwardConfig(min_train_days=20, test_window_days=20, step_days=10),
                        BacktestConfig(one_bet_per_runner=True))
        run = bt.run({"all": (Strategy("all"), FlatStake(10))})
        keys = run.scored[["race_date", "venue", "horse_id"]]
        assert not keys.duplicated().any()


# ── degenerate inputs ─────────────────────────────────────────────────────────

class TestGuards:
    def test_raises_without_race_date(self):
        with pytest.raises(ValueError):
            Backtester(pd.DataFrame({"x": [1]}), _factory(_LeakSpy()), WalkForwardConfig())

    def test_raises_when_span_too_short_for_a_fold(self):
        spy = _LeakSpy()
        bt = Backtester(_panel(days=10), _factory(spy),
                        WalkForwardConfig(min_train_days=365, test_window_days=30))
        with pytest.raises(ValueError):
            bt.run({"all": (Strategy("all"), FlatStake(10))})

    def test_model_metrics_present(self):
        spy = _LeakSpy()
        bt = Backtester(_panel(), _factory(spy),
                        WalkForwardConfig(min_train_days=20, test_window_days=10))
        run = bt.run({"all": (Strategy("all"), FlatStake(10))})
        m = run.model_metrics
        assert m["n"] > 0
        assert 0.0 <= m["auc"] <= 1.0
        assert m["base_rate"] == pytest.approx(0.2, abs=0.05)  # 1 winner / 5 runners

    def test_kelly_strategy_also_runs(self):
        spy = _LeakSpy()
        bt = Backtester(_panel(), _factory(spy),
                        WalkForwardConfig(min_train_days=20, test_window_days=10))
        run = bt.run({"kelly": (Strategy("ev", min_ev=0.0), KellyStake(0.25))})
        assert "kelly" in run.strategies
