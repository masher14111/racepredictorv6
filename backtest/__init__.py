"""Rigorous, leak-free walk-forward backtester for the value-betting layer.

Public API
----------
    from backtest import (
        load_panel, Backtester, BacktestConfig, WalkForwardConfig,
        Strategy, FlatStake, KellyStake, CatBoostFactory,
        render_report, save_run,
    )

    panel = load_panel()                       # bet candidates from training.parquet
    bt = Backtester(panel, CatBoostFactory(),  # price-free model, calibrated per fold
                    WalkForwardConfig(min_train_days=365, test_window_days=30))
    run = bt.run({
        "ev5_flat":  (Strategy("ev5", min_ev=0.05, min_odds=2, max_odds=26), FlatStake(10)),
        "edge_kelly":(Strategy("edge10", min_edge_pct=0.10, max_odds=26), KellyStake(0.25)),
    })
    print(render_report(run))
    save_run(run)                              # -> data/backtests/<ts>/ for the UI

Design invariants (see module docstrings for detail):
* the model is **price-free** — its probability never depends on the market price;
* bets execute at a **pre-off** price (ppwap/morningwap), CLV is scored vs the
  **closing** Betfair SP (odds_finish); the leaky training ``implied_prob`` is
  never used for the bet decision;
* training/calibration for a fold use only data on/before that fold's origin.
"""
from backtest.data import PanelConfig, load_panel
from backtest.engine import Backtester, BacktestConfig, BacktestRun
from backtest.metrics import (
    clv_pct,
    devig,
    edge,
    expected_value,
    implied_prob,
    kelly_fraction,
    max_drawdown,
    settle,
    summarize_bets,
)
from backtest.model import CallableModelFactory, CatBoostFactory
from backtest.report import compare_table, render_report, save_run
from backtest.splitter import Fold, WalkForwardConfig, walk_forward_folds
from backtest.staking import FlatStake, KellyStake, Strategy

__all__ = [
    "load_panel", "PanelConfig",
    "Backtester", "BacktestConfig", "BacktestRun",
    "WalkForwardConfig", "Fold", "walk_forward_folds",
    "Strategy", "FlatStake", "KellyStake",
    "CatBoostFactory", "CallableModelFactory",
    "render_report", "compare_table", "save_run",
    "implied_prob", "expected_value", "edge", "kelly_fraction", "devig",
    "settle", "clv_pct", "max_drawdown", "summarize_bets",
]
